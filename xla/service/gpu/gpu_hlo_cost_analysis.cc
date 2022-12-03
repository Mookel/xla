/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gpu_hlo_cost_analysis.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla {
namespace gpu {

// Use the "reserved" keys for these properties so lookups are fast.
static constexpr absl::string_view kIRSizeKey = HloCostAnalysis::kReserved0Key;
static constexpr absl::string_view kBasicBlockSplitCountKey =
    HloCostAnalysis::kReserved1Key;

Status GpuHloCostAnalysis::Preprocess(const HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(HloCostAnalysis::Preprocess(hlo));

  current_properties_[kIRSizeKey] = 1;
  current_properties_[kBasicBlockSplitCountKey] =
      ElementalIrEmitter::OpInvalidatesCache(hlo);

  return OkStatus();
}

int64_t GpuHloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  float utilization = hlo_properties_.at(hlo)[kUtilizationKey];
  if (!options_.count_multiple_input_accesses) {
    utilization = fmin(utilization, 1.0);
  }
  return GetShapeSize(hlo->shape()) * utilization;
}

Status GpuHloCostAnalysis::FusionCalculateUtilizations(
    const HloInstruction* fusion) {
  const HloInstruction* root = fusion->fused_expression_root();
  // Traverse through the computation from the root till parameters propagating
  // the utilization of operands; store utilization of each node
  // in hlo_properties_. All consumers of an instruction are processed before
  // the instruction itself.
  std::vector<HloInstruction*> instructions =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instructions);

  // To estimate where within the computation an instruction output can be
  // reused and where it has to be recomputed again we group accesses to the
  // instruction by their origin from "element-wise use roots". All access
  // paths from such a root to the instruction are element-wise.
  // Whenever we account a non-element-wise operation we forget about
  // element-wise roots encountered so far and provisionally set its operands
  // as new element-wise roots.
  struct InstrInfo {
    absl::flat_hash_set<const InstrInfo*> elementwise_use_roots;
    int64_t ir_size = 0;
    float utilization = 0;
  };
  absl::node_hash_map<const HloInstruction*, InstrInfo> infos;

  // For the purpose of operand utilization analysis, no matter how the fusion
  // outputs are used, we assume that fusion is always executed completely
  // producing 100% of its outputs.
  infos[root] = {/*elementwise_use_roots=*/{}, /*ir_size=*/1,
                 /*utilization=*/1.0};
  infos[root].elementwise_use_roots.insert(&infos[root]);

  current_properties_[kFlopsKey] = 0;
  current_properties_[kBasicBlockSplitCountKey] = 0;
  current_properties_[kIRSizeKey] = 0;

  for (const HloInstruction* instr : instructions) {
    Properties& props = hlo_properties_[instr];
    InstrInfo& info = infos[instr];

    // Overwrite IRSize = 1 from Preprocess().
    props[kIRSizeKey] = 0;

    for (const InstrInfo* root_info : info.elementwise_use_roots) {
      props[kUtilizationKey] += root_info->utilization;
      props[kIRSizeKey] += root_info->ir_size;
    }

    VLOG(8) << instr->name() << ":";
    VLOG(8) << "Total utilization: " << props[kUtilizationKey];
    VLOG(8) << "Times emitted: " << props[kIRSizeKey];

    current_properties_[kFlopsKey] += props[kUtilizationKey] * props[kFlopsKey];
    current_properties_[kIRSizeKey] += props[kIRSizeKey];
    current_properties_[kBasicBlockSplitCountKey] +=
        props[kIRSizeKey] * ElementalIrEmitter::OpInvalidatesCache(instr);

    for (int operand_idx = 0; operand_idx < instr->operand_count();
         ++operand_idx) {
      const HloInstruction* operand = instr->operand(operand_idx);
      if (instr->IsElementwise() || instr->opcode() == HloOpcode::kTuple ||
          instr->opcode() == HloOpcode::kGetTupleElement) {
        auto& operand_use_roots = infos[operand].elementwise_use_roots;
        for (const InstrInfo* r : info.elementwise_use_roots) {
          operand_use_roots.insert(r);
        }
      } else {
        float cur_operand_utilization =
            props[kUtilizationKey] * operand_utilization(*instr, operand_idx);
        // The utilization is always a best-effort estimate, but in some cases
        // cannot be precise due to dynamic nature of operations - dynamic
        // slice is one such example. We do an average estimate in these
        // cases and this can sometimes produce fractional utilizations which
        // should be at least rounded up to a whole number of produced elements
        // to be more realistic.
        int64_t operand_elements =
            ShapeUtil::ElementsInRecursive(operand->shape());
        cur_operand_utilization =
            std::ceil(cur_operand_utilization * operand_elements) /
            operand_elements;

        auto& operand_info = infos[operand];
        operand_info.utilization += cur_operand_utilization;
        operand_info.ir_size += props[kIRSizeKey];
        operand_info.elementwise_use_roots.insert(&operand_info);
      }
    }
  }

  return OkStatus();
}

bool GpuHloCostAnalysis::ProducerConsumerMergedTooLarge(
    const HloInstruction& producer, const HloInstruction& consumer) {
  int64_t producer_replication = 1;
  // Fusing 'producer' into 'consumer' fusion currently results in replicating
  // its IR the number of times the consumer replicates the access
  // to the parameter corresponding to the producer.
  if (consumer.opcode() == HloOpcode::kFusion) {
    producer_replication =
        IrSize(*consumer.fused_parameter(consumer.operand_index(&producer)));
  }
  VLOG(5) << producer.name() << " would be emitted by " << consumer.name()
          << " x" << producer_replication;
  int64_t n_splits = producer_replication * IrBasicBlockSplitCount(producer) +
                     IrBasicBlockSplitCount(consumer);
  VLOG(5) << "Basic block split counts: " << IrBasicBlockSplitCount(producer)
          << ", " << IrBasicBlockSplitCount(consumer) << " -> " << n_splits;
  if (n_splits > kMaxBasicBlockSplitsPerFusion) {
    return true;
  }
  int64_t merged_ir_size =
      (IrSize(producer) * producer_replication + IrSize(consumer)) *
      (1 << n_splits);
  VLOG(5) << "IR sizes: " << IrSize(producer) << ", " << IrSize(consumer)
          << " -> " << merged_ir_size;
  return merged_ir_size > kMaxIRSize;
}

Status GpuHloCostAnalysis::HandleCustomCall(const HloInstruction* custom_call) {
  if (IsCublasGemm(*custom_call)) {
    // The naming conventions and meanings of gemm parameters are documented
    // here:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    TF_ASSIGN_OR_RETURN(auto gemm_config,
                        custom_call->backend_config<gpu::GemmBackendConfig>());

    // Technically, in addition to the dot product (A * B), cuBLAS gemm also
    // performs additional scaling (by factor 'alpha') and addition with a
    // scaled third matrix (beta * C), which will introduce additional
    // multiplications and additions. But total FLOPS will be dominated by the
    // dot product, so we don't include these extra multiplications and
    // additions in the FLOPS calculation.

    // Also, this calculation assumes that the strides for the gemm are
    // properly set such that none of the inputs in a batch overlap with any
    // other batches. If they do, this will undercount the FLOPS, because it
    // assumes that the strides are implicit in the sizes of the batch
    // dimensions.

    // Finally, this is technically incorrect if the element type of this
    // gemm is an integer type, because in that case no floating point
    // operations are involved at all! But we still calculate FLOPS because the
    // number is sometimes required for ad-hoc calculations.

    // cublasLt supports auxiliary outputs, so output may be tuple.
    const Shape& output_shape = custom_call->shape().IsTuple()
                                    ? custom_call->shape().tuple_shapes(0)
                                    : custom_call->shape();

    current_properties_[kFlopsKey] =
        GetDotFlops(custom_call->operand(0)->shape(), output_shape,
                    gemm_config.dot_dimension_numbers());
    return OkStatus();
  }

  if (IsCustomCallToDnnConvolution(*custom_call)) {
    // As with dots, this flops calculation has the following inaccuracies.
    //
    //  - We may have a fused conv which does additional ops (multiplying by a
    //    scalar `alpha`, adding a bias or side-input, doing a relu, etc).  But
    //    we can safely ignore this because the overall computation is dominated
    //    by the convolution itself.
    //
    //  - cudnn may use complex conv algorithms that do fewer (or more!) flops
    //    than we calculate.
    //
    //  - for int8_t convs, these aren't *fl*ops, but we fudge it.
    current_properties_[kFlopsKey] = GetConvolutionFlops(custom_call);

    // conv custom-calls return a tuple (real_output, temp_bytes).  Count just
    // the real_output in output bytes accessed.  The main purpose of
    // hlo_cost_analysis is to figure out if ops are running "as fast as
    // possible", and if we were to include temp memory in here, we'd
    // essentially be *rewarding* convs that use additional temp memory!
    if (custom_call->shape().IsTuple()) {
      float output_size =
          options_.shape_size(custom_call->shape().tuple_shapes(0));
      // 'Bytes accessed' are estimated in HloCostAnalysis::Preprocess() as
      // input + output. As the output size is being adjusted here it has
      // to propagate to the total bytes accessed.
      current_properties_[kBytesAccessedKey] -=
          current_properties_.output_bytes_accessed();
      current_properties_.set_output_bytes_accessed(output_size);
      current_properties_[kBytesAccessedKey] += output_size;
    }
    return OkStatus();
  }

  return HloCostAnalysis::HandleCustomCall(custom_call);
}

int64_t GpuHloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = [&]() -> const Shape& {
    // convolution custom-calls return a tuple of (actual_result, temp_buffer).
    const Shape& shape = convolution->shape();
    if (IsCustomCallToDnnConvolution(*convolution) &&
        convolution->shape().IsTuple()) {
      return shape.tuple_shapes(0);
    }
    return shape;
  }();

  return HloCostAnalysis::GetConvolutionFlops(convolution, lhs_shape, rhs_shape,
                                              result_shape);
}

Status GpuHloCostAnalysis::HandleElementwiseOp(const HloInstruction* hlo) {
  const HloOpcode opcode = hlo->opcode();
  const auto& shape = hlo->shape();
  const PrimitiveType type = shape.element_type();

  // These are clock cycle estimates of some of the most common expensive
  // operations. They most likely vary a lot from GPU to GPU but should
  // at least provide reasonable comparisons for the computation cost analysis.
  // HLOs used to measure these can be found in gpu_performance_model_test.cc
  // This list is far from complete yet.
  // TODO(b/256570878): Make a tool to measure these numbers and store them
  // separately from the code where possible.

  // Typical elementwise instructions take about 3 clock cycles.
  int64_t flop_per_element = 3;
  switch (opcode) {
    case HloOpcode::kTanh:
      if (type == F32) {
        flop_per_element = 30;
      } else if (type == F64) {
        flop_per_element = 2000;
      }
      break;
    case HloOpcode::kDivide:
      if (type == S32) {
        flop_per_element = 80;
      } else if (type == F64) {
        flop_per_element = 3200;
      } else if (type == C128) {
        flop_per_element = 20000;
      }
      break;
    // Expands to multiple instructions.
    case HloOpcode::kExp:
      if (type == F64) {
        flop_per_element = 2200;
      }
      break;
    case HloOpcode::kSqrt:
      if (type == F64) {
        flop_per_element = 1100;
      } else if (type == C128) {
        flop_per_element = 25000;
      }
      break;
    case HloOpcode::kRsqrt:
      if (type == F64) {
        flop_per_element = 900;
      }
      break;
    case HloOpcode::kAdd:
      if (type == F64) {
        flop_per_element = 120;
      } else if (type == C128) {
        flop_per_element = 240;
      }
      break;
    case HloOpcode::kMultiply:
      if (type == F64) {
        flop_per_element = 120;
      } else if (type == C128) {
        flop_per_element = 650;
      }
      break;
    case HloOpcode::kPower:
      if (type == F64) {
        flop_per_element = 11000;
      } else if (type == C128) {
        flop_per_element = 28000;
      }
      break;
    case HloOpcode::kLog:
      if (type == F32) {
        flop_per_element = 45;
      } else if (type == F64) {
        flop_per_element = 1000;
      }
      break;
    default:
      // Raise default cost of all unlisted F64 and C128 ops.
      if (type == F64) {
        flop_per_element = 10;
      } else if (type == C128) {
        flop_per_element = 20;
      }
      break;
  }
  current_properties_[kFlopsKey] =
      flop_per_element * ShapeUtil::ElementsInRecursive(shape);
  return OkStatus();
}

Status GpuHloCostAnalysis::HandleElementwiseUnary(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status GpuHloCostAnalysis::HandleElementwiseBinary(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

std::unique_ptr<HloCostAnalysis>
GpuHloCostAnalysis::CreateNestedCostAnalysis() {
  return std::make_unique<GpuHloCostAnalysis>(options_);
}

bool GpuHloCostAnalysis::KeyToCopyFromSubcomputation(
    absl::string_view key) const {
  return !absl::StartsWith(key, kBytesAccessedKey) &&
         !absl::StartsWith(key, kUtilizationKey) &&
         !absl::StartsWith(key, kIRSizeKey) &&
         !absl::StartsWith(key, kBasicBlockSplitCountKey);
}

float GpuHloCostAnalysis::IrBasicBlockSplitCount(
    const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kBasicBlockSplitCountKey, hlo_properties_);
}

float GpuHloCostAnalysis::IrSize(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kIRSizeKey, hlo_properties_);
}

}  // namespace gpu
}  // namespace xla
