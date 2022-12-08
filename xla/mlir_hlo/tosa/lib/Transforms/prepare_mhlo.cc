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

#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo_tosa/Transforms/passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TOSAPREPAREMHLOPASS
#include "mhlo_tosa/Transforms/passes.h.inc"

#define PASS_NAME "tosa-prepare-mhlo"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

class PrepareMhlo : public ::impl::TosaPrepareMhloPassBase<PrepareMhlo> {
 public:
  explicit PrepareMhlo() = default;
  void runOnOperation() override;
};

// Rewrites the DotGeneral op to a standard DotOp, for the case where there are
// no batch dimensions and the lhs contracting dimension equals lhs.rank - 1.
// The standard dot op only supports tensors of rank 1 and 2.
struct DotGeneralToDot : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    auto lhsType = op.getLhs().getType().dyn_cast<RankedTensorType>();
    auto rhsType = op.getRhs().getType().dyn_cast<RankedTensorType>();
    if (!lhsType | !rhsType) {
      return rewriter.notifyMatchFailure(op, "input tensors are not ranked");
    }

    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    if ((lhsRank != 1 && lhsRank != 2) || (rhsRank != 1 && rhsRank != 2)) {
      return failure();
    }

    auto dimensionAttr = op.getDotDimensionNumbers();
    if (!dimensionAttr.getLhsBatchingDimensions().empty()) return failure();
    if (!dimensionAttr.getRhsBatchingDimensions().empty()) return failure();

    auto lhsContractingDim = dimensionAttr.getLhsContractingDimensions();
    auto rhsContractingDim = dimensionAttr.getRhsContractingDimensions();

    if (lhsContractingDim.size() != 1 || rhsContractingDim.size() != 1)
      return failure();
    if (rhsContractingDim.front() != 0) return failure();

    if (lhsRank == 1 && lhsContractingDim.front() != 0) return failure();
    if (lhsRank == 2 && lhsContractingDim.front() != 1) return failure();

    rewriter.replaceOpWithNewOp<mhlo::DotOp>(
        op, op.getType(), op.getLhs(), op.getRhs(),
        op.getPrecisionConfig().value_or(nullptr));

    return success();
  }
};

void PrepareMhlo::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<DotGeneralToDot>(ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPrepareMhloPass() {
  return std::make_unique<PrepareMhlo>();
}

}  // namespace tosa
}  // namespace mlir
