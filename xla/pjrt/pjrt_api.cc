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

#include "xla/pjrt/pjrt_api.h"

#include <dlfcn.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"

namespace pjrt {

static auto* pjrt_apis =
    new absl::flat_hash_map<std::string, const PJRT_Api*>{};

static std::string CanonicalizeDeviceType(absl::string_view device_type) {
  return absl::AsciiStrToLower(device_type);
}

xla::StatusOr<const PJRT_Api*> PjrtApi(absl::string_view device_type) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  auto iter = pjrt_apis->find(canonicalize_device_type);

  // TODO(b/261601433): the block below is for backward compatibiality. Remove
  // this block after pytorch adds the call to LoadPjrtPlugin.
  if (iter == pjrt_apis->end() && canonicalize_device_type == "tpu") {
    const char* env_value = getenv("TPU_LIBRARY_PATH");
    const char* libtpu_path =
        env_value && strlen(env_value) > 0 ? env_value : "libtpu.so";
    TF_RETURN_IF_ERROR(LoadPjrtPlugin("tpu", libtpu_path));
    iter = pjrt_apis->find("tpu");
  }

  if (iter == pjrt_apis->end()) {
    return tsl::errors::NotFound("PJRT_Api not found for device type ",
                                 canonicalize_device_type);
  }
  return iter->second;
}

xla::Status SetPjrtApi(absl::string_view device_type, const PJRT_Api* api) {
  std::string canonicalize_device_type = CanonicalizeDeviceType(device_type);
  if (auto iter = pjrt_apis->find(canonicalize_device_type);
      iter != pjrt_apis->end()) {
    return tsl::errors::AlreadyExists(
        "PJRT_Api already exists for device type ", canonicalize_device_type);
  }
  (*pjrt_apis)[canonicalize_device_type] = api;
  LOG(INFO) << "PJRT_Api is set for device type " << canonicalize_device_type;
  return tsl::OkStatus();
}

xla::Status LoadPjrtPlugin(absl::string_view device_type,
                           absl::string_view library_path) {
  void* library = dlopen(library_path.data(), RTLD_NOW);
  if (library == nullptr) {
    return tsl::errors::Internal("Failed to open ", library_path);
  }
  const PJRT_Api* (*fptr)();
  *reinterpret_cast<void**>(&fptr) = dlsym(library, "GetPjrtApi");
  if (fptr == nullptr) {
    return tsl::errors::NotFound("GetPjrtApi not found in ", library_path);
  }
  LOG(INFO) << "GetPjrtApi was found for " << device_type << " at "
            << library_path;

  const PJRT_Api* pjrt_api = fptr();
  TF_RETURN_IF_ERROR(pjrt::CheckMatchingStructSizes(
      "PJRT_Api", PJRT_Api_STRUCT_SIZE, pjrt_api->struct_size));
  TF_RETURN_IF_ERROR(pjrt::SetPjrtApi(device_type, pjrt_api));
  return tsl::OkStatus();
}

}  // namespace pjrt