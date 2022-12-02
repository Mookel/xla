"""Wrapper around proto libraries used inside the XLA codebase."""

load(
    "@tsl//:tsl.bzl",
    "clean_dep",
    "if_tsl_link_protobuf",
)
load("@tsl//platform:build_config.bzl", "tsl_cc_test")
load(
    "//third_party/tensorflow:tensorflow.bzl",
    "tf_copts",
)

def xla_py_proto_library(**kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    _ignore = kwargs
    pass

def xla_py_grpc_library(**kwargs):
    # Note: we don't currently define any special targets for Python GRPC in OSS.
    _ignore = kwargs
    pass

ORC_JIT_MEMORY_MAPPER_TARGETS = []

def xla_py_test_deps():
    return []

def xla_cc_binary(deps = None, **kwargs):
    if not deps:
        deps = []

    # TODO(ddunleavy): some of these should be removed from here and added to
    # specific targets.
    deps += [
        clean_dep("//google/protobuf"),
        "//xla:xla_proto_cc_impl",
        "//xla:xla_data_proto_cc_impl",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service/gpu:backend_configs_cc_impl",
        "//xla/stream_executor:dnn_proto_cc_impl",
        "@tsl//platform:env_impl",
        "@tsl//profiler/utils:time_utils_impl",
        "@tsl//profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//protobuf:protos_all_cc_impl",
    ]
    native.cc_binary(deps = deps, **kwargs)

def xla_cc_test(
        name,
        copts = [],
        deps = [],
        **kwargs):
    tsl_cc_test(
        name = name,
        copts = copts + tf_copts(),
        deps = deps + if_tsl_link_protobuf(
            [],
            [
                # TODO(zacmustin): remove these in favor of more granular dependencies in each test.
                "//xla:xla_proto_cc_impl",
                "//xla:xla_data_proto_cc_impl",
                "//xla/service:hlo_proto_cc_impl",
                "//xla/service/gpu:backend_configs_cc_impl",
                "//xla/stream_executor:device_description_proto_cc_impl",
                "//xla/stream_executor:dnn_proto_cc_impl",
                "//xla/stream_executor:stream_executor_impl",
                "//xla/stream_executor/cuda:cublas_plugin",
                "//xla/stream_executor/gpu:gpu_init_impl",
                "//xla/stream_executor/host:host_platform",
                "//xla/stream_executor/host:host_gpu_executor",
                "//xla/stream_executor/host:host_platform_id",
                "@tsl//framework:allocator",
                "@tsl//framework:allocator_registry_impl",
                "@tsl//platform:env_impl",
                "@tsl//platform:tensor_float_32_utils",
                "@tsl//profiler/utils:time_utils_impl",
                "@tsl//profiler/backends/cpu:annotation_stack_impl",
                "@tsl//profiler/backends/cpu:traceme_recorder_impl",
                "@tsl//protobuf:autotuning_proto_cc_impl",
                "@tsl//protobuf:dnn_proto_cc_impl",
                "@tsl//protobuf:protos_all_cc_impl",
                "@tsl//util:determinism",
            ],
        ),
        **kwargs
    )
