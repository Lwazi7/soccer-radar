#pragma once

#include <onnxruntime_cxx_api.h>
#include <dlfcn.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <string>
#include <cstring>
#include <cstdint>

namespace soccer_radar {

// Fast inline FP32 to FP16 conversion using ONNX Runtime built-in C++ API
inline Ort::Float16_t float_to_half(float f) {
    return Ort::Float16_t(f);
}

// Fast inline FP16 to FP32 conversion using ONNX Runtime built-in C++ API
inline float half_to_float(Ort::Float16_t fp16_val) {
    return fp16_val.ToFloat();
}

inline void configure_session_options(Ort::SessionOptions& session_options, int num_threads = 4) {
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads <= 0) hw_threads = num_threads;
    
    int target_threads = std::min(4, hw_threads);
    session_options.SetIntraOpNumThreads(target_threads);
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();

    // Dynamically attempt XNNPACK (NEON SIMD) and NNAPI (Android NPU) acceleration via dlsym
    typedef OrtStatus* (*AppendFn)(OrtSessionOptions*, const void*);
    typedef OrtStatus* (*AppendNnapiFn)(OrtSessionOptions*, uint32_t);

    void* handle = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("libonnxruntime.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = RTLD_DEFAULT;

    AppendFn xnnpack_fn = (AppendFn)dlsym(handle, "OrtSessionOptionsAppendExecutionProvider_XNNPACK");
    if (xnnpack_fn) {
        OrtStatus* status = xnnpack_fn(session_options, nullptr);
        if (status == nullptr) {
            std::cout << "[Hardware] XNNPACK execution provider enabled dynamically via dlsym!" << std::endl;
            if (handle != RTLD_DEFAULT && handle) dlclose(handle);
            return;
        } else {
            Ort::GetApi().ReleaseStatus(status);
        }
    }

    AppendNnapiFn nnapi_fn = (AppendNnapiFn)dlsym(handle, "OrtSessionOptionsAppendExecutionProvider_Nnapi");
    if (nnapi_fn) {
        OrtStatus* status = nnapi_fn(session_options, 0);
        if (status == nullptr) {
            std::cout << "[Hardware] NNAPI (Android NPU) execution provider enabled dynamically via dlsym!" << std::endl;
            if (handle != RTLD_DEFAULT && handle) dlclose(handle);
            return;
        } else {
            Ort::GetApi().ReleaseStatus(status);
        }
    }

    if (handle != RTLD_DEFAULT && handle) dlclose(handle);
}

} // namespace soccer_radar
