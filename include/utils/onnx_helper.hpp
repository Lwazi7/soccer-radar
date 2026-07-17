#pragma once

#include <onnxruntime_cxx_api.h>
#include <dlfcn.h>
#include <iostream>
#include <thread>
#include <algorithm>

namespace soccer_radar {

inline void configure_session_options(Ort::SessionOptions& session_options, int num_threads = 4) {
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads <= 0) hw_threads = num_threads;
    
    // Set up to 4 threads for optimal performance core utilization on mobile ARM big.LITTLE CPUs
    session_options.SetIntraOpNumThreads(std::min(4, hw_threads));
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();

    // Dynamically attempt to enable XNNPACK or NNAPI without requiring compile-time factory headers
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
