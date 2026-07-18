cmake_minimum_required(VERSION 3.14)
project(SoccerRadar VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimizations for high-performance mobile/desktop C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -ffast-math -fomit-frame-pointer -Wno-nan-infinity-disabled")

# Architecture optimization (auto-detect ARM64 / Termux NEON SIMD)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "ARM64 Architecture detected. Enabling NEON intrinsics.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d)

# Locate ONNX Runtime seamlessly across Termux, command line (-DONNXRUNTIME_ROOT=...), environment variables, or Linux paths
if(DEFINED ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}")
    # Retain user passed -DONNXRUNTIME_ROOT=...
elseif(DEFINED ENV{ONNXRUNTIME_ROOT} AND EXISTS "$ENV{ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
elseif(EXISTS "/data/data/com.termux/files/usr/include/onnxruntime")
    set(ONNXRUNTIME_ROOT "/data/data/com.termux/files/usr")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/onnxruntime")
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
elseif(EXISTS "/usr/local/include/onnxruntime")
    set(ONNXRUNTIME_ROOT "/usr/local")
else()
    set(ONNXRUNTIME_ROOT "/usr")
endif()

message(STATUS "Using ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

# Comprehensive list of all possible OpenCV header directories
set(ALL_OPENCV_INCLUDE_PATHS
    ${OpenCV_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
    "/data/data/com.termux/files/usr/include/opencv4"
    "/data/data/com.termux/files/usr/include"
    "/usr/include/opencv4"
    "/usr/local/include/opencv4"
    "/opt/homebrew/include/opencv4"
)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ALL_OPENCV_INCLUDE_PATHS}
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime
)

add_executable(soccer_radar
    src/main.cpp
    src/utils/letterbox.cpp
    src/utils/video_io.cpp
    src/object_detection/object_detector.cpp
    src/keypoint_detection/keypoint_detector.cpp
    src/object_tracking/tracker.cpp
    src/object_annotations/annotator.cpp
    src/player_clustering/embeddings.cpp
    src/player_clustering/clustering.cpp
    src/tactical_analysis/homography.cpp
    src/pipelines/processing_pipeline.cpp
    src/pipelines/detection_pipeline.cpp
    src/pipelines/keypoint_pipeline.cpp
    src/pipelines/tracking_pipeline.cpp
    src/pipelines/tactical_pipeline.cpp
)

target_include_directories(soccer_radar PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ALL_OPENCV_INCLUDE_PATHS}
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime
)

set(OPENCV_TERMUX_SO
    "/data/data/com.termux/files/usr/lib/libopencv_core.so"
    "/data/data/com.termux/files/usr/lib/libopencv_imgproc.so"
    "/data/data/com.termux/files/usr/lib/libopencv_imgcodecs.so"
    "/data/data/com.termux/files/usr/lib/libopencv_videoio.so"
    "/data/data/com.termux/files/usr/lib/libopencv_calib3d.so"
)

if(EXISTS "/data/data/com.termux/files/usr/lib/libopencv_core.so")
    set(FINAL_OPENCV_LIBS ${OPENCV_TERMUX_SO} ${OpenCV_LIBS} ${OpenCV_LIBRARIES})
else()
    set(FINAL_OPENCV_LIBS ${OpenCV_LIBS} ${OpenCV_LIBRARIES})
endif()

file(GLOB ONNX_SO "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
file(GLOB ONNX_DYLIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib*")

if(ONNX_SO)
    list(GET ONNX_SO 0 FIRST_ONNX_SO)
    target_link_libraries(soccer_radar PRIVATE ${FINAL_OPENCV_LIBS} "${FIRST_ONNX_SO}")
elseif(ONNX_DYLIB)
    list(GET ONNX_DYLIB 0 FIRST_ONNX_DYLIB)
    target_link_libraries(soccer_radar PRIVATE ${FINAL_OPENCV_LIBS} "${FIRST_ONNX_DYLIB}")
else()
    target_link_libraries(soccer_radar PRIVATE ${FINAL_OPENCV_LIBS} onnxruntime)
endif()
