#!/data/data/com.termux/files/usr/bin/bash
# =============================================================================
# Soccer Radar v2.0 - Termux Build Script
# =============================================================================
# This script sets up and builds Soccer Radar on Termux (Android/ARM64).
#
# Usage:
#   chmod +x build_termux.sh
#   ./build_termux.sh
# =============================================================================

set -e

echo "=== Soccer Radar v2.0 - Termux Build Script ==="
echo ""

# Step 1: Update packages
echo "[1/5] Checking Termux packages..."
pkg update -y
pkg install -y cmake make clang opencv git wget python || true

# Step 2: Install ONNX Runtime
echo ""
echo "[2/5] Setting up ONNX Runtime..."

ONNXRUNTIME_VERSION="1.17.1"
ONNXRUNTIME_DIR="${HOME}/onnxruntime"

if [ ! -d "${ONNXRUNTIME_DIR}" ]; then
    echo "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} for ARM64..."

    if pkg show onnxruntime &>/dev/null; then
        pkg install -y onnxruntime || true
        echo "ONNX Runtime checked from Termux repos"
    fi

    ARCH=$(uname -m)
    if [ ! -d "${ONNXRUNTIME_DIR}" ] && [ ! -f "/data/data/com.termux/files/usr/lib/libonnxruntime.so" ]; then
        if [ "${ARCH}" = "aarch64" ]; then
            ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
        else
            ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
        fi

        mkdir -p "${ONNXRUNTIME_DIR}"
        cd "${ONNXRUNTIME_DIR}"
        wget -q "${ONNXRUNTIME_URL}" -O onnxruntime.tgz
        tar xzf onnxruntime.tgz --strip-components=1
        rm onnxruntime.tgz
        cd -
        echo "ONNX Runtime extracted to ${ONNXRUNTIME_DIR}"
    fi
else
    echo "ONNX Runtime already present at ${ONNXRUNTIME_DIR}"
fi

# Step 3: Check/Download models
echo ""
echo "[3/5] Checking required models..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
python3 setup_models.py || true

# Step 4: Configure and Build Soccer Radar
echo ""
echo "[4/5] Building Soccer Radar..."

BUILD_DIR="${SCRIPT_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake with explicit Termux OpenCV and ONNX Runtime paths
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

if [ -f "/data/data/com.termux/files/usr/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DOpenCV_DIR=/data/data/com.termux/files/usr/lib/cmake/opencv4"
fi

if [ -d "${ONNXRUNTIME_DIR}/include" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DONNXRUNTIME_ROOT=${ONNXRUNTIME_DIR}"
elif [ -f "/data/data/com.termux/files/usr/include/onnxruntime/onnxruntime_c_api.h" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DONNXRUNTIME_ROOT=/data/data/com.termux/files/usr"
fi

echo "Running CMake: cmake ${CMAKE_ARGS} .. "
cmake ${CMAKE_ARGS} "${SCRIPT_DIR}"

# Build
NPROC=$(nproc 2>/dev/null || echo 2)
echo "Compiling with ${NPROC} threads..."
make -j${NPROC}

echo ""
echo "[5/5] Build complete!"
echo ""
echo "Binary location: ${BUILD_DIR}/soccer_radar"
echo ""
echo "Usage:"
echo "  cd ${SCRIPT_DIR}"
echo "  ./build/soccer_radar --video videos/test_720p.mp4"
