#!/usr/bin/env bash
# =============================================================================
# Soccer Radar v2.0 - Model Download Script
# =============================================================================
# Downloads and prepares the exact ONNX models required:
#   1. soccana_object_720p.onnx  (Adit-jain/soccana)
#   2. soccana_keypoint_720p.onnx (Adit-jain/Soccana_Keypoint)
#   3. mobilenetv4_conv_small.onnx (onnx-community/mobilenetv4_conv_small)
# =============================================================================

set -e
cd "$(dirname "$0")"

mkdir -p models

echo "Running model setup (setup_models.py)..."
python3 setup_models.py

echo "Model setup complete."
