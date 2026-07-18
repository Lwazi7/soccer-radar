#!/usr/bin/env python3
"""
Remaining 2 Models FP16 Exporter & Downloader (Colab / Standalone)
==================================================================
Converts and downloads the remaining two models in clean FP16 (16-bit float) format:
1. `football_tv2radar.onnx` (~3.1 MB in FP16) from `amirrezarajabi/Football-TV2Radar`.
2. `mobilenetv4_conv_small.onnx` (~5.2 MB in FP16) from `onnx-community/mobilenetv4_conv_small.e2400_r224_in1k`.

Google Colab Usage (Copy & paste into a Colab cell):
----------------------------------------------------
!pip install -q ultralytics onnx onnxsim onnxconverter-common

import urllib.request
import shutil
import os
from ultralytics import YOLO

# =============================================================================
# MODEL 1: Football-TV2Radar Corner Detector -> FP16 ONNX (~3.1 MB)
# =============================================================================
url_pt = "https://raw.githubusercontent.com/amirrezarajabi/Football-TV2Radar/main/corner_detection_model.pt"
pt_filename = "corner_detection_model.pt"
tv2radar_onnx = "football_tv2radar.onnx"

print(f"[1/4] Downloading {pt_filename} from GitHub...")
urllib.request.urlretrieve(url_pt, pt_filename)

print("\n[2/4] Exporting Football-TV2Radar to static 1280x736 FP16 ONNX...")
model = YOLO(pt_filename)
exported_path = model.export(
    format="onnx",
    imgsz=(736, 1280),
    simplify=True,
    dynamic=False,
    half=True,    # <--- FP16 (Half precision: 2x smaller, native NPU acceleration)
    opset=12
)
shutil.move(exported_path, tv2radar_onnx)
if os.path.exists(pt_filename): os.remove(pt_filename)
print(f"      Saved {tv2radar_onnx} ({os.path.getsize(tv2radar_onnx) / (1024*1024):.2f} MB)")

# =============================================================================
# MODEL 2: MobileNetV4 Conv Small Embeddings -> FP16 ONNX (~5.2 MB)
# =============================================================================
print("\n[3/4] Fetching MobileNetV4 FP16 ONNX model from HuggingFace...")
# HuggingFace directly hosts an exact FP16 export in `onnx/model_fp16.onnx`
url_mobilenet_fp16 = "https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/onnx/model_fp16.onnx"
mobilenet_onnx = "mobilenetv4_conv_small.onnx"

try:
    urllib.request.urlretrieve(url_mobilenet_fp16, mobilenet_onnx)
    print(f"      Saved {mobilenet_onnx} ({os.path.getsize(mobilenet_onnx) / (1024*1024):.2f} MB)")
except Exception as e:
    print(f"      Direct FP16 fetch failed ({e}). Downloading FP32 and converting to FP16 via onnxconverter-common...")
    url_fp32 = "https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/onnx/model.onnx"
    fp32_tmp = "mobilenet_fp32_tmp.onnx"
    urllib.request.urlretrieve(url_fp32, fp32_tmp)
    import onnx
    from onnxconverter_common import float16
    model_fp32 = onnx.load(fp32_tmp)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, mobilenet_onnx)
    if os.path.exists(fp32_tmp): os.remove(fp32_tmp)
    print(f"      Converted and saved {mobilenet_onnx} ({os.path.getsize(mobilenet_onnx) / (1024*1024):.2f} MB)")

# =============================================================================
# DOWNLOAD BOTH MODELS IN GOOGLE COLAB
# =============================================================================
print("\n[4/4] Triggering browser downloads in Google Colab...")
try:
    from google.colab import files
    files.download(tv2radar_onnx)
    files.download(mobilenet_onnx)
except ImportError:
    print(f"      Not in Google Colab. Models saved locally at:\n        - {os.path.abspath(tv2radar_onnx)}\n        - {os.path.abspath(mobilenet_onnx)}")
"""

import os
import sys
import shutil
import urllib.request

def main():
    print("=== Remaining 2 Models FP16 Exporter & Downloader ===")
    try:
        from ultralytics import YOLO
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q ultralytics onnx onnxsim onnxconverter-common")
        from ultralytics import YOLO

    # 1. Football-TV2Radar -> FP16
    url_pt = "https://raw.githubusercontent.com/amirrezarajabi/Football-TV2Radar/main/corner_detection_model.pt"
    pt_filename = "corner_detection_model.pt"
    tv2radar_onnx = "football_tv2radar.onnx"

    print(f"\n[1/4] Downloading {pt_filename} from GitHub...")
    urllib.request.urlretrieve(url_pt, pt_filename)

    print("\n[2/4] Exporting Football-TV2Radar to static 1280x736 FP16 ONNX...")
    model = YOLO(pt_filename)
    exported_path = model.export(
        format="onnx",
        imgsz=(736, 1280),
        simplify=True,
        dynamic=False,
        half=True,
        opset=12
    )
    shutil.move(exported_path, tv2radar_onnx)
    if os.path.exists(pt_filename): os.remove(pt_filename)
    print(f"      Saved {tv2radar_onnx} ({os.path.getsize(tv2radar_onnx) / (1024*1024):.2f} MB)")

    # 2. MobileNetV4 -> FP16
    print("\n[3/4] Fetching MobileNetV4 FP16 ONNX model from HuggingFace...")
    url_mobilenet_fp16 = "https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/onnx/model_fp16.onnx"
    mobilenet_onnx = "mobilenetv4_conv_small.onnx"

    try:
        urllib.request.urlretrieve(url_mobilenet_fp16, mobilenet_onnx)
        print(f"      Saved {mobilenet_onnx} ({os.path.getsize(mobilenet_onnx) / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"      Direct FP16 fetch failed ({e}). Converting FP32 via onnxconverter-common...")
        url_fp32 = "https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/onnx/model.onnx"
        fp32_tmp = "mobilenet_fp32_tmp.onnx"
        urllib.request.urlretrieve(url_fp32, fp32_tmp)
        import onnx
        from onnxconverter_common import float16
        model_fp32 = onnx.load(fp32_tmp)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, mobilenet_onnx)
        if os.path.exists(fp32_tmp): os.remove(fp32_tmp)
        print(f"      Converted and saved {mobilenet_onnx} ({os.path.getsize(mobilenet_onnx) / (1024*1024):.2f} MB)")

    print("\n[4/4] Triggering downloads...")
    try:
        from google.colab import files
        files.download(tv2radar_onnx)
        files.download(mobilenet_onnx)
    except ImportError:
        print(f"Saved locally: {os.path.abspath(tv2radar_onnx)}, {os.path.abspath(mobilenet_onnx)}")

if __name__ == "__main__":
    main()
