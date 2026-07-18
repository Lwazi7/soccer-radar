#!/usr/bin/env python3
"""
Soccana 720p Clean/FP16 Model Exporter (Colab / Standalone)
===========================================================
Re-exports the exact 10 MB PyTorch weights (`best.pt` from `Adit-jain/soccana`)
into a clean, static, simplified 1280x736 ONNX graph (`simplify=True, dynamic=False`).
Removing embedded dynamic loop nodes allows Android NNAPI / XNNPACK to evaluate
`soccana_object_720p.onnx` natively on mobile NPU/SIMD cores in ~50-150 ms!

Google Colab Usage (Copy & paste into a Colab cell):
----------------------------------------------------
!pip install -q ultralytics onnx onnxsim

import urllib.request
import shutil
import os
from ultralytics import YOLO

# 1. Download PyTorch model weights (`best.pt` from `Adit-jain/soccana`)
url = "https://huggingface.co/Adit-jain/soccana/resolve/main/Model/weights/best.pt"
pt_filename = "soccana_object.pt"
onnx_filename = "soccana_object_720p_clean.onnx"

print(f"[1/3] Downloading {pt_filename} from HuggingFace...")
urllib.request.urlretrieve(url, pt_filename)
print(f"      Downloaded {os.path.getsize(pt_filename) / (1024*1024):.2f} MB")

# 2. Export to ONNX (exact 1280x736 resolution: height=736, width=1280)
# Note: Set half=True below if you want an FP16 (~5 MB) model for native NPU/GPU binding!
print("\n[2/3] Exporting clean static ONNX graph (height=736, width=1280)...")
model = YOLO(pt_filename)
exported_path = model.export(
    format="onnx",
    imgsz=(736, 1280),
    simplify=True,
    dynamic=False,
    half=False,  # Set to True for FP16 (~5 MB)
    opset=12
)

# 3. Rename and download
shutil.move(exported_path, onnx_filename)
if os.path.exists(pt_filename):
    os.remove(pt_filename)

print(f"\n[3/3] Export complete: {onnx_filename} ({os.path.getsize(onnx_filename) / (1024*1024):.2f} MB)")

try:
    from google.colab import files
    print("      Triggering browser download in Google Colab...")
    files.download(onnx_filename)
except ImportError:
    print(f"      Saved locally at: {os.path.abspath(onnx_filename)}")
"""

import os
import sys
import shutil
import urllib.request

def main():
    print("=== Soccana Clean 720p ONNX Exporter ===")
    try:
        from ultralytics import YOLO
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q ultralytics onnx onnxsim")
        from ultralytics import YOLO

    url = "https://huggingface.co/Adit-jain/soccana/resolve/main/Model/weights/best.pt"
    pt_filename = "soccana_object.pt"
    onnx_filename = "soccana_object_720p_clean.onnx"

    print(f"\n[1/3] Downloading {pt_filename} from HuggingFace...")
    urllib.request.urlretrieve(url, pt_filename)

    print("\n[2/3] Exporting clean static ONNX graph (imgsz=736,1280)...")
    model = YOLO(pt_filename)
    exported_path = model.export(
        format="onnx",
        imgsz=(736, 1280),
        simplify=True,
        dynamic=False,
        half=False,
        opset=12
    )

    shutil.move(exported_path, onnx_filename)
    if os.path.exists(pt_filename):
        os.remove(pt_filename)

    print(f"\n[3/3] Export complete: {onnx_filename} ({os.path.getsize(onnx_filename) / (1024*1024):.2f} MB)")

    try:
        from google.colab import files
        files.download(onnx_filename)
    except ImportError:
        print(f"Saved locally: {os.path.abspath(onnx_filename)}")

if __name__ == "__main__":
    main()
