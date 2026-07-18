#!/usr/bin/env python3
"""
Football-TV2Radar Corner Detection Model Colab Exporter
=======================================================
Run this script inside a Google Colab notebook cell (`!python export_colab.py`
or copy/paste the Python code directly) to automatically:
1. Download `corner_detection_model.pt` (~6.2 MB) from GitHub (`amirrezarajabi/Football-TV2Radar`).
2. Export it to ONNX (`1280x736` letterbox size: height=736, width=1280) with graph simplification.
3. Rename the output to `football_tv2radar.onnx` and automatically download it to your computer.

Google Colab Usage (Copy & paste into a Colab cell):
----------------------------------------------------
!pip install -q ultralytics onnx onnxsim

import urllib.request
import shutil
import os
from ultralytics import YOLO

# 1. Download PyTorch model weights (~6.2 MB)
url = "https://raw.githubusercontent.com/amirrezarajabi/Football-TV2Radar/main/corner_detection_model.pt"
pt_filename = "corner_detection_model.pt"
onnx_filename = "football_tv2radar.onnx"

print(f"[1/3] Downloading {pt_filename} from GitHub...")
urllib.request.urlretrieve(url, pt_filename)
print(f"      Downloaded {os.path.getsize(pt_filename) / (1024*1024):.2f} MB")

# 2. Export to ONNX (1280x736 letterbox size)
print("\n[2/3] Exporting YOLOv8 model to ONNX (height=736, width=1280)...")
model = YOLO(pt_filename)
# Note: Ultralytics imgsz tuple is (height, width) -> (736, 1280)
exported_path = model.export(
    format="onnx",
    imgsz=(736, 1280),
    simplify=True,
    dynamic=False,
    opset=12
)

# 3. Rename and download
shutil.move(exported_path, onnx_filename)
print(f"\n[3/3] Export complete: {onnx_filename} ({os.path.getsize(onnx_filename) / (1024*1024):.2f} MB)")

try:
    from google.colab import files
    print("      Triggering browser download in Google Colab...")
    files.download(onnx_filename)
except ImportError:
    print(f"      Not running inside Google Colab. The file is saved locally at ./{onnx_filename}")
"""

import os
import sys
import shutil
import urllib.request

def main():
    print("=== Football-TV2Radar Model Exporter (Colab/Standalone) ===")
    
    # Check dependencies
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing required packages (ultralytics, onnx, onnxsim)...")
        os.system(f"{sys.executable} -m pip install -q ultralytics onnx onnxsim")
        from ultralytics import YOLO

    url = "https://raw.githubusercontent.com/amirrezarajabi/Football-TV2Radar/main/corner_detection_model.pt"
    pt_filename = "corner_detection_model.pt"
    onnx_filename = "football_tv2radar.onnx"

    print(f"\n[1/3] Downloading {pt_filename} from GitHub...")
    urllib.request.urlretrieve(url, pt_filename)
    print(f"      Downloaded {os.path.getsize(pt_filename) / (1024*1024):.2f} MB")

    print("\n[2/3] Exporting YOLOv8 model to ONNX (height=736, width=1280)...")
    model = YOLO(pt_filename)
    exported_path = model.export(
        format="onnx",
        imgsz=(736, 1280),
        simplify=True,
        dynamic=False,
        opset=12
    )

    shutil.move(exported_path, onnx_filename)
    if os.path.exists(pt_filename):
        os.remove(pt_filename)
        
    print(f"\n[3/3] Export complete: {onnx_filename} ({os.path.getsize(onnx_filename) / (1024*1024):.2f} MB)")

    try:
        from google.colab import files
        print("      Triggering Google Colab browser download...")
        files.download(onnx_filename)
    except ImportError:
        print(f"      Saved locally: {os.path.abspath(onnx_filename)}")

if __name__ == "__main__":
    main()
