#!/usr/bin/env python3
"""
Soccer Radar v2.0 Model Setup & Auto-Downloader
===============================================
Downloads and converts models from the exact HuggingFace repositories provided:
1. Soccana Object Detection: https://huggingface.co/Adit-jain/soccana
2. Soccana Keypoint Detection: https://huggingface.co/Adit-jain/Soccana_Keypoint
3. MobileNetV4 Embeddings: https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k

Usage:
    python3 setup_models.py
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Target ONNX model paths
OBJECT_MODEL = MODELS_DIR / "soccana_object_720p.onnx"
KEYPOINT_MODEL = MODELS_DIR / "soccana_keypoint_720p.onnx"
EMBEDDING_MODEL = MODELS_DIR / "mobilenetv4_conv_small.onnx"


def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name} from {url} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    print(f"Saved to {dest_path}")


def setup_object_model():
    if OBJECT_MODEL.exists():
        print(f"[*] {OBJECT_MODEL} already exists.")
        return
    print("\n[1/3] Setting up Soccana Object Detection Model...")
    url_pt = "https://huggingface.co/Adit-jain/soccana/resolve/main/Model/weights/best.pt"
    pt_path = MODELS_DIR / "soccana_object.pt"
    try:
        download_file(url_pt, pt_path)
        print("Exporting YOLO PyTorch model to ONNX (1280x736 letterbox)...")
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        export_path = model.export(format="onnx", imgsz=(736, 1280), simplify=True)
        shutil.move(export_path, OBJECT_MODEL)
        pt_path.unlink(missing_ok=True)
        print(f"[+] Successfully exported {OBJECT_MODEL}")
    except Exception as e:
        print(f"[-] Could not auto-export PyTorch weights ({e}).")
        print("Please export manually: yolo export model=best.pt format=onnx imgsz=736,1280")


def setup_keypoint_model():
    if KEYPOINT_MODEL.exists():
        print(f"[*] {KEYPOINT_MODEL} already exists.")
        return
    print("\n[2/3] Setting up Soccana Keypoint Detection Model...")
    url_pt = "https://huggingface.co/Adit-jain/Soccana_Keypoint/resolve/main/Model/weights/best.pt"
    pt_path = MODELS_DIR / "soccana_keypoint.pt"
    try:
        download_file(url_pt, pt_path)
        print("Exporting YOLO Pose PyTorch model to ONNX (1280x736 letterbox)...")
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        export_path = model.export(format="onnx", imgsz=(736, 1280), simplify=True)
        shutil.move(export_path, KEYPOINT_MODEL)
        pt_path.unlink(missing_ok=True)
        print(f"[+] Successfully exported {KEYPOINT_MODEL}")
    except Exception as e:
        print(f"[-] Could not auto-export PyTorch weights ({e}).")
        print("Please export manually: yolo export model=best.pt format=onnx imgsz=736,1280")


def setup_embedding_model():
    if EMBEDDING_MODEL.exists():
        print(f"[*] {EMBEDDING_MODEL} already exists.")
        return
    print("\n[3/3] Setting up MobileNetV4 Embedding Model...")
    # Direct ONNX download from onnx-community repository
    url_onnx = "https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/onnx/model.onnx"
    try:
        download_file(url_onnx, EMBEDDING_MODEL)
        print(f"[+] Successfully downloaded {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"[-] Error downloading MobileNetV4 ONNX: {e}")


def main():
    print("=== Soccer Radar v2.0 Model Setup ===")
    setup_object_model()
    setup_keypoint_model()
    setup_embedding_model()
    print("\n=== All required models checked/setup in ./models/ ===")


if __name__ == "__main__":
    main()
