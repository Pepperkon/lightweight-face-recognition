import os
import urllib.request


def main():
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    save_path = "../models/face_detection_yunet_2023mar.onnx"

    if os.path.exists(save_path):
        print("MODEL ALREADY ON DISK.")
        return

    print("DOWNLOADING YUNET (ok. 340 KB)...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"[+] SUCCES: {save_path}")
    except Exception as e:
        print(f"[-] ERROR: {e}")


if __name__ == "__main__":
    main()