import cv2
import os

cameraIp="http://192.168.1.32:8080/video"
MODEL_PATH = "../models/face_detection_yunet_2023mar.onnx"
SAVE_FOLDER = "../dataset/raw"

def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=0.9
    )
    cap = cv2.VideoCapture(cameraIp)
    if not cap.isOpened():
        return

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize((w_frame, h_frame))

    photo_id=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        _, faces = detector.detect(frame)
        for face in faces:
            box = face[:4].astype(int)
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Akwizycja Danych (Wersja z YuNet)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            photo_path = os.path.join(SAVE_FOLDER, f"zdjecie_{photo_id}.jpg")
            cv2.imwrite(photo_path, frame)
            photo_id += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
