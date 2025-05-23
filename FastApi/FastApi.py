import cv2
import torch

def main():
    # Load YOLOv5s model (auto-downloads weights if needed)
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open default webcam (device 0)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        results.render()
        frame = results.ims[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('XPro Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
