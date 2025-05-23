from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded.")

def detect_objects():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    print("Starting detection loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 expects RGB images
        results = model(frame)

        # Detect if any objects found
        detected = len(results.xyxy[0]) > 0

        # Emit detection results
        socketio.emit('object_detection', {'detected': detected})

        # Emit camera active status (always true while streaming)
        socketio.emit('camera_status', {'active': True})

        time.sleep(0.2)  # control detection rate ~5fps

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Run detection in background thread
    thread = threading.Thread(target=detect_objects)
    thread.daemon = True
    thread.start()

    socketio.run(app, host='0.0.0.0', port=5000)
