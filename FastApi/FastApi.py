from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import torch
import asyncio

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

detection_status = {"detected": False, "camera_active": False}

@app.get("/")
async def get_home():
    html_path = Path(__file__).parent / "templates" / "index.html"
    html_content = html_path.read_text()
    return HTMLResponse(content=html_content, status_code=200)

def gen_frames():
    cap = cv2.VideoCapture(0)
    detection_status["camera_active"] = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detected = len(results.xyxy[0]) > 0
        detection_status["detected"] = detected

        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    detection_status["camera_active"] = False

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json({
            "object_detection": {"detected": detection_status["detected"]},
            "camera_status": {"active": detection_status["camera_active"]}
        })
        await asyncio.sleep(1)
