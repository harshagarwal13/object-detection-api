from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io
from ultralytics import YOLO  # Install using `pip install ultralytics`

app = FastAPI()

model = YOLO("yolov8n.pt")


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(image)
        detections = []
        for result in results:
            for box, confidence, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detections.append({
                    "label": model.names[int(label)],  # Map class index to name
                    "score": round(float(confidence), 2),
                    "bbox": [round(float(coord), 2) for coord in box.tolist()]
                })

        return {"detections": detections}
    except Exception as e:
        return {"error": str(e)}

