from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import io

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

app = FastAPI()

model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# COCO label names (class 0 is 'background')
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "dog", "horse", "dog", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
transform = transforms.Compose([
    transforms.ToTensor()
])

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(img_tensor)

        results = []
        for idx, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][idx].item()
            if score >= 0.5:
                label_id = predictions[0]['labels'][idx].item()
                if label_id<len(COCO_CLASSES):
                    label_name = COCO_CLASSES[label_id]
                    results.append({
                        "label": label_name,
                        "score": score
                    })
                else:
                    results.append({
                        "label": "Not able to recognize",
                        "score": score
                    })

        return {"detections": results}
    except Exception as e:
        return {"error": str(e)}
