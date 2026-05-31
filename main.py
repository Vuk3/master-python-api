from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
from typing import Any, Dict, List

app = FastAPI()

MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
model = None
load_err = "YOLO_MODEL_PATH is not set."

if MODEL_PATH:
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        load_err = str(e)

@app.get("/health")
def health_check():
    return "OK from PYTHON"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if model is None:
        raise HTTPException(status_code=500, detail=f"YOLO model not loaded: {load_err}")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        image_width, image_height = img.size
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        results = model.predict(img, verbose=False)  
        r = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    detections: List[Dict[str, Any]] = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.tolist()
        conf = r.boxes.conf.tolist()
        cls_ids = r.boxes.cls.tolist()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(conf[i])
            cls_id = int(cls_ids[i])
            label = r.names.get(cls_id, str(cls_id))

            detections.append({
                "label": label,
                "score": score,
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            })

    return {
        "model": "yolov8",
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "detections": detections,
        "fileName": file.filename,
        "contentType": file.content_type,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
