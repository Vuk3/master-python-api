import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from services.model_service import get_model

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    model, load_err = get_model()
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
        result = results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.tolist()
        conf = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()

        for index in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[index]
            score = float(conf[index])
            cls_id = int(cls_ids[index])
            label = result.names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "label": label,
                    "score": score,
                    "box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                }
            )

    return {
        "model": "yolov8",
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "detections": detections,
        "fileName": file.filename,
        "contentType": file.content_type,
    }
