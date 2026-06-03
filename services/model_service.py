import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "half-annotated"
    / "yolov8s_640_e50_best.pt"
)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
model = None
load_err = f"YOLO model file was not found at {MODEL_PATH}."

if Path(MODEL_PATH).is_file():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        load_err = str(e)


def get_model():
    return model, load_err
