import os

from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
model = None
load_err = "YOLO_MODEL_PATH is not set."

if MODEL_PATH:
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        load_err = str(e)


def get_model():
    return model, load_err
