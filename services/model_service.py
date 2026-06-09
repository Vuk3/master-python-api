import os
import re
from pathlib import Path

from ultralytics import YOLO

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
MODEL_EXTENSIONS = {".pt"}
DEFAULT_MODEL_PATH = MODELS_ROOT / "half-annotated" / "yolov8s_640_e50_best.pt"
CONFIGURED_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
YOLO_SIZE_PATTERN = re.compile(r"([nslmx])$", re.IGNORECASE)
model_cache = {}


def get_available_models():
    default_path = get_default_model_path()
    models = [
        create_model_summary(model_path)
        for model_path in find_model_paths()
    ]

    if default_path is not None:
        models.sort(key=lambda model: model["id"] != get_model_id(default_path))

    return {
        "models": models,
        "defaultModelId": get_model_id(default_path) if default_path else None,
    }


def get_model(model_id=None):
    model_path = resolve_model_path(model_id)
    if model_path is None:
        return None, None, f"Requested YOLO model was not found: {model_id}"

    cached_model = model_cache.get(model_path)
    if cached_model is not None:
        return cached_model, create_model_summary(model_path), None

    try:
        loaded_model = YOLO(str(model_path))
    except Exception as error:
        return None, create_model_summary(model_path), str(error)

    model_cache[model_path] = loaded_model
    return loaded_model, create_model_summary(model_path), None


def create_model_summary(model_path):
    model_name = get_model_name(model_path)
    annotation_type = get_annotation_type(model_path)

    return {
        "id": get_model_id(model_path),
        "name": model_name,
        "family": get_model_family(model_name),
        "annotationType": annotation_type,
        "isDefault": model_path == get_default_model_path(),
    }


def find_model_paths():
    model_paths = []
    if MODELS_ROOT.exists():
        model_paths.extend(
            path.resolve()
            for path in MODELS_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
        )

    configured_path = CONFIGURED_MODEL_PATH.expanduser().resolve()
    if (
        configured_path.is_file()
        and configured_path.suffix.lower() in MODEL_EXTENSIONS
        and configured_path not in model_paths
    ):
        model_paths.append(configured_path)

    return sorted(
        path.resolve()
        for path in model_paths
    )


def get_default_model_path():
    configured_path = CONFIGURED_MODEL_PATH.expanduser().resolve()
    if configured_path.is_file():
        return configured_path

    model_paths = find_model_paths()
    if model_paths:
        return model_paths[0]

    return None


def resolve_model_path(model_id=None):
    available_paths = {get_model_id(path): path for path in find_model_paths()}

    if model_id:
        return available_paths.get(model_id)

    default_model_path = get_default_model_path()
    if default_model_path is not None:
        return default_model_path

    return None


def get_model_id(model_path):
    try:
        return model_path.relative_to(MODELS_ROOT).as_posix()
    except ValueError:
        return model_path.name


def get_model_group(model_path):
    try:
        return model_path.parent.relative_to(MODELS_ROOT).as_posix()
    except ValueError:
        return "external"


def get_annotation_type(model_path):
    group = get_model_group(model_path)

    if "/" in group:
        return group.split("/", 1)[0]

    return group


def get_model_name(model_path):
    first_token = model_path.stem.split("_", 1)[0]
    normalized_token = first_token.strip()

    if normalized_token.lower().startswith("yolo"):
        return f"YOLO{normalized_token[4:]}"

    return normalized_token.replace("-", " ").replace("_", " ")


def get_model_family(model_name):
    normalized_name = model_name.strip()
    if not normalized_name.lower().startswith("yolo"):
        return "Custom model"

    family_name = YOLO_SIZE_PATTERN.sub("", normalized_name)
    return f"Ultralytics {family_name}"
