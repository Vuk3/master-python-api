from fastapi import APIRouter

from services.model_service import get_available_models

router = APIRouter()


@router.get("/models")
def models():
    return get_available_models()
