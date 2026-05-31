from fastapi import FastAPI
from routes.health import router as health_router
from routes.predict import router as predict_router

app = FastAPI()
app.include_router(health_router)
app.include_router(predict_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8123)
