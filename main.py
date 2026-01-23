from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health_check():
    return "OK from PYTHON"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content="No file uploaded")

    return {
        "message": "Prediction result from PYTHON API",
        "fileName": file.filename,
        "contentType": file.content_type
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123)
