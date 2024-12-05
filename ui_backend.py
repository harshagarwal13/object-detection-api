from fastapi import FastAPI, File, UploadFile
import requests

app = FastAPI()
AI_BACKEND_URL = "http://ai-backend:8005/detect/"  # Update with your AI backend URL

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Forward the image to the AI backend
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = requests.post(AI_BACKEND_URL, files=files)
        response.raise_for_status()

        # Return results from the AI backend
        return response.json()
    except Exception as e:
        return {"error": str(e)}
