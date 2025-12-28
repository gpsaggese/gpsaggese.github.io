import io

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="CLIP Embedding API")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.post("/embed/image")
async def embed_image(image: UploadFile = File(...)):
    """
    Generate CLIP embedding for an uploaded image.

    Args:
        image: Uploaded image file.

    Returns:
        JSONResponse containing image filename and embedding vector, or error message.
    """
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.cpu().tolist()[0]

        return JSONResponse({"image_filename": image.filename, "embedding": embedding})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/embed/text")
async def embed_text(text: str = Form(...)):
    """
    Generate CLIP embedding for input text.

    Args:
        text: Input text string.

    Returns:
        JSONResponse containing text and embedding vector, or error message.
    """
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        embedding = outputs.cpu().tolist()[0]

        return JSONResponse({"text": text, "embedding": embedding})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    """
    Root endpoint to check API status.

    Returns:
        Dictionary with API status message.
    """
    return {"message": "CLIP embedding API is running!"}
