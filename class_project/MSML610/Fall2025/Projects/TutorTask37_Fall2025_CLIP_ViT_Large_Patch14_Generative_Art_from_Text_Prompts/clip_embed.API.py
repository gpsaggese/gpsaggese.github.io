from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI(title="CLIP Embedding API")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/embed/image")
async def embed_image(image: UploadFile = File(...)):
    """
    Upload an image and get its CLIP embedding.
    """
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Convert tensor to list for JSON serialization
        embedding = outputs.cpu().tolist()[0]

        return JSONResponse({
            "image_filename": image.filename,
            "embedding": embedding
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/embed/text")
async def embed_text(text: str = Form(...)):
    """
    Send text and get its CLIP embedding.
    """
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        embedding = outputs.cpu().tolist()[0]

        return JSONResponse({
            "text": text,
            "embedding": embedding
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "CLIP embedding API is running!"}