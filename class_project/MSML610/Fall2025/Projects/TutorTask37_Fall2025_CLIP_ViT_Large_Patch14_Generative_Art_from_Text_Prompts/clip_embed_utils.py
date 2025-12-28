import faiss
import numpy as np
import requests
from PIL import Image

API_URL = "http://172.16.7.95:8000"
API_URL = "http://localhost:8000"

image_index = faiss.read_index("clip_embed.image_index.faiss")
text_index = faiss.read_index("clip_embed.text_index.faiss")


def get_image_embedding(image_path: str):
    """
    Retrieve CLIP embedding for an image file via API.

    Args:
        image_path: Path to the image file.

    Returns:
        Embedding vector as a list, or None if request fails.
    """
    url = f"{API_URL}/embed/image"
    with open(image_path, "rb") as img_file:
        files = {"image": (image_path, img_file, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def get_image_embedding_from_bytes(bytes_data: bytes):
    """
    Retrieve CLIP embedding for an image from bytes data via API.

    Args:
        bytes_data: Image data as bytes.

    Returns:
        Embedding vector as a list, or None if request fails.
    """
    url = f"{API_URL}/embed/image"
    files = {"image": ("image.jpg", bytes_data, "image/jpeg")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def get_text_embedding(text: str):
    """
    Retrieve CLIP embedding for text via API.

    Args:
        text: Input text string.

    Returns:
        Embedding vector as a list, or None if request fails.
    """
    url = f"{API_URL}/embed/text"
    data = {"text": text}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        data = response.json()
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def search_vec_db(embedding, vectorDB="image", k=3):
    """
    Search for similar vectors in FAISS index.

    Args:
        embedding: Query embedding vector.
        vectorDB: Type of database to search, either "image" or "text".
        k: Number of nearest neighbors to retrieve.

    Returns:
        Tuple of (distances, indices) arrays from FAISS search.
    """
    norm_ = np.linalg.norm(embedding)
    if norm_ > 0:
        embedding = embedding / norm_
    else:
        embedding = embedding
    embedding = np.expand_dims(embedding, axis=0)
    if vectorDB == "image":
        distances, indices = image_index.search(embedding, k)
    else:
        distances, indices = text_index.search(embedding, k)

    return distances, indices
