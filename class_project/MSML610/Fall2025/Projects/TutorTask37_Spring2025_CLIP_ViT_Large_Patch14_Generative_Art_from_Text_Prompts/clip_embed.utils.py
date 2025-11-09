from PIL import Image
import requests
import faiss
import numpy as np

API_URL = "http://172.16.7.95:8000"  # change if running remotely
API_URL = "http://localhost:8000"

image_index = faiss.read_index("image_index.faiss")
text_index = faiss.read_index("text_index.faiss")


def get_image_embedding(image_path: str):
    """
    Sends an image to the /embed/image endpoint and returns its embedding.
    """
    url = f"{API_URL}/embed/image"
    with open(image_path, "rb") as img_file:
        files = {"image": (image_path, img_file, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        # print(f"Image embedding received for: {data['image_filename']}")
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def get_image_embedding_from_bytes(bytes_data: bytes):
    """
    Sends an image to the /embed/image endpoint and returns its embedding.
    """
    url = f"{API_URL}/embed/image"
    files = {"image": ("image.jpg", bytes_data, "image/jpeg")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        # print(f"Image embedding received for: {data['image_filename']}")
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def get_text_embedding(text: str):
    """
    Sends text to the /embed/text endpoint and returns its embedding.
    """
    url = f"{API_URL}/embed/text"
    data = {"text": text}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        data = response.json()
        # print(f"Text embedding received for: {data['text']}")
        return data["embedding"]
    else:
        print("Error:", response.text)
        return None


def search_vec_db(embedding, vectorDB="image", k=3):
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
