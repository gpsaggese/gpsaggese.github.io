# CLIP Embeddings Search App - MSML610 Fall25#37

This project demonstrates how to build and query **text and image embeddings** using **CLIP** and **FAISS**, all packaged with **Docker** and a **Streamlit** interface.

The project contains the following:
- Generate and store embeddings for images and text using CLIP.
- Use FAISS for fast similarity search.
- Build a simple Streamlit app for querying by text or image.

---

## Run Instructions

1. Navigate to the project directory:
   ```bash
   cd class_project/instructions/clip_faiss_streamlit

Work in Progress.

## Project Structure

| File/Folder                        | Description                                       |
| ----------------------------------- | ------------------------------------------------- |
| **clip_embed.API.py**               | API for generating embeddings                     |
| **clip_embed.dataset/**             | Dataset folder containing text and images         |
| **clip_embed.dataset/captions.txt** | Text captions for image-text pairs                |
| **clip_embed.dataset/Images/**     | Folder containing sample images                   |
| **clip_embed.example.ipynb**        | Example notebook demonstrating embedding generation|
| **clip_embed.image_index.faiss**    | FAISS index for image embeddings                  |
| **clip_embed.text_index.faiss**     | FAISS index for text embeddings                   |
| **clip_embed.streamlit.py**         | Streamlit web app for querying                    |
| **clip_embed.utils.py**             | Helper functions for embeddings and indexing      |
| **Dockerfile**                      | Docker environment setup                          |
| **README.md**                       | Project documentation                            |
| **requirements.txt**                | Python dependencies                               |

## Note
This is a work in progress project: additional features, improved UI, and dataset integration are still being developed. However, the model has been successfully executed.