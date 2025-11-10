# TutorTask72 — JAX Wildlife Image Classification
### Name - Ravi Vignesh LNU

## Deliverables
- `JAX_wildlife_utils.py` — data I/O, CNN definition, training loop, evaluation, and plotting.
- `JAX_wildlife.API.ipynb` / `.md` — tutorial showing how to use the utils.
- `JAX_wildlife.example.ipynb` / `.md` — end-to-end workflow
- `Dockerfile` — JAX environment used for all notebooks.

## Docker workflow
1. Build the image:
   ```bash
   docker build -t msml610/tutortask72_jax .
   ```
2. Run Jupyter with this folder mounted:
   ```powershell
   docker run --rm -it -p 8888:8888 -v "${PWD}:/data" msml610/tutortask72_jax `
   bash -lc "cd /data && jupyter-notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"
   ```
3. Open `http://localhost:8888` in your browser and run either notebook top-to-bottom via “Restart & Run All”.

     

