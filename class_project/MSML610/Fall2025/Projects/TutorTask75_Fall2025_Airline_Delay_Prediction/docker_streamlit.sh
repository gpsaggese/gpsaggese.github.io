#!/bin/bash
# run the Streamlit app
docker run --rm -it -p 8501:8501 \
  -v "$(pwd)"/models:/app/models \
  airline-delay-prediction
