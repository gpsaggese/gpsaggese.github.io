# Use a small image `alpine` with Python inside.
FROM python:3.6-alpine
# Copy current dir `counter_app` into image.
ADD . /code
# Set working dir.
WORKDIR /code
# Install requirements.
RUN pip install -r requirements.txt
# Set the default app.
CMD ["python", "app.py"]
