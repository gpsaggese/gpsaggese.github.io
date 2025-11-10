# Predicting House Prices

## Quickstart
```bash
./docker_build.sh

# Train (detached)
docker rm -f housing-train 2>/dev/null
docker run -d --name housing-train \
  -v "$(pwd)":/app \
  -e DATA_PATH=/app/data/melb_data.csv \
  azua-housing:latest \
  python train.py
docker logs -f housing-train
````

Artifacts on success:

```
artifacts/model.joblib
artifacts/metrics.json
```

## Serve API

```bash
docker rm -f housing-api 2>/dev/null
docker run -d --name housing-api \
  -p 8000:8000 \
  -v "$(pwd)":/app \
  azua-housing:latest \
  uvicorn serve:app --host 0.0.0.0 --port 8000

curl -s http://localhost:8000/
curl -s http://localhost:8000/schema
```

## Predict

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"Rooms":3,"Bathroom":1,"Car":1,"Landsize":350,"Suburb":"Richmond","Type":"h","Method":"S","Regionname":"Southern Metropolitan","Date":"2017-05-15"}}'
```

See `AzuaHousing.API.md` and `AzuaHousing.example.md` for details.

