from fastapi import FastAPI
import pickle

app = FastAPI()

with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

@app.post("/predict")
def predict(client: dict):
    probability = pipeline.predict_proba([client])[0, 1]
    return {"probability": float(probability)}