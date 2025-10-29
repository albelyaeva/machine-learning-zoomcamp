import pickle
import requests

with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

probability = pipeline.predict_proba([client])[0, 1]
print(f"Probability: {probability:.3f}")

url = "http://localhost:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
response = requests.post(url, json=client).json()
print(response)