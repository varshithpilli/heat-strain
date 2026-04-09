import os
import json
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

twilio_sid   = os.getenv("TWILIO_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_from  = os.getenv("TWILIO_FROM")
twilio_to    = os.getenv("TWILIO_TO")

app = FastAPI(title="HeatGuard API")

origins = [
    "http://localhost:5173",
    "https://varzone.in",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow OPTIONS
    allow_headers=["*"],
)

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

LABEL_NAMES = {0: "Normal", 1: "Moderate", 2: "High"}

MODEL_DISPLAY = {
    "custom_nn",
    "random_forest",
    "gradient_boost",
    "svm",
    "logistic_reg"
}

FEATURES = [
    "body_temp", "ambient_temp", "humidity", "heart_rate",
    "skin_resistance", "resp_rate", "movement",
    "avg_sensor_temp", "sensor_spread",
    "temp_humidity_index", "heat_index",
    "hr_temp_product", "skin_resistance_normalized",
    "body_amb_diff", "iaq", "lux", "sound"
]

class SensorInput(BaseModel):
    body_temp: float
    ambient_temp: float
    humidity: float
    heart_rate: float
    skin_resistance: float
    resp_rate: float = 18
    movement: float = 1
    model: str = "random_forest"

def load_models():
    models, features = {}, {}

    for target in ["heat_stress_label", "dehydration_label"]:
        models[target] = {}

        for name in MODEL_DISPLAY:
            path = os.path.join(MODEL_DIR, f"{target}_{name}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[target][name] = pickle.load(f)

        feat_path = os.path.join(MODEL_DIR, f"{target}_features.json")
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                features[target] = json.load(f)

    return models, features

MODELS, FEATURES_MAP = load_models()

def compute_features(inputs):

    bt  = inputs["body_temp"]
    at  = inputs["ambient_temp"]
    hum = inputs["humidity"]
    hr  = inputs["heart_rate"]
    sr  = inputs["skin_resistance"]
    rr  = inputs["resp_rate"]
    mv  = inputs["movement"]

    thi = bt + 0.33 * (hum / 100 * 6.105 * np.exp(17.27 * at / (at + 237.3))) - 4.0

    hi  = (-8.78 + 1.611 * at + 2.339 * hum
           - 0.1461 * at * hum
           - 0.0123 * at**2
           - 0.0164 * hum**2
           + 0.00221 * at**2 * hum
           + 0.000725 * at * hum**2
           - 3.58e-6 * at**2 * hum**2)

    return {
        "body_temp": bt,
        "ambient_temp": at,
        "humidity": hum,
        "heart_rate": hr,
        "skin_resistance": sr,
        "resp_rate": rr,
        "movement": mv,
        "avg_sensor_temp": bt,
        "sensor_spread": 0.2,
        "temp_humidity_index": thi,
        "heat_index": hi,
        "hr_temp_product": hr * bt / 100.0,
        "skin_resistance_normalized": sr / 500.0,
        "body_amb_diff": bt - at,
        "iaq": 0.0,
        "lux": 0.0,
        "sound": 0.0
    }

def predict_risk(model, feat_row, feature_names):

    X = np.array([[feat_row.get(f, 0.0) for f in feature_names]])

    proba = model.predict_proba(X)[0]
    pred  = int(proba.argmax())

    return pred, proba.tolist()

def send_alert(message):

    # if not twilio_sid:
    #     return

    # client = Client(twilio_sid, twilio_token)

    # client.messages.create(
    #     from_=twilio_from,
    #     body=message,
    #     to=twilio_to
    # )
    
    # import requests
    # BOT_TOKEN=os.getenv("TG_BOT_TOKEN")
    # CHAT_ID=os.getenv("TG_CHAT_ID")
    # payload = {
    #     "chat_id": CHAT_ID,
    #     "text": message
    # }
    # url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # r = requests.post(url, json=payload, timeout=15)
    # r.raise_for_status()
    return

@app.get("/")
def root():
    return {"message": "HeatGuard API running"}
    
@app.get("/health")
def health():
    return {"message": "HeatGuard API running"}

@app.post("/predict")
async def predict(data: SensorInput):

    model_name = data.model

    inputs = data.dict()
    feat = compute_features(inputs)

    results = {}

    for target in ["heat_stress_label", "dehydration_label"]:

        if model_name not in MODELS[target]:
            continue

        model = MODELS[target][model_name]
        feats = FEATURES_MAP.get(target, FEATURES)

        pred, proba = predict_risk(model, feat, feats)

        results[target] = {
            "class": int(pred),
            "label": LABEL_NAMES[pred],
            "probabilities": proba
        }

    heat_pred  = results["heat_stress_label"]["class"]
    dehyd_pred = results["dehydration_label"]["class"]
    
    heat_proba  = results["heat_stress_label"]["probabilities"]
    dehyd_proba = results["dehydration_label"]["probabilities"]

    overall = max(heat_pred, dehyd_pred)

    if overall >= 1:
        heat_risk  = heat_proba[heat_pred] * 100
        dehyd_risk = dehyd_proba[dehyd_pred] * 100

        message = f"""
HeatGuard Alert

Risk Level: {LABEL_NAMES[overall]}

Heat Stress: {results['heat_stress_label']['label']}
Dehydration: {results['dehydration_label']['label']}

Heat Risk: {heat_risk:.2f}%
Dehydration Risk: {dehyd_risk:.2f}%

Please take immediate precautions:
- Stay hydrated.
- Avoid direct sunlight.
- Take rest if feeling unwell.

Stay safe
"""
        send_alert(message)

    return {
        "overall_risk": LABEL_NAMES[overall],
        "results": results,
        "message": message
    }