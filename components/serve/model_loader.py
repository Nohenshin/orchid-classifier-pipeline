import joblib
import numpy as np
from common.feature_extractor import extract_features

model = None
scaler = None
label_to_name = None

def load_model(model_dir):
    global model, scaler, label_to_name
    model = joblib.load(f"{model_dir}/model.joblib")
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    label_to_name = joblib.load(f"{model_dir}/label_to_name.joblib")

def predict(image_path):
    feat = extract_features(image_path).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred_id = model.predict(feat_scaled)[0]
    proba = model.predict_proba(feat_scaled)[0].tolist() if hasattr(model, "predict_proba") else None
    return {"class": label_to_name[pred_id], "class_id": int(pred_id), "probabilities": proba}