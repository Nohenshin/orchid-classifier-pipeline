import os
import argparse
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from common.feature_extractor import extract_features

def load_data(data_dir):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    name_to_label = {name: i for i, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            img_path = os.path.join(class_dir, img_file)
            try:
                X.append(extract_features(img_path))
                y.append(name_to_label[class_name])
            except:
                continue
    return np.array(X), np.array(y), class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    
    model = joblib.load(os.path.join(args.model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.joblib'))
    label_to_name = joblib.load(os.path.join(args.model_dir, 'label_to_name.joblib'))
    class_names = list(label_to_name.values())
    
    X, y_true, _ = load_data(args.data_dir)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    acc = np.mean(y_pred == y_true)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()