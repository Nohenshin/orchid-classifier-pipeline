import os
import argparse
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from common.feature_extractor import extract_features

def load_data(data_dir):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_to_name = {i: name for i, name in enumerate(class_names)}
    name_to_label = {name: i for i, name in label_to_name.items()}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        label = name_to_label[class_name]
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            img_path = os.path.join(class_dir, img_file)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Lỗi {img_path}: {e}")
    return np.array(X), np.array(y), label_to_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_type', default='svm', choices=['svm','rf'])
    parser.add_argument('--output_dir', default='/model')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading training data...")
    X_train, y_train, label_to_name = load_data(args.data_dir)
    print(f"Samples: {len(X_train)}, Classes: {len(label_to_name)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if args.model_type == 'svm':
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training...")
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, os.path.join(args.output_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(label_to_name, os.path.join(args.output_dir, 'label_to_name.joblib'))
    
    y_pred = model.predict(X_train_scaled)
    acc = accuracy_score(y_train, y_pred)
    print(f"Train accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()