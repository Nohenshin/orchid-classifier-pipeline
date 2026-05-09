import os
import argparse
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import model_loader

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    temp_path = f"/tmp/{filename}"
    file.save(temp_path)
    try:
        result = model_loader.predict(temp_path)
    finally:
        os.remove(temp_path)
    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    model_loader.load_model(args.model_dir)
    app.run(host='0.0.0.0', port=args.port)