# Flower Classifier Pipeline

## Giới thiệu

Flower Classifier Pipeline là hệ thống phân loại hình ảnh hoa sử dụng kiến trúc pipeline machine learning, được container hóa bằng Docker cho từng thành phần riêng biệt.

Pipeline bao gồm các bước chính:

1. Data Augmentation – tăng cường dữ liệu ảnh
2. Model Training – huấn luyện mô hình phân loại
3. Model Evaluation – đánh giá mô hình
4. Model Serving – triển khai API dự đoán

Hệ thống hỗ trợ phân loại 5 loại hoa:

* Cúc (Cuc)
* Đào (Dao)
* Lan (Lan)
* Mai (Mai)
* Thọ (Tho)

## Cấu trúc thư mục

flower-classifier-pipeline/
├── common/
│   └── feature_extractor.py

├── components/
│   ├── augment/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── augment.py

│   ├── train/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── train.py

│   ├── evaluate/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── evaluate.py

│   └── serve/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── app.py
│       └── model_loader.py

├── pipeline/
│   └── pipeline.py

├── data/
│   ├── train/
│   │   ├── Cuc/
│   │   ├── Dao/
│   │   ├── Lan/
│   │   ├── Mai/
│   │   └── Tho/

│   └── test/
│       ├── Cuc/
│       ├── Dao/
│       ├── Lan/
│       ├── Mai/
│       └── Tho/

└── .github/workflows/

## Yêu cầu hệ thống

* Python 3.10+
* Docker
* Docker Compose (khuyến nghị)
* Git

## Dataset

### Training Data

Mỗi loại hoa gồm:

* 30 ảnh train / class

Tổng cộng:

* 150 ảnh train

### Testing Data

Mỗi loại hoa gồm:

* 10 ảnh test / class

Tổng cộng:

* 50 ảnh test

### Cấu trúc dữ liệu

data/
├── train/
│   ├── Cuc/
│   ├── Dao/
│   ├── Lan/
│   ├── Mai/
│   └── Tho/

└── test/
├── Cuc/
├── Dao/
├── Lan/
├── Mai/
└── Tho/

## Cài đặt project

### Clone repository

git clone <your-repository-url>
cd flower-classifier-pipeline

## Chạy từng component bằng Docker

### 1. Data Augmentation

cd components/augment
docker build -t flower-augment .
docker run flower-augment

### 2. Model Training

cd components/train
docker build -t flower-train .
docker run flower-train

### 3. Model Evaluation

cd components/evaluate
docker build -t flower-evaluate .
docker run flower-evaluate

### 4. Model Serving

cd components/serve
docker build -t flower-serve .
docker run -p 8000:8000 flower-serve

API sẽ chạy tại:

http://localhost:8000

## Chạy toàn bộ pipeline

cd pipeline
python pipeline.py

Pipeline sẽ tự động thực hiện:

* Augmentation
* Training
* Evaluation
* Serving

## API Inference

Ví dụ gọi API dự đoán:

curl -X POST \
-F "file=@flower.jpg" \
http://localhost:8000/predict

Kết quả trả về:

{
"prediction": "Lan",
"confidence": 0.96
}

## CI/CD (Tuỳ chọn)

Thư mục:

.github/workflows/

dùng để cấu hình GitHub Actions cho:

* tự động test
* build Docker image
* deploy pipeline

## Tác giả

Project phục vụ mục đích học tập về:

* MLOps
* Machine Learning Pipeline
* Dockerized ML Components
* Model Serving
* CI/CD cho Machine Learning

## Ghi chú

Khuyến nghị sử dụng:

* FastAPI cho serve API
* Scikit-learn hoặc TensorFlow/PyTorch cho training
* Docker Compose để orchestration
* GitHub Actions cho CI/CD
