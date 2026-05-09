FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY common/ /app/common/
COPY components/train/train.py .

ENTRYPOINT ["python", "train.py"]