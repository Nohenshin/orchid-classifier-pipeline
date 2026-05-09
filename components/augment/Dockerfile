FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY augment.py .
ENTRYPOINT ["python", "augment.py"]