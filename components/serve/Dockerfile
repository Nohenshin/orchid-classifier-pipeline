FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common/ /app/common/
COPY components/serve/model_loader.py components/serve/app.py ./
ENTRYPOINT ["python", "app.py"]