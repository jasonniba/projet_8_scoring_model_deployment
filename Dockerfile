FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY models/credit_scoring_model_onnx/ ./models/credit_scoring_model_onnx/

COPY src/ ./src/


EXPOSE 8000


CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]