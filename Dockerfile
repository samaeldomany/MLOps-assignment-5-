FROM python:3.10-slim
ARG RUN_ID
ENV MODEL_ID=${RUN_ID}

RUN echo "Packaging model from MLflow Run: ${MODEL_ID}"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .