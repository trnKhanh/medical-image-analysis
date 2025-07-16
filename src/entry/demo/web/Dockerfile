FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/models data/datasets

EXPOSE 8000

CMD ["uvicorn", "entry.demo.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
