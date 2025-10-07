FROM python:3.11-slim

# System deps (CBC solver, compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY backend/app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
