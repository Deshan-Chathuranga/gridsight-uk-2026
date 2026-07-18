# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/apps/frontend
COPY apps/frontend/package*.json ./
RUN npm install
COPY apps/frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend
FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file first to cache pip layers
COPY requirements.txt ./

# Install PyTorch CPU first to avoid heavy CUDA wheels (saves >2GB image size)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and other codebase folders
COPY . .

# Copy the built frontend dist to the correct location
COPY --from=frontend-builder /app/apps/frontend/dist /app/apps/frontend/dist

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "apps.backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
