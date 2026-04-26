FROM python:3.11-slim

# System deps for PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the FastEmbed model during build so it's cached in the image
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-base-en-v1.5').embed(['warmup']))"

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p /app/data

# HuggingFace Spaces runs as a non-root user — ensure data dir is writable
RUN chmod -R 777 /app/data

# HuggingFace Spaces requires port 7860
# Build version: 2
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]