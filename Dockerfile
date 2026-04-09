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
# This avoids a cold-start delay on first request
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]