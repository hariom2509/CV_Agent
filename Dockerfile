FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY cv_agent/ ./cv_agent/
COPY cli.py .
COPY app.py .

# Create data and output directories
RUN mkdir -p data output

# Default command: CLI
CMD ["python", "cli.py"]

EXPOSE 8501
