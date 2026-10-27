# ==============================================================================
# FlightFair: Production Dockerfile
# Multi-stage optimized container for Flask REST API & Predictive Pricing Engine
# ==============================================================================

FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Image
FROM python:3.11-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Copy application source code and model artifacts
COPY src/ /app/src/
COPY static/ /app/static/
COPY templates/ /app/templates/
COPY model/ /app/model/
COPY data/ /app/data/
COPY app.py /app/
COPY setup.py /app/
COPY README.md /app/

# Install flightfair package in container
RUN pip install --no-cache-dir -e .

EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI Server entrypoint using waitress
CMD ["waitress-serve", "--port=5000", "--call", "app:app"]
