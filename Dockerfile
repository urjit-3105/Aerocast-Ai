# Production Containerization for Aero-Cast AI
FROM python:3.11-slim

WORKDIR /app

# Prevent python from buffering stdout/stderr and writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck probe
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Execute Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
