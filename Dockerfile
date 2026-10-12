# ---- Loan Approval Prediction: production image ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only what the app needs at runtime (see .dockerignore)
COPY app.py loan_model.py ./
COPY model/ ./model/

# Run as a non-root user (safer)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# The git commit id is passed in by the CI pipeline and shown in the app sidebar
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8501

# Streamlit has a built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
