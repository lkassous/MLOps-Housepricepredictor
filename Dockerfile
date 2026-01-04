# Multi-stage Dockerfile pour production MLOps
# Stage 1: Builder - installe les dépendances
FROM python:3.11-slim as builder

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Créer un virtual environment et installer les dépendances
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - image finale légère
FROM python:3.11-slim

WORKDIR /app

# Copier le virtual environment depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

# Copier les fichiers de l'application  
COPY --chown=mlops:mlops app.py .
COPY --chown=mlops:mlops train_simple_model.py .
COPY --chown=mlops:mlops train.csv .
COPY --chown=mlops:mlops static/ ./static/

# Créer le dossier model_artifacts
RUN mkdir -p model_artifacts

# Entraîner le modèle pendant le build (génère model_artifacts/)
RUN python train_simple_model.py

# Variables d'environnement
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Exposer le port
EXPOSE 8080

# Utiliser l'utilisateur non-root
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Commande de démarrage de l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
