FROM python:3.12-slim

WORKDIR /code

# 1) Dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code et modèle sérialisé
COPY src/ src/
COPY app/ app/
COPY models/modele_final.joblib models/modele_final.joblib

# Bonne pratique : ne pas exécuter en root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
