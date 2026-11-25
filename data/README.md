# AutoRachat — Prédiction du prix d'une voiture d'occasion (Groupe 8)

Phase de déploiement : modèle sérialisé avec MLflow, API FastAPI, conteneur Docker.

## Structure du projet
```
autorachat/
├── data/            car_data.csv (brut), car_data_nettoye.csv (généré)
├── notebooks/       Projet_Prix_Voiture_Occasion.ipynb (exploration, modèles, export)
├── src/             preparation.py · modele.py · prediction.py · chargement.py
├── scripts/         sauvegarder_modele.py  (entraîne + sérialise avec MLflow)
├── app/             api.py (FastAPI) ; app.py (prototype Streamlit, jalon 3)
├── models/          modele_final.joblib · reference_suivi.json
├── tests/           test_api.py · test_serialisation.py
├── reports/         figures/ · resultats.json
├── Dockerfile · requirements.txt · .dockerignore · .gitignore
```

## 1. Sérialiser le modèle (MLflow)
```bash
pip install -r requirements-dev.txt
python scripts/sauvegarder_modele.py            # ou --modele ridge
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```
Le script enregistre dans MLflow : paramètres, métriques, modèle scikit-learn (avec *signature* et
*input example*), artefacts (paquet joblib, domaine de validité) et un modèle **enregistré** nommé
`AutoRachat`, avec l'alias `production`. Rechargement :
```python
import mlflow.sklearn
modele = mlflow.sklearn.load_model("models:/AutoRachat@production")
```
Pour utiliser DagsHub plutôt que le stockage local : définir `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` avant de lancer le script.

## 2. Lancer l'API
```bash
uvicorn app.api:app --reload      # documentation interactive : http://127.0.0.1:8000/docs
```
| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | le service tourne et le modèle est chargé |
| GET | `/info` | version, métriques et domaine de validité du modèle |
| POST | `/predict` | prix estimé + intervalle à 80 % |

Exemple :
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "annee": 2015, "prix_neuf_cad": 14700, "kms": 40000, "carburant": "Petrol",
  "vendeur": "Dealer", "transmission": "Manual", "proprietaires": 0, "marque": "Honda"}'
```

## 3. Docker
```bash
docker build -t autorachat-api .
docker run -p 8000:8000 autorachat-api
```

## 4. Tests
```bash
pytest -q
```

## Risques de déploiement identifiés
- **Version de scikit-learn différente** entre entraînement et production → figée dans `requirements.txt`.
- **Entrées hors domaine** (prix du neuf > 60 000 $, années hors 2003–2018) → validation Pydantic + `avertissements`.
- **Dérive des données** → `reference_suivi.json` (PSI), à brancher sur le suivi.
- **Taux de change INR→CAD figé** → à mettre à jour à chaque réentraînement.
- **Peu de données (≈ 197 voitures)** → intervalle de prédiction toujours communiqué.
