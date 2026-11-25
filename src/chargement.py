"""Chargement du modèle sérialisé.

Deux sources possibles :
  1. MLflow : variable MLFLOW_RUN_ID -> télécharge l'artefact 'modele_deploiement/modele_final.joblib'
     du run (le paquet contient le modèle + l'intervalle de prédiction + le domaine de validité).
  2. Fichier local (défaut) : models/modele_final.joblib, ou le chemin donné par MODEL_PATH.
"""
import os
from pathlib import Path

import joblib

RACINE = Path(__file__).resolve().parents[1]


def charger_paquet() -> dict:
    run_id = os.getenv("MLFLOW_RUN_ID")
    if run_id:
        from mlflow.artifacts import download_artifacts
        chemin = download_artifacts(
            artifact_uri=f"runs:/{run_id}/modele_deploiement/modele_final.joblib")
        return joblib.load(chemin)

    chemin = Path(os.getenv("MODEL_PATH", RACINE / "models" / "modele_final.joblib"))
    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}. Lancez d'abord scripts/sauvegarder_modele.py "
            "(ou la section 3.2 du notebook).")
    return joblib.load(chemin)
