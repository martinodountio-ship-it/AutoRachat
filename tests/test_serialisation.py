"""Vérifie que le modèle sérialisé se recharge (joblib ET registre MLflow) avec les mêmes prédictions."""
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest

from src.chargement import charger_paquet
from src.preparation import preparer_entree

RACINE = Path(__file__).resolve().parents[1]
ENTREE = preparer_entree(2015, 14700, 40000, "Petrol", "Dealer", "Manual", 0, "Honda")


def test_paquet_contient_les_cles():
    p = charger_paquet()
    for cle in ["modele", "q_log", "niveau", "domaine", "version", "features"]:
        assert cle in p


def test_rechargement_mlflow_identique_au_joblib():
    base = RACINE / "mlflow.db"
    if not base.exists():
        pytest.skip("Pas de base MLflow locale : lancez scripts/sauvegarder_modele.py")
    mlflow.set_tracking_uri(f"sqlite:///{base}")
    depuis_mlflow = mlflow.sklearn.load_model("models:/AutoRachat@production")
    depuis_joblib = charger_paquet()["modele"]
    assert np.allclose(depuis_mlflow.predict(ENTREE), depuis_joblib.predict(ENTREE))
