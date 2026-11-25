"""Tests de l'API. Nécessitent models/modele_final.joblib (voir scripts/sauvegarder_modele.py)."""
import pytest
from fastapi.testclient import TestClient

from app.api import app

VEHICULE = {"annee": 2015, "prix_neuf_cad": 14700, "kms": 40000, "carburant": "Petrol",
            "vendeur": "Dealer", "transmission": "Manual", "proprietaires": 0, "marque": "Honda"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # déclenche le chargement du modèle (lifespan)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["modele_charge"] is True


def test_predict_ok(client):
    r = client.post("/predict", json=VEHICULE)
    assert r.status_code == 200
    j = r.json()
    assert j["prix_estime"] > 0
    assert j["intervalle_bas"] <= j["prix_estime"] <= j["intervalle_haut"]
    assert j["devise"] == "CAD"


def test_predict_plus_vieux_moins_cher(client):
    recent = client.post("/predict", json={**VEHICULE, "annee": 2017}).json()["prix_estime"]
    ancien = client.post("/predict", json={**VEHICULE, "annee": 2008}).json()["prix_estime"]
    assert ancien < recent


@pytest.mark.parametrize("champ,valeur", [("carburant", "Electrique"), ("marque", "Tesla"),
                                          ("kms", -5), ("annee", 1900), ("prix_neuf_cad", 0)])
def test_predict_entree_invalide(client, champ, valeur):
    r = client.post("/predict", json={**VEHICULE, champ: valeur})
    assert r.status_code == 422


def test_predict_champ_manquant(client):
    incomplet = {k: v for k, v in VEHICULE.items() if k != "kms"}
    assert client.post("/predict", json=incomplet).status_code == 422


def test_avertissement_hors_domaine(client):
    r = client.post("/predict", json={**VEHICULE, "prix_neuf_cad": 150000})
    assert r.status_code == 200 and r.json()["avertissements"]
