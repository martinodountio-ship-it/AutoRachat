"""API FastAPI d'AutoRachat : estimation du prix d'une voiture d'occasion.

Lancer en local :   uvicorn app.api:app --reload
Documentation :     http://127.0.0.1:8000/docs   (Swagger généré automatiquement)
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chargement import charger_paquet
from src.prediction import avertissements_domaine, predire

ETAT = {}  # contient le modèle chargé au démarrage (chargé une seule fois, pas à chaque requête)


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    ETAT["paquet"] = charger_paquet()
    yield
    ETAT.clear()


app = FastAPI(
    title="AutoRachat - Estimation du prix d'une voiture d'occasion",
    description="Envoie les caractéristiques d'un véhicule et reçoit un prix estimé en CAD "
                "avec un intervalle de prédiction à 80 %.",
    version="1.0.0",
    lifespan=cycle_de_vie,
)


class Vehicule(BaseModel):
    """Corps de la requête POST /predict (validé automatiquement par Pydantic)."""
    annee: int = Field(..., ge=1990, le=2019, description="Année du véhicule", examples=[2015])
    prix_neuf_cad: float = Field(..., gt=0, le=200_000, description="Prix du modèle neuf (CAD)",
                                 examples=[14700])
    kms: int = Field(..., ge=0, le=1_000_000, description="Kilométrage", examples=[40000])
    carburant: Literal["Petrol", "Diesel", "CNG"] = Field(..., examples=["Petrol"])
    vendeur: Literal["Dealer", "Individual"] = Field(..., examples=["Dealer"])
    transmission: Literal["Manual", "Automatic"] = Field(..., examples=["Manual"])
    proprietaires: int = Field(..., ge=0, le=3, description="Nombre de propriétaires précédents",
                               examples=[0])
    marque: Literal["Maruti", "Toyota", "Hyundai", "Honda"] = Field(..., examples=["Honda"])


class Prediction(BaseModel):
    prix_estime: int
    intervalle_bas: int
    intervalle_haut: int
    niveau_intervalle: float
    devise: str
    version_modele: str
    avertissements: list[str] = []


@app.get("/health", tags=["Suivi"])
def health():
    """Vérifie que le service tourne et que le modèle est chargé (utile pour Docker / le monitoring)."""
    return {"statut": "ok", "modele_charge": "paquet" in ETAT}


@app.get("/info", tags=["Suivi"])
def info():
    """Métadonnées du modèle en production."""
    p = ETAT["paquet"]
    return {"modele": p["nom"], "version": p["version"], "date_entrainement": p["date_entrainement"],
            "devise": p["devise"], "metriques_test": p["metriques_test"], "domaine": p["domaine"]}


@app.post("/predict", response_model=Prediction, tags=["Prédiction"])
def predict(v: Vehicule):
    """Retourne le prix estimé et son intervalle pour un véhicule."""
    p = ETAT.get("paquet")
    if p is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        res = predire(p, annee=v.annee, prix_neuf_cad=v.prix_neuf_cad, kms=v.kms,
                      carburant=v.carburant, vendeur=v.vendeur, transmission=v.transmission,
                      proprietaires=v.proprietaires, marque=v.marque)
    except Exception as e:  # erreur inattendue du modèle : ne pas exposer les détails internes
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.") from e
    res["version_modele"] = p["version"]
    res["avertissements"] = avertissements_domaine(p, annee=v.annee, prix_neuf_cad=v.prix_neuf_cad,
                                                   kms=v.kms)
    return res
