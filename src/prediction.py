"""Logique de prédiction partagée (API, Streamlit, tests) : prix + intervalle."""
import numpy as np

from src.preparation import preparer_entree


def predire(paquet: dict, *, annee, prix_neuf_cad, kms, carburant, vendeur,
            transmission, proprietaires, marque) -> dict:
    """Renvoie le prix estimé et l'intervalle de prédiction (en CAD)."""
    x = preparer_entree(annee, prix_neuf_cad, kms, carburant, vendeur, transmission,
                        proprietaires, marque)
    prix = float(paquet["modele"].predict(x)[0])
    q = paquet["q_log"]
    return {
        "prix_estime": round(prix),
        "intervalle_bas": round(float(np.expm1(np.log1p(prix) - q))),
        "intervalle_haut": round(float(np.expm1(np.log1p(prix) + q))),
        "niveau_intervalle": paquet["niveau"],
        "devise": paquet.get("devise", "CAD"),
    }


def avertissements_domaine(paquet: dict, *, annee, prix_neuf_cad, kms) -> list:
    """Signale les entrées hors du domaine d'entraînement (le modèle extrapole alors)."""
    d = paquet["domaine"]
    alertes = []
    if not d["Year"][0] <= annee <= d["Year"][1]:
        alertes.append(f"Année hors du domaine d'entraînement {d['Year']}.")
    if not d["Prix_neuf_CAD"][0] <= prix_neuf_cad <= d["Prix_neuf_CAD"][1]:
        alertes.append("Prix du neuf hors du domaine d'entraînement : estimation moins fiable.")
    if not d["Kms_Driven"][0] <= kms <= d["Kms_Driven"][1]:
        alertes.append("Kilométrage hors du domaine d'entraînement : estimation moins fiable.")
    return alertes
