"""Fonctions de préparation des données (reprises du notebook, partie 2)."""
import numpy as np
import pandas as pd

ANNEE_REFERENCE = 2019            # année de collecte du jeu CarDekho
TAUX_INR_CAD = 0.01488            # 1 INR en CAD (Banque du Canada, 2026-10-21)
CAD_PAR_LAKH = 100_000 * TAUX_INR_CAD
CIBLE = "Prix_vente_CAD"
VARIABLES_NUMERIQUES = ["Prix_neuf_CAD", "Age_vehicule", "Kms_Driven", "Owner"]
VARIABLES_CATEGORIELLES = ["Fuel_Type", "Seller_Type", "Transmission", "Marque"]
FEATURES = VARIABLES_NUMERIQUES + VARIABLES_CATEGORIELLES

MARQUES = {
    "Maruti": ["800", "alto 800", "alto k10", "baleno", "ciaz", "dzire", "ertiga", "ignis", "omni",
               "ritz", "s cross", "swift", "sx4", "vitara brezza", "wagon r"],
    "Toyota": ["camry", "corolla", "corolla altis", "etios cross", "etios g", "etios gd", "etios liva",
               "fortuner", "innova", "land cruiser"],
    "Hyundai": ["creta", "elantra", "eon", "grand i10", "i10", "i20", "verna", "xcent"],
    "Honda": ["amaze", "brio", "city", "jazz"],
}
MODELE_VERS_MARQUE = {m: marque for marque, modeles in MARQUES.items() for m in modeles}


def ajouter_variables_derivees(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Marque"] = d["Car_Name"].str.strip().str.lower().map(MODELE_VERS_MARQUE)
    d["Type_vehicule"] = np.where(d["Marque"].notna(), "Voiture", "Deux-roues")
    d["Age_vehicule"] = ANNEE_REFERENCE - d["Year"]
    d["Prix_vente_CAD"] = (d["Selling_Price"] * CAD_PAR_LAKH).round()
    d["Prix_neuf_CAD"] = (d["Present_Price"] * CAD_PAR_LAKH).round()
    return d


def nettoyer(d: pd.DataFrame):
    """Doublons -> variables dérivées -> exclusion des deux-roues. Renvoie (données, journal)."""
    journal = {"lignes_initiales": len(d), "doublons_supprimes": int(d.duplicated().sum())}
    d = ajouter_variables_derivees(d.drop_duplicates())
    journal["deux_roues_exclus"] = int((d["Type_vehicule"] == "Deux-roues").sum())
    d = d[d["Type_vehicule"] == "Voiture"].reset_index(drop=True)
    journal["valeurs_manquantes"] = int(d.isna().sum().sum())
    journal["lignes_finales"] = len(d)
    return d, journal


def preparer_entree(annee, prix_neuf_cad, kms, carburant, vendeur, transmission,
                    proprietaires, marque_) -> pd.DataFrame:
    """Construit UNE ligne au format attendu par le modèle (même logique que le notebook)."""
    return pd.DataFrame([{
        "Prix_neuf_CAD": prix_neuf_cad, "Age_vehicule": ANNEE_REFERENCE - annee,
        "Kms_Driven": kms, "Owner": proprietaires, "Fuel_Type": carburant,
        "Seller_Type": vendeur, "Transmission": transmission, "Marque": marque_,
    }])[FEATURES]
