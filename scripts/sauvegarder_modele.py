"""Entraîne le modèle final et le SÉRIALISE avec MLflow (+ copie joblib locale).

Usage :
    python scripts/sauvegarder_modele.py                       # régression linéaire (log)
    python scripts/sauvegarder_modele.py --modele ridge

Suivi MLflow :
    - par défaut, une base SQLite locale ./mlflow.db + ./mlruns (aucune configuration)
    - DagsHub : définir MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
      (ou exécuter dagshub.init(...) comme dans le notebook)

Ce que le script produit :
    models/modele_final.joblib    paquet utilisé par l'API (modèle + intervalle + domaine)
    models/reference_suivi.json   (référence de dérive, voir notebook 3.4)
    MLflow : run + modèle sklearn (avec signature) + artefacts + modèle enregistré « AutoRachat »
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permet `import src`

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from src.modele import cible_log, pipeline
from src.preparation import (ANNEE_REFERENCE, CIBLE, FEATURES, TAUX_INR_CAD, nettoyer)

RACINE = Path(__file__).resolve().parents[1]
SEED = 42
NIVEAU = 0.80
SEUIL_PRIX_NEUF = 60_000
NOM_MODELE_MLFLOW = "AutoRachat"


def main(choix: str, donnees: Path):
    df, _ = nettoyer(pd.read_csv(donnees))
    df = df[df["Prix_neuf_CAD"] <= SEUIL_PRIX_NEUF].reset_index(drop=True)
    X, y = df[FEATURES], df[CIBLE]

    algo = {"lineaire": LinearRegression(), "ridge": Ridge(alpha=1.0)}[choix]
    nom = {"lineaire": "Régression linéaire (log)", "ridge": "Ridge (log)"}[choix]

    # Évaluation honnête : test mis de côté, comme dans le notebook (80/20 puis réentraînement)
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    modele = pipeline(cible_log(algo)).fit(X_trval, y_trval)
    pred = modele.predict(X_test)
    metriques = {"MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred),
                 "MAPE_pourcent": 100 * mean_absolute_percentage_error(y_test, pred)}

    # Intervalle de prédiction (split conformal, échelle log)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(modele), X_trval, y_trval, cv=cv)
    err = np.abs(np.log1p(y_trval) - np.log1p(oof))
    n = len(err)
    q = float(np.quantile(err, min(1, np.ceil((n + 1) * NIVEAU) / n)))

    # Modèle déployé : réentraîné sur toutes les données
    modele_deploye = clone(modele).fit(X, y)
    domaine = {
        "Prix_neuf_CAD": [float(X.Prix_neuf_CAD.min()), float(X.Prix_neuf_CAD.max())],
        "Age_vehicule": [int(X.Age_vehicule.min()), int(X.Age_vehicule.max())],
        "Kms_Driven": [int(X.Kms_Driven.min()), int(X.Kms_Driven.max())],
        "Year": [int(df.Year.min()), int(df.Year.max())],
        "Fuel_Type": sorted(X.Fuel_Type.unique()), "Seller_Type": sorted(X.Seller_Type.unique()),
        "Transmission": sorted(X.Transmission.unique()), "Marque": sorted(X.Marque.unique()),
    }
    paquet = {"modele": modele_deploye, "nom": nom, "q_log": q, "niveau": NIVEAU,
              "features": FEATURES, "annee_reference": ANNEE_REFERENCE, "domaine": domaine,
              "devise": "CAD", "taux_inr_cad": TAUX_INR_CAD,
              "metriques_test": metriques, "version": "1.0.0",
              "date_entrainement": time.strftime("%Y-%m-%d")}

    (RACINE / "models").mkdir(exist_ok=True)
    chemin_joblib = RACINE / "models" / "modele_final.joblib"
    joblib.dump(paquet, chemin_joblib)

    # ---- Sérialisation MLflow ----
    if not os.getenv("MLFLOW_TRACKING_URI"):  # sinon : base SQLite locale (supporte le registre)
        mlflow.set_tracking_uri(f"sqlite:///{RACINE / 'mlflow.db'}")
    mlflow.set_experiment("AutoRachat - deploiement")

    exemple = X.head(5)
    signature = infer_signature(exemple, modele_deploye.predict(exemple))

    with mlflow.start_run(run_name=f"Modele final - {nom}") as run:
        mlflow.log_params({"modele_final": nom, "version": paquet["version"], "devise": "CAD",
                           "niveau_intervalle": NIVEAU, "n_lignes_entrainement": len(X)})
        mlflow.log_metrics({k: float(v) for k, v in metriques.items()})
        mlflow.log_metric("q_log", q)
        info = mlflow.sklearn.log_model(
            sk_model=modele_deploye, name="modele_final", signature=signature,
            input_example=exemple, registered_model_name=NOM_MODELE_MLFLOW)
        mlflow.log_artifact(str(chemin_joblib), artifact_path="modele_deploiement")
        mlflow.log_dict(domaine, "domaine_validite.json")
        run_id = run.info.run_id

    # Alias « production » sur la version qui vient d'être enregistrée (si le registre le permet)
    try:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(NOM_MODELE_MLFLOW, "production", info.registered_model_version)
    except Exception as e:  # le registre peut être restreint selon le serveur
        print("Alias non défini :", e)

    print(f"Modèle : {nom} | MAE test = {metriques['MAE']:.0f} $ | R2 = {metriques['R2']:.3f}")
    print("Joblib :", chemin_joblib)
    print("MLflow run_id :", run_id)
    print(f"Pour recharger : mlflow.sklearn.load_model('models:/{NOM_MODELE_MLFLOW}@production')")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=["lineaire", "ridge"], default="lineaire")
    ap.add_argument("--donnees", type=Path, default=RACINE / "data" / "car_data.csv")
    a = ap.parse_args()
    main(a.modele, a.donnees)
