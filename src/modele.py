"""Construction du pipeline scikit-learn (identique au notebook, partie 2.3)."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.preparation import VARIABLES_CATEGORIELLES


def creer_preprocesseur():
    log_scale = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())])
    return ColumnTransformer([
        ("log_num", log_scale, ["Prix_neuf_CAD", "Kms_Driven"]),
        ("num", StandardScaler(), ["Age_vehicule", "Owner"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), VARIABLES_CATEGORIELLES),
    ])


def cible_log(modele):
    return TransformedTargetRegressor(regressor=modele, func=np.log1p, inverse_func=np.expm1)


def pipeline(modele):
    return Pipeline([("prep", creer_preprocesseur()), ("modele", modele)])
