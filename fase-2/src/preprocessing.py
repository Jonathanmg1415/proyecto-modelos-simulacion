# Construcción del ColumnTransformer de preprocesamiento (fase-1/notebook.ipynb, sección 5.3).
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import NUM_FEATURES, CAT_LOW, CAT_HIGH


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), NUM_FEATURES),
            ("cat_low", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CAT_LOW),
            ("cat_high", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=0.01, sparse_output=False)),
            ]), CAT_HIGH),
        ],
        remainder="drop",
    )