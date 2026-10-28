#Construcción del modelo principal (Gradient Boosting) y del modelo base.

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import clean_features
from .preprocessing import build_preprocessor


def build_model(ref_year: int, seed: int, max_iter: int = 400, learning_rate: float = 0.05) -> TransformedTargetRegressor:
    feature_builder = FunctionTransformer(clean_features, kw_args={"ref_year": ref_year})
    pipeline = Pipeline([
        ("features", feature_builder),
        ("prep", build_preprocessor()),
        ("reg", HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, random_state=seed,
        )),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def build_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="median")