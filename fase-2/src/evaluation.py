# Métricas de evaluación

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate(y_true, y_pred) -> dict:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr < 200_000
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(root_mean_squared_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae_lt_200k": float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask])),
        "r2_lt_200k": float(r2_score(y_true_arr[mask], y_pred_arr[mask])),
    }