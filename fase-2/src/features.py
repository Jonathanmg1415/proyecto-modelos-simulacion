from __future__ import annotations

import re

import numpy as np
import pandas as pd

KNOWN_FUEL = {
    "Gasoline", 
    "Hybrid", 
    "E85 Flex Fuel", 
    "Diesel", 
    "Plug-In Hybrid"
    }

NUM_FEATURES = [
    "milage_num", 
    "antiguedad", 
    "engine_hp", 
    "engine_liters",
    "engine_cylinders", 
    "transmission_speeds", 
    "has_clean_title",
]

CAT_LOW = ["brand", "fuel_type_clean", "accident_clean", "transmission_type"]

CAT_HIGH = ["ext_col", "int_col"]


def parse_money(serie: pd.Series) -> pd.Series:
    """'$10,300' -> 10300.0 ; '51,000 mi.' -> 51000.0. Determinista, sin ajuste."""
    return pd.to_numeric(
        serie.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )


def clean_features(data: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    out["milage_num"] = parse_money(data["milage"])
    out["antiguedad"] = ref_year - pd.to_numeric(data["model_year"], errors="coerce")

    eng = data["engine"].astype(str)
    out["engine_hp"] = pd.to_numeric(eng.str.extract(r"([\d.]+)\s*HP", flags=re.I)[0], errors="coerce")
    out["engine_liters"] = pd.to_numeric(eng.str.extract(r"([\d.]+)\s*L", flags=re.I)[0], errors="coerce")
    cyl = pd.to_numeric(eng.str.extract(r"(\d+)\s*Cylinder", flags=re.I)[0], errors="coerce")
    cyl_alt = pd.to_numeric(eng.str.extract(r"\b[VIH](\d{1,2})\b", flags=re.I)[0], errors="coerce")
    out["engine_cylinders"] = cyl.fillna(cyl_alt)

    trans = data["transmission"].astype(str)
    out["transmission_speeds"] = pd.to_numeric(trans.str.extract(r"(\d+)-Speed", flags=re.I)[0], errors="coerce")
    t = trans.str.lower()
    out["transmission_type"] = np.select(
        [t.str.contains("cvt"), t.str.contains(r"m/t|manual"), t.str.contains(r"a/t|automatic|dual shift")],
        ["CVT", "Manual", "Automatic"], default="Other",
    )

    out["brand"] = data["brand"].astype(str)
    out["fuel_type_clean"] = data["fuel_type"].where(data["fuel_type"].isin(KNOWN_FUEL), "Unknown").astype(str)
    out["accident_clean"] = data["accident"].fillna("Unknown").astype(str)
    out["has_clean_title"] = (data["clean_title"] == "Yes").astype("int64")
    out["ext_col"] = data["ext_col"].astype(str)
    out["int_col"] = data["int_col"].astype(str)
    return out