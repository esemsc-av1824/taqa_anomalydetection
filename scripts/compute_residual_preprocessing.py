#!/usr/bin/env python3
"""
Compute and persist residual preprocessing scalers to ensure inference uses the
same transforms as training:
- Pressures: signed log1p only (no scaler saved)
- Temperatures: RobustScaler (center, scale)
- Choke-Position: StandardScaler (mean, scale)

Outputs: models_4/residual_preprocessing.json

This script infers feature groups from the ONNX model's feature_names metadata
and the conventional column names used in this project.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import onnx
except Exception:
    onnx = None


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models_4"
TRAIN_PATHS = [
    ROOT / "training_data" / "wide36_tools_flat.parquet",
    ROOT / "training_data" / "wide36_tools_flat.parq",
]


def signed_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def load_training_df() -> pd.DataFrame:
    for p in TRAIN_PATHS:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"Could not find training parquet in {TRAIN_PATHS}")


def get_feature_names_from_onnx(onnx_path: Path) -> List[str]:
    if onnx is None:
        raise RuntimeError("onnx package not available to read metadata")
    m = onnx.load(str(onnx_path))
    feats = None
    for kv in m.metadata_props:
        if kv.key == "feature_names":
            feats = kv.value.split(",")
            break
    if not feats:
        raise RuntimeError("feature_names metadata missing in ONNX model")
    return feats


def group_features(feature_names: List[str]) -> Dict[str, List[str]]:
    # Project conventions
    pressure_aliases = {
        "Upstream-Pressure",
        "Downstream-Pressure",
        "Downstream-Upstream-Difference",
    }
    temp_aliases = {"Upstream-Temperature", "Downstream-Temperature"}
    choke_aliases = {"Choke-Position"}

    pressures = [c for c in feature_names if c in pressure_aliases or "Pressure" in c]
    temps = [c for c in feature_names if c in temp_aliases or "Temperature" in c]
    chokes = [c for c in feature_names if c in choke_aliases or "Choke" in c]

    # Ensure stable order as in feature_names
    pressures = [c for c in feature_names if c in pressures]
    temps = [c for c in feature_names if c in temps]
    chokes = [c for c in feature_names if c in chokes]

    return {"pressures": pressures, "temps": temps, "chokes": chokes}


def main():
    models_dir = MODELS_DIR
    onnx_model = models_dir / "residual_battery.onnx"
    if not onnx_model.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")

    feature_names = get_feature_names_from_onnx(onnx_model)
    groups = group_features(feature_names)

    df = load_training_df()

    # Derive pressure difference if missing
    if "Downstream-Upstream-Difference" in feature_names and \
       "Downstream-Upstream-Difference" not in df.columns:
        if {"Downstream-Pressure", "Upstream-Pressure"}.issubset(df.columns):
            df["Downstream-Upstream-Difference"] = (
                df["Downstream-Pressure"] - df["Upstream-Pressure"]
            )
        else:
            df["Downstream-Upstream-Difference"] = 0.0

    # Prepare arrays
    df_feats = df[feature_names].copy()

    # Apply log to pressures (no scaler saved for them)
    for c in groups["pressures"]:
        if c in df_feats.columns:
            df_feats[c] = signed_log1p(df_feats[c].astype(float))

    # Fit scalers
    robust = RobustScaler()
    std = StandardScaler()

    temp_cols = groups["temps"]
    choke_cols = groups["chokes"]

    scalers_info: Dict[str, Dict] = {
        "pressure": {"method": "log1p", "cols": groups["pressures"]},
        "temperature": {"method": "robust", "cols": temp_cols, "center": {}, "scale": {}},
        "choke": {"method": "standard", "cols": choke_cols, "center": {}, "scale": {}},
        "feature_order": feature_names,
    }

    if temp_cols:
        robust.fit(df_feats[temp_cols].astype(float).values)
        center = robust.center_.tolist()
        scale = robust.scale_.tolist()
        for i, c in enumerate(temp_cols):
            scalers_info["temperature"]["center"][c] = center[i]
            scalers_info["temperature"]["scale"][c] = scale[i]

    if choke_cols:
        std.fit(df_feats[choke_cols].astype(float).values)
        center = std.mean_.tolist()
        scale = std.scale_.tolist()
        for i, c in enumerate(choke_cols):
            scalers_info["choke"]["center"][c] = center[i]
            scalers_info["choke"]["scale"][c] = scale[i]

    out_path = models_dir / "residual_preprocessing.json"
    out_path.write_text(json.dumps(scalers_info, indent=2))
    print(f"✓ Wrote {out_path} with:")
    print(f"  pressures: {groups['pressures']}")
    print(f"  temps: {temp_cols} (robust)")
    print(f"  chokes: {choke_cols} (standard)")


if __name__ == "__main__":
    main()
