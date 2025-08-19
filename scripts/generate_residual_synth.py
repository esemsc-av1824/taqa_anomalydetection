#!/usr/bin/env python3
"""
Generate a synthetic dataset for residual models that preserves the X–y
relationship for normals and calibrates residual magnitudes to the training scale.

Design:
- Normals:
    • Bootstrap X rows from the central training band (5–95% per feature).
    • Set target y ≈ model(X) + ε, where ε is sampled from the empirical training
        residuals (normal subset). This makes normal residuals match training MAD.
- Anomalies (50/50 mix):
    • Covariate-only perturbations: perturb X (scaled to per-feature std), keep y from
        original row (so model(X_pert) ≠ y), inducing residual.
    • Target-only drift: set y = model(X) + k·ε, where k depends on difficulty tier.
- Difficulty mix: 40% easy, 40% medium, 20% hard.

Output: anomaly_detection_analysis/synthetic_data/challenging_residual_battery_500pts.parquet
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import onnx
except Exception:
    onnx = None
import onnxruntime as ort

BASE = Path(__file__).resolve().parents[1]
TRAIN = BASE / 'training_data' / 'wide36_tools_flat.parquet'
MODELS = BASE / 'models_4'
OUT = BASE / 'anomaly_detection_analysis' / 'synthetic_data' / 'challenging_residual_battery_500pts.parquet'

MODEL_NAME = 'residual_battery.onnx'
TARGET = 'Battery-Voltage'


def get_feature_names_from_onnx(onnx_path: Path):
    if onnx is None:
        raise RuntimeError('onnx package is required to read feature_names metadata')
    m = onnx.load(str(onnx_path))
    for kv in m.metadata_props:
        if kv.key == 'feature_names':
            return kv.value.split(',')
    raise RuntimeError('feature_names metadata missing in ONNX')


def ensure_pressure_diff(df: pd.DataFrame):
    if 'Downstream-Upstream-Difference' not in df.columns and {
        'Downstream-Pressure','Upstream-Pressure'
    }.issubset(df.columns):
        df['Downstream-Upstream-Difference'] = df['Downstream-Pressure'] - df['Upstream-Pressure']
    return df


def load_training_preprocessing(prep_path: Path):
    """Load saved training-time preprocessing stats (for temps/choke) and feature order."""
    if not prep_path.exists():
        return None
    with open(prep_path) as f:
        prep = json.load(f)
    # Normalize schema for convenience
    result = {
        'pressure_cols': prep.get('pressure', {}).get('cols', []),
        'pressure_method': prep.get('pressure', {}).get('method', 'log1p'),
        'robust_center': prep.get('temperature', {}).get('center', {}),
        'robust_scale': prep.get('temperature', {}).get('scale', {}),
        'standard_mean': prep.get('choke', {}).get('center', {}),
        'standard_scale': prep.get('choke', {}).get('scale', {}),
        'feature_order': prep.get('feature_order'),
    }
    return result


def apply_training_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Apply the exact training-time preprocessing to a frame in-place and return it.
    - Pressure: signed log1p
    - Temperature: robust using center/scale
    - Choke: standard using mean/scale
    """
    dfx = df.copy()
    # Pressure
    for c in prep.get('pressure_cols', []):
        if c in dfx.columns:
            x = dfx[c].astype(float).values
            dfx[c] = np.sign(x) * np.log1p(np.abs(x))
    # Temperature
    for c, center in prep.get('robust_center', {}).items():
        if c in dfx.columns:
            scale = float(prep.get('robust_scale', {}).get(c, 1.0)) or 1.0
            dfx[c] = (dfx[c].astype(float) - float(center)) / scale
    # Choke
    for c, mean in prep.get('standard_mean', {}).items():
        if c in dfx.columns:
            scale = float(prep.get('standard_scale', {}).get(c, 1.0)) or 1.0
            dfx[c] = (dfx[c].astype(float) - float(mean)) / scale
    return dfx


def create_dataset(n_samples=500, anomaly_rate=0.15, random_state=42):
    assert TRAIN.exists(), f'Missing training parquet: {TRAIN}'
    assert (MODELS / MODEL_NAME).exists(), f'Missing model: {MODELS/MODEL_NAME}'

    df_tr = pd.read_parquet(TRAIN)
    df_tr = ensure_pressure_diff(df_tr.copy())

    # Feature list from ONNX (exclude target if present there)
    feats = get_feature_names_from_onnx(MODELS / MODEL_NAME)
    feats = [f for f in feats if f != TARGET]

    cols_needed = [c for c in feats if c in df_tr.columns] + ([TARGET] if TARGET in df_tr.columns else [])
    df_tr = df_tr[cols_needed].dropna().reset_index(drop=True)

    # Central band (5–95%) to draw normals
    q05 = df_tr.quantile(0.05)
    q95 = df_tr.quantile(0.95)
    mask = np.ones(len(df_tr), dtype=bool)
    for c in cols_needed:
        if c in q05 and c in q95:
            mask &= (df_tr[c] >= q05[c]) & (df_tr[c] <= q95[c])
    df_central = df_tr[mask].reset_index(drop=True)

    # Prepare ONNX runtime and preprocessing
    onnx_path = MODELS / MODEL_NAME
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])  # type: ignore
    order = get_feature_names_from_onnx(onnx_path)

    prep = load_training_preprocessing(MODELS / 'residual_preprocessing.json')
    assert prep is not None, 'Missing residual_preprocessing.json with training scalers.'

    # Helper to predict target given raw df rows
    def predict_target(df_rows: pd.DataFrame) -> np.ndarray:
        dfx = df_rows.copy()
        # Ensure pressure diff col exists
        if 'Downstream-Upstream-Difference' not in dfx.columns and {
            'Downstream-Pressure','Upstream-Pressure'
        }.issubset(dfx.columns):
            dfx['Downstream-Upstream-Difference'] = dfx['Downstream-Pressure'] - dfx['Upstream-Pressure']
        # Apply exact training preprocessing
        dfx = apply_training_preprocessing(dfx, prep)
        # Ensure all expected columns
        for c in order:
            if c not in dfx.columns:
                dfx[c] = 0.0
        X = dfx[order].astype(np.float32).values
        pred = sess.run(None, {sess.get_inputs()[0].name: X})[0].squeeze()
        return np.asarray(pred, dtype=np.float32)

    # Compute training residuals on central band
    y_tr = df_central[TARGET].astype(np.float32).values
    yhat_tr = predict_target(df_central[feats])
    resid_tr = np.abs(y_tr - yhat_tr)
    # Empirical residuals for normals; MAD and distribution
    resid_tr = resid_tr[np.isfinite(resid_tr)]
    if len(resid_tr) == 0:
        raise RuntimeError('Could not compute training residuals; got empty array')
    med = float(np.median(resid_tr))
    mad = float(np.median(np.abs(resid_tr - med)))
    # Fallback if mad is degenerate
    if mad <= 0:
        mad = float(np.percentile(resid_tr, 75) - np.percentile(resid_tr, 25)) / 1.349 or 1e-3

    rng = np.random.default_rng(random_state)
    n_anom = max(1, int(n_samples * anomaly_rate))
    n_norm = n_samples - n_anom

    # Normals: bootstrap X from central, set y = model(X) + eps (eps ~ empirical resid)
    idx_norm = rng.choice(len(df_central), size=n_norm, replace=True)
    normals_X = df_central.iloc[idx_norm][feats].copy()
    normals = normals_X.copy()
    normals[TARGET] = predict_target(normals_X)
    # Sample noise from empirical residuals (with sign)
    # Use symmetric noise by sampling residual magnitude and random sign
    resid_pool = resid_tr.copy()
    if len(resid_pool) < n_norm:
        resid_pool = np.tile(resid_pool, int(np.ceil(n_norm / len(resid_pool))))
    eps = rng.choice(resid_pool, size=n_norm, replace=True)
    signs = rng.choice([-1.0, 1.0], size=n_norm)
    normals[TARGET] = normals[TARGET].values + signs * eps
    normals['is_anomaly'] = 0
    normals['difficulty'] = 'normal'

    # Anomalies: mix of covariate perturbation and target drift
    idx_base = rng.choice(len(df_tr), size=n_anom, replace=True)
    anoms = df_tr.iloc[idx_base][feats + [TARGET]].copy()
    anoms['is_anomaly'] = 1

    n_easy = int(0.4 * n_anom); n_med = int(0.4 * n_anom); n_hard = n_anom - n_easy - n_med
    difficulties = np.array(['easy']*n_easy + ['medium']*n_med + ['hard']*n_hard)
    rng.shuffle(difficulties)
    anoms['difficulty'] = difficulties

    feat_scales = {c: (np.nanstd(df_tr[c].values) or 1.0) for c in feats}

    for i in range(len(anoms)):
        diff = anoms.iloc[i]['difficulty']
        row = anoms.iloc[i:i+1]
        if rng.random() < 0.5:
            # Feature-inconsistent anomaly: perturb X, keep y from original row
            for c in feats:
                scale = float(feat_scales.get(c, 1.0))
                if diff == 'easy':
                    row.at[row.index[0], c] = row[c].values[0] + rng.normal(0, 0.15*scale)
                elif diff == 'medium':
                    row.at[row.index[0], c] = row[c].values[0] + rng.normal(0, 0.4*scale)
                else:
                    row.at[row.index[0], c] = row[c].values[0] + rng.normal(0, 0.8*scale)
            # assign back
            for c in feats:
                anoms.iat[i, anoms.columns.get_loc(c)] = row[c].values[0]
            # y stays as original TARGET value
        else:
            # Target-drift anomaly: set y = model(X) + k * eps, k by difficulty
            yhat = float(predict_target(row[feats]).ravel()[0])
            k = 3.0 if diff == 'easy' else 6.0 if diff == 'medium' else 12.0
            eps_mag = rng.choice(resid_tr)
            sign = rng.choice([-1.0, 1.0])
            row.at[row.index[0], TARGET] = yhat + sign * k * eps_mag
            anoms.iat[i, anoms.columns.get_loc(TARGET)] = row[TARGET].values[0]

    # Stitch and shuffle
    df_syn = pd.concat([normals, anoms], ignore_index=True)
    # Ensure pressure diff exists
    df_syn = ensure_pressure_diff(df_syn)
    df_syn = df_syn.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_syn


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_syn = create_dataset(n_samples=500, anomaly_rate=0.15)
    df_syn.to_parquet(OUT)
    print(f'Saved: {OUT} (shape={df_syn.shape}, anomaly_rate={df_syn.is_anomaly.mean():.3f})')


if __name__ == '__main__':
    main()
