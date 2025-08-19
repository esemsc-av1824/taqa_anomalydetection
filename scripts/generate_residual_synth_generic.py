#!/usr/bin/env python3
"""
Generic synthetic generator for residual models (battery, pressures, temperatures, target-pos).

Goal: Produce synthetic data whose normal residual distribution matches the training
scale (via empirical residual sampling), so saved MAD-based cutoffs are meaningful.

Usage examples:
  python scripts/generate_residual_synth_generic.py \
    --model models_4/residual_upT.onnx \
    --target "Upstream-Temperature" \
    --out anomaly_detection_analysis/synthetic_data/challenging_residual_upT_500pts.parquet

  python scripts/generate_residual_synth_generic.py \
    --model models_4/residual_downT.onnx \
    --target "Downstream-Temperature" \
    --out anomaly_detection_analysis/synthetic_data/challenging_residual_downT_500pts.parquet
"""
from __future__ import annotations
import argparse
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


def get_feature_names_from_onnx(onnx_path: Path):
    if onnx is None:
        raise RuntimeError('onnx package is required to read feature_names metadata')
    m = onnx.load(str(onnx_path))
    for kv in m.metadata_props:
        if kv.key == 'feature_names':
            # try JSON list first
            val = kv.value
            if isinstance(val, str) and val.strip().startswith('['):
                try:
                    return json.loads(val)
                except Exception:
                    pass
            return val.split(',')
    raise RuntimeError('feature_names metadata missing in ONNX')


def ensure_pressure_diff(df: pd.DataFrame):
    if 'Downstream-Upstream-Difference' not in df.columns and {
        'Downstream-Pressure','Upstream-Pressure'
    }.issubset(df.columns):
        df['Downstream-Upstream-Difference'] = df['Downstream-Pressure'] - df['Upstream-Pressure']
    return df


def load_training_preprocessing(prep_path: Path):
    if not prep_path.exists():
        return None
    prep = json.loads(prep_path.read_text())
    return {
        'pressure_cols': prep.get('pressure', {}).get('cols', []),
        'pressure_method': prep.get('pressure', {}).get('method', 'log1p'),
        'robust_center': prep.get('temperature', {}).get('center', {}),
        'robust_scale': prep.get('temperature', {}).get('scale', {}),
        'standard_mean': prep.get('choke', {}).get('center', {}),
        'standard_scale': prep.get('choke', {}).get('scale', {}),
        'feature_order': prep.get('feature_order'),
    }


def apply_training_preprocessing(df: pd.DataFrame, prep: dict) -> pd.DataFrame:
    dfx = df.copy()
    # Pressure: signed log1p
    for c in prep.get('pressure_cols', []):
        if c in dfx.columns:
            x = dfx[c].astype(float).values
            dfx[c] = np.sign(x) * np.log1p(np.abs(x))
    # Temps: robust
    for c, center in prep.get('robust_center', {}).items():
        if c in dfx.columns:
            scale = float(prep.get('robust_scale', {}).get(c, 1.0)) or 1.0
            dfx[c] = (dfx[c].astype(float) - float(center)) / scale
    # Choke: standard
    for c, mean in prep.get('standard_mean', {}).items():
        if c in dfx.columns:
            scale = float(prep.get('standard_scale', {}).get(c, 1.0)) or 1.0
            dfx[c] = (dfx[c].astype(float) - float(mean)) / scale
    return dfx


def create_dataset(model_path: Path, target: str, n_samples=500, anomaly_rate=0.15, random_state=42) -> pd.DataFrame:
    assert TRAIN.exists(), f'Missing training parquet: {TRAIN}'
    assert model_path.exists(), f'Missing model: {model_path}'

    df_tr = pd.read_parquet(TRAIN)
    df_tr = ensure_pressure_diff(df_tr.copy())

    # Feature list from ONNX (exclude target if accidentally present)
    feats = get_feature_names_from_onnx(model_path)
    feats = [f for f in feats if f != target]

    cols_needed = [c for c in feats if c in df_tr.columns] + ([target] if target in df_tr.columns else [])
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
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])  # type: ignore
    # Prefer ONNX feature order; fall back to training-prep order if absent
    order = get_feature_names_from_onnx(model_path)

    prep = load_training_preprocessing(BASE / 'models_4' / 'residual_preprocessing.json')
    assert prep is not None, 'Missing residual_preprocessing.json with training scalers.'

    # Helper to predict target given raw df rows
    def predict_target(df_rows: pd.DataFrame) -> np.ndarray:
        dfx = df_rows.copy()
        dfx = ensure_pressure_diff(dfx)
        dfx = apply_training_preprocessing(dfx, prep)
        for c in order:
            if c not in dfx.columns:
                dfx[c] = 0.0
        X = dfx[order].astype(np.float32).values
        pred = sess.run(None, {sess.get_inputs()[0].name: X})[0].squeeze()
        return np.asarray(pred, dtype=np.float32)

    # Compute training residuals on central band
    y_tr = df_central[target].astype(np.float32).values
    yhat_tr = predict_target(df_central[feats])
    resid_tr = np.abs(y_tr - yhat_tr)
    resid_tr = resid_tr[np.isfinite(resid_tr)]
    if len(resid_tr) == 0:
        raise RuntimeError('Could not compute training residuals; got empty array')

    rng = np.random.default_rng(random_state)
    n_anom = max(1, int(n_samples * anomaly_rate))
    n_norm = n_samples - n_anom

    # Normals: y ≈ model(X) + ε, ε ~ empirical residuals
    idx_norm = rng.choice(len(df_central), size=n_norm, replace=True)
    normals_X = df_central.iloc[idx_norm][feats].copy()
    normals = normals_X.copy()
    normals[target] = predict_target(normals_X)
    resid_pool = resid_tr.copy()
    if len(resid_pool) < n_norm:
        resid_pool = np.tile(resid_pool, int(np.ceil(n_norm / len(resid_pool))))
    eps = rng.choice(resid_pool, size=n_norm, replace=True)
    signs = rng.choice([-1.0, 1.0], size=n_norm)
    normals[target] = normals[target].values + signs * eps
    normals['is_anomaly'] = 0
    normals['difficulty'] = 'normal'

    # Anomalies: mix of covariate perturbation and target drift
    idx_base = rng.choice(len(df_tr), size=n_anom, replace=True)
    anoms = df_tr.iloc[idx_base][feats + [target]].copy()
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
            for c in feats:
                anoms.iat[i, anoms.columns.get_loc(c)] = row[c].values[0]
        else:
            # Target-drift anomaly: set y = model(X) + k * ε
            yhat = float(predict_target(row[feats]).ravel()[0])
            k = 3.0 if diff == 'easy' else 6.0 if diff == 'medium' else 12.0
            eps_mag = rng.choice(resid_tr)
            sign = rng.choice([-1.0, 1.0])
            row.at[row.index[0], target] = yhat + sign * k * eps_mag
            anoms.iat[i, anoms.columns.get_loc(target)] = row[target].values[0]

    # Stitch and shuffle
    df_syn = pd.concat([normals, anoms], ignore_index=True)
    df_syn = ensure_pressure_diff(df_syn)
    df_syn = df_syn.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_syn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to residual ONNX model (e.g., models_4/residual_upT.onnx)')
    p.add_argument('--target', required=True, help='Target column name (e.g., "Upstream-Temperature")')
    p.add_argument('--out', required=True, help='Output parquet path')
    p.add_argument('--n-samples', type=int, default=500)
    p.add_argument('--anomaly-rate', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    model_path = (BASE / args.model) if not args.model.startswith('/') else Path(args.model)
    out_path = (BASE / args.out) if not args.out.startswith('/') else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_syn = create_dataset(model_path, args.target, n_samples=args.n_samples, anomaly_rate=args.anomaly_rate, random_state=args.seed)
    df_syn.to_parquet(out_path)
    print(f'Saved: {out_path} (shape={df_syn.shape}, anomaly_rate={df_syn.is_anomaly.mean():.3f})')


if __name__ == '__main__':
    main()
