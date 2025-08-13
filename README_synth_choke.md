# Synthetic Dataset: choke_position.onnx

File: `training_data/synth_choke_position.parquet`
Rows: 10

Features used by model (IsolationForest 3-D):
- Choke-Position
- ToolStateNum
- Downstream-Temperature

Label column added: `is_anomaly` (1 = anomaly, 0 = normal)

Heuristics derived from original training data (`wide36_tools_flat.parquet` ~1.29M rows):
- Normal Choke-Position central band (1st–99th pct): [-1.189, 101.189]
- Rare ToolStateNum identified (<0.1% freq): 7680
- Downstream-Temperature normally ~15–17°C median; > ~96°C is extreme (>= 95th pct ~96.8, 99th pct ~108)

Constructed anomalies:
1. Choke-Position slightly below typical low quantile (-1.8)
2. Choke-Position above 99th percentile (102.5)
3. Rare ToolStateNum = 7680
4. Extremely high Downstream-Temperature (105.0°C)

Normal rows sampled within common ranges and frequent ToolStateNum values (2 dominates, plus 1,5,6) with moderate temperatures.

## Next Steps
- Run ONNX inference to confirm IsolationForest flags rows 6–9 (0-index) as anomalies.
- Extend same methodology to other models (define feature lists, pull quantiles, craft boundary / rare-state / cross-feature inconsistencies).

