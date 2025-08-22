# Simple Data Profiling Script (Crash Fix)

This is a minimal replacement for the crashed data_profile.ipynb that implements
the resource-limited approach to prevent notebook controller disposal.

## Key Changes Made:
- Limited number of tools processed (MAX_TOOLS = 20)
- Reduced RandomForest estimators (50 instead of 200)
- Limited parallelism (n_jobs=2 instead of n_jobs=-1) 
- Data sampling for large datasets (max 10,000 rows)
- Reduced cross-validation folds (3 instead of 5)
- Better error handling and progress reporting

## Usage:
Run this as a Python script instead of the crashed notebook:

```python
python data_profile_fixed.py
```

The original notebook crashed due to excessive memory usage when running
RandomForest cross-validation on large datasets with high parallelism.