# Anomaly Detection Analysis Framework

This directory contains a streamlined framework for evaluating and optimizing anomaly detection models in the TAQA system.

## 📁 Project Structure

```
anomaly_detection_analysis/
├── notebooks/
│   ├── multi_model_evaluation.ipynb          # Main evaluation notebook
│   ├── model_retraining_optimization.ipynb   # Optimization and retraining
│   ├── choke_model_verification.ipynb        # Original choke position analysis
│   └── create_expanded_synthetic_choke.ipynb # Synthetic data generation
├── synthetic_data/
│   ├── synth_choke_position_100pts.parquet   # Choke position test data
│   ├── synth_delta_temp_open_100pts.parquet  # Temperature test data (generated)
│   └── synth_full_vectors_if_100pts.parquet  # Full vector test data (generated)
├── reports/
│   ├── THESIS_Multi_Model_Evaluation.md     # Comprehensive thesis report
│   ├── multi_model_performance_summary.csv  # Performance metrics summary
│   ├── multi_model_evaluation_report.json   # Detailed evaluation results
│   └── optimization_results.json            # Retraining results
└── README.md                                 # This file
```

## 🎯 Framework Overview

### Three Target Models

1. **choke_position.onnx**
   - Detects anomalies in choke position behavior
   - Key features: choke_position, temp, tool_state
   - Known issues: Extreme value detection, rare tool states

2. **delta_temp_open.onnx** 
   - Detects temperature differential anomalies during open operations
   - Key features: temp_up, temp_down, delta_temp_open, open_duration
   - Known issues: Temperature sensitivity, temporal patterns

3. **full_vectors_if.onnx**
   - Multi-dimensional isolation forest for comprehensive anomaly detection
   - Features: All available sensor data
   - Known issues: High dimensionality, feature correlation

### Workflow Process

#### Phase 1: Baseline Evaluation
```bash
# Run the main evaluation notebook
jupyter notebook notebooks/multi_model_evaluation.ipynb
```

This notebook will:
- Load baseline models (models/) and improved models (models_3/)
- Generate synthetic test datasets (100 samples, 15% anomaly rate)
- Compare performance metrics (Precision, Recall, F1-score)
- Identify improvement opportunities
- Generate comprehensive reports

#### Phase 2: Optimization and Retraining
```bash
# Run the optimization notebook
jupyter notebook notebooks/model_retraining_optimization.ipynb
```

This notebook will:
- Load evaluation results and recommendations
- Apply advanced feature engineering
- Optimize hyperparameters using grid search
- Retrain models with improved configurations
- Save optimized models to `models_optimized/`

#### Phase 3: Validation and Deployment
- Re-run evaluation notebook with optimized models
- Compare baseline → improved → optimized performance
- Deploy best-performing models to production

## 📊 Key Metrics and Results

### Performance Improvements (Latest Run)

| Model | Baseline Recall | Improved Recall | Improvement |
|-------|-----------------|-----------------|-------------|
| choke_position | 0.267 | 0.400 | +50.0% |
| delta_temp_open | TBD | TBD | TBD |
| full_vectors_if | TBD | TBD | TBD |

### Optimization Strategies Applied

#### choke_position
- ✅ RobustScaler preprocessing
- ✅ 2% contamination rate optimization
- ✅ 200 tree ensemble
- 🔄 Feature engineering (ratios, deltas, extreme indicators)

#### delta_temp_open
- 🔄 Temperature normalization
- 🔄 Temporal feature engineering
- 🔄 Contamination rate tuning (1.5-3.5%)

#### full_vectors_if
- 🔄 Feature selection and correlation removal
- 🔄 Ensemble approach with soft voting
- 🔄 Bootstrap sampling optimization

## 🔧 Technical Implementation

### Synthetic Data Generation
- **Model-specific patterns**: Each model gets tailored anomaly types
- **Realistic distributions**: Based on training data statistics
- **Balanced datasets**: 85 normal + 15 anomalous samples
- **Reproducible**: Fixed random seeds for consistent testing

### Model Optimization
- **Grid search**: Systematic hyperparameter tuning
- **Feature engineering**: Model-specific transformations
- **Scaling strategies**: RobustScaler vs StandardScaler comparison
- **ONNX export**: Production-ready model format

### Evaluation Framework
- **Automated scaling detection**: Loads and applies preprocessing automatically
- **Comprehensive metrics**: Precision, Recall, F1, Confusion Matrix
- **Missed anomaly analysis**: Root cause investigation
- **Visualization**: Performance plots and score distributions

## 📝 Thesis Integration

### Reports Generated
1. **THESIS_Multi_Model_Evaluation.md**: Comprehensive analysis suitable for thesis
2. **Performance summaries**: CSV and JSON formats for data analysis
3. **Methodology documentation**: Reproducible research framework

### Key Contributions
- **Systematic evaluation framework** for multiple anomaly detection models
- **Synthetic data generation** with realistic anomaly patterns
- **Automated optimization pipeline** with hyperparameter tuning
- **Production-ready deployment** with ONNX model export

## 🚀 Quick Start

1. **Initial Evaluation**:
   ```bash
   cd anomaly_detection_analysis/notebooks
   jupyter notebook multi_model_evaluation.ipynb
   ```

2. **Review Results**: Check `reports/` directory for performance analysis

3. **Optimize Models**:
   ```bash
   jupyter notebook model_retraining_optimization.ipynb
   ```

4. **Final Validation**: Re-run evaluation with optimized models

## 📚 Dependencies

- pandas, numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- onnxruntime: Model inference
- skl2onnx: Model conversion
- matplotlib, seaborn: Visualization
- jupyter: Interactive notebooks

## 🎯 Expected Outcomes

### For choke_position Model
- **Target**: >20% recall improvement
- **Current**: 50% improvement achieved ✅
- **Next**: Feature engineering for extreme values

### For delta_temp_open Model  
- **Target**: Establish baseline and identify improvements
- **Focus**: Temperature differential patterns and temporal features

### For full_vectors_if Model
- **Target**: Optimize high-dimensional anomaly detection
- **Focus**: Feature selection and ensemble methods

## 📞 Support

This framework is designed to be self-contained and well-documented. Each notebook includes:
- Clear markdown explanations
- Step-by-step execution guidance
- Error handling and debugging information
- Comprehensive output and visualization

For questions or issues, refer to the detailed comments and documentation within each notebook.
