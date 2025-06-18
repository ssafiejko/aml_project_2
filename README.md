# Advanced Machine Learning: Feature Selection Project

## Overview
This repository contains code for a machine learning project focused on **feature selection** in high-dimensional customer data (500 variables, 5k samples). The goal is to build a parsimonious binary classifier to identify households exceeding an electricity usage threshold while minimizing costly input variables. Models are evaluated using a utility-based scoring framework that balances predictive accuracy (precision@top-20%) with feature cost constraints.

**Key Result**:  
The optimal solution uses **only 1 feature** (Variable `2`) with an SVM classifier, achieving a **score of 0.748**.

## Dataset
- **Size**: 5,000 training samples + 5,000 test samples
- **Features**: 500 anonymized variables
- **Target**: Binary classification (exceed electricity threshold)
- **Evaluation Metric**: Precision@top-20% (`score`) with cost penalty for additional features

## Repository Structure
### Core Scripts
| File | Description |
|------|-------------|
| `PAFS.py` | Precision-Aware Feature Selector implementation |
| `custom_scaler.py` | Custom preprocessing scalers (outlier clipping, log transforms) |
| `mlp.py` | Multi-Layer Perceptron model utilities |
| `utils.py` | Helper functions for data processing and visualization |

### Jupyter Notebooks
| Notebook | Purpose |
|----------|---------|
| `basic_feature_selection.ipynb` | Tests L1-LR, XGBoost, MARS feature selection |
| `feature_distributions.ipynb` | Analyzes feature distributions (Fig. 1 in report) |
| `feature_selection.ipynb` | Main feature selection experiments |
| `paifs_testing.ipynb` | PAFS method evaluation (Sec. 5) |
| `evolution_solution.ipynb` | Evolutionary Algorithm implementation (Sec. 6) |
| `shap_testing.ipynb` | SHAP-based feature engineering (Sec. 7) |
| `final_model_selection.ipynb` | Final model comparison (Sec. 8) |

## Key Methodologies
### Preprocessing
- **Scalers**: `StandardScaler` (normal), `QuantileTransformer` (exponential), `MinMaxScaler` (uniform), `PowerTransformer` (skewed)
- **Outlier Handling**: Clipping + logarithmic transforms for exponential distributions

### Feature Selection Techniques
1. **Filter Methods** (Initial screening):
   - VIF (remove multicollinear features)
   - ANOVA F-test
   - ReliefF
   - Mutual Information

2. **Advanced Methods**:
   - **PAFS**: Hybrid approach using RF/GB/LR + greedy forward selection
   - **Evolutionary Algorithm**: Genetic algorithm with RF classifier
   - **SHAP**: TreeExplainer + PCA for correlated features

### Model Evaluation
- **Constraint**: Each new feature must yield ≥2% `score` gain
- **Validation**: 5-10 fold CV per feature subset
- **Tradeoff Analysis**: Net reward = `score` - feature cost penalty

## Results Summary
| Method | Best Score | Features Selected |
|--------|------------|-------------------|
| **Evolutionary Algorithm** | 0.77 | {2, 7, 215} |
| **PAFS** | 0.783 | {2, 374} |
| **SHAP** | 0.715 | {2, 462} |
| **XGBoost/LR/MARS** | 0.722-0.738 | Singleton {2} |
| **Final Model (SVM)** | **0.748** | **{2}** |

## How to Reproduce
1. **Preprocessing**: Run `feature_distributions.ipynb` to analyze distributions
2. **Feature Selection**: Execute notebooks in order:
   - `basic_feature_selection.ipynb`
   - `paifs_testing.ipynb` / `evolution_solution.ipynb` / `shap_testing.ipynb`
3. **Final Model**: Run `final_model_selection.ipynb` to validate SVM performance
4. **Output**: Results exported to `320628_vars.txt`

