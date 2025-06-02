from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   PowerTransformer, QuantileTransformer, RobustScaler, FunctionTransformer)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

normal_cols = list(range(0, 200))
exponential_cols = list(range(200, 300))
uniform_cols = list(range(300, 400))
skewed_normal_cols = list(range(400, 500))

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.lower_bounds_ = X.quantile(self.lower_quantile)
            self.upper_bounds_ = X.quantile(self.upper_quantile)
        else:
            self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
            self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped
    
# ==================== unique transformation for each distribution ====================
column_transformer = ColumnTransformer([
    ('standard', Pipeline([
        ('clip', OutlierClipper()),
        ('scaler', StandardScaler())
    ]), normal_cols),

    ('quantile', Pipeline([
        ('clip', OutlierClipper()),
        ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=20))
    ]), exponential_cols),

    ('minmax', Pipeline([
        ('clip', OutlierClipper()),
        ('minmax', MinMaxScaler())
    ]), uniform_cols),

    ('power', Pipeline([
        ('clip', OutlierClipper()),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ]), skewed_normal_cols)
], remainder='passthrough')

# No transformations, only outlier clipping
# Only log for Exponentials
tree_column_transformer = ColumnTransformer([
        ('standard', Pipeline([
        ('clip', OutlierClipper()),
    ]), normal_cols),

    ('log', Pipeline([
        ('clip', OutlierClipper()),
        ('quantile', FunctionTransformer(func=np.log1p, inverse_func=np.expm1)) 
    ]), exponential_cols),

    ('robust', Pipeline([  # Consider RobustScaler instead of MinMax
        ('clip', OutlierClipper()),
    ]), uniform_cols),

    ('power', Pipeline([
        ('clip', OutlierClipper()),
    ]), skewed_normal_cols)
])