import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.base import clone

from utils import VIF_SELECTED_VARIABES

import warnings
warnings.filterwarnings('ignore')

class PrecisionAwareFeatureSelector:
    
    def __init__(self, 
                 max_features=10,
                 cv_folds=5,
                 n_iterations=20,
                 stability_threshold=0.7,
                 early_stopping_patience=5,
                 top_percentile=20,
                 scaler=None,
                 column_transformer=None,
                 alternative_column_transformer=None,
                 exclude_features=None,
                 random_state=42):
        
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.n_iterations = n_iterations
        self.stability_threshold = stability_threshold
        self.early_stopping_patience = early_stopping_patience
        self.top_percentile = top_percentile
        self.scaler = scaler
        self.column_transformer = column_transformer
        self.alternative_column_transformer = alternative_column_transformer
        self.exclude_features = exclude_features if exclude_features is not None else []
        self.random_state = random_state
        
        self.base_models = [
            RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            LogisticRegression(random_state=random_state, max_iter=1000, penalty='l1', solver='liblinear')
        ]
        
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_history_ = []
        self.fitted_scalers_ = {}  # Store fitted scalers for each feature subset
        self.allowed_features_ = None  # Features after exclusion
        self.feature_mapping_ = None  # Mapping from reduced to original indices
    
    def _apply_feature_exclusion(self, X):
        """Apply feature exclusion and return reduced dataset with mapping"""
        n_features = X.shape[1]
        exclude_set = set(self.exclude_features)
        self.allowed_features_ = [i for i in range(n_features) if i not in exclude_set]
        self.feature_mapping_ = {i: orig_idx for i, orig_idx in enumerate(self.allowed_features_)}
        print(f"Excluded {len(exclude_set)} features. Remaining: {len(self.allowed_features_)} features")
        return X[:, self.allowed_features_]
    
    def _create_subset_scaler(self, reduced_feature_indices):
        """Create a scaler that works with the selected feature subset from reduced data"""
        from sklearn.base import clone
        from sklearn.compose import ColumnTransformer
        
        if self.scaler is None:
            return None
        
        original_feature_indices = [self.feature_mapping_[i] for i in reduced_feature_indices]
        orig_to_reduced = {orig: i for i, orig in enumerate(original_feature_indices)}
        if self.scaler == "ct":
            if self.column_transformer is None:
                raise ValueError("column_transformer must be provided when scaler='ct'")
            
            subset_transformer = clone(self.column_transformer)
            
            new_transformers = []
            for name, transformer, cols in subset_transformer.transformers:
                if cols in ("remainder", "drop"):
                    new_cols = cols                    
                else:
                    if isinstance(cols, (list, np.ndarray)):
                        # translate every column number that survives into the 0-based
                        # position inside the slice
                        new_cols = [orig_to_reduced[c] for c in cols if c in orig_to_reduced]
                        if not new_cols:
                            continue                     # nothing from this sub-pipe in slice
                    else:                                # single int
                        if cols in orig_to_reduced:
                            new_cols = orig_to_reduced[cols]
                        else:
                            continue
                new_transformers.append((name, transformer, new_cols))

            if not new_transformers:
                return None

            from sklearn.compose import ColumnTransformer
            subset_transformer = ColumnTransformer(
                transformers=new_transformers, remainder="drop"
            )
            return subset_transformer
            
        elif self.scaler == "act":
            if self.alternative_column_transformer is None:
                raise ValueError("alternative_column_transformer must be provided when scaler='act'")
            
            subset_transformer = clone(self.alternative_column_transformer)
            
            new_transformers = []
            for name, transformer, cols in subset_transformer.transformers:
                if cols in ("remainder", "drop"):
                    new_cols = cols                      # keep as-is
                else:
                    if isinstance(cols, (list, np.ndarray)):
                        # translate every column number that survives into the 0-based
                        # position inside the slice
                        new_cols = [orig_to_reduced[c] for c in cols if c in orig_to_reduced]
                        if not new_cols:
                            continue                     # nothing from this sub-pipe in slice
                    else:                                # single int
                        if cols in orig_to_reduced:
                            new_cols = orig_to_reduced[cols]
                        else:
                            continue
                new_transformers.append((name, transformer, new_cols))

            if not new_transformers:
                return None
            from sklearn.compose import ColumnTransformer
            subset_transformer = ColumnTransformer(
                transformers=new_transformers, remainder="drop"
            )
            return subset_transformer
    
    def _scale(self, X_train, X_val, reduced_feature_indices):
        """Scale the data using the appropriate scaler for the feature subset"""
        if self.scaler is None:
            return X_train, X_val
        
        # Create a key for this feature subset
        subset_key = tuple(sorted(reduced_feature_indices))
        
        # Get or create scaler for this subset
        if subset_key not in self.fitted_scalers_:
            subset_scaler = self._create_subset_scaler(reduced_feature_indices)
            if subset_scaler is not None:
                subset_scaler.fit(X_train)
                self.fitted_scalers_[subset_key] = subset_scaler
            else:
                self.fitted_scalers_[subset_key] = None
        
        scaler = self.fitted_scalers_[subset_key]
        
        if scaler is None:
            return X_train, X_val
        else:
            return scaler.transform(X_train), scaler.transform(X_val)
        
    def _calculate_precision_at_top_percentile(self, y_true, y_proba):
        """Calculate precision at top percentile of predicted probabilities"""
        n_top = int(len(y_true) * (self.top_percentile / 100))
        if n_top == 0:
            n_top = 1
            
        top_indices = np.argsort(y_proba)[-n_top:]
        

        top_predictions = y_true[top_indices]
        if len(top_predictions) == 0:
            return 0.0
        
        return np.mean(top_predictions)
    
    def _get_ensemble_feature_importance(self, X, y):
        """Get feature importance from ensemble of models"""
        n_features = X.shape[1]
        importance_matrix = np.zeros((len(self.base_models), n_features))
        
        for i, model in enumerate(self.base_models):
            model_copy = clone(model)
            model_copy.fit(X, y)
            
            if hasattr(model_copy, 'feature_importances_'):
                importance_matrix[i] = model_copy.feature_importances_
            elif hasattr(model_copy, 'coef_'):
                importance_matrix[i] = np.abs(model_copy.coef_[0])
        
        # Normalize importances for each model
        for i in range(len(self.base_models)):
            if importance_matrix[i].sum() > 0:
                importance_matrix[i] /= importance_matrix[i].sum()
        
        # Average across models
        avg_importance = np.mean(importance_matrix, axis=0)
        return avg_importance
    
    def _evaluate_feature_subset(self, X, y, feature_indices, reduced_feature_indices=None):
        """Evaluate a feature subset using cross-validation"""
        if len(feature_indices) == 0:
            return 0.0, 0.0
            
        X_subset = X[:, feature_indices]
        
        # Use the reduced indices for scaling (these are indices in the reduced dataset)
        scaling_indices = reduced_feature_indices if reduced_feature_indices is not None else feature_indices
            
        cv_scores = []
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in skf.split(X_subset, y):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train_scaled, X_val_scaled = self._scale(X_train, X_val, scaling_indices)
            
            # Use best performing model from ensemble
            best_score = 0
            for model in self.base_models:
                model_copy = clone(model)
                model_copy.fit(X_train_scaled, y_train)
                y_proba = model_copy.predict_proba(X_val_scaled)[:, 1]
                score = self._calculate_precision_at_top_percentile(y_val, y_proba)
                best_score = max(best_score, score)
            
            cv_scores.append(best_score)
        
        return np.mean(cv_scores), np.std(cv_scores)
    
    def _forward_selection_step(self, X, y, current_features):
        """Perform one step of forward feature selection"""
        n_features = X.shape[1]
        remaining_features = [i for i in range(n_features) if i not in current_features]
        
        if not remaining_features:
            return None, 0.0
        
        best_feature = None
        best_score = -1.0
        
        for feature in remaining_features:
            candidate_features = current_features + [feature]
            score, _ = self._evaluate_feature_subset(X, y, candidate_features, candidate_features)
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        return best_feature, best_score
    
    def _check_stability(self, feature_history, window_size=5):
        """Check if feature selection has stabilized"""
        if len(feature_history) < window_size:
            return False
        
        recent_selections = feature_history[-window_size:]
        
        # Calculate Jaccard similarity between consecutive selections
        similarities = []
        for i in range(1, len(recent_selections)):
            set1 = set(recent_selections[i-1])
            set2 = set(recent_selections[i])
            if len(set1.union(set2)) == 0:
                similarity = 1.0
            else:
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
            similarities.append(similarity)
        
        return np.mean(similarities) >= self.stability_threshold
    
    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)
        
        print(f"Starting PAIFS with {X.shape[1]} features, targeting {self.max_features} features")
        if self.exclude_features:
            print(f"Excluding {len(self.exclude_features)} features: {self.exclude_features}")
        print(f"Optimizing precision at top {self.top_percentile}% of predictions")
        
        X_reduced = self._apply_feature_exclusion(X)
        
        # Initial importance-based filtering on reduced dataset
        print("\nPhase 1: Initial importance-based filtering...")
        importance_scores = self._get_ensemble_feature_importance(X_reduced, y)
        
        # Select top features based on importance (5x target to allow refinement)
        initial_k = min(self.max_features * 5, X_reduced.shape[1])
        top_features = np.argsort(importance_scores)[-initial_k:]
        
        print(f"Selected top {len(top_features)} features based on ensemble importance")
        
        # Iterative precision-aware selection
        print("\nPhase 2: Iterative precision-aware selection...")
        X_filtered = X_reduced[:, top_features]
        
        current_features = []
        best_overall_score = 0.0
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            if len(current_features) >= self.max_features:
                break
                
            next_feature, score = self._forward_selection_step(X_filtered, y, current_features)
            
            if next_feature is None:
                print(f"No more features to add at iteration {iteration}")
                break
            
            current_features.append(next_feature)
            
            original_idx = self.allowed_features_[top_features[next_feature]]
            print(f"Iteration {iteration + 1}: Added feature {original_idx}, "
                  f"Score: {score:.4f}, Total features: {len(current_features)}")
            
            self.selection_history_.append(current_features.copy())
            
            # Early stopping check
            if score > best_overall_score:
                best_overall_score = score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at iteration {iteration + 1} (patience: {self.early_stopping_patience})")
                break
            
            # Stability check
            if self._check_stability(self.selection_history_):
                print(f"Feature selection stabilized at iteration {iteration + 1}")
                break
        
        # Map back to original feature indices
        selected_reduced_indices = [top_features[i] for i in current_features]
        self.selected_features_ = [self.allowed_features_[i] for i in selected_reduced_indices]
        self.feature_scores_ = {feat: importance_scores[selected_reduced_indices[i]] 
                              for i, feat in enumerate(self.selected_features_)}
        
        print(f"\nFinal selection: {len(self.selected_features_)} features")
        print(f"Selected features (original indices): {self.selected_features_}")
        print(f"Final precision@top{self.top_percentile}%: {best_overall_score:.4f}")
        
        return self
    
    def transform(self, X):
        """Transform X by selecting the chosen features and applying scaling if needed"""
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")
        
        X = np.array(X)
        X_selected = X[:, self.selected_features_]
        
        # Apply scaling if scaler was used during fitting
        if self.scaler is not None:
            # Create indices for the selected features in the reduced space
            # We need to map selected features back to their positions in the reduced dataset
            reduced_indices = []
            for feat in self.selected_features_:
                if feat in self.allowed_features_:
                    reduced_indices.append(self.allowed_features_.index(feat))
            
            subset_key = tuple(sorted(reduced_indices))
            if subset_key in self.fitted_scalers_ and self.fitted_scalers_[subset_key] is not None:
                scaler = self.fitted_scalers_[subset_key]
                X_selected = scaler.transform(X_selected)
        
        return X_selected
    
    def fit_transform(self, X, y):
        """Fit the selector and transform X"""
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices=False):
        """Get boolean mask or integer indices of selected features"""
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")
        
        if indices:
            return np.array(self.selected_features_)
        else:
            n_features = max(self.selected_features_) + 1
            mask = np.zeros(n_features, dtype=bool)
            mask[self.selected_features_] = True
            return mask