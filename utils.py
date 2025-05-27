import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from custom_scaler import column_transformer
from sklearn.preprocessing import StandardScaler
import copy

# Zmienne wybrane przez VIF (deterministyczne i długo się liczy)
VIF_SELECTED_VARIABES = [13, 178, 194, 298, 305, 117, 228, 462, 414, 425, 0, 1, 3, 4, 5, 6, 7, 8, 9]

class ModelComparator:

    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_splits = n_splits
        self.scaler = column_transformer
        self.random_state = random_state
    
    def top_k_accuracy_curve(self, y_true, y_prob):
        y_prob = np.array(y_prob).flatten()
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_prob_sorted = y_prob[sorted_indices]

        accuracies = np.cumsum(y_true_sorted == 1) / (np.arange(1, len(y_true_sorted) + 1))
        thresholds = y_prob_sorted

        return accuracies, thresholds

    def train_acc(self, y_true, y_pred):
        return np.mean(y_pred == y_true)
    
    def scale(self, X_train, X_test):
        # sclr = StandardScaler()
        # self.scaler = sclr
        self.scaler.fit(X_train)
        return self.scaler.transform(X_train), self.scaler.transform(X_test)

    def plot_evaluation(self, all_accuracies, all_thresholds, avg_train_acc):
        half_threshold = len(all_accuracies[0]) // 2 # We are interested only in y==1, no inspection

        mean_acc = np.mean(all_accuracies, axis=0)[:half_threshold]
        std_acc = np.std(all_accuracies, axis=0)[:half_threshold]

        mean_thr = np.mean(all_thresholds, axis=0)[:half_threshold]
        std_thr = np.std(all_thresholds, axis=0)[:half_threshold]

        k_vals = np.arange(1, len(mean_acc) + 1)[:half_threshold]

        plt.figure(figsize=(12, 6))

        # Accuracy curve with std
        plt.plot(k_vals, mean_acc, label='Mean Top-k Accuracy', linewidth=2)
        plt.fill_between(k_vals, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, label='Std Top-k Accuracy')

        # Threshold curve with std
        plt.plot(k_vals, mean_thr, label='Mean Probability Threshold', linewidth=2)
        plt.fill_between(k_vals, mean_thr - std_thr, mean_thr + std_thr, alpha=0.2, label='Std Threshold')

        # Highlight top 20% and 50%
        for perc in [0.2, 0.5]: 
            k = int(len(k_vals) * (perc*2)) # Everything is halved
            plt.scatter(k, mean_acc[k - 1], color='red')
            plt.text(k, mean_acc[k - 1] + 0.04, f'Acc@{int(perc * 100)}%: {mean_acc[k - 1]:.2f}', ha='center', fontsize=12)

        plt.xlabel('Top-k')
        plt.ylabel('Accuracy / Probability Threshold')
        plt.title(f'Cross-Validated Top-k Accuracy & Thresholds\nAvg Train Accuracy: {avg_train_acc:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_variables_histograms(self):
        pass
    def evaluate_model(self, model, variables=None):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        acc_curves = []
        threshold_curves = []
        train_accuracies = []
        
        test_accuracies = []

        for train_idx, test_idx in skf.split(self.X, self.y):
            X_train_fold = self.X[train_idx]
            X_test_fold = self.X[test_idx]

            X_train_fold, X_test_fold = self.scale(X_train_fold, X_test_fold)

            if variables is not None:
                X_train_fold = X_train_fold[:, variables]
                X_test_fold = X_test_fold[:, variables]
            
            y_train_fold = self.y[train_idx]
            y_test_fold = self.y[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_train_pred = model.predict(X_train_fold)
            y_test_proba = model.predict_proba(X_test_fold)
            y_test_pred  = model.predict(X_test_fold).flatten()
            if y_test_proba.ndim > 1 and y_test_proba.shape[1] > 1:
                y_test_proba = y_test_proba[:, 1]

            acc_curve, threshold_curve = self.top_k_accuracy_curve(y_test_fold, y_test_proba)

            acc_curves.append(acc_curve)
            threshold_curves.append(threshold_curve)
            train_accuracies.append(self.train_acc(y_train_fold, y_train_pred))
            test_accuracies.append(np.mean(y_test_pred == y_test_fold))

        # Ensure all curves have same length (trim to min length for plotting)
        min_len = min(map(len, acc_curves))
        acc_curves = np.array([ac[:min_len] for ac in acc_curves])
        threshold_curves = np.array([tc[:min_len] for tc in threshold_curves])

        avg_train_acc = np.mean(train_accuracies)

        self.plot_evaluation(acc_curves, threshold_curves, avg_train_acc)

        test_accuracy_mean = np.mean(test_accuracies)
        test_accuract_std  = np.std(test_accuracies)
        print(f"Test Accuracies (Overall): {test_accuracy_mean:.2f} ± {test_accuract_std:.2f}")

        return np.mean(acc_curves, axis=0)[int(len(test_idx)*0.2)] # Acc@20%
    

class ModelEnsemble:

    def __init__(self, models, voting, weights=None):
        assert voting in ("mean", "median", "weighted")
        if voting == "weighted":
            if weights is None:
                raise ValueError("weighted voting requires non-Null weights argument")
        self.models = [copy.deepcopy(model) for model in models]
        self.n_models = len(models)
        self.voting = voting

        if weights is not None:
            weights_raw = np.asarray(weights)
            self.weights = weights_raw / np.sum(weights_raw) # Normalization

    def fit(self, X, y):
        # Trains all models independently
        for model in self.models:
            model.fit(X,y)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)
        return (y_proba > 0.5).astype(np.int8).squeeze()
    
    def predict_proba(self, X):
        predictions = []

        for model in self.models:
            y_proba_curr = model.predict_proba(X)
            if y_proba_curr.ndim > 1 and y_proba_curr.shape[1] > 1:
                    y_proba_curr = y_proba_curr[:, 1]
            predictions.append(y_proba_curr.squeeze())
        
        predictions = np.asarray(predictions)
        if self.voting == "mean":
            return np.mean(predictions, axis=0)
        elif self.voting == "median":
            return np.median(predictions, axis=0)
        elif self.voting == "weighted":
            return np.dot(self.weights, predictions)

