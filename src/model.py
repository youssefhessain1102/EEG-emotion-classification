import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


def create_model(config: dict) -> LinearDiscriminantAnalysis:
    """Factory function for model instantiation."""
    m_cfg = config["model"]
    return LinearDiscriminantAnalysis(solver=m_cfg["solver"], shrinkage=m_cfg["shrinkage"])


def evaluate_loso(
    X: np.ndarray, y: np.ndarray, ranges: np.ndarray, best_mask: np.ndarray, config: dict
) -> tuple[float, list[float]]:
    """Evaluate Leave-One-Subject-Out Cross Validation performance."""
    n_participants = len(ranges) - 1
    preds = np.zeros(len(y))
    per_participant_auc = []

    for i in range(n_participants):
        test_idx = np.arange(ranges[i], ranges[i + 1], dtype=int)
        train_idx = np.concatenate([np.arange(0, ranges[i], dtype=int), np.arange(ranges[i + 1], len(y), dtype=int)])

        X_train, y_train = X[train_idx][:, best_mask], y[train_idx]
        X_test, y_test = X[test_idx][:, best_mask], y[test_idx]

        clf = create_model(config)
        clf.fit(X_train, y_train)

        preds[test_idx] = clf.predict_proba(X_test)[:, 1]
        part_auc = roc_auc_score(y_test, preds[test_idx])
        per_participant_auc.append(part_auc)

    total_auc = roc_auc_score(y, preds)
    return total_auc, per_participant_auc