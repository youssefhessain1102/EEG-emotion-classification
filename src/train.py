import numpy as np

from src.data_loader import get_train_files, load_train_participant
from src.model import create_model, evaluate_loso
from src.preprocessor import preprocess_participant_trials, select_top_features


def run_training(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Execute full data loading, feature extraction, LOSO cross-validation, and final model training."""
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    train_files = get_train_files(config["paths"]["train_path"])
    print(f"Found {len(train_files)} training participants")

    train_data_list, train_labels_list, train_counts = [], [], []

    for subj_file in train_files:
        subj_id = subj_file.split("_")[1]
        print(f"Processing participant {subj_id}...")

        trials, labels, _ = load_train_participant(config["paths"]["train_path"], subj_file)
        trials = trials.astype(np.float32)
        features = preprocess_participant_trials(trials, config)

        train_data_list.append(features)
        train_labels_list.append(labels)
        train_counts.append(len(labels))

    X_train_all = np.concatenate(train_data_list, axis=0)
    y_train_all = np.concatenate(train_labels_list, axis=0)
    ranges = np.concatenate([[0], np.cumsum(train_counts)])

    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT FEATURE SELECTION & VALIDATION")
    print("=" * 70)

    n_participants = len(train_counts)
    coef_matrix = np.zeros((n_participants, config["features"]["n_cov_features"]))

    # Pass 1: Collect model coefficients per LOSO fold
    for i in range(n_participants):
        train_idx = np.concatenate(
            [np.arange(0, ranges[i], dtype=int), np.arange(ranges[i + 1], len(y_train_all), dtype=int)]
        )

        clf = create_model(config)
        clf.fit(X_train_all[train_idx], y_train_all[train_idx])
        coef_matrix[i] = clf.coef_[0]

    # Feature selection based on coefficient stability
    top_k = config["model"]["feature_selection"]["best_k"]
    best_mask = select_top_features(coef_matrix, top_k=top_k)

    # Pass 2: Evaluate LOSO CV performance using best feature mask
    total_auc, per_part_auc = evaluate_loso(X_train_all, y_train_all, ranges, best_mask, config)

    for idx, auc in enumerate(per_part_auc):
        print(f"Participant {idx+1}/{n_participants}: AUC = {auc:.4f}")

    print(f"\nTotal LOSO AUC: {total_auc:.4f}")

    # Train final model on complete dataset using selected features
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL TRAINING DATA")
    print("=" * 70)

    final_model = create_model(config)
    final_model.fit(X_train_all[:, best_mask], y_train_all)

    return X_train_all, y_train_all, best_mask, final_model