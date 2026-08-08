import os

import numpy as np
import pandas as pd

from src.data_loader import get_test_files, load_hdf5_data
from src.preprocessor import preprocess_participant_trials


def run_inference(config: dict, final_model: object, best_mask: np.ndarray) -> None:
    """Run batch prediction on test participants and produce valid Kaggle submission."""
    print("=" * 70)
    print("LOADING TEST DATA & GENERATING PREDICTIONS")
    print("=" * 70)

    test_files = get_test_files(config["paths"]["test_path"])
    output_file = config["paths"]["submission_file"]
    n_timepoints = config["signal_processing"]["n_timepoints"]

    if not test_files:
        print("No test files found. Creating placeholder submission.csv...")
        submission_df = pd.DataFrame(columns=["id", "prediction"])
        submission_df.to_csv(output_file, index=False)
        return

    submission_rows = []

    for file_path in test_files:
        basename = os.path.basename(file_path)
        subj_id = basename.split("_")[2].split(".")[0]
        print(f"Predicting test subject {subj_id}...")

        data = load_hdf5_data(file_path)
        trials = data["trial"]

        features = preprocess_participant_trials(trials, config)
        features_selected = np.nan_to_num(features[:, best_mask], nan=0.0)

        # Predict probabilities
        y_pred_proba = final_model.predict_proba(features_selected)[:, 1]

        # Tile across timepoints
        n_trials_test = trials.shape[0]
        subj_preds = np.tile(y_pred_proba[:, np.newaxis], (1, n_timepoints))

        # Format tabular submission IDs: {subject}_{trial}_{timepoint}
        t_grid, tr_grid = np.meshgrid(np.arange(n_timepoints), np.arange(n_trials_test))
        ids = [f"{subj_id}_{tr}_{t}" for tr, t in zip(tr_grid.flatten(), t_grid.flatten())]

        block_df = pd.DataFrame({"id": ids, "prediction": subj_preds.flatten()})
        submission_rows.append(block_df)

    submission_df = pd.concat(submission_rows, ignore_index=True)
    submission_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("SUBMISSION CREATED & VALIDATED")
    print("=" * 70)
    print(f"Output File: {output_file}")
    print(f"Total Rows: {len(submission_df)}")

    # Validation check
    if submission_df["prediction"].min() >= 0.0 and submission_df["prediction"].max() <= 1.0:
        print("PASS: Predictions strictly within valid probability range [0, 1]")
    else:
        print("FAIL: Prediction values outside expected range [0, 1]")