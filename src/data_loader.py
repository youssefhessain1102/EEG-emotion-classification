import glob
import os

import h5py
import numpy as np


def load_hdf5_data(filepath: str) -> dict:
    """Load HDF5 MATLAB v7.3 file structure."""
    def load_field(f, data_ref, field_name):
        field = data_ref[field_name]
        if isinstance(field, h5py.Dataset):
            ref_value = field[()]
            if isinstance(ref_value, h5py.Reference):
                return f[ref_value]
            elif hasattr(ref_value, "shape") and ref_value.shape == (1, 1):
                ref = ref_value.item()
                if isinstance(ref, h5py.Reference):
                    return f[ref]
                else:
                    if isinstance(ref, bytes):
                        ref = ref.decode("utf-8")
                    return f[ref]
            else:
                return field
        else:
            return field

    with h5py.File(filepath, "r") as f:
        data_ref = f["data"]

        try:
            trialinfo_data = load_field(f, data_ref, "trialinfo")
            trialinfo = np.array(trialinfo_data).T
        except (KeyError, ValueError, TypeError):
            trialinfo = None

        time_data = np.array(load_field(f, data_ref, "time")).flatten()
        trial_data = np.array(load_field(f, data_ref, "trial")).T  # (Trials, Channels, Time)

        mask = time_data >= 0
        if np.any(~mask):
            time_data = time_data[mask]
            trial_data = trial_data[:, :, mask]

        return {"trial": trial_data, "trialinfo": trialinfo, "time": time_data}


def get_train_files(train_path: str) -> list[str]:
    """Retrieve sorted participant file paths for training."""
    neu_path = os.path.join(train_path, "sleep_neu")
    if not os.path.exists(neu_path):
        raise FileNotFoundError(f"Training directory not found at: {neu_path}")
    return sorted([f for f in os.listdir(neu_path) if f.endswith(".mat")])


def load_train_participant(train_path: str, subj_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate neutral and emotional datasets for a single training participant."""
    subj_id = subj_file.split("_")[1]

    neu_data = load_hdf5_data(os.path.join(train_path, "sleep_neu", subj_file))
    emo_data = load_hdf5_data(os.path.join(train_path, "sleep_emo", subj_file))

    if neu_data["trialinfo"] is None or emo_data["trialinfo"] is None:
        raise ValueError(f"Missing trialinfo for participant {subj_id}")

    neu_data["trialinfo"][:, 0] = 1
    emo_data["trialinfo"][:, 0] = 2

    combined_trials = np.concatenate([neu_data["trial"], emo_data["trial"]], axis=0)
    combined_labels = np.concatenate([neu_data["trialinfo"][:, 0], emo_data["trialinfo"][:, 0]], axis=0)

    return combined_trials, combined_labels, neu_data["time"]


def get_test_files(test_path: str) -> list[str]:
    """Retrieve test subject filenames."""
    if not os.path.exists(test_path):
        return []
    return sorted(glob.glob(os.path.join(test_path, "test_subject_*.mat")))