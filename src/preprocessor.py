import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import RobustScaler


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter using Second-Order Sections (SOS)."""
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def extract_covariance_matrix(arr: np.ndarray, win_min: int, win_max: int) -> np.ndarray:
    """Extract covariance matrices across trials within a specified window."""
    arr = arr[:, :, win_min:win_max]
    window_size = arr.shape[-1]
    covariances = np.einsum("ijk,ilk->ijl", arr, arr) / window_size
    return covariances


def extract_136_features(covs: np.ndarray, n_channels: int = 16) -> np.ndarray:
    """Extract upper-triangular elements from covariance matrices."""
    triu_row, triu_col = np.triu_indices(n_channels)
    return covs[:, triu_row, triu_col]

# def extract_asymmetry_features(covs: np.ndarray, pair_indices: list, eps: float = 1e-3) -> np.ndarray:
#     """
#     DASM = log(P_left) - log(P_right)      -> always numerically safe (subtraction)
#     RASM = log(P_left) / log(P_right)      -> literature definition (Zheng & Lu style DE ratio)
#     """
#     raw_powers = np.diagonal(covs, axis1=1, axis2=2)  # (n_trials, 16), always > 0 by construction
#     log_powers = np.log(np.maximum(raw_powers, eps))  # DE-like proxy, CAN be negative/near-zero

#     dasm_list, rasm_list = [], []
#     for left_idx, right_idx in pair_indices:
#         p_left = log_powers[:, left_idx]
#         p_right = log_powers[:, right_idx]

#         dasm = p_left - p_right

#         # safe_denom = np.where(p_right >= 0, 1.0, -1.0) * np.maximum(np.abs(p_right), eps)
#         # rasm = p_left / safe_denom

#         dasm_list.append(dasm)
#         # rasm_list.append(rasm)

#     return np.column_stack(dasm_list)
    
def extract_raw_asymmetry_features(trials: np.ndarray, pair_indices: list, win_min: int, win_max: int, eps: float = 1e-3) -> np.ndarray:
    """DASM/RASM computed from RAW (pre-RobustScaler) channel variance."""
    raw_powers = np.var(trials[:, :, win_min:win_max], axis=-1)   # (n_trials, 16)
    log_powers = np.log(np.maximum(raw_powers, eps))

    dasm_list, rasm_list = [], []
    for left_idx, right_idx in pair_indices:
        p_left = log_powers[:, left_idx]
        p_right = log_powers[:, right_idx]

        dasm_list.append(p_left - p_right)

        safe_denom = np.where(p_right >= 0, 1.0, -1.0) * np.maximum(np.abs(p_right), eps)
        rasm_list.append(p_left / safe_denom)

    return np.column_stack(dasm_list + rasm_list)

def preprocess_participant_trials(trials: np.ndarray, config: dict) -> np.ndarray:
    """End-to-end signal processing and feature extraction pipeline for one participant."""
    n_trials, n_channels, _ = trials.shape
    sp_cfg = config["signal_processing"]
    feat_cfg = config["features"]

    # Bandpass filtering


    data_filtered_alpha = np.zeros_like(trials)
    for trial in range(n_trials):
        for ch in range(n_channels):
            data_filtered_alpha[trial, ch, :] = butter_bandpass_filter(
                trials[trial, ch, :],
                8,
                13,
                sp_cfg["sampling_rate"],
            )
    asymmetric_features = extract_raw_asymmetry_features(data_filtered_alpha, feat_cfg["asymmetry_pairs"], feat_cfg["window_min"], feat_cfg["window_max"])
    del data_filtered_alpha

    data_filtered_theta = np.zeros_like(trials)
    for trial in range(n_trials):
        for ch in range(n_channels):
            data_filtered_theta[trial, ch, :] = butter_bandpass_filter(
                trials[trial, ch, :],
                sp_cfg["freq_band"]["min"],
                sp_cfg["freq_band"]["max"],
                sp_cfg["sampling_rate"],
            )
    
    # Robust scaling across channel-time dimensions per participant
    orig_shape = data_filtered_theta.shape
    data_filtered_theta = data_filtered_theta.reshape(orig_shape[0], -1)
    scaler = RobustScaler()
    data_filtered_theta = scaler.fit_transform(data_filtered_theta).reshape(orig_shape)

    # mean = np.mean(data_filtered, axis=0)
    # std = np.std(data_filtered, axis=0)
    # data_filtered = (data_filtered - mean) / (std + 1e-10)

    # Covariance and upper triangle extraction
    cov_matrices = extract_covariance_matrix(data_filtered_theta, feat_cfg["window_min"], feat_cfg["window_max"])
    features = extract_136_features(cov_matrices, n_channels=sp_cfg["n_channels"])

    features = np.hstack([features, asymmetric_features])

    # Z-score normalization per participant
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)


# Here, with respect to feature selection(Top-75)
# I used the excellent Eng. Mohamed Samy's idea and code
# https://github.com/MohamedQiqa/eeg-emotion-classification-pipeline/blob/master/_research_architecture/01_pipeline_blueprint.md

def select_top_features(coef_matrix: np.ndarray, top_k: int = 75) -> np.ndarray:
    """Select feature indices based on coefficient stability across LOPO folds."""
    direction_agreement = np.abs(np.sign(coef_matrix).mean(axis=0)) # Direction(+/-) agreement per folds
    mean_absolute_coef = np.mean(np.abs(coef_matrix), axis=0) # Mean Absolute coefficient(How much is that feature important for us?)
    coefficient_cv = np.std(np.abs(coef_matrix), axis=0) / (mean_absolute_coef + 1e-10)

    final_score = direction_agreement * mean_absolute_coef / (coefficient_cv + 0.1)
    return np.argsort(-final_score)[:top_k]