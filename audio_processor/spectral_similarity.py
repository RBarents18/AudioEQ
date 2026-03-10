import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity


def compute_spectral_similarity(input_samples, ref_samples, sample_rate):
    """
    Compute spectral similarity using MFCCs and cosine similarity.

    Parameters
    ----------
    input_samples : np.ndarray
        Input audio samples.
    ref_samples : np.ndarray
        Reference audio samples.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    dict
        - mfcc_cosine_similarity: float 0-1 (1 = identical spectra)
        - spectral_centroid_diff: difference in spectral centroid (Hz)
        - spectral_bandwidth_diff: difference in spectral bandwidth (Hz)
        - spectral_rolloff_diff: difference in spectral rolloff (Hz)
        - mfcc_distance: Euclidean distance between mean MFCCs
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    n_mfcc = 13

    # MFCCs
    input_mfcc = librosa.feature.mfcc(y=input_samples, sr=sample_rate, n_mfcc=n_mfcc)
    ref_mfcc = librosa.feature.mfcc(y=ref_samples, sr=sample_rate, n_mfcc=n_mfcc)

    input_mean_mfcc = np.mean(input_mfcc, axis=1).reshape(1, -1)
    ref_mean_mfcc = np.mean(ref_mfcc, axis=1).reshape(1, -1)

    cos_sim = float(cosine_similarity(input_mean_mfcc, ref_mean_mfcc)[0, 0])
    # Clamp to [0, 1]
    cos_sim = float(np.clip(cos_sim, 0.0, 1.0))

    mfcc_distance = float(np.linalg.norm(input_mean_mfcc - ref_mean_mfcc))

    # Spectral features
    def _mean_feature(fn, samples):
        return float(np.mean(fn(y=samples, sr=sample_rate)))

    input_centroid = _mean_feature(librosa.feature.spectral_centroid, input_samples)
    ref_centroid = _mean_feature(librosa.feature.spectral_centroid, ref_samples)

    input_bandwidth = _mean_feature(librosa.feature.spectral_bandwidth, input_samples)
    ref_bandwidth = _mean_feature(librosa.feature.spectral_bandwidth, ref_samples)

    input_rolloff = _mean_feature(librosa.feature.spectral_rolloff, input_samples)
    ref_rolloff = _mean_feature(librosa.feature.spectral_rolloff, ref_samples)

    return {
        "mfcc_cosine_similarity": cos_sim,
        "spectral_centroid_diff": float(input_centroid - ref_centroid),
        "spectral_bandwidth_diff": float(input_bandwidth - ref_bandwidth),
        "spectral_rolloff_diff": float(input_rolloff - ref_rolloff),
        "mfcc_distance": mfcc_distance,
    }
