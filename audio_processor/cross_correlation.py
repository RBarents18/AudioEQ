import numpy as np
from scipy.signal import correlate


def compute_cross_correlation(input_samples, ref_samples, sample_rate):
    """
    Compute cross-correlation between input and reference audio.

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
        - max_correlation: peak cross-correlation value, normalised to [0, 1]
        - lag_samples: lag (in samples) at peak correlation
        - lag_ms: lag in milliseconds
        - structural_similarity: overall structural similarity score 0-1
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    eps = 1e-10

    correlation = correlate(input_samples, ref_samples, mode="full")

    # Normalise so that identical signals give 1.0
    norm = float(
        np.sqrt(np.sum(input_samples ** 2) * np.sum(ref_samples ** 2)) + eps
    )
    norm_corr = correlation / norm

    max_idx = int(np.argmax(np.abs(norm_corr)))
    max_correlation = float(np.clip(np.abs(norm_corr[max_idx]), 0.0, 1.0))

    # Convert to signed lag
    lag_samples = max_idx - (min_len - 1)
    lag_ms = float(lag_samples / sample_rate * 1000)

    # Structural similarity: Pearson correlation of the aligned signals
    if lag_samples > 0:
        a = input_samples[lag_samples:]
        b = ref_samples[: min_len - lag_samples]
    elif lag_samples < 0:
        a = input_samples[: min_len + lag_samples]
        b = ref_samples[-lag_samples:]
    else:
        a = input_samples
        b = ref_samples

    align_len = min(len(a), len(b))
    a = a[:align_len]
    b = b[:align_len]

    if align_len > 1:
        pearson = float(np.corrcoef(a, b)[0, 1])
        structural_similarity = float(np.clip((pearson + 1) / 2, 0.0, 1.0))
    else:
        structural_similarity = 0.0

    return {
        "max_correlation": max_correlation,
        "lag_samples": int(lag_samples),
        "lag_ms": lag_ms,
        "structural_similarity": structural_similarity,
    }
