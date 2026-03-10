import numpy as np
from scipy.signal import correlate


def calculate_snr(input_samples, ref_samples):
    """
    Estimate SNR by treating the reference as signal and (input - aligned_ref)
    as noise.  The two signals are aligned first using cross-correlation so
    that a small timing offset does not artificially inflate the noise floor.

    Parameters
    ----------
    input_samples : np.ndarray
        Input audio samples.
    ref_samples : np.ndarray
        Reference audio samples.

    Returns
    -------
    dict
        - snr_db: signal-to-noise ratio in dB
        - noise_rms: RMS of the noise component
        - signal_rms: RMS of the signal (reference)
        - quality_rating: "Excellent", "Good", "Fair", or "Poor"
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    # Align input to reference using cross-correlation
    correlation = correlate(input_samples, ref_samples, mode="full")
    lag = int(np.argmax(np.abs(correlation))) - (min_len - 1)

    if lag > 0:
        aligned_input = input_samples[lag:]
        aligned_ref = ref_samples[: min_len - lag]
    elif lag < 0:
        aligned_input = input_samples[: min_len + lag]
        aligned_ref = ref_samples[-lag:]
    else:
        aligned_input = input_samples
        aligned_ref = ref_samples

    # Truncate to same length after alignment
    align_len = min(len(aligned_input), len(aligned_ref))
    aligned_input = aligned_input[:align_len]
    aligned_ref = aligned_ref[:align_len]

    eps = 1e-10
    signal_rms = float(np.sqrt(np.mean(aligned_ref ** 2)))
    noise = aligned_input - aligned_ref
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))

    snr_db = float(20 * np.log10((signal_rms + eps) / (noise_rms + eps)))

    if snr_db >= 40:
        quality_rating = "Excellent"
    elif snr_db >= 25:
        quality_rating = "Good"
    elif snr_db >= 10:
        quality_rating = "Fair"
    else:
        quality_rating = "Poor"

    return {
        "snr_db": snr_db,
        "noise_rms": noise_rms,
        "signal_rms": signal_rms,
        "quality_rating": quality_rating,
    }
