import numpy as np


def analyze_dynamic_range(input_samples, ref_samples, sample_rate):
    """
    Analyze dynamic range of input vs reference audio.

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
        - input_dynamic_range_db: dynamic range of input in dB
        - ref_dynamic_range_db: dynamic range of reference in dB
        - dynamic_range_diff_db: difference (input - ref)
        - input_crest_factor_db: crest factor (peak/RMS) of input in dB
        - ref_crest_factor_db: crest factor of reference in dB
        - overcompression_detected: bool, True if input is significantly compressed
        - loudness_difference_db: difference in perceived loudness (RMS-based)
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    eps = 1e-10

    def _dynamic_range_db(samples):
        """Compute dynamic range as the difference between peak and noise floor."""
        peak = np.max(np.abs(samples))
        # Noise floor estimated as the 5th percentile of absolute values
        # (excluding near-silence frames)
        abs_samples = np.abs(samples)
        noise_floor = np.percentile(abs_samples[abs_samples > eps], 5)
        return float(20 * np.log10((peak + eps) / (noise_floor + eps)))

    def _crest_factor_db(samples):
        peak = np.max(np.abs(samples))
        rms = np.sqrt(np.mean(samples ** 2))
        return float(20 * np.log10((peak + eps) / (rms + eps)))

    input_dr = _dynamic_range_db(input_samples)
    ref_dr = _dynamic_range_db(ref_samples)
    dr_diff = float(input_dr - ref_dr)

    input_cf = _crest_factor_db(input_samples)
    ref_cf = _crest_factor_db(ref_samples)

    # Over-compression: input has noticeably lower dynamic range
    overcompression_detected = bool(dr_diff < -6.0)

    input_rms = float(np.sqrt(np.mean(input_samples ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_samples ** 2)))
    loudness_diff_db = float(20 * np.log10((input_rms + eps) / (ref_rms + eps)))

    return {
        "input_dynamic_range_db": input_dr,
        "ref_dynamic_range_db": ref_dr,
        "dynamic_range_diff_db": dr_diff,
        "input_crest_factor_db": input_cf,
        "ref_crest_factor_db": ref_cf,
        "overcompression_detected": overcompression_detected,
        "loudness_difference_db": loudness_diff_db,
    }
