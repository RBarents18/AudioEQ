import numpy as np
from scipy.fft import fft, fftfreq


def compute_fft(samples, sample_rate):
    """
    Compute FFT of audio samples.

    Parameters
    ----------
    samples : np.ndarray
        1-D audio samples.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, magnitudes) for positive frequencies only.
    """
    n = len(samples)
    freqs = fftfreq(n, 1 / sample_rate)
    fft_vals = np.abs(fft(samples))
    pos_mask = freqs >= 0
    return freqs[pos_mask], fft_vals[pos_mask]


def compare_frequency_spectra(input_samples, ref_samples, sample_rate):
    """
    Compare frequency spectra of input vs reference audio.

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
        - input_freqs: list of frequencies (Hz)
        - input_magnitudes: list of FFT magnitudes for input
        - ref_freqs: list of frequencies (Hz)
        - ref_magnitudes: list of FFT magnitudes for reference
        - deviation_db: mean absolute deviation in dB between spectra
        - prominent_deviations: list of [freq, deviation_db] for top 5 deviating frequencies
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    input_freqs, input_mags = compute_fft(input_samples, sample_rate)
    ref_freqs, ref_mags = compute_fft(ref_samples, sample_rate)

    eps = 1e-10
    input_db = 20 * np.log10(input_mags + eps)
    ref_db = 20 * np.log10(ref_mags + eps)

    deviation_db_arr = np.abs(input_db - ref_db)
    mean_deviation_db = float(np.mean(deviation_db_arr))

    # Top 5 most deviating frequencies
    top5_idx = np.argsort(deviation_db_arr)[-5:][::-1]
    prominent_deviations = [
        [float(input_freqs[i]), float(deviation_db_arr[i])] for i in top5_idx
    ]

    # Downsample for JSON transport (max 2048 points)
    max_points = 2048
    step = max(1, len(input_freqs) // max_points)

    return {
        "input_freqs": input_freqs[::step].tolist(),
        "input_magnitudes": input_mags[::step].tolist(),
        "ref_freqs": ref_freqs[::step].tolist(),
        "ref_magnitudes": ref_mags[::step].tolist(),
        "deviation_db": mean_deviation_db,
        "prominent_deviations": prominent_deviations,
    }
