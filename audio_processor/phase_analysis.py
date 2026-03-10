import numpy as np
from scipy.fft import fft, fftfreq


def analyze_phase(input_samples, ref_samples, sample_rate):
    """
    Analyze phase differences between input and reference audio using an
    FFT-based approach.

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
        - mean_phase_difference_rad: mean phase difference in radians
        - mean_phase_difference_deg: mean phase difference in degrees
        - phase_coherence: 0-1 score of phase alignment (1 = perfectly aligned)
        - estimated_delay_samples: estimated sample delay between signals
        - estimated_delay_ms: estimated delay in milliseconds
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    # Use cross-spectrum to estimate phase difference
    input_fft = fft(input_samples)
    ref_fft = fft(ref_samples)

    eps = 1e-10

    # Cross-power spectrum
    cross_spectrum = input_fft * np.conj(ref_fft)

    # Phase of the cross-spectrum gives point-wise phase difference
    phase_diff = np.angle(cross_spectrum)

    # Only consider positive frequencies with significant energy
    n = min_len
    freqs = fftfreq(n, 1 / sample_rate)
    pos_mask = freqs > 0

    ref_magnitude = np.abs(ref_fft[pos_mask])
    energy_threshold = np.percentile(ref_magnitude, 50)
    significant_mask = ref_magnitude > energy_threshold

    if significant_mask.sum() > 0:
        weighted_phase_diff = phase_diff[pos_mask][significant_mask]
        mean_phase_diff_rad = float(np.mean(weighted_phase_diff))
    else:
        mean_phase_diff_rad = float(np.mean(phase_diff[pos_mask]))

    mean_phase_diff_deg = float(np.degrees(mean_phase_diff_rad))

    # Phase coherence: magnitude of the mean unit phasor (circular mean)
    unit_phasors = np.exp(1j * phase_diff[pos_mask])
    phase_coherence = float(np.abs(np.mean(unit_phasors)))

    # Estimate delay via peak of IFFT of the normalised cross-spectrum
    normalised_cross = cross_spectrum / (np.abs(cross_spectrum) + eps)
    gcc = np.real(np.fft.ifft(normalised_cross))
    lag = int(np.argmax(gcc))
    # Convert to signed lag
    if lag > n // 2:
        lag = lag - n

    estimated_delay_samples = int(lag)
    estimated_delay_ms = float(lag / sample_rate * 1000)

    return {
        "mean_phase_difference_rad": mean_phase_diff_rad,
        "mean_phase_difference_deg": mean_phase_diff_deg,
        "phase_coherence": phase_coherence,
        "estimated_delay_samples": estimated_delay_samples,
        "estimated_delay_ms": estimated_delay_ms,
    }
