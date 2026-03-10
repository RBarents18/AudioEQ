"""
Audio quality comparison module.
Provides modular analysis functions to compare an input audio signal
against a reference (studio-quality) audio signal.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter array so both have the same length."""
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)))
    return a, b


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert to mono by averaging channels if necessary."""
    if samples.ndim > 1:
        return samples.mean(axis=1)
    return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file.

    Returns
    -------
    samples : np.ndarray  (float32, mono, normalised to [-1, 1])
    sample_rate : int
    """
    samples, sr = librosa.load(filepath, sr=None, mono=True)
    return samples.astype(np.float32), sr


def frequency_analysis(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
    sr: int = 22050,
) -> dict:
    """
    FFT-based frequency comparison.

    Returns
    -------
    dict with keys:
        freq_bins       : frequency axis in Hz
        input_magnitude : magnitude spectrum of input (dB)
        ref_magnitude   : magnitude spectrum of reference (dB)
        deviation_db    : per-bin deviation (input - ref) in dB
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    n = len(input_audio)
    freq_bins = fftfreq(n, d=1.0 / sr)[: n // 2]

    eps = 1e-10
    input_fft = np.abs(fft(input_audio))[: n // 2]
    ref_fft = np.abs(fft(ref_audio))[: n // 2]

    input_mag_db = 20 * np.log10(input_fft + eps)
    ref_mag_db = 20 * np.log10(ref_fft + eps)
    deviation_db = input_mag_db - ref_mag_db

    return {
        "freq_bins": freq_bins.tolist(),
        "input_magnitude": input_mag_db.tolist(),
        "ref_magnitude": ref_mag_db.tolist(),
        "deviation_db": deviation_db.tolist(),
    }


def time_domain_analysis(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
) -> dict:
    """
    Analyse waveform differences in the time domain.

    Returns
    -------
    dict with keys:
        rms_input        : RMS of input signal
        rms_ref          : RMS of reference signal
        rms_diff_db      : difference in RMS levels (dB)
        clipping_detected: True if any sample |x| > 0.99
        amplitude_ratio  : rms_input / rms_ref
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    eps = 1e-10
    rms_input = float(np.sqrt(np.mean(input_audio ** 2)))
    rms_ref = float(np.sqrt(np.mean(ref_audio ** 2)))
    rms_diff_db = float(20 * np.log10((rms_input + eps) / (rms_ref + eps)))
    clipping_detected = bool(np.any(np.abs(input_audio) > 0.99))
    amplitude_ratio = float((rms_input + eps) / (rms_ref + eps))

    return {
        "rms_input": rms_input,
        "rms_ref": rms_ref,
        "rms_diff_db": rms_diff_db,
        "clipping_detected": clipping_detected,
        "amplitude_ratio": amplitude_ratio,
    }


def phase_analysis(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
) -> dict:
    """
    Measure phase differences using the cross-spectrum.

    Returns
    -------
    dict with keys:
        mean_phase_diff_deg : mean phase difference in degrees
        std_phase_diff_deg  : standard deviation of phase differences
        max_phase_diff_deg  : maximum absolute phase difference
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    n = len(input_audio)
    input_fft = fft(input_audio)[: n // 2]
    ref_fft = fft(ref_audio)[: n // 2]

    cross_spectrum = input_fft * np.conj(ref_fft)
    phase_diff_rad = np.angle(cross_spectrum)
    phase_diff_deg = np.degrees(phase_diff_rad)

    return {
        "mean_phase_diff_deg": float(np.mean(phase_diff_deg)),
        "std_phase_diff_deg": float(np.std(phase_diff_deg)),
        "max_phase_diff_deg": float(np.max(np.abs(phase_diff_deg))),
    }


def snr_calculation(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
) -> dict:
    """
    Signal-to-Noise Ratio calculation.
    Reference is treated as the clean signal; the difference is noise.

    Returns
    -------
    dict with keys:
        snr_db         : SNR in dB
        noise_floor_db : RMS of the noise (difference) in dB
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    eps = 1e-10
    noise = input_audio - ref_audio
    signal_power = float(np.mean(ref_audio ** 2))
    noise_power = float(np.mean(noise ** 2))

    snr_db = float(10 * np.log10((signal_power + eps) / (noise_power + eps)))
    noise_floor_db = float(10 * np.log10(noise_power + eps))

    return {
        "snr_db": snr_db,
        "noise_floor_db": noise_floor_db,
    }


def dynamic_range_analysis(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
) -> dict:
    """
    Compare dynamic ranges of both signals.

    Returns
    -------
    dict with keys:
        input_dynamic_range_db : peak-to-noise-floor range of input
        ref_dynamic_range_db   : peak-to-noise-floor range of reference
        crest_factor_input     : peak / RMS of input (dB)
        crest_factor_ref       : peak / RMS of reference (dB)
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)

    eps = 1e-10

    def _dynamic_range(audio):
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # approximate noise floor as the 5th percentile of absolute values
        noise_floor = float(np.percentile(np.abs(audio), 5))
        dr_db = 20 * np.log10((peak + eps) / (noise_floor + eps))
        cf_db = 20 * np.log10((peak + eps) / (rms + eps))
        return dr_db, cf_db

    input_dr, input_cf = _dynamic_range(input_audio)
    ref_dr, ref_cf = _dynamic_range(ref_audio)

    return {
        "input_dynamic_range_db": float(input_dr),
        "ref_dynamic_range_db": float(ref_dr),
        "crest_factor_input": float(input_cf),
        "crest_factor_ref": float(ref_cf),
    }


def spectral_similarity(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
    sr: int = 22050,
) -> dict:
    """
    Cosine similarity and MFCC-based comparison.

    Returns
    -------
    dict with keys:
        cosine_similarity : 0-1 (1 = identical spectra)
        mfcc_distance     : Euclidean distance between mean MFCC vectors
        mfcc_correlation  : Pearson correlation of mean MFCC vectors
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    eps = 1e-10
    n = len(input_audio)
    input_fft = np.abs(fft(input_audio))[: n // 2]
    ref_fft = np.abs(fft(ref_audio))[: n // 2]

    dot = float(np.dot(input_fft, ref_fft))
    norm = float(np.linalg.norm(input_fft) * np.linalg.norm(ref_fft))
    cosine_sim = float(dot / (norm + eps))

    # MFCC comparison
    input_mfcc = librosa.feature.mfcc(y=input_audio, sr=sr, n_mfcc=13)
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
    input_mean = input_mfcc.mean(axis=1)
    ref_mean = ref_mfcc.mean(axis=1)

    mfcc_distance = float(np.linalg.norm(input_mean - ref_mean))

    if np.std(input_mean) < eps or np.std(ref_mean) < eps:
        mfcc_correlation = 1.0 if np.allclose(input_mean, ref_mean) else 0.0
    else:
        mfcc_correlation = float(np.corrcoef(input_mean, ref_mean)[0, 1])

    return {
        "cosine_similarity": cosine_sim,
        "mfcc_distance": mfcc_distance,
        "mfcc_correlation": mfcc_correlation,
    }


def cross_correlation_analysis(
    input_audio: np.ndarray,
    ref_audio: np.ndarray,
    sr: int = 22050,
) -> dict:
    """
    Cross-correlation for timing alignment assessment.

    Returns
    -------
    dict with keys:
        max_correlation  : peak normalised cross-correlation value
        lag_samples      : lag at peak correlation (samples)
        lag_ms           : lag in milliseconds
        similarity_score : 0-1 score based on correlation peak
    """
    input_audio = _to_mono(input_audio)
    ref_audio = _to_mono(ref_audio)
    input_audio, ref_audio = _pad_to_match(input_audio, ref_audio)

    eps = 1e-10
    # Normalise to unit energy before correlating
    in_norm = input_audio / (np.linalg.norm(input_audio) + eps)
    ref_norm = ref_audio / (np.linalg.norm(ref_audio) + eps)

    corr = signal.correlate(in_norm, ref_norm, mode="full")
    lags = signal.correlation_lags(len(in_norm), len(ref_norm), mode="full")

    peak_idx = int(np.argmax(np.abs(corr)))
    max_corr = float(corr[peak_idx])
    lag_samples = int(lags[peak_idx])
    lag_ms = float(lag_samples / sr * 1000)
    similarity_score = float(np.clip(np.abs(max_corr), 0.0, 1.0))

    return {
        "max_correlation": max_corr,
        "lag_samples": lag_samples,
        "lag_ms": lag_ms,
        "similarity_score": similarity_score,
    }


def aggregate_quality_metrics(input_path: str, ref_path: str) -> dict:
    """
    Run all analyses and return aggregated metrics plus an overall quality
    score (0-100).

    Parameters
    ----------
    input_path : path to the input audio file
    ref_path   : path to the reference audio file
    """
    input_audio, input_sr = load_audio(input_path)
    ref_audio, ref_sr = load_audio(ref_path)

    # Use the higher sample rate as the working SR
    sr = max(input_sr, ref_sr)
    if input_sr != sr:
        input_audio = librosa.resample(input_audio, orig_sr=input_sr, target_sr=sr)
    if ref_sr != sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sr)

    freq = frequency_analysis(input_audio, ref_audio, sr)
    time_dom = time_domain_analysis(input_audio, ref_audio)
    phase = phase_analysis(input_audio, ref_audio)
    snr = snr_calculation(input_audio, ref_audio)
    dyn = dynamic_range_analysis(input_audio, ref_audio)
    spec_sim = spectral_similarity(input_audio, ref_audio, sr)
    xcorr = cross_correlation_analysis(input_audio, ref_audio, sr)

    # ------------------------------------------------------------------
    # Overall quality score (0–100) – weighted average of sub-scores
    # ------------------------------------------------------------------
    # SNR score: 0 dB → 0, 40 dB → 100
    snr_score = float(np.clip(snr["snr_db"] / 40.0 * 100, 0, 100))

    # Cosine similarity already in [0, 1]
    cosine_score = float(np.clip(spec_sim["cosine_similarity"] * 100, 0, 100))

    # MFCC correlation in [-1, 1] → [0, 100]
    mfcc_score = float(np.clip((spec_sim["mfcc_correlation"] + 1) / 2 * 100, 0, 100))

    # Amplitude ratio (1.0 is perfect)
    amp_score = float(max(0.0, 100.0 - abs(time_dom["amplitude_ratio"] - 1.0) * 100))

    # Clipping penalty
    clipping_penalty = 20.0 if time_dom["clipping_detected"] else 0.0

    weights = {"snr": 0.35, "cosine": 0.30, "mfcc": 0.20, "amp": 0.15}
    overall = (
        weights["snr"] * snr_score
        + weights["cosine"] * cosine_score
        + weights["mfcc"] * mfcc_score
        + weights["amp"] * amp_score
        - clipping_penalty
    )
    overall_quality_score = float(np.clip(overall, 0, 100))

    return {
        "frequency_analysis": freq,
        "time_domain_analysis": time_dom,
        "phase_analysis": phase,
        "snr": snr,
        "dynamic_range": dyn,
        "spectral_similarity": spec_sim,
        "cross_correlation": xcorr,
        "overall_quality_score": round(overall_quality_score, 2),
        "sample_rate": sr,
        "duration_input_s": float(len(input_audio) / sr),
        "duration_ref_s": float(len(ref_audio) / sr),
    }
