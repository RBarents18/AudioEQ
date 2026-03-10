import numpy as np

from .frequency_analysis import compare_frequency_spectra
from .time_domain import analyze_time_domain
from .phase_analysis import analyze_phase
from .snr import calculate_snr
from .dynamic_range import analyze_dynamic_range
from .spectral_similarity import compute_spectral_similarity
from .cross_correlation import compute_cross_correlation


def compute_quality_report(input_samples, ref_samples, sample_rate):
    """
    Aggregate all analysis results into a comprehensive quality report.

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
        {
          "overall_score": float 0-100,
          "frequency": <frequency_analysis results>,
          "time_domain": <time_domain results>,
          "phase": <phase_analysis results>,
          "snr": <snr results>,
          "dynamic_range": <dynamic_range results>,
          "spectral_similarity": <spectral_similarity results>,
          "cross_correlation": <cross_correlation results>,
          "feedback": [list of actionable string recommendations],
          "quality_rating": "Excellent" | "Good" | "Fair" | "Poor"
        }
    """
    freq_results = compare_frequency_spectra(input_samples, ref_samples, sample_rate)
    td_results = analyze_time_domain(input_samples, ref_samples, sample_rate)
    phase_results = analyze_phase(input_samples, ref_samples, sample_rate)
    snr_results = calculate_snr(input_samples, ref_samples)
    dr_results = analyze_dynamic_range(input_samples, ref_samples, sample_rate)
    spec_results = compute_spectral_similarity(input_samples, ref_samples, sample_rate)
    xcorr_results = compute_cross_correlation(input_samples, ref_samples, sample_rate)

    # ------------------------------------------------------------------
    # Overall score: weighted average of normalised sub-scores (0-100)
    # ------------------------------------------------------------------

    # 1. SNR score (0-100): map SNR from [-10 dB, 60 dB] → [0, 100]
    snr_score = float(np.clip((snr_results["snr_db"] + 10) / 70 * 100, 0, 100))

    # 2. Spectral similarity score (already 0-1, scale to 0-100)
    spectral_score = float(spec_results["mfcc_cosine_similarity"] * 100)

    # 3. Cross-correlation score (already 0-1)
    xcorr_score = float(xcorr_results["max_correlation"] * 100)

    # 4. Phase coherence score (already 0-1)
    phase_score = float(phase_results["phase_coherence"] * 100)

    # 5. Frequency deviation score: map deviation_db [0, 30] → [100, 0]
    freq_dev = freq_results["deviation_db"]
    freq_score = float(np.clip(100 - (freq_dev / 30 * 100), 0, 100))

    # 6. Dynamic range score: penalise over-compression
    dr_diff = dr_results["dynamic_range_diff_db"]
    # Full score if diff > -3 dB, zero at -30 dB
    if dr_diff < -3:
        dr_score_raw = 100 + (dr_diff / 27 * 100)
    else:
        dr_score_raw = 100
    dr_score = float(np.clip(dr_score_raw, 0, 100))

    weights = {
        "snr": 0.30,
        "spectral": 0.25,
        "xcorr": 0.20,
        "phase": 0.10,
        "freq": 0.10,
        "dr": 0.05,
    }

    overall_score = float(
        snr_score * weights["snr"]
        + spectral_score * weights["spectral"]
        + xcorr_score * weights["xcorr"]
        + phase_score * weights["phase"]
        + freq_score * weights["freq"]
        + dr_score * weights["dr"]
    )
    overall_score = float(np.clip(overall_score, 0.0, 100.0))

    # Quality rating
    if overall_score >= 80:
        quality_rating = "Excellent"
    elif overall_score >= 60:
        quality_rating = "Good"
    elif overall_score >= 40:
        quality_rating = "Fair"
    else:
        quality_rating = "Poor"

    # ------------------------------------------------------------------
    # Feedback messages
    # ------------------------------------------------------------------
    feedback = []

    snr_db = snr_results["snr_db"]
    if snr_db < 10:
        feedback.append(
            f"Very low SNR ({snr_db:.1f} dB) – check for background noise or encoding artifacts."
        )
    elif snr_db < 25:
        feedback.append(
            f"Moderate SNR ({snr_db:.1f} dB) – consider noise reduction or higher-quality encoding."
        )

    if td_results["clipping_detected"]:
        ratio_pct = td_results["input_clipping_ratio"] * 100
        feedback.append(
            f"Clipping detected in input ({ratio_pct:.2f}% of samples) – reduce input gain to prevent distortion."
        )

    if dr_results["overcompression_detected"]:
        feedback.append(
            f"Over-compression detected: input dynamic range is "
            f"{abs(dr_results['dynamic_range_diff_db']):.1f} dB lower than reference. "
            "Consider using a gentler compressor setting."
        )

    amp_diff = td_results["amplitude_ratio_db"]
    if abs(amp_diff) > 3:
        direction = "louder" if amp_diff > 0 else "quieter"
        feedback.append(
            f"Input is {abs(amp_diff):.1f} dB {direction} than reference – "
            "apply loudness normalization for a fair comparison."
        )

    cos_sim = spec_results["mfcc_cosine_similarity"]
    if cos_sim < 0.7:
        feedback.append(
            f"Low spectral similarity ({cos_sim:.2f}) – tonal character differs significantly from reference."
        )

    phase_coh = phase_results["phase_coherence"]
    if phase_coh < 0.5:
        feedback.append(
            f"Low phase coherence ({phase_coh:.2f}) – signals may be time-shifted or phase-inverted."
        )

    delay_ms = abs(phase_results["estimated_delay_ms"])
    if delay_ms > 5:
        feedback.append(
            f"Estimated timing offset of {delay_ms:.1f} ms detected – align the signals before comparison."
        )

    freq_dev_val = freq_results["deviation_db"]
    if freq_dev_val > 10:
        feedback.append(
            f"High frequency deviation ({freq_dev_val:.1f} dB) – significant spectral colouration present."
        )

    dc_input = td_results["dc_offset_input"]
    if abs(dc_input) > 0.01:
        feedback.append(
            f"DC offset detected in input ({dc_input:.4f}) – apply a high-pass filter to remove it."
        )

    if not feedback:
        feedback.append("Audio quality looks great – no significant issues detected.")

    return {
        "overall_score": overall_score,
        "frequency": freq_results,
        "time_domain": td_results,
        "phase": phase_results,
        "snr": snr_results,
        "dynamic_range": dr_results,
        "spectral_similarity": spec_results,
        "cross_correlation": xcorr_results,
        "feedback": feedback,
        "quality_rating": quality_rating,
    }
