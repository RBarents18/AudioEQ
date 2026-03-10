import numpy as np


def analyze_time_domain(input_samples, ref_samples, sample_rate):
    """
    Analyze time-domain characteristics of input vs reference audio.

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
        - input_rms: RMS amplitude of input
        - ref_rms: RMS amplitude of reference
        - input_peak: peak amplitude of input
        - ref_peak: peak amplitude of reference
        - amplitude_ratio_db: difference in RMS expressed in dB
        - input_clipping_ratio: fraction of input samples near clipping (> 0.99)
        - ref_clipping_ratio: fraction of reference samples near clipping
        - clipping_detected: bool, True if input has significant clipping
        - dc_offset_input: mean value (DC offset) of input
        - dc_offset_ref: mean value (DC offset) of reference
        - input_waveform: downsampled input waveform (list)
        - ref_waveform: downsampled reference waveform (list)
    """
    min_len = min(len(input_samples), len(ref_samples))
    input_samples = input_samples[:min_len]
    ref_samples = ref_samples[:min_len]

    eps = 1e-10

    input_rms = float(np.sqrt(np.mean(input_samples ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_samples ** 2)))

    input_peak = float(np.max(np.abs(input_samples)))
    ref_peak = float(np.max(np.abs(ref_samples)))

    amplitude_ratio_db = float(20 * np.log10((input_rms + eps) / (ref_rms + eps)))

    clip_threshold = 0.99
    input_clipping_ratio = float(np.mean(np.abs(input_samples) > clip_threshold))
    ref_clipping_ratio = float(np.mean(np.abs(ref_samples) > clip_threshold))
    clipping_detected = bool(input_clipping_ratio > 0.001)

    dc_offset_input = float(np.mean(input_samples))
    dc_offset_ref = float(np.mean(ref_samples))

    # Downsample waveforms for JSON transport (max 2048 points)
    max_points = 2048
    step = max(1, min_len // max_points)
    input_waveform = input_samples[::step].tolist()
    ref_waveform = ref_samples[::step].tolist()

    return {
        "input_rms": input_rms,
        "ref_rms": ref_rms,
        "input_peak": input_peak,
        "ref_peak": ref_peak,
        "amplitude_ratio_db": amplitude_ratio_db,
        "input_clipping_ratio": input_clipping_ratio,
        "ref_clipping_ratio": ref_clipping_ratio,
        "clipping_detected": clipping_detected,
        "dc_offset_input": dc_offset_input,
        "dc_offset_ref": dc_offset_ref,
        "input_waveform": input_waveform,
        "ref_waveform": ref_waveform,
    }
