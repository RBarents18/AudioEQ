import io
import numpy as np
import librosa
import soundfile as sf


def load_audio(file_path, sr=None):
    """
    Load audio from a file path using librosa.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    sr : int or None
        Target sample rate. If None, the native sample rate is used.

    Returns
    -------
    tuple[np.ndarray, int]
        (samples, sample_rate)
    """
    samples, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    validate_audio(samples, sample_rate)
    return samples, sample_rate


def load_audio_from_bytes(audio_bytes, sr=None):
    """
    Load audio from a bytes object (e.g., from an uploaded file).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content.
    sr : int or None
        Target sample rate. If None, the native sample rate is used.

    Returns
    -------
    tuple[np.ndarray, int]
        (samples, sample_rate)
    """
    buf = io.BytesIO(audio_bytes)
    samples, sample_rate = librosa.load(buf, sr=sr, mono=True)
    validate_audio(samples, sample_rate)
    return samples, sample_rate


def validate_audio(samples, sample_rate):
    """
    Validate audio data.

    Parameters
    ----------
    samples : np.ndarray
        Audio samples.
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    ValueError
        If the audio data is invalid.
    """
    if samples is None or not isinstance(samples, np.ndarray):
        raise ValueError("samples must be a numpy array")
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D, got shape {samples.shape}")
    if len(samples) == 0:
        raise ValueError("audio contains no samples")
    if not np.isfinite(samples).all():
        raise ValueError("audio contains NaN or infinite values")
    if sample_rate is None or sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")
