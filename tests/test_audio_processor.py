"""
Tests for the audio_processor package.

All tests use synthetically generated audio (sine waves / noise) so there are
no external dependencies and no mocking required.
"""
import numpy as np
import pytest

# ── Helpers ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
DURATION    = 1.0  # seconds


def _sine(freq=440.0, sr=SAMPLE_RATE, duration=DURATION, amplitude=0.5):
    """Return a mono sine-wave array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(sr=SAMPLE_RATE, duration=DURATION, amplitude=0.1, seed=42):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(sr * duration))).astype(np.float32)


def _clipping_signal(sr=SAMPLE_RATE, duration=DURATION):
    """Sine wave with amplitude > 1.0 so many samples clip at ±0.99."""
    return np.clip(_sine(amplitude=2.0, sr=sr, duration=duration), -1.0, 1.0).astype(np.float32)


# ── Loader ─────────────────────────────────────────────────────────────────

class TestLoader:
    def test_validate_valid(self):
        from audio_processor.loader import validate_audio
        validate_audio(_sine(), SAMPLE_RATE)  # should not raise

    def test_validate_empty(self):
        from audio_processor.loader import validate_audio
        with pytest.raises(ValueError, match="no samples"):
            validate_audio(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_validate_non_finite(self):
        from audio_processor.loader import validate_audio
        bad = _sine().copy()
        bad[10] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_audio(bad, SAMPLE_RATE)

    def test_validate_bad_sr(self):
        from audio_processor.loader import validate_audio
        with pytest.raises(ValueError, match="invalid sample rate"):
            validate_audio(_sine(), -1)

    def test_load_audio_from_bytes(self, tmp_path):
        import soundfile as sf
        from audio_processor.loader import load_audio_from_bytes
        audio_file = tmp_path / "test.wav"
        sig = _sine()
        sf.write(str(audio_file), sig, SAMPLE_RATE)
        audio_bytes = audio_file.read_bytes()
        samples, sr = load_audio_from_bytes(audio_bytes)
        assert len(samples) > 0
        assert sr > 0


# ── Frequency Analysis ─────────────────────────────────────────────────────

class TestFrequencyAnalysis:
    EXPECTED_KEYS = {
        "input_freqs", "input_magnitudes",
        "ref_freqs",   "ref_magnitudes",
        "deviation_db", "prominent_deviations",
    }

    def test_returns_expected_keys(self):
        from audio_processor.frequency_analysis import compare_frequency_spectra
        result = compare_frequency_spectra(_sine(), _sine(freq=880), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_low_deviation(self):
        from audio_processor.frequency_analysis import compare_frequency_spectra
        sig = _sine()
        result = compare_frequency_spectra(sig, sig, SAMPLE_RATE)
        assert result["deviation_db"] < 1.0

    def test_prominent_deviations_length(self):
        from audio_processor.frequency_analysis import compare_frequency_spectra
        result = compare_frequency_spectra(_sine(), _sine(freq=880), SAMPLE_RATE)
        assert len(result["prominent_deviations"]) == 5

    def test_different_lengths_handled(self):
        from audio_processor.frequency_analysis import compare_frequency_spectra
        short = _sine(duration=0.5)
        long  = _sine(duration=1.0)
        # Should not raise
        result = compare_frequency_spectra(short, long, SAMPLE_RATE)
        assert "deviation_db" in result

    def test_compute_fft_positive_freqs(self):
        from audio_processor.frequency_analysis import compute_fft
        freqs, mags = compute_fft(_sine(), SAMPLE_RATE)
        assert np.all(freqs >= 0)
        assert len(freqs) == len(mags)


# ── Time Domain ────────────────────────────────────────────────────────────

class TestTimeDomain:
    EXPECTED_KEYS = {
        "input_rms", "ref_rms", "input_peak", "ref_peak",
        "amplitude_ratio_db", "input_clipping_ratio", "ref_clipping_ratio",
        "clipping_detected", "dc_offset_input", "dc_offset_ref",
        "input_waveform", "ref_waveform",
    }

    def test_returns_expected_keys(self):
        from audio_processor.time_domain import analyze_time_domain
        result = analyze_time_domain(_sine(), _noise(), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_detects_clipping(self):
        from audio_processor.time_domain import analyze_time_domain
        clipped = _clipping_signal()
        result  = analyze_time_domain(clipped, _sine(), SAMPLE_RATE)
        assert result["clipping_detected"] is True

    def test_no_clipping_for_normal_signal(self):
        from audio_processor.time_domain import analyze_time_domain
        result = analyze_time_domain(_sine(amplitude=0.3), _sine(amplitude=0.3), SAMPLE_RATE)
        assert result["clipping_detected"] is False

    def test_rms_positive(self):
        from audio_processor.time_domain import analyze_time_domain
        result = analyze_time_domain(_sine(), _sine(), SAMPLE_RATE)
        assert result["input_rms"] > 0
        assert result["ref_rms"]   > 0

    def test_identical_amplitude_ratio_near_zero(self):
        from audio_processor.time_domain import analyze_time_domain
        sig    = _sine()
        result = analyze_time_domain(sig, sig, SAMPLE_RATE)
        assert abs(result["amplitude_ratio_db"]) < 0.01

    def test_waveform_lists_returned(self):
        from audio_processor.time_domain import analyze_time_domain
        result = analyze_time_domain(_sine(), _sine(), SAMPLE_RATE)
        assert isinstance(result["input_waveform"], list)
        assert isinstance(result["ref_waveform"],   list)
        assert len(result["input_waveform"]) > 0


# ── Phase Analysis ─────────────────────────────────────────────────────────

class TestPhaseAnalysis:
    EXPECTED_KEYS = {
        "mean_phase_difference_rad", "mean_phase_difference_deg",
        "phase_coherence", "estimated_delay_samples", "estimated_delay_ms",
    }

    def test_returns_expected_keys(self):
        from audio_processor.phase_analysis import analyze_phase
        result = analyze_phase(_sine(), _noise(), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_high_coherence(self):
        from audio_processor.phase_analysis import analyze_phase
        sig    = _sine()
        result = analyze_phase(sig, sig, SAMPLE_RATE)
        assert result["phase_coherence"] > 0.9

    def test_coherence_in_range(self):
        from audio_processor.phase_analysis import analyze_phase
        result = analyze_phase(_sine(), _noise(), SAMPLE_RATE)
        assert 0.0 <= result["phase_coherence"] <= 1.0

    def test_identical_signals_near_zero_delay(self):
        from audio_processor.phase_analysis import analyze_phase
        sig    = _sine()
        result = analyze_phase(sig, sig, SAMPLE_RATE)
        assert abs(result["estimated_delay_ms"]) < 1.0

    def test_deg_conversion(self):
        from audio_processor.phase_analysis import analyze_phase
        result = analyze_phase(_sine(), _sine(), SAMPLE_RATE)
        expected = np.degrees(result["mean_phase_difference_rad"])
        assert abs(result["mean_phase_difference_deg"] - expected) < 1e-6


# ── SNR ────────────────────────────────────────────────────────────────────

class TestSNR:
    EXPECTED_KEYS = {"snr_db", "noise_rms", "signal_rms", "quality_rating"}

    def test_returns_expected_keys(self):
        from audio_processor.snr import calculate_snr
        result = calculate_snr(_sine(), _sine())
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_high_snr(self):
        from audio_processor.snr import calculate_snr
        sig    = _sine()
        result = calculate_snr(sig, sig)
        assert result["snr_db"] > 60

    def test_noisy_signal_lower_snr(self):
        from audio_processor.snr import calculate_snr
        sig    = _sine(amplitude=0.5)
        noisy  = sig + _noise(amplitude=0.3)
        result = calculate_snr(noisy, sig)
        assert result["snr_db"] < 60

    def test_quality_rating_values(self):
        from audio_processor.snr import calculate_snr
        result = calculate_snr(_sine(), _sine())
        assert result["quality_rating"] in ("Excellent", "Good", "Fair", "Poor")

    def test_signal_rms_positive(self):
        from audio_processor.snr import calculate_snr
        result = calculate_snr(_sine(), _sine())
        assert result["signal_rms"] > 0

    def test_different_lengths_handled(self):
        from audio_processor.snr import calculate_snr
        short = _sine(duration=0.5)
        long  = _sine(duration=1.0)
        result = calculate_snr(short, long)
        assert "snr_db" in result


# ── Dynamic Range ──────────────────────────────────────────────────────────

class TestDynamicRange:
    EXPECTED_KEYS = {
        "input_dynamic_range_db", "ref_dynamic_range_db", "dynamic_range_diff_db",
        "input_crest_factor_db", "ref_crest_factor_db",
        "overcompression_detected", "loudness_difference_db",
    }

    def test_returns_expected_keys(self):
        from audio_processor.dynamic_range import analyze_dynamic_range
        result = analyze_dynamic_range(_sine(), _noise(), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_zero_diff(self):
        from audio_processor.dynamic_range import analyze_dynamic_range
        sig    = _sine()
        result = analyze_dynamic_range(sig, sig, SAMPLE_RATE)
        assert abs(result["dynamic_range_diff_db"]) < 0.01

    def test_overcompression_flag(self):
        from audio_processor.dynamic_range import analyze_dynamic_range
        # Heavily compressed: constant amplitude (very low dynamic range)
        compressed = np.full(int(SAMPLE_RATE * DURATION), 0.5, dtype=np.float32)
        result = analyze_dynamic_range(compressed, _sine(amplitude=0.8), SAMPLE_RATE)
        assert result["overcompression_detected"] is True

    def test_no_overcompression_identical(self):
        from audio_processor.dynamic_range import analyze_dynamic_range
        sig    = _sine()
        result = analyze_dynamic_range(sig, sig, SAMPLE_RATE)
        assert result["overcompression_detected"] is False


# ── Spectral Similarity ────────────────────────────────────────────────────

class TestSpectralSimilarity:
    EXPECTED_KEYS = {
        "mfcc_cosine_similarity", "spectral_centroid_diff",
        "spectral_bandwidth_diff", "spectral_rolloff_diff", "mfcc_distance",
    }

    def test_returns_expected_keys(self):
        from audio_processor.spectral_similarity import compute_spectral_similarity
        result = compute_spectral_similarity(_sine(), _noise(), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_similarity_near_one(self):
        from audio_processor.spectral_similarity import compute_spectral_similarity
        sig    = _sine()
        result = compute_spectral_similarity(sig, sig, SAMPLE_RATE)
        assert result["mfcc_cosine_similarity"] > 0.99

    def test_similarity_in_range(self):
        from audio_processor.spectral_similarity import compute_spectral_similarity
        result = compute_spectral_similarity(_sine(), _noise(), SAMPLE_RATE)
        assert 0.0 <= result["mfcc_cosine_similarity"] <= 1.0

    def test_identical_signals_zero_mfcc_distance(self):
        from audio_processor.spectral_similarity import compute_spectral_similarity
        sig    = _sine()
        result = compute_spectral_similarity(sig, sig, SAMPLE_RATE)
        assert result["mfcc_distance"] < 1e-4

    def test_different_lengths_handled(self):
        from audio_processor.spectral_similarity import compute_spectral_similarity
        short = _sine(duration=0.5)
        long  = _sine(duration=1.0)
        result = compute_spectral_similarity(short, long, SAMPLE_RATE)
        assert "mfcc_cosine_similarity" in result


# ── Cross-Correlation ──────────────────────────────────────────────────────

class TestCrossCorrelation:
    EXPECTED_KEYS = {
        "max_correlation", "lag_samples", "lag_ms", "structural_similarity",
    }

    def test_returns_expected_keys(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        result = compute_cross_correlation(_sine(), _noise(), SAMPLE_RATE)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_identical_signals_max_correlation_near_one(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        sig    = _sine()
        result = compute_cross_correlation(sig, sig, SAMPLE_RATE)
        assert result["max_correlation"] > 0.99

    def test_identical_signals_zero_lag(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        sig    = _sine()
        result = compute_cross_correlation(sig, sig, SAMPLE_RATE)
        assert result["lag_samples"] == 0

    def test_correlation_in_range(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        result = compute_cross_correlation(_sine(), _noise(), SAMPLE_RATE)
        assert 0.0 <= result["max_correlation"] <= 1.0

    def test_structural_similarity_in_range(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        result = compute_cross_correlation(_sine(), _noise(), SAMPLE_RATE)
        assert 0.0 <= result["structural_similarity"] <= 1.0

    def test_identical_signals_structural_similarity_near_one(self):
        from audio_processor.cross_correlation import compute_cross_correlation
        sig    = _sine()
        result = compute_cross_correlation(sig, sig, SAMPLE_RATE)
        assert result["structural_similarity"] > 0.99


# ── Quality Metrics ────────────────────────────────────────────────────────

class TestQualityMetrics:
    TOP_LEVEL_KEYS = {
        "overall_score", "frequency", "time_domain", "phase", "snr",
        "dynamic_range", "spectral_similarity", "cross_correlation",
        "feedback", "quality_rating",
    }

    def test_returns_expected_top_level_keys(self):
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _noise(), SAMPLE_RATE)
        assert self.TOP_LEVEL_KEYS.issubset(result.keys())

    def test_overall_score_in_range(self):
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _noise(), SAMPLE_RATE)
        assert 0.0 <= result["overall_score"] <= 100.0

    def test_identical_signals_high_score(self):
        from audio_processor.quality_metrics import compute_quality_report
        sig    = _sine()
        result = compute_quality_report(sig, sig, SAMPLE_RATE)
        assert result["overall_score"] >= 70.0

    def test_quality_rating_valid(self):
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _noise(), SAMPLE_RATE)
        assert result["quality_rating"] in ("Excellent", "Good", "Fair", "Poor")

    def test_feedback_is_non_empty_list(self):
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _noise(), SAMPLE_RATE)
        assert isinstance(result["feedback"], list)
        assert len(result["feedback"]) >= 1

    def test_all_sub_dicts_present(self):
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _sine(), SAMPLE_RATE)
        for key in ("frequency", "time_domain", "phase", "snr",
                    "dynamic_range", "spectral_similarity", "cross_correlation"):
            assert isinstance(result[key], dict), f"{key} should be a dict"

    def test_scores_are_python_floats(self):
        """Ensure all numeric top-level values are JSON-serialisable Python floats."""
        import json
        from audio_processor.quality_metrics import compute_quality_report
        result = compute_quality_report(_sine(), _sine(), SAMPLE_RATE)
        # If any numpy scalar leaks this will raise TypeError
        json.dumps(result)
