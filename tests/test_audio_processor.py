"""
Unit tests for audio_processor module.
All tests generate synthetic audio in memory — no real files required
(except aggregate_quality_metrics which writes temp .wav files).
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import audio_processor as ap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050  # default sample rate for tests


def _sine(freq_hz: float = 440.0, duration_s: float = 1.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def _write_wav(samples: np.ndarray, sr: int = SR) -> str:
    """Write a mono float32 array to a temp .wav file. Returns the file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    # scipy.io.wavfile expects int16 or float32
    wavfile.write(tmp.name, sr, samples)
    return tmp.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFrequencyAnalysis(unittest.TestCase):
    def test_returns_expected_keys(self):
        a = _sine(440)
        b = _sine(880)
        result = ap.frequency_analysis(a, b, SR)
        for key in ("freq_bins", "input_magnitude", "ref_magnitude", "deviation_db"):
            self.assertIn(key, result)

    def test_deviation_array_shape(self):
        a = _sine(440)
        b = _sine(440)
        result = ap.frequency_analysis(a, b, SR)
        n = len(a)
        expected_len = n // 2
        self.assertEqual(len(result["deviation_db"]), expected_len)
        self.assertEqual(len(result["freq_bins"]), expected_len)

    def test_deviation_near_zero_for_identical(self):
        a = _sine(440)
        result = ap.frequency_analysis(a, a.copy(), SR)
        dev = np.array(result["deviation_db"])
        self.assertTrue(np.allclose(dev, 0.0, atol=1e-3))


class TestTimeDomainAnalysis(unittest.TestCase):
    def test_clipping_detected_above_threshold(self):
        a = np.ones(SR, dtype=np.float32) * 0.995  # |x| > 0.99 → clipping
        b = _sine(440)
        result = ap.time_domain_analysis(a, b)
        self.assertTrue(result["clipping_detected"])

    def test_no_clipping_below_threshold(self):
        a = _sine(440, amplitude=0.5)
        b = _sine(440, amplitude=0.5)
        result = ap.time_domain_analysis(a, b)
        self.assertFalse(result["clipping_detected"])

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.time_domain_analysis(a, a.copy())
        for key in ("rms_input", "rms_ref", "rms_diff_db", "clipping_detected", "amplitude_ratio"):
            self.assertIn(key, result)

    def test_amplitude_ratio_one_for_identical(self):
        a = _sine(440)
        result = ap.time_domain_analysis(a, a.copy())
        self.assertAlmostEqual(result["amplitude_ratio"], 1.0, places=4)


class TestSNRCalculation(unittest.TestCase):
    def test_high_snr_for_identical_signals(self):
        a = _sine(440)
        result = ap.snr_calculation(a, a.copy())
        self.assertGreater(result["snr_db"], 30.0)

    def test_lower_snr_with_added_noise(self):
        rng = np.random.default_rng(42)
        a = _sine(440)
        noisy = a + rng.normal(0, 0.15, len(a)).astype(np.float32)
        snr_clean = ap.snr_calculation(a, a.copy())["snr_db"]
        snr_noisy = ap.snr_calculation(noisy, a)["snr_db"]
        self.assertGreater(snr_clean, snr_noisy)

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.snr_calculation(a, a.copy())
        for key in ("snr_db", "noise_floor_db"):
            self.assertIn(key, result)


class TestDynamicRangeAnalysis(unittest.TestCase):
    def test_positive_dynamic_range(self):
        a = _sine(440)
        b = _sine(880)
        result = ap.dynamic_range_analysis(a, b)
        self.assertGreater(result["input_dynamic_range_db"], 0)
        self.assertGreater(result["ref_dynamic_range_db"], 0)

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.dynamic_range_analysis(a, a.copy())
        for key in ("input_dynamic_range_db", "ref_dynamic_range_db", "crest_factor_input", "crest_factor_ref"):
            self.assertIn(key, result)


class TestSpectralSimilarity(unittest.TestCase):
    def test_cosine_similarity_one_for_identical(self):
        a = _sine(440)
        result = ap.spectral_similarity(a, a.copy(), SR)
        self.assertAlmostEqual(result["cosine_similarity"], 1.0, places=4)

    def test_mfcc_correlation_high_for_identical(self):
        a = _sine(440)
        result = ap.spectral_similarity(a, a.copy(), SR)
        self.assertGreater(result["mfcc_correlation"], 0.99)

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.spectral_similarity(a, a.copy(), SR)
        for key in ("cosine_similarity", "mfcc_distance", "mfcc_correlation"):
            self.assertIn(key, result)


class TestCrossCorrelationAnalysis(unittest.TestCase):
    def test_near_zero_lag_for_identical(self):
        a = _sine(440)
        result = ap.cross_correlation_analysis(a, a.copy(), SR)
        self.assertAlmostEqual(result["lag_samples"], 0, delta=2)
        self.assertAlmostEqual(result["lag_ms"], 0.0, delta=0.5)

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.cross_correlation_analysis(a, a.copy(), SR)
        for key in ("max_correlation", "lag_samples", "lag_ms", "similarity_score"):
            self.assertIn(key, result)

    def test_similarity_score_high_for_identical(self):
        a = _sine(440)
        result = ap.cross_correlation_analysis(a, a.copy(), SR)
        self.assertGreater(result["similarity_score"], 0.99)


class TestPhaseAnalysis(unittest.TestCase):
    def test_near_zero_phase_diff_for_identical(self):
        a = _sine(440)
        result = ap.phase_analysis(a, a.copy())
        self.assertAlmostEqual(result["mean_phase_diff_deg"], 0.0, places=3)

    def test_returns_expected_keys(self):
        a = _sine(440)
        result = ap.phase_analysis(a, a.copy())
        for key in ("mean_phase_diff_deg", "std_phase_diff_deg", "max_phase_diff_deg"):
            self.assertIn(key, result)


class TestAggregateQualityMetrics(unittest.TestCase):
    def setUp(self):
        self.input_path = _write_wav(_sine(440))
        self.ref_path   = _write_wav(_sine(440))

    def tearDown(self):
        for p in (self.input_path, self.ref_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    def test_end_to_end_returns_quality_score(self):
        result = ap.aggregate_quality_metrics(self.input_path, self.ref_path)
        self.assertIn("overall_quality_score", result)
        self.assertGreaterEqual(result["overall_quality_score"], 0)
        self.assertLessEqual(result["overall_quality_score"], 100)

    def test_high_score_for_identical_audio(self):
        result = ap.aggregate_quality_metrics(self.input_path, self.ref_path)
        self.assertGreater(result["overall_quality_score"], 80)

    def test_all_top_level_keys_present(self):
        result = ap.aggregate_quality_metrics(self.input_path, self.ref_path)
        for key in (
            "frequency_analysis", "time_domain_analysis", "phase_analysis",
            "snr", "dynamic_range", "spectral_similarity", "cross_correlation",
            "overall_quality_score", "sample_rate",
        ):
            self.assertIn(key, result)

    def test_different_lengths_handled(self):
        """Shorter input should be zero-padded without error."""
        short_path = _write_wav(_sine(440, duration_s=0.5))
        try:
            result = ap.aggregate_quality_metrics(short_path, self.ref_path)
            self.assertIn("overall_quality_score", result)
        finally:
            os.unlink(short_path)


if __name__ == "__main__":
    unittest.main()
