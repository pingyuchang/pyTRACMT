from pathlib import Path
import importlib.util, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pytracmt_core", ROOT / "pyTRACMT.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)

def test_mt19937_64_is_reproducible():
    a, b = m.MT19937_64(1234), m.MT19937_64(1234)
    assert [a.uint64() for _ in range(10)] == [b.uint64() for _ in range(10)]

def test_robust_filter_replaces_isolated_spike():
    x = np.zeros(101); x[50] = 1e6
    y = m.robust_filter_channel(x, m.ThresholdsForRobustFilter(3, 5, 10))
    assert abs(y[50]) < 1.0

def test_section_spectra_frequency_bin_shape():
    fs, L, k = 16.0, 64, 4
    t = np.arange(L) / fs
    sec = np.column_stack([np.sin(2*np.pi*(k*fs/L)*t), np.cos(2*np.pi*(k*fs/L)*t)])
    out = m.section_spectra(sec, fs, m.SegmentSpec(L, [k]), 0.5)
    assert k in out and out[k].shape == (1, 2)

def test_tukey_weight_is_bounded():
    r = np.array([0.0, 1.0, 100.0])
    w = m.tukey_weight(r, c=4.685)
    assert np.all((w >= 0) & (w <= 1))
    assert w[-1] == 0.0
