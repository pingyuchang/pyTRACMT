from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pytracmt", ROOT / "pyTRACMT.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

def test_tukey_table_loads_and_has_expected_shape():
    cs, bs = m._load_tracmt_tukey_table()
    assert cs.shape == (1001,)
    assert bs.shape == (101, 1001)
    assert np.all(np.diff(cs) > 0)
    assert np.isfinite(bs).all()

def test_tukey_table_anchor_values():
    cs, bs = m._load_tracmt_tukey_table()
    assert cs[0] == 0.0
    assert np.isclose(cs[1], 0.1)
    assert np.isclose(cs[-1], 100.0)

def test_tukey_params_are_finite_positive():
    for dim, n in [(1, 100), (2, 100), (6, 256), (20, 1000)]:
        b, c = m.tukey_params(dim, n)
        assert np.isfinite(b) and b > 0
        assert np.isfinite(c) and c > 0
