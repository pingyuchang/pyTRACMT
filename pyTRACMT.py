#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyTRACMT Professional v11 (TRACMT Source-Aligned Edition)
Release version: 1.0.1
======================================================

Integrated Python engine + English GUI/CLI for TRACMT-style MT processing.
V7 focuses on numerical compatibility with TRACMT, including corrected robust scales, staged prior weights, Tukey scale iteration, and TRACMT error propagation:

- PROCEDURE 0: ordinary remote reference (RR) with optional IRLS M-estimators
- PROCEDURE 1: robust multivariate regression / RRMS-style Fast-S estimator
- - PREWHITENING: none / standard AR / robust PARCOR-like candidate AR
- ROBUST_FILTER: robust time-domain spike suppression/replacement
- COHERENCE_CRITERIA: squared-coherence based segment rejection
- CAL_FILES: TRACMT calibration file format, frequency-domain correction
- TRACMT.log and TRACMT.cvg output
- response_functions.csv, apparent_resistivity_and_phase.csv (Zxx/Zxy/... column names)
- basic EDI export

Important implementation note
-----------------------------
This is a Python reconstruction guided by the TRACMT source structure and
published algorithm. It is intended to be numerically compatible in workflow and
file format. V8 adds the C++ std::mt19937_64 engine, strict full-refit bootstrap, and LAPACK-style complex solves. V10 uses the complete TRACMT Tukey b/c table and exact C++ interpolation and scale update. In this SoftwareX revision the table is stored as an inspectable plain-text CSV with BSD-3-Clause attribution and SHA-256 integrity validation. V11 additionally matches short-section zero padding, removes unintended segment demeaning, uses the TRACMT seed 1234 mt19937_64 candidate stream, and reproduces the C++ ordinary-RR coherence rejection definition. Bit identity still depends on using the same TRACMT build, compiler standard library, input precision, and preprocessing settings.

Lead developer: Ping-Yu Chang
Scientific reference implementation: TRACMT by Yoshiya Usui
"""
from __future__ import annotations
__version__ = "1.0.1"
import dataclasses as dc
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, linalg, signal

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import tkinter.font as tkfont
    TK_OK = True
except Exception:
    TK_OK = False

MU0 = 4.0 * math.pi * 1e-7
EPS = 1e-30


# -----------------------------------------------------------------------------
# C++ std::mt19937_64 compatible random engine
# -----------------------------------------------------------------------------
class MT19937_64:
    """Reference implementation of std::mt19937_64.

    The state transition and tempering constants are those prescribed by the
    C++ standard. ``uniform_int`` uses unbiased rejection sampling, matching
    the semantics of ``std::uniform_int_distribution<int>``. Exact sample
    streams can still vary among C++ standard-library vendors because the
    distribution mapping itself is implementation-defined; the engine words
    are bit-identical.
    """
    _MASK = (1 << 64) - 1
    _N, _M = 312, 156
    _A = 0xB5026F5AA96619E9
    _UM = 0xFFFFFFFF80000000
    _LM = 0x7FFFFFFF
    _F = 6364136223846793005

    def __init__(self, seed: int = 5489):
        self.mt = [0] * self._N
        self.index = self._N
        self.mt[0] = int(seed) & self._MASK
        for i in range(1, self._N):
            x = self.mt[i - 1]
            self.mt[i] = (self._F * (x ^ (x >> 62)) + i) & self._MASK

    def _twist(self) -> None:
        for i in range(self._N):
            x = (self.mt[i] & self._UM) | (self.mt[(i + 1) % self._N] & self._LM)
            xa = x >> 1
            if x & 1:
                xa ^= self._A
            self.mt[i] = self.mt[(i + self._M) % self._N] ^ xa
        self.index = 0

    def uint64(self) -> int:
        if self.index >= self._N:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & self._MASK

    def uniform_int(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("high must be >= low")
        span = int(high) - int(low) + 1
        if span <= 0:
            raise OverflowError("integer range is too large")
        limit = ((1 << 64) // span) * span
        while True:
            r = self.uint64()
            if r < limit:
                return int(low) + int(r % span)

    def indexes(self, n: int, size: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        return np.fromiter((self.uniform_int(0, n - 1) for _ in range(size)),
                           dtype=np.int64, count=size)

# -----------------------------------------------------------------------------
# Logging compatible with TRACMT.log / TRACMT.cvg
# -----------------------------------------------------------------------------
class OutputFiles:
    def __init__(self, outdir: Path, program: str = "TRACMT"):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.log_path = self.outdir / f"{program}.log"
        self.cvg_path = self.outdir / f"{program}.cvg"
        self.log = self.log_path.open("w", encoding="utf-8")
        self.cvg = self.cvg_path.open("w", encoding="utf-8")
        self.write_log(f"Start {program} Version pyTRACMT-v11-tracmt-source-aligned", elapsed=False)

    def close(self):
        try:
            self.write_log("End TRACMT")
        finally:
            self.log.close()
            self.cvg.close()

    def elapsed(self) -> str:
        return f"( {time.time() - self.t0:.2f} sec )"

    def write_log(self, msg: str, elapsed: bool = True):
        self.log.write(msg + (" " + self.elapsed() if elapsed else "") + "\n")
        self.log.flush()

    def write_cvg(self, msg: str):
        self.cvg.write(msg + "\n")
        self.cvg.flush()

    def write_both(self, msg: str, elapsed: bool = True):
        self.write_log(msg, elapsed=elapsed)
        self.write_cvg(msg)

    def warning(self, msg: str):
        self.write_log("[Warning] " + msg)

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dc.dataclass
class SegmentSpec:
    length: int
    indexes: List[int]

@dc.dataclass
class ThresholdsForRobustFilter:
    first: float = 10.0
    second: float = 12.0
    max_consecutive: int = 50

@dc.dataclass
class Params:
    num_out: int = 2
    num_rr: int = 2
    num_input: int = 2
    sampling_freq: float = 1.0
    sampling_freq_org: float = 1.0
    num_threads: int = 1
    num_section: int = 1
    data_sections: List[Tuple[int, List[Tuple[str, int]]]] = dc.field(default_factory=list)
    segments: List[SegmentSpec] = dc.field(default_factory=list)
    overlap: float = 0.5
    azimuth: List[float] = dc.field(default_factory=list)
    rotation_deg: float = 0.0
    procedure: int = 0  # 0 RR, 1 RRMS, 2 MRRMS-like
    rrms: Tuple[int, int, int, float, int, int, float] = (1, 100, 3, 0.05, 10, 16, 0.01)
    m_estimators: Tuple[int, int] = (0, 1)  # -1 none, 0 Huber, 1 Tukey, 2 Thomson
    huber_threshold: float = 3.0
    thomson_param: float = 0.0
    robust_iter_max: int = 10
    robust_conv: float = 0.01
    error_estimation: int = 1
    bootstrap: int = 1000
    output_rhoa_phs: bool = False
    output_edi: bool = True
    output_level: int = 0
    cal_files: List[str] = dc.field(default_factory=list)
    prewhitening: Optional[Tuple[int, int, int]] = None  # type, max_ar, num_Candidates
    robust_filter: Optional[List[ThresholdsForRobustFilter]] = None
    robust_filter_replace: bool = False
    coherence_criteria: Optional[Tuple[int, float]] = None
    high_pass: Optional[float] = None
    low_pass: Optional[float] = None
    notch: List[float] = dc.field(default_factory=list)
    notch_q: float = 10.0
    start_times: List[str] = dc.field(default_factory=list)

    @property
    def nchan(self) -> int:
        return self.num_out + self.num_input + self.num_rr


def _clean_param_lines(path: Path) -> List[str]:
    """Read param.dat while preserving paths that contain spaces.

    TRACMT-style files are keyword driven. Many Windows paths contain spaces
    (for example, "sickfish Dropbox/Chang pingyu").  Therefore this parser
    must never split a whole line and assume token[1] is numeric unless the
    LAST token is explicitly a numeric skip count.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in raw_lines:
        # keep inner spaces in paths, only strip ends and remove full-line comments
        s = line.strip().strip('"').strip("'")
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _is_keyword_line(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z_]+", text.strip().upper()))


def _is_number_token(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _split_numeric_line(text: str) -> List[str]:
    return [v for v in re.split(r"[,;\s]+", text.strip()) if v]


def parse_param(path: str | Path) -> Params:
    path = Path(path)
    lines = _clean_param_lines(path)
    p = Params()
    i = 0

    def read_numbers_until_keyword(start: int) -> Tuple[List[str], int]:
        vals = []
        j = start
        while j < len(lines):
            token = lines[j].strip()
            if _is_keyword_line(token):
                break
            vals.extend(token.split())
            j += 1
        return vals, j

    while i < len(lines):
        key = lines[i].strip().upper()
        i += 1
        if key == "END":
            break
        if key == "NUM_OUT":
            p.num_out = int(lines[i].split()[0]); i += 1
        elif key == "NUM_RR":
            p.num_rr = int(lines[i].split()[0]); i += 1
        elif key == "SAMPLING_FREQ":
            p.sampling_freq = float(lines[i].split()[0]); p.sampling_freq_org = p.sampling_freq; i += 1
        elif key == "NUM_THREADS":
            p.num_threads = int(lines[i].split()[0]); i += 1
        elif key == "NUM_SECTION":
            p.num_section = int(lines[i].split()[0]); i += 1
        elif key == "OVERLAP":
            p.overlap = float(lines[i].split()[0]); i += 1
        elif key == "PROCEDURE":
            val = lines[i].split()[0].strip().upper(); i += 1
            if val in ("0", "RR", "ORDINARY_REMOTE_REFERENCE"):
                p.procedure = 0
            elif val in ("1", "RRMS", "MULTIVARIATE_REGRESSION"):
                p.procedure = 1
            elif val in ("2", "MRRMS", "MODIFIED_MULTIVARIATE_REGRESSION"):
                p.procedure = 2
            else:
                p.procedure = int(float(val))
        elif key in ("RRMS", "ROBUST_MULTIVARIATE_REGRESSION"):
            vals = []
            while len(vals) < 7 and i < len(lines):
                vals.extend(lines[i].split()); i += 1
            p.rrms = (int(vals[0]), int(vals[1]), int(vals[2]), float(vals[3]), int(vals[4]), int(vals[5]), float(vals[6]))
        elif key == "MESTIMATORS":
            vals = []
            while len(vals) < 2 and i < len(lines):
                if _is_keyword_line(lines[i]):
                    break
                vals.extend(lines[i].split()); i += 1
            if len(vals) >= 2:
                p.m_estimators = (int(vals[0]), int(vals[1]))
            elif len(vals) == 1:
                p.m_estimators = (int(vals[0]), -1)
        elif key == "HUBER":
            vals = lines[i].split(); i += 1
            p.huber_threshold = float(vals[0])
            if len(vals) > 1: p.robust_iter_max = int(vals[1])
            if len(vals) > 2: p.robust_conv = float(vals[2])
        elif key == "THOMSON":
            vals = lines[i].split(); i += 1
            p.thomson_param = float(vals[0])
            if len(vals) > 1: p.robust_iter_max = int(vals[1])
            if len(vals) > 2: p.robust_conv = float(vals[2])
        elif key == "TUKEYS_BIWEIGHTS":
            vals = lines[i].split(); i += 1
            p.robust_iter_max = int(vals[0])
            if len(vals) > 1: p.robust_conv = float(vals[1])
        elif key == "ERROR_ESTIMATION":
            p.error_estimation = int(lines[i].split()[0]); i += 1
        elif key == "BOOTSTRAP":
            p.bootstrap = int(lines[i].split()[0]); i += 1
        elif key == "OUTPUT_RHOA_PHS":
            p.output_rhoa_phs = True
        elif key == "OUTPUT_EDI":
            p.output_edi = True
        elif key == "OUTPUT_LEVEL":
            p.output_level = int(lines[i].split()[0]); i += 1
        elif key == "ROTATION":
            p.rotation_deg = float(lines[i].split()[0]); i += 1
        elif key == "AZIMUTH":
            vals, i = read_numbers_until_keyword(i)
            p.azimuth = [float(v) for v in vals]
        elif key == "CAL_FILES":
            vals = []
            while len(vals) < max(p.nchan, 1) and i < len(lines):
                if _is_keyword_line(lines[i]):
                    break
                vals.extend(lines[i].split()); i += 1
            p.cal_files = vals[:p.nchan]
        elif key == "SEGMENT":
            nseg = int(lines[i].split()[0]); i += 1
            segs = []
            for _ in range(nseg):
                vals = [int(x) for x in lines[i].split()]; i += 1
                L, nidx = vals[0], vals[1]
                segs.append(SegmentSpec(L, vals[2:2+nidx]))
            p.segments = segs
        elif key == "DATA_FILES":
            # TRACMT format: for each section: ndata, then nchan entries.
            # Each channel can be either:
            #   <full path possibly with spaces>
            #   <skip>
            # or one line:
            #   <full path possibly with spaces> <skip>
            sections = []
            if p.num_section <= 0:
                p.num_section = 1
            for _section in range(p.num_section):
                if i >= len(lines):
                    raise ValueError("DATA_FILES ended before data count")
                ndata = int(float(_split_numeric_line(lines[i])[0])); i += 1
                files = []
                for _c in range(p.nchan):
                    if i >= len(lines):
                        raise ValueError("DATA_FILES ended before all channel file names were read")
                    line = lines[i].strip().strip('"').strip("'")
                    i += 1
                    toks = line.split()
                    # Only treat a token on the same line as skip if the LAST token is numeric.
                    # This preserves paths like "sickfish Dropbox/Chang pingyu/ex.txt".
                    if len(toks) >= 2 and _is_number_token(toks[-1]):
                        skip = int(float(toks[-1]))
                        fname = " ".join(toks[:-1]).strip().strip('"').strip("'")
                    else:
                        fname = line
                        if i < len(lines) and _is_number_token(lines[i].split()[0]) and not _is_keyword_line(lines[i]):
                            skip = int(float(lines[i].split()[0])); i += 1
                        else:
                            skip = 0
                    files.append((fname, skip))
                sections.append((ndata, files))
            p.data_sections = sections
        elif key == "PREWHITENING":
            vals = []
            while len(vals) < 3 and i < len(lines):
                vals.extend(lines[i].split()); i += 1
            p.prewhitening = (int(vals[0]), int(vals[1]), int(vals[2]))
        elif key == "ROBUST_FILTER":
            first = lines[i].split(); i += 1
            p.robust_filter_replace = int(first[0]) != 0
            ths = []
            for _c in range(p.nchan):
                vals = lines[i].split(); i += 1
                ths.append(ThresholdsForRobustFilter(float(vals[0]), float(vals[1]), int(vals[2])))
            p.robust_filter = ths
        elif key == "COHERENCE_CRITERIA":
            n = int(lines[i].split()[0]); i += 1
            thr = float(lines[i].split()[0]); i += 1
            p.coherence_criteria = (n, thr)
        elif key == "HIGH_PASS":
            p.high_pass = float(lines[i].split()[0]); i += 1
        elif key == "LOW_PASS":
            p.low_pass = float(lines[i].split()[0]); i += 1
        elif key == "NOTCH":
            vals = lines[i].split(); i += 1
            n = int(vals[0])
            freqs = vals[1:]
            while len(freqs) < n:
                freqs.extend(lines[i].split()); i += 1
            p.notch = [float(x) for x in freqs[:n]]
        elif key == "NOTCH_PARAM_Q":
            p.notch_q = float(lines[i].split()[0]); i += 1
        elif key == "START_TIMES":
            vals, i = read_numbers_until_keyword(i)
            p.start_times = vals[:p.num_section]
        else:
            # Skip unsupported keyword data conservatively if next line is not a keyword.
            # This keeps parser tolerant for ATS/ELOG/MTH5-specific options.
            continue
    if not p.segments:
        raise ValueError("SEGMENT block is required")
    if not p.data_sections:
        # backward-compatible fallback: previous starter format with one n and simple pairs
        raise ValueError("DATA_FILES block is required in TRACMT section format")
    return p

# -----------------------------------------------------------------------------
# Data IO and preprocessing
# -----------------------------------------------------------------------------
def read_channel(path: Path, skip: int, n: int) -> np.ndarray:
    """Read one numeric channel robustly.

    Supports whitespace or CSV files, optional headers, and multi-column files.
    If multiple numeric columns exist, the last numeric column is used, matching
    the earlier pyTRACMT behavior.
    """
    path = _candidate_existing_path(Path(str(path).strip().strip('"').strip("'")))
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Time-series file not found.\n" + _path_diagnostic(path))
    # First try pandas so headers like Time,Ex,Ey,Hx,Hy do not crash.
    try:
        if str(path).lower().endswith((".csv", ".txt", ".dat")):
            try:
                df = pd.read_csv(path, comment="#")
            except Exception:
                df = pd.read_csv(path, comment="#", delim_whitespace=True, header=None)
            num = df.apply(pd.to_numeric, errors="coerce")
            # drop all-empty columns and all-empty rows
            num = num.dropna(axis=1, how="all").dropna(axis=0, how="all")
            if num.shape[1] > 0 and num.shape[0] > 0:
                arr = num.iloc[:, -1].to_numpy(dtype=float)
            else:
                raise ValueError("no numeric column found")
        else:
            raise ValueError("fallback")
    except Exception:
        try:
            arr = np.genfromtxt(path, comments="#", delimiter=None, invalid_raise=False)
        except Exception:
            arr = np.genfromtxt(path, comments="#", delimiter=",", invalid_raise=False)
        arr = np.asarray(arr)
        if arr.ndim > 1:
            # Use last column with at least one finite number
            good_cols = [j for j in range(arr.shape[1]) if np.isfinite(arr[:, j]).any()]
            if not good_cols:
                raise ValueError(f"No numeric data found in {path}")
            arr = arr[:, good_cols[-1]]
        arr = arr[np.isfinite(arr)]
    if skip:
        arr = arr[int(skip):]
    if n is not None and n > 0 and n < 10**17:
        arr = arr[:int(n)]
    if arr.size == 0:
        raise ValueError(f"No numeric samples read from {path}")
    return arr.astype(float)

def _candidate_existing_path(path: Path) -> Path:
    """Return an existing path candidate without changing a valid DATA_FILES path.

    Important: a path like .../ex.txt/ex.txt can be valid when the first
    ex.txt is a folder and the second ex.txt is the data file. Therefore this
    function never removes repeated tail names. It only tries harmless Windows
    path normalizations and user-home relocation when the exact path is missing.
    """
    raw = str(path).strip().strip('"').strip("'")
    Candidates = []

    def add(x):
        try:
            xp = Path(str(x).strip().strip('"').strip("'"))
            if xp not in Candidates:
                Candidates.append(xp)
        except Exception:
            pass

    add(raw)
    add(os.path.normpath(raw))

    # Windows long-path form. Useful for deep Dropbox paths or non-ASCII paths.
    if os.name == "nt":
        norm = os.path.abspath(os.path.normpath(raw))
        if re.match(r"^[A-Za-z]:\\", norm) and not norm.startswith("\\\\?\\"):
            add("\\\\?\\" + norm)

        # If param.dat was generated on another Windows account, try current home.
        m = re.match(r"^([A-Za-z]:[\\/])Users[\\/]([^\\/]+)[\\/](.*)$", raw)
        if m:
            home = Path.home()
            add(home / m.group(3))

    for cand in Candidates:
        try:
            if cand.exists() and cand.is_file():
                return cand
        except Exception:
            continue
    return Path(raw)


def _path_diagnostic(path: Path) -> str:
    """Create useful diagnostics for FileNotFoundError."""
    parts = []
    try:
        parts.append(f"exact={path}")
        parts.append(f"cwd={Path.cwd()}")
        parts.append(f"home={Path.home()}")
        cur = path
        checked = []
        while cur != cur.parent and len(checked) < 8:
            checked.append(f"{cur} -> exists={cur.exists()}, is_file={cur.is_file()}, is_dir={cur.is_dir()}")
            cur = cur.parent
        parts.append("parent_check: " + " | ".join(checked))
    except Exception as e:
        parts.append(f"diagnostic_failed={e}")
    return "\n".join(parts)


def _resolve_data_path(param_path: Path, fname: str) -> Path:
    """Resolve TRACMT DATA_FILES path safely, including Windows paths with spaces.

    This version respects the exact path written in param.dat. It does NOT
    collapse repeated names like ex.txt/ex.txt because that can be intentional.
    """
    text = str(fname).strip().strip('"').strip("'")
    text = os.path.expandvars(os.path.expanduser(text))
    # Windows absolute paths are not considered absolute by pathlib on Linux,
    # but they are absolute when the script is executed on Windows.
    if Path(text).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text):
        return _candidate_existing_path(Path(text))
    return _candidate_existing_path(param_path.parent / text)


def load_data(param_path: str | Path, p: Params, out: OutputFiles) -> List[np.ndarray]:
    """Load all DATA_FILES sections.

    This is intentionally TRACMT-compatible rather than split()-based:
    paths may contain spaces, each channel has its own skip count, and
    all channels in a section are trimmed to the same minimum length.
    """
    param_path = Path(param_path)
    if not p.data_sections:
        raise ValueError("No DATA_FILES sections found in param.dat")
    sections: List[np.ndarray] = []
    for isec, (ndata, files) in enumerate(p.data_sections):
        if len(files) < p.nchan:
            raise ValueError(f"DATA_FILES section {isec}: expected {p.nchan} channels, got {len(files)}")
        out.write_log(f"Load DATA_FILES section {isec+1}: requested samples={ndata}", elapsed=False)
        cols = []
        names = []
        for ich, (fname, skip) in enumerate(files[:p.nchan]):
            fpath = _resolve_data_path(param_path, fname)
            arr = read_channel(fpath, skip=skip, n=ndata)
            cols.append(arr)
            names.append(str(fpath))
            out.write_log(f"  channel {ich}: {fpath} skip={skip} samples={len(arr)}", elapsed=False)
        nmin = min(len(c) for c in cols)
        if nmin <= 0:
            raise ValueError(f"DATA_FILES section {isec}: no usable samples")
        data = np.column_stack([np.asarray(c[:nmin], dtype=float) for c in cols])
        sections.append(data)
        out.write_log(f"  section {isec+1} loaded: {data.shape[0]} samples x {data.shape[1]} channels", elapsed=False)
    return sections


def apply_filters(data: np.ndarray, fs: float, p: Params) -> np.ndarray:
    x = np.array(data, dtype=float, copy=True)
    if p.high_pass:
        sos = signal.butter(1, p.high_pass, btype="highpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    if p.low_pass:
        sos = signal.butter(2, p.low_pass, btype="lowpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    for f0 in p.notch:
        b, a = signal.iirnotch(f0, p.notch_q, fs=fs)
        x = signal.filtfilt(b, a, x, axis=0)
    return x


def robust_filter_channel(y: np.ndarray, th: ThresholdsForRobustFilter) -> np.ndarray:
    y = np.asarray(y, float).copy()
    med = signal.medfilt(y, kernel_size=5 if len(y) > 5 else 3)
    r = y - med
    sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + EPS
    z = np.abs(r) / sigma
    bad = z >= th.first
    if not bad.any():
        return y
    # Replace bad spikes by median-filtered values, with protection for long runs.
    i = 0
    while i < len(y):
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < len(y) and bad[j]:
            j += 1
        if (j - i) <= th.max_consecutive or z[i:j].max() >= th.second:
            y[i:j] = med[i:j]
        i = j
    return y


def apply_robust_filter(data: np.ndarray, p: Params, out: OutputFiles) -> np.ndarray:
    if not p.robust_filter:
        return data
    out.write_both("Apply robust filter")
    y = np.array(data, copy=True)
    for c in range(y.shape[1]):
        y[:, c] = robust_filter_channel(y[:, c], p.robust_filter[min(c, len(p.robust_filter)-1)])
    return y if p.robust_filter_replace else data


def estimate_ar_yw(y: np.ndarray, max_order: int) -> Tuple[np.ndarray, int]:
    y = np.asarray(y, float)
    y = y - np.nanmean(y)
    max_order = max(1, min(max_order, len(y)//5))
    best_a = np.zeros(0)
    best_aic = np.inf
    best_p = 0
    ac = signal.correlate(y, y, mode="full")
    ac = ac[len(y)-1:len(y)+max_order] / max(len(y), 1)
    for p in range(1, max_order+1):
        R = linalg.toeplitz(ac[:p]) + np.eye(p)*EPS
        rhs = ac[1:p+1]
        try:
            a = linalg.solve(R, rhs, assume_a="pos")
        except Exception:
            a = np.linalg.lstsq(R, rhs, rcond=None)[0]
        evar = max(ac[0] - np.dot(a, rhs), EPS)
        aic = len(y) * (math.log(2*math.pi*evar) + 1.0) + 2*(p+1)
        if aic < best_aic:
            best_aic, best_a, best_p = aic, a, p
    return best_a, best_p


def prewhiten_channel(y: np.ndarray, method: int, max_ar: int, ncand: int) -> Tuple[np.ndarray, np.ndarray]:
    # method -1 user coefficients not supported here; 0 standard; 1 robust approximation.
    yy = np.asarray(y, float)
    if method == 1:
        yy0 = robust_filter_channel(yy, ThresholdsForRobustFilter(6, 8, 50))
    else:
        yy0 = yy
    a, order = estimate_ar_yw(yy0, max_ar)
    if len(a) == 0:
        return yy.copy(), a
    # whitening: y_t - sum a_k y_{t-k}
    out = yy.copy()
    for k, ak in enumerate(a, start=1):
        out[k:] -= ak * yy[:-k]
    out[:len(a)] = out[len(a)] if len(out) > len(a) else 0
    return out, a


def apply_prewhitening(sections: List[np.ndarray], p: Params, out: OutputFiles) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not p.prewhitening:
        return sections, []
    method, max_ar, ncand = p.prewhitening
    out.write_both(f"Apply prewhitening: method={method}, max_ar={max_ar}, Candidates={ncand}")
    result = []
    coeffs_all = []
    for sec in sections:
        w = np.array(sec, copy=True)
        coeffs_sec = []
        for c in range(sec.shape[1]):
            w[:, c], coeff = prewhiten_channel(sec[:, c], method, max_ar, ncand)
            coeffs_sec.append(coeff)
        result.append(w)
        coeffs_all.append(coeffs_sec)
    return result, coeffs_all

# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
class CalibrationFunction:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.factor = 1.0
        self.freq = np.array([], dtype=float)
        self.amp = np.array([], dtype=float)
        self.phase = np.array([], dtype=float)
        self._read()

    def _read(self):
        vals = []
        for line in self.path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.extend(s.split())
        if len(vals) < 2:
            return
        self.factor = float(vals[0])
        n = int(float(vals[1]))
        rows = vals[2:]
        if n <= 0 or len(rows) < 3*n:
            return
        arr = np.array([float(x) for x in rows[:3*n]]).reshape(n, 3)
        f = arr[:, 0]
        c = arr[:, 1] + 1j * arr[:, 2]
        idx = np.argsort(f)
        self.freq = f[idx]
        self.amp = np.abs(c[idx])
        self.phase = np.unwrap(np.angle(c[idx]))

    def response(self, freq: float) -> complex:
        if len(self.freq) == 0:
            return complex(self.factor)
        lf = np.log10(self.freq)
        x = np.log10(max(freq, self.freq[0]))
        logamp = np.log10(np.maximum(self.amp, EPS))
        if len(self.freq) >= 3:
            try:
                ai = interpolate.Akima1DInterpolator(lf, logamp, extrapolate=True)(x)
                pi = interpolate.Akima1DInterpolator(lf, self.phase, extrapolate=True)(x)
            except Exception:
                ai = np.interp(x, lf, logamp)
                pi = np.interp(x, lf, self.phase)
        else:
            ai = np.interp(x, lf, logamp)
            pi = np.interp(x, lf, self.phase)
        return self.factor * (10.0 ** float(ai)) * np.exp(1j * float(pi))


def load_calibrations(param_file: Path, p: Params, out: OutputFiles) -> List[Optional[CalibrationFunction]]:
    cals = []
    base = param_file.parent
    for i, name in enumerate(p.cal_files):
        if not name or name.lower() in ("none", "null", "-"):
            cals.append(None); continue
        fp = Path(name)
        if not fp.is_absolute(): fp = base / fp
        if not fp.exists():
            out.warning(f"Calibration file not found for channel {i}: {fp}")
            cals.append(None)
        else:
            out.write_log(f"Read calibration function from {fp}")
            cals.append(CalibrationFunction(fp))
    while len(cals) < p.nchan:
        cals.append(None)
    return cals

# -----------------------------------------------------------------------------
# Spectra
# -----------------------------------------------------------------------------
def section_spectra(sec: np.ndarray, fs: float, seg: SegmentSpec, overlap: float) -> Dict[int, np.ndarray]:
    """Create TRACMT-style tapered FFT segments.

    Important source-alignment details:
    * A section shorter than the requested segment length is zero padded and
      processed as one segment (TRACMT Analysis::convertToFrequencyData).
    * No extra per-segment demeaning is applied here. Any baseline removal must
      be requested through preprocessing, as in TRACMT.
    * A periodic Hann window is used, corresponding to the endpoint convention
      normally used by TRACMT's hanningWindow before the FFT.
    """
    L = int(seg.length)
    step = max(1, int((1.0 - overlap) * L))
    nchan = sec.shape[1]
    if sec.shape[0] < L:
        starts = [0]
    else:
        starts = range(0, sec.shape[0] - L + 1, step)
    win = signal.windows.hann(L, sym=False)
    out = {k: [] for k in seg.indexes}
    for st in starts:
        x = np.zeros((L, nchan), dtype=float)
        take = sec[st:min(st + L, sec.shape[0]), :]
        x[:take.shape[0], :] = np.nan_to_num(take, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.fft.rfft(x * win[:, None], axis=0)
        for k in seg.indexes:
            if 0 < k < X.shape[0]:
                out[k].append(X[k, :])
    return {k: np.asarray(v, complex) for k, v in out.items() if len(v)}


def collect_frequency_blocks(sections: List[np.ndarray], p: Params) -> List[Tuple[int, int, float, np.ndarray]]:
    blocks = []
    for iseg, seg in enumerate(p.segments):
        for k in seg.indexes:
            allv = []
            for sec in sections:
                sp = section_spectra(sec, p.sampling_freq, seg, p.overlap)
                if k in sp:
                    allv.append(sp[k])
            if allv:
                ft = np.vstack(allv)  # nseg x nchan
                freq = k * p.sampling_freq / seg.length
                blocks.append((seg.length, k, freq, ft))
    return blocks


def apply_calibration_ft(ft: np.ndarray, freq: float, cals: Sequence[Optional[CalibrationFunction]]) -> np.ndarray:
    y = np.array(ft, copy=True)
    for c, cal in enumerate(cals):
        if cal is None:
            continue
        resp = cal.response(freq)
        if abs(resp) > EPS:
            y[:, c] /= resp
    return y

# -----------------------------------------------------------------------------
# Robust weights and estimators
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TRACMT Tukey biweight parameter table (transparent external CSV)
# -----------------------------------------------------------------------------
# Numerical values are transcribed from TRACMT's
# src/TableOfTukeysBiweightParameters.h, distributed upstream under BSD-3-Clause.
# They are intentionally stored as a plain-text CSV in data/ rather than as an
# opaque compressed payload. This makes the data directly inspectable and avoids
# executing or decoding embedded binary content. See THIRD_PARTY_NOTICES.md.
_TUKEY_TABLE_CACHE = None
_TUKEY_TABLE_FILENAME = "tukey_biweight_parameters.csv"
_TUKEY_TABLE_SHA256 = "8a5c6b494a700138bc7610e9afab97220d92251286c4bc20f663343c106d68bb"

def _tukey_table_path() -> Path:
    """Return the bundled, human-readable Tukey parameter table path."""
    return Path(__file__).resolve().parent / "data" / _TUKEY_TABLE_FILENAME

def _load_tracmt_tukey_table() -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate the BSD-3-Clause TRACMT Tukey parameter table.

    Security/transparency: this loader reads numeric text only with NumPy; it
    does not unpickle, execute, decompress, or dynamically import any content.
    A SHA-256 check detects accidental or unauthorized modification.
    """
    global _TUKEY_TABLE_CACHE
    if _TUKEY_TABLE_CACHE is None:
        import hashlib
        path = _tukey_table_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Tukey parameter table not found: {path}. "
                "Install/copy the data directory together with pyTRACMT."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _TUKEY_TABLE_SHA256:
            raise RuntimeError(
                "Tukey parameter table failed SHA-256 integrity check: "
                f"{digest} != {_TUKEY_TABLE_SHA256}"
            )
        table = np.loadtxt(path, delimiter=",", comments="#", skiprows=5, dtype=np.float64)
        if table.shape != (1001, 102):
            raise RuntimeError(f"Invalid Tukey table shape: {table.shape} != (1001, 102)")
        if not np.all(np.isfinite(table)):
            raise RuntimeError("Tukey parameter table contains non-finite values")
        cs = table[:, 0].copy()
        bs = table[:, 1:].T.copy()
        if np.any(np.diff(cs) <= 0):
            raise RuntimeError("Tukey c values must be strictly increasing")
        _TUKEY_TABLE_CACHE = (cs, bs)
    return _TUKEY_TABLE_CACHE

def median_abs_deviation_raw(x: np.ndarray) -> float:
    """Raw median absolute deviation, matching TRACMT Util::calculateMAD."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def madn(x: np.ndarray) -> float:
    """TRACMT Util::calculateMADN: MAD / 0.675."""
    return median_abs_deviation_raw(x) / 0.675


def tukey_params(dim: int, n: int) -> Tuple[float, float]:
    """Return TRACMT's exact finite-sample Tukey ``b`` and ``c`` parameters.

    This is a line-for-line translation of
    ``RobustWeightTukeysBiweights::calculateParams``.  TRACMT searches the
    precomputed table for the two adjacent c values bracketing
    ``(n-p)/(2n)`` and linearly interpolates both b and c.
    """
    dim = int(dim)
    n = int(n)
    if dim <= 0:
        raise ValueError("Tukey dimension must be positive")
    if dim > 101:
        raise ValueError("Tukey dimension exceeds TRACMT table limit (101)")
    if n <= dim:
        raise ValueError("Number of data must exceed Tukey dimension")

    cs, bs = _load_tracmt_tukey_table()
    row = bs[dim - 1]
    rhs = 0.5 * (1.0 - float(dim) / float(n))
    for ic in range(1, len(cs)):
        c0, c1 = float(cs[ic - 1]), float(cs[ic])
        b0, b1 = float(row[ic - 1]), float(row[ic])
        lhs0 = 1.0 if ic == 1 else 6.0 * b0 / (c0 * c0)
        lhs1 = 6.0 * b1 / (c1 * c1)
        diff0, diff1 = lhs0 - rhs, lhs1 - rhs
        if diff0 >= 0.0 and diff1 <= 0.0:
            den = abs(lhs1 - lhs0)
            weight = abs(diff0) / den if den > EPS else 0.0
            c = (1.0 - weight) * c0 + weight * c1
            b = (1.0 - weight) * b0 + weight * b1
            return max(float(b), EPS), float(c)
    raise RuntimeError("TRACMT Tukey parameters could not be determined")


def _rr_scale_factor(absr: np.ndarray) -> float:
    """TRACMT RobustWeight::calculateScaleByMADN: raw MAD / 0.448453."""
    scale = median_abs_deviation_raw(np.asarray(absr, dtype=float)) / 0.448453
    if not np.isfinite(scale) or scale <= EPS:
        scale = max(float(np.median(np.asarray(absr, dtype=float))), EPS)
    return float(scale)


def tukey_loss(val: np.ndarray, c: float) -> np.ndarray:
    v = np.asarray(val, dtype=float)
    av = np.abs(v)
    return np.where(av > c, c*c/6.0, v*v/2.0-v**4/(2*c*c)+v**6/(6*c**4))


def tukey_scale_update(absr: np.ndarray, scale_pre: float, b: float, c: float) -> float:
    """One exact TRACMT RobustWeightTukeysBiweights scale update."""
    r = np.asarray(absr, dtype=float)
    sp = max(float(scale_pre), EPS)
    val = r / sp
    terms = tukey_loss(val, c) * sp * sp
    near0 = np.abs(val) < EPS
    terms[near0] = 0.5 * r[near0] ** 2
    square_scale = float(np.mean(terms) / max(b, EPS))
    return math.sqrt(max(square_scale, EPS))

def _rr_weighted_power(y: np.ndarray, X: np.ndarray, z_row: np.ndarray, weights: np.ndarray) -> float:
    resid = y - X @ z_row
    w = np.asarray(weights, float)
    return float(np.sum(w * np.abs(resid) ** 2) / max(np.sum(w), EPS))


def _rr_solve_row(y: np.ndarray, X: np.ndarray, Rr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, float)
    P = (X.T * w) @ Rr.conj()
    q = (y * w) @ Rr.conj()
    try:
        # q @ inv(P) == solve(P.T, q.T).  Avoiding an explicit inverse follows
        # TRACMT/LAPACK more closely and reduces short-period conditioning error.
        return np.asarray(linalg.solve(P.T, q.T, assume_a="gen", check_finite=False)).ravel()
    except Exception:
        return np.asarray(q @ np.linalg.pinv(P, rcond=1e-14)).ravel()


def _rr_report_row(
    z_row: np.ndarray,
    out_idx: int,
    p: Params,
    coeffs_all,
    freq: float,
) -> np.ndarray:
    """Map whitened-domain row impedance to TRACMT-reported physical units."""
    if not coeffs_all or freq <= 0:
        return z_row
    Ztmp = np.zeros((p.num_out, p.num_input), dtype=complex)
    Ztmp[out_idx, :] = z_row
    return _dewhiten_local_impedance(Ztmp, p, coeffs_all, freq)[out_idx]


def _tracmt_freq_header(freq: float) -> str:
    period = 1.0 / freq
    return f"Now Frequency(Hz): {freq:.8g}, Period(s): {period:.6g}"


def _rr_log_response_block(
    out: OutputFiles,
    z_row_report: np.ndarray,
    z_row_whitened: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    out_idx: int,
    p: Params,
    ft: np.ndarray,
) -> None:
    wsum = float(np.sum(weights))
    Z = np.zeros((p.num_out, p.num_input), dtype=complex)
    if out_idx < Z.shape[0]:
        Z[out_idx, :] = z_row_whitened
    coh = _output_squared_coherence(ft, Z, p, out_idx, weights=weights)
    out.write_cvg(f"\tSum of weights: {wsum:g}")
    out.write_cvg(f"\tSquared coherence: {coh:.6g}")
    pairs = ", ".join(f"( {z.real:.4e}, {z.imag:.4e})" for z in z_row_report)
    out.write_cvg(f"\tEstimated response function: {pairs}")
    amps = ", ".join(f" {abs(z):.4e}" for z in z_row_report)
    out.write_cvg(f"\tAmplitude of the estimated response function:{amps}")
    phases = ", ".join(f" {math.degrees(math.atan2(z.imag, z.real)):5.1f}" for z in z_row_report)
    out.write_cvg(f"\tPhase(deg.) of the estimated response function:{phases}")


def _rr_estimator_name(mtype: int) -> str:
    if mtype == 0:
        return "Huber"
    if mtype == 1:
        return "Tukey's biweights"
    if mtype == 2:
        return "Thomson"
    return "none"


def rr_estimate(
    ft: np.ndarray,
    p: Params,
    out: OutputFiles,
    coeffs_all=None,
    freq: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """Ordinary remote reference following TRACMT's staged IRLS workflow."""
    no, ni = p.num_out, p.num_input
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + 2]
    resp = np.zeros((no, ni), dtype=complex)
    weights = np.ones(ft.shape[0], dtype=float)
    weights_by_out: List[np.ndarray] = []
    absr = np.ones(ft.shape[0], dtype=float)

    for out_idx in range(no):
        y = Y[:, out_idx]
        weights = np.ones(ft.shape[0], dtype=float)
        z_row = _rr_solve_row(y, X, Rr, weights)
        out.write_cvg("-" * 80)
        out.write_cvg(f"Calculate response functions for output variable {out_idx}")
        out.write_cvg("-" * 80)
        out.write_cvg("Calculate response functions by the ordinary least square method")
        _rr_log_response_block(out, _rr_report_row(z_row, out_idx, p, coeffs_all, freq),
                               z_row, y, X, weights, out_idx, p, ft)

        for mtype in p.m_estimators:
            if mtype < 0:
                continue
            est_name = _rr_estimator_name(mtype)
            out.write_cvg(f"Calculate response functions by iteratively reweighted remote reference using {est_name} weight")
            # C++ weightsPrior remain fixed during one M-estimator stage.
            weights_prior = np.asarray(weights, dtype=float).copy()
            resid = y - X @ z_row
            absr = np.abs(resid)
            scale = _rr_scale_factor(absr)
            if mtype == 1:
                b_tukey, c_tukey = tukey_params(2, len(y))
            wrp_prev = None

            for it in range(p.robust_iter_max):
                resid = y - X @ z_row
                absr = np.abs(resid)
                if mtype == 1:
                    scale = tukey_scale_update(absr, scale, b_tukey, c_tukey)
                    factor = tukey_weight(absr / max(scale, EPS), c_tukey)
                elif mtype == 0:
                    scale = _rr_scale_factor(absr)
                    factor = huber_weight(absr, scale, p.huber_threshold)
                else:
                    scale = _rr_scale_factor(absr)
                    factor = thomson_weight(absr, scale, len(absr), p.thomson_param)
                wnew = weights_prior * factor
                z_new = _rr_solve_row(y, X, Rr, wnew)
                wrp = _rr_weighted_power(y, X, z_new, wnew)

                out.write_cvg(f"Iteration number = {it}")
                out.write_cvg(f"\tScale factor: {scale:g}")
                if mtype == 1:
                    out.write_cvg(f"\tParameter c: {c_tukey:g}")
                _rr_log_response_block(out, _rr_report_row(z_new, out_idx, p, coeffs_all, freq),
                                       z_new, y, X, wnew, out_idx, p, ft)
                out.write_cvg(f"\tWeighted residual power: {wrp:g}")

                dz_rel = float(np.max(np.abs(z_new-z_row) / np.maximum(np.abs(z_row), EPS)))
                wrp_rel = (abs(wrp-wrp_prev)/max(abs(wrp_prev), EPS)) if wrp_prev is not None else np.inf
                z_row, weights = z_new, wnew
                if wrp_prev is not None and dz_rel < p.robust_conv and wrp_rel < p.robust_conv:
                    out.write_cvg(f"Iteration using {est_name} weight converged")
                    break
                wrp_prev = wrp
            else:
                out.write_cvg(f"Iteration using {est_name} weight reached max_iter={p.robust_iter_max}")

        resp[out_idx, :] = z_row
        weights_by_out.append(np.asarray(weights, float).copy())

    resid = Y - np.einsum("ni,ij->nj", X, resp.T)
    absr = np.sqrt(np.sum(np.abs(resid) ** 2, axis=1))
    return resp, weights, absr, np.var(resid, axis=0).real, weights_by_out

def tukey_weight(u: np.ndarray, c: float) -> np.ndarray:
    u = np.asarray(u, float)
    z = u / max(c, EPS)
    w = np.zeros_like(z)
    m = np.abs(z) < 1.0
    w[m] = (1.0 - z[m] ** 2) ** 2
    return w


def huber_weight(absres: np.ndarray, scale: float, threshold: float = 3.0) -> np.ndarray:
    x = np.asarray(absres, float) / max(scale, EPS)
    w = np.ones_like(x)
    m = x > threshold
    w[m] = threshold / np.maximum(x[m], EPS)
    return w


def thomson_weight(absres: np.ndarray, scale: float, nseg: int, param: float = 0.0) -> np.ndarray:
    if param < -EPS:
        probability = -param
    else:
        probability = (float(nseg) - param - 0.5) / max(float(nseg), 1.0)
    probability = min(max(probability, EPS), 1.0 - 1e-12)
    Q = math.sqrt(-2.0 * math.log(1.0 - probability))
    x = np.asarray(absres, float) / max(scale, EPS)
    return math.exp(math.exp(-Q*Q)) * np.exp(-np.exp(Q * (np.abs(x) - Q)))


def weighted_ls(Y: np.ndarray, R: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    # Y: n x q, R: n x 2, returns q x 2 resp satisfying Y ~= R @ resp.T
    if weights is None:
        w = np.ones(R.shape[0])
    else:
        w = np.asarray(weights, float)
    P = (R.conj().T * w) @ R
    Q = (Y.conj().T * w) @ R  # q x 2 using conjugate convention; convert below
    # We want sum Y * conj(R) * inv(sum R * conj(R)); use direct expression.
    Q = (Y.T * w) @ R.conj()
    P = (R.T * w) @ R.conj()
    try:
        resp = Q @ np.linalg.inv(P)
    except Exception:
        resp = Q @ np.linalg.pinv(P)
    return resp


def residuals_multivar(Y: np.ndarray, R: np.ndarray, resp: np.ndarray) -> np.ndarray:
    return Y - R @ resp.T


def md_diagonal(resid: np.ndarray, variances: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(resid) ** 2 / np.maximum(variances[None, :], EPS), axis=1))


def md_covariance(resid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    invc = np.linalg.pinv(cov)
    vals = np.einsum("ni,ij,nj->n", resid.conj(), invc, resid).real
    return np.sqrt(np.maximum(vals, 0.0))


def robust_scale_tukey(md: np.ndarray, scale0: Optional[float], b: float, c: float, itmax: int = 30) -> float:
    """Iterate the exact TRACMT Tukey scale equation for multivariate MD."""
    values = np.asarray(md, dtype=float)
    scale = float(scale0) if scale0 is not None and scale0 > EPS else _rr_scale_factor(values)
    for _ in range(itmax):
        new = tukey_scale_update(values, scale, b, c)
        if abs(new - scale) / max(abs(scale), EPS) < 0.01:
            return new
        scale = new
    return scale


def initial_resp_from_two_samples(Y: np.ndarray, R: np.ndarray, i: int, j: int) -> Optional[np.ndarray]:
    A = np.vstack([R[i], R[j]])
    if abs(np.linalg.det(A)) < 1e-20:
        return None
    try:
        return Y[[i, j], :].T @ np.linalg.inv(A).T
    except Exception:
        return None


def improve_rrms_candidate(Y: np.ndarray, R: np.ndarray, resp: np.ndarray, p: Params, cov_full: bool,
                           max_iter: int, convcrit: float, out: OutputFiles) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    q = Y.shape[1]
    b, c = tukey_params(q, len(Y))
    scale = None
    weights = np.ones(Y.shape[0])
    variances = np.ones(q)
    cov = np.eye(q)
    for it in range(max_iter):
        resid = residuals_multivar(Y, R, resp)
        if cov_full:
            cov = (resid.conj().T @ resid).real / max(len(resid), 1)
            cov += np.eye(q) * np.trace(cov) * 1e-10
            md = md_covariance(resid, cov)
        else:
            variances = np.mean(np.abs(resid)**2, axis=0).real + EPS
            md = md_diagonal(resid, variances)
        scale_new = robust_scale_tukey(md, scale, b, c)
        w = tukey_weight(md / max(scale_new, EPS), c)
        resp_new = weighted_ls(Y, R, w)
        conv = np.linalg.norm(resp_new - resp) / max(np.linalg.norm(resp_new), EPS)
        out.write_cvg(f"    I-step iter={it+1}, scale={scale_new:.6e}, conv={conv:.6e}, mean_w={w.mean():.6e}")
        resp, weights, scale = resp_new, w, scale_new
        if conv < convcrit:
            break
    resid = residuals_multivar(Y, R, resp)
    if cov_full:
        cov = (resid.conj().T * weights) @ resid / max(weights.sum(), EPS)
        cov = cov.real + np.eye(q)*EPS
        md = md_covariance(resid, cov)
        variances = np.diag(cov).real
    else:
        variances = ((np.abs(resid)**2).T @ weights / max(weights.sum(), EPS)).real + EPS
        cov = np.diag(variances)
        md = md_diagonal(resid, variances)
    return resp, weights, scale or 1.0, variances, md


def rrms_estimate(ft: np.ndarray, p: Params, out: OutputFiles, cov_full: bool = False) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    no, ni = p.num_out, p.num_input
    q = no + ni
    Y = ft[:, :q]              # Ex,Ey,Hx,Hy as dependent variables
    R = ft[:, q:q+2]           # RRx,RRy
    n = len(Y)
    if n < 3:
        resp = weighted_ls(Y, R)
        resid = residuals_multivar(Y, R, resp)
        return resp, np.ones(n), 1.0, np.var(resid, axis=0).real, np.ones(n)

    select_random, nmax, it1, conv1, nbest, it2, conv2 = p.rrms
    rng = MT19937_64(1234)
    pairs = set()
    if select_random:
        attempts = 0
        while len(pairs) < min(nmax, n*(n-1)//2) and attempts < nmax*20:
            i = rng.uniform_int(0, n - 1)
            j = rng.uniform_int(0, n - 1)
            if i == j:
                attempts += 1
                continue
            if i > j:
                i, j = j, i
            pairs.add((int(i), int(j)))
            attempts += 1
    else:
        for i in range(n):
            for j in range(i+1, n):
                pairs.add((i, j))
                if len(pairs) >= nmax: break
            if len(pairs) >= nmax: break
    out.write_cvg(f"RRMS candidate pairs: {len(pairs)}")

    Candidates = []
    for k, (i, j) in enumerate(pairs):
        r0 = initial_resp_from_two_samples(Y, R, i, j)
        if r0 is None:
            continue
        out.write_cvg(f"  candidate={k+1}, pair=({i},{j}) first I-step")
        r1, w1, s1, v1, md1 = improve_rrms_candidate(Y, R, r0, p, cov_full, it1, conv1, out)
        Candidates.append((s1, r1, w1, v1, md1, (i, j)))
    if not Candidates:
        r0 = weighted_ls(Y, R)
        Candidates.append((1.0, r0, np.ones(n), np.ones(q), np.ones(n), (-1, -1)))
    Candidates.sort(key=lambda x: x[0])
    bests = Candidates[:max(1, min(nbest, len(Candidates)))]
    final = []
    for k, (_, r, _, _, _, pair) in enumerate(bests):
        out.write_cvg(f"  second I-step candidate={k+1}, pair={pair}")
        final.append((*improve_rrms_candidate(Y, R, r, p, cov_full, it2, conv2, out), pair))
    final.sort(key=lambda x: x[2])  # scale
    resp, weights, scale, variances, md, pair = final[0]
    out.write_cvg(f"RRMS selected pair={pair}, scale={scale:.6e}, kept_weight_sum={weights.sum():.3f}")
    return resp, weights, scale, variances, md


def _ordinary_rr_group_coherence(block: np.ndarray, p: Params) -> float:
    """Exact coherence definition used by TRACMT's selection helper.

    TRACMT fits ordinary RR responses in the group and evaluates
    1 - residual_power / output_power, clipped to [0, 1]. The C++ helper
    overwrites the value for each output and therefore returns the final output
    variable's coherence; this behavior is intentionally reproduced.
    """
    no, ni = p.num_out, p.num_input
    X = block[:, no:no + ni]
    Rr = block[:, no + ni:no + ni + 2]
    coherence = 0.0
    weights = np.ones(block.shape[0], dtype=float)
    for out_idx in range(no):
        y = block[:, out_idx]
        z = _rr_solve_row(y, X, Rr, weights)
        syn = X @ z
        gvar = float(np.sum(np.abs(y) ** 2))
        gres = float(np.sum(np.abs(y - syn) ** 2))
        coherence = 1.0 - gres / max(gvar, EPS)
        coherence = min(1.0, max(0.0, coherence))
    return float(coherence)


def coherence_mask(ft: np.ndarray, p: Params, out: OutputFiles) -> np.ndarray:
    n = ft.shape[0]
    mask = np.ones(n, dtype=bool)
    if not p.coherence_criteria:
        return mask
    group_n, threshold = p.coherence_criteria
    if group_n < 2 or n < group_n:
        return mask
    out.write_cvg(f"Apply COHERENCE_CRITERIA: nseg={group_n}, threshold={threshold}")
    rejected = 0
    # C++ processes consecutive complete groups only; a trailing incomplete
    # group is left untouched.
    for st in range(0, n - group_n + 1, group_n):
        ed = st + group_n
        coherence = _ordinary_rr_group_coherence(ft[st:ed, :], p)
        out.write_cvg(f"  segments={st}:{ed-1}, squared coherence={coherence:.8g}")
        if coherence < threshold:
            mask[st:ed] = False
            rejected += group_n
    out.write_cvg(f"  coherence rejected segments={rejected}/{n}")
    return mask

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def impedance_from_rrms_istf(resp: np.ndarray, p: Params) -> np.ndarray:
    no = p.num_out
    U = resp[:no, :]
    V = resp[no:no+2, :]
    try:
        return U @ np.linalg.inv(V)
    except Exception:
        return U @ np.linalg.pinv(V)


def rhoa_phase(z: complex, freq: float) -> Tuple[float, float]:
    """TRACMT/Usui-compatible apparent resistivity and phase.

    TRACMT output response units are E/H in (mV/km)/nT.  For these units,

        rho_a[ohm m] = 0.2 * T[s] * |Z|^2

    where T=1/f.  The previous Python version used the SI formula
    |Z|^2/(mu0*omega) directly and overestimated rho_a by about 6.33e5.
    """
    if freq <= 0 or not np.isfinite(freq):
        return np.nan, np.nan
    rho = 0.2 * (1.0 / freq) * (abs(z) ** 2)
    ph = math.degrees(math.atan2(z.imag, z.real))
    return rho, ph


TRACMT_CSV_FLOAT_FORMAT = "%.10e"
TRACMT_Z_COMPONENTS = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
Z_COMPONENT_LABELS = {(0, 0): "Zxx", (0, 1): "Zxy", (1, 0): "Zyx", (1, 1): "Zyy"}
OUTPUT_FIELD_LABELS = ("Ex", "Ey", "Ez", "E3", "E4")


def _z_component_label(out_idx: int, inp_idx: int) -> str:
    """Human-readable MT impedance name, e.g. Zxy."""
    if (out_idx, inp_idx) in Z_COMPONENT_LABELS:
        return Z_COMPONENT_LABELS[(out_idx, inp_idx)]
    return f"Z_{out_idx}_{inp_idx}"


def _output_field_label(out_idx: int) -> str:
    """E-field output channel label for coherence columns."""
    if 0 <= out_idx < len(OUTPUT_FIELD_LABELS):
        return OUTPUT_FIELD_LABELS[out_idx]
    return f"E{out_idx}"


def _coherence_column_name(out_idx: int) -> str:
    return f"coherence_{_output_field_label(out_idx)}"


def _tracmt_input_channel_indices(p: Params) -> List[int]:
    return list(range(p.num_out, p.num_out + p.num_input))


def _tracmt_coherence_key(p: Params, out_idx: int) -> str:
    """Legacy TRACMT index-style coherence key (kept for internal references)."""
    ij_pair = "+".join(str(x) for x in _tracmt_input_channel_indices(p))
    return f"coherence_{out_idx}_{ij_pair}"


def _csv_response_column_names(p: Params) -> List[str]:
    cols = ["frequency", "period"]
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            cols += [f"{lab}_real", f"{lab}_imag"]
        cols.append(_coherence_column_name(oi))
    for oi in range(p.num_out):
        for j in range(p.num_input):
            cols.append(f"d{_z_component_label(oi, j)}")
    return cols


def _tracmt_response_column_names(p: Params) -> List[str]:
    return _csv_response_column_names(p)


def _csv_rho_column_names(p: Params) -> List[str]:
    cols = ["frequency", "period"]
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            cols += [f"rho_{lab}", f"phase_{lab}"]
        cols.append(_coherence_column_name(oi))
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            cols += [f"drho_{lab}", f"dphase_{lab}"]
    return cols


def _tracmt_rho_column_names(p: Params) -> List[str]:
    return _csv_rho_column_names(p)


def _output_squared_coherence(
    ft: np.ndarray,
    Z: np.ndarray,
    p: Params,
    out_idx: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Multiple squared coherence gamma^2 between E and Z-predicted E.

    TRACMT accumulates cross-spectra with the final IRLS segment weights w_i:

        S_xy = sum(w_i y_i conj(yhat_i)) / sum(w_i)
        S_xx = sum(w_i |y_i|^2) / sum(w_i)
        S_yy = sum(w_i |yhat_i|^2) / sum(w_i)
        gamma^2 = |S_xy|^2 / (S_xx S_yy)

    Z and ft must both be in the same domain (whitened if prewhitening is on).
    """
    if out_idx >= Z.shape[0] or Z.shape[1] < p.num_input or len(ft) < 2:
        return np.nan
    hx_idx = p.num_out
    hy_idx = p.num_out + 1
    if hy_idx >= ft.shape[1]:
        return np.nan
    y = ft[:, out_idx]
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, float)
    if len(w) != len(y):
        w = np.ones(len(y), dtype=float)
    sw = float(np.sum(w))
    if sw <= EPS:
        return np.nan
    y_hat = np.zeros_like(y, dtype=complex)
    for j in range(min(p.num_input, Z.shape[1])):
        col = hx_idx + j
        if col < ft.shape[1]:
            y_hat += Z[out_idx, j] * ft[:, col]
    sxy = np.sum(w * y * np.conj(y_hat)) / sw
    sxx = np.sum(w * np.abs(y) ** 2) / sw
    syy = np.sum(w * np.abs(y_hat) ** 2) / sw
    den = sxx * syy + EPS
    return float((abs(sxy) ** 2) / den)


def _build_tracmt_response_row(
    freq: float,
    period: float,
    ft: np.ndarray,
    Z: np.ndarray,
    err_bars: Dict[str, Tuple[float, float, float]],
    p: Params,
    Zw: Optional[np.ndarray] = None,
    weights_by_out: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    row: Dict[str, float] = {"frequency": freq, "period": period}
    zkey = {(oi, j): name for oi, j, name in TRACMT_Z_COMPONENTS}
    Zcoh = Z if Zw is None else Zw
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            if oi < Z.shape[0] and j < Z.shape[1]:
                row[f"{lab}_real"] = Z[oi, j].real
                row[f"{lab}_imag"] = Z[oi, j].imag
            else:
                row[f"{lab}_real"] = np.nan
                row[f"{lab}_imag"] = np.nan
        w_out = None
        if weights_by_out and oi < len(weights_by_out):
            w_out = weights_by_out[oi]
        row[_coherence_column_name(oi)] = _output_squared_coherence(ft, Zcoh, p, oi, weights=w_out)
    for oi in range(p.num_out):
        for j in range(p.num_input):
            dz, _, _ = err_bars.get(zkey.get((oi, j), ""), (np.nan, np.nan, np.nan))
            row[f"d{_z_component_label(oi, j)}"] = dz
    return row


def _build_tracmt_rho_row(
    freq: float,
    period: float,
    ft: np.ndarray,
    Z: np.ndarray,
    err_bars: Dict[str, Tuple[float, float, float]],
    p: Params,
    Zw: Optional[np.ndarray] = None,
    weights_by_out: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    row: Dict[str, float] = {"frequency": freq, "period": period}
    zkey = {(oi, j): name for oi, j, name in TRACMT_Z_COMPONENTS}
    Zcoh = Z if Zw is None else Zw
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            if oi < Z.shape[0] and j < Z.shape[1]:
                rho, ph = rhoa_phase(Z[oi, j], freq)
            else:
                rho, ph = np.nan, np.nan
            row[f"rho_{lab}"] = rho
            row[f"phase_{lab}"] = ph
        w_out = None
        if weights_by_out and oi < len(weights_by_out):
            w_out = weights_by_out[oi]
        row[_coherence_column_name(oi)] = _output_squared_coherence(ft, Zcoh, p, oi, weights=w_out)
    for oi in range(p.num_out):
        for j in range(p.num_input):
            lab = _z_component_label(oi, j)
            _, drho, dph = err_bars.get(zkey.get((oi, j), ""), (np.nan, np.nan, np.nan))
            row[f"drho_{lab}"] = drho
            row[f"dphase_{lab}"] = dph
    return row


def _build_engine_response_row(
    freq: float,
    period: float,
    istf: np.ndarray,
    Z: np.ndarray,
    err_bars: Dict[str, Tuple[float, float, float]],
    scale: float,
    weights: np.ndarray,
    n_segments: int,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "freq": freq,
        "period": period,
        "n_segments": n_segments,
        "scale": scale,
        "mean_weight": float(np.mean(weights)),
    }
    for i in range(istf.shape[0]):
        for j in range(istf.shape[1]):
            row[f"T{i+1}{j+1}_real"] = istf[i, j].real
            row[f"T{i+1}{j+1}_imag"] = istf[i, j].imag
    for i, j, name in TRACMT_Z_COMPONENTS:
        if i < Z.shape[0] and j < Z.shape[1]:
            lab = _z_component_label(i, j)
            row[f"{lab}_real"] = Z[i, j].real
            row[f"{lab}_imag"] = Z[i, j].imag
            dz_abs, _, _ = err_bars.get(name, (np.nan, np.nan, np.nan))
            row[f"d{lab}"] = dz_abs
            row[f"{name}_real"] = Z[i, j].real
            row[f"{name}_imag"] = Z[i, j].imag
            row[f"{name}_err_abs"] = dz_abs
    return row


def _write_tracmt_csv(path: Path, rows: List[Dict[str, float]], columns: List[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False, float_format=TRACMT_CSV_FLOAT_FORMAT, encoding="utf-8-sig",
    )


def _prewhite_response(coeff: np.ndarray, freq: float, fs: float) -> complex:
    """Frequency response of the AR prewhitening filter y' = A(f) y."""
    try:
        a = np.asarray(coeff, dtype=float)
        if a.size == 0:
            return 1.0 + 0j
        k = np.arange(1, a.size + 1, dtype=float)
        return complex(1.0 - np.sum(a * np.exp(-2j * math.pi * freq * k / fs)))
    except Exception:
        return 1.0 + 0j


def _prewhite_channel_responses(coeffs_all, freq: float, fs: float, nchan: int) -> np.ndarray:
    """Return A_c(f) for each channel. Uses the first section by default.

    In TRACMT-style prewhitening, each channel may be filtered by a different
    AR filter.  Transfer functions estimated from whitened data must be
    dewhitened.  With channels ordered Ex,Ey,Hx,Hy,RRx,RRy, the local impedance
    correction is Z_ij = Z'_ij * A_Hj / A_Ei.
    """
    A = np.ones(nchan, dtype=complex)
    try:
        if not coeffs_all:
            return A
        coeffs_sec = coeffs_all[0]
        for c in range(min(nchan, len(coeffs_sec))):
            A[c] = _prewhite_response(coeffs_sec[c], freq, fs)
    except Exception:
        pass
    return A


def _dewhiten_local_impedance(Zw: np.ndarray, p: Params, coeffs_all, freq: float) -> np.ndarray:
    """Undo per-channel prewhitening for local impedance matrix."""
    if not coeffs_all:
        return Zw
    Z = np.array(Zw, dtype=complex, copy=True)
    A = _prewhite_channel_responses(coeffs_all, freq, p.sampling_freq, p.nchan)
    no = p.num_out
    ni = min(2, p.num_input)
    for i in range(min(no, Z.shape[0])):
        Ae = A[i] if abs(A[i]) > EPS else 1.0 + 0j
        for j in range(min(ni, Z.shape[1])):
            Ah = A[no + j] if abs(A[no + j]) > EPS else 1.0 + 0j
            Z[i, j] = Z[i, j] * Ah / Ae
    return Z


class _SilentOutput:
    def write_cvg(self, msg: str):
        pass
    def write_log(self, msg: str, elapsed: bool = True):
        pass
    def write_both(self, msg: str, elapsed: bool = True):
        pass
    def warning(self, msg: str):
        pass


def _rr_uses_robust(p: Params) -> bool:
    return any(int(m) >= 0 for m in p.m_estimators)


def _rr_dewhiten_scale(p: Params, coeffs_all, freq: float, out_idx: int, inp_idx: int) -> float:
    """|A_H / A_E| scaling from whitened to physical impedance uncertainty."""
    if not coeffs_all or freq <= 0:
        return 1.0
    A = _prewhite_channel_responses(coeffs_all, freq, p.sampling_freq, p.nchan)
    no = p.num_out
    Ae = abs(A[out_idx]) if out_idx < len(A) and abs(A[out_idx]) > EPS else 1.0
    ch = no + inp_idx
    Ah = abs(A[ch]) if ch < len(A) and abs(A[ch]) > EPS else 1.0
    return Ah / Ae


def _rr_solve_all_fixed_weights(ft: np.ndarray, p: Params, weights: np.ndarray) -> np.ndarray:
    """Solve all RR output rows once with fixed segment weights (whitened domain)."""
    no, ni = p.num_out, p.num_input
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + 2]
    w = np.asarray(weights, float)
    resp = np.zeros((no, ni), dtype=complex)
    for out_idx in range(no):
        resp[out_idx] = _rr_solve_row(Y[:, out_idx], X, Rr, w)
    return resp


def _estimate_Z_row_fixed_weights(
    ft: np.ndarray,
    p: Params,
    out_idx: int,
    weights: np.ndarray,
    coeffs_all=None,
    freq: float = 0.0,
) -> np.ndarray:
    """Re-estimate one RR output row with fixed segment weights (physical units)."""
    if len(ft) < 3 or out_idx >= p.num_out:
        return np.full(p.num_input, np.nan + 1j * np.nan)
    no, ni = p.num_out, p.num_input
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + 2]
    z_row = _rr_solve_row(Y[:, out_idx], X, Rr, weights)
    Z = np.full((no, ni), np.nan + 1j * np.nan, dtype=complex)
    Z[out_idx, :] = z_row
    return _dewhiten_local_impedance(Z, p, coeffs_all, freq)[out_idx]


def _estimate_Z_fixed_weights(
    ft: np.ndarray,
    p: Params,
    weights: np.ndarray,
    coeffs_all=None,
    freq: float = 0.0,
) -> np.ndarray:
    """Re-estimate Z with fixed robust weights — TRACMT fixed-weights bootstrap step."""
    if len(ft) < 3:
        return np.full((p.num_out, p.num_input), np.nan + 1j * np.nan)
    z = _rr_solve_all_fixed_weights(ft, p, weights)
    return _dewhiten_local_impedance(z, p, coeffs_all, freq)


def _estimate_Z_quiet(ft: np.ndarray, p: Params, coeffs_all=None, freq: float = 0.0) -> np.ndarray:
    """Re-estimate Z from one frequency block without writing cvg/log output.

    Used only for robust-bootstrap uncertainty (mode 3), where the full
    RR/RRMS estimator is re-run on each resample.
    """
    if len(ft) < 3:
        return np.full((2, 2), np.nan + 1j*np.nan)
    qout = _SilentOutput()
    if p.procedure == 0:
        z, _weights, _absr, _var, _weights_by_out = rr_estimate(ft, p, qout, coeffs_all=coeffs_all, freq=freq)
        return _dewhiten_local_impedance(z, p, coeffs_all, freq)
    istf, _weights, _scale, _var, _md = rrms_estimate(ft, p, qout, cov_full=(p.procedure == 2))
    return _dewhiten_local_impedance(impedance_from_rrms_istf(istf, p), p, coeffs_all, freq)


def _unwrap_phase_samples(phases_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Unwrap phase samples around the reported phase, in degrees."""
    ph = np.asarray(phases_deg, dtype=float)
    return center_deg + ((ph - center_deg + 180.0) % 360.0 - 180.0)


def _tracmt_complex_bootstrap_std(vals: np.ndarray) -> float:
    """TRACMT C++ complex-response bootstrap standard error.

    The original code centers each complex bootstrap chain on its bootstrap
    mean and divides the summed squared complex deviations by ``2*B - 4``.
    This treats each complex response as two real quantities and removes four
    real degrees of freedom for the two complex RR coefficients in one output
    row.
    """
    vals = np.asarray(vals, dtype=complex)
    vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
    b = int(vals.size)
    if b <= 2:
        return np.nan
    mean_z = np.mean(vals)
    denom = float(2 * b - 4)
    if denom <= 0.0:
        return np.nan
    variance = float(np.sum(np.abs(vals - mean_z) ** 2) / denom)
    return math.sqrt(max(variance, 0.0))


def _tracmt_propagate_impedance_error(z: complex, freq: float, dz: float) -> Tuple[float, float]:
    """C++ TRACMT propagation from absolute impedance error to rho/phase.

    d(rho_a) = 0.4 |Z| dZ / f
    d(phi)   = asin(dZ/|Z|) in degrees; TRACMT writes 360 degrees when
               dZ/|Z| is not smaller than one.
    """
    if freq <= 0.0 or not np.isfinite(freq) or not np.isfinite(dz):
        return np.nan, np.nan
    amp = abs(z)
    d_rho = 0.4 * amp * dz / freq
    ratio = dz / max(amp, EPS)
    d_phase = math.degrees(math.asin(ratio)) if ratio < 1.0 else 360.0
    return float(d_rho), float(d_phase)


def _complex_component_std(vals: np.ndarray, z0: complex) -> float:
    """Backward-compatible alias using the exact TRACMT bootstrap definition."""
    return _tracmt_complex_bootstrap_std(vals)


def _rho_phase_std(vals: np.ndarray, freq: float, center_phase_deg: float) -> Tuple[float, float]:
    """Legacy helper retained for non-TRACMT experimental resampling modes."""
    rho_samples = np.array([rhoa_phase(v, freq)[0] for v in vals], dtype=float)
    ph_samples = _unwrap_phase_samples(
        np.array([rhoa_phase(v, freq)[1] for v in vals], dtype=float),
        center_phase_deg,
    )
    d_rho = float(np.nanstd(rho_samples, ddof=1)) if len(rho_samples) >= 2 else np.nan
    d_ph = float(np.nanstd(ph_samples, ddof=1)) if len(ph_samples) >= 2 else np.nan
    return d_rho, d_ph

def _parametric_rr_errors(
    ft: np.ndarray,
    p: Params,
    Z: np.ndarray,
    freq: float,
    coeffs_all=None,
    weights_by_out: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """TRACMT parametric 1-sigma errors from weighted RR covariance."""
    labels = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
    empty = {name: (np.nan, np.nan, np.nan) for _i, _j, name in labels}
    no, ni = p.num_out, p.num_input
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + 2]
    zkey = {(oi, j): name for oi, j, name in labels}
    out = dict(empty)
    for out_idx in range(no):
        if weights_by_out and out_idx < len(weights_by_out):
            w = np.asarray(weights_by_out[out_idx], float)
        elif weights is not None:
            w = np.asarray(weights, float)
        else:
            w = np.ones(len(ft), dtype=float)
        y = Y[:, out_idx]
        z_row = _rr_solve_row(y, X, Rr, w)
        resid = y - X @ z_row
        wsum = float(np.sum(w))
        dof = max(wsum - float(ni), 1.0)
        sigma2 = float(np.sum(w * np.abs(resid) ** 2) / dof)
        P = (X.T * w) @ Rr.conj()
        try:
            Pinv = np.linalg.inv(P)
        except Exception:
            Pinv = np.linalg.pinv(P)
        for j in range(min(ni, Z.shape[1])):
            name = zkey.get((out_idx, j))
            if name is None:
                continue
            z0 = Z[out_idx, j]
            rho0, ph0 = rhoa_phase(z0, freq)
            scale = _rr_dewhiten_scale(p, coeffs_all, freq, out_idx, j)
            se2 = max(float(np.real(Pinv[j, j]) * sigma2), 0.0)
            dZ = math.sqrt(se2) * scale
            if not np.isfinite(dZ) or dZ <= 0:
                dZ = abs(z0) / math.sqrt(max(wsum, 1.0))
            d_rho, d_ph = _tracmt_propagate_impedance_error(z0, freq, dZ)
            out[name] = (float(dZ), float(d_rho), float(d_ph))
    return out


def _bootstrap_rr_errors(
    ft: np.ndarray,
    p: Params,
    Z: np.ndarray,
    freq: float,
    coeffs_all,
    weights_by_out: Sequence[np.ndarray],
    mode: int,
    rng: np.random.Generator,
    user_nrep: int,
) -> Dict[str, Tuple[float, float, float]]:
    """TRACMT-compatible fixed-weight bootstrap or jackknife per output row.

    For ERROR_ESTIMATION=1 this follows AnalysisOrdinaryRemoteReference.cpp:
    robust weights are held fixed, segments are sampled with replacement,
    each output row is solved independently, the chain is centered on its own
    bootstrap mean, and complex variance uses denominator ``2*B-4``.  Rhoa and
    phase errors are then propagated analytically from dZ exactly as TRACMT.
    """
    labels = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
    empty = {name: (np.nan, np.nan, np.nan) for _i, _j, name in labels}
    n = int(len(ft))
    no = p.num_out
    zkey = {(oi, j): name for oi, j, name in labels}
    result = dict(empty)

    if mode == 1:
        nrep = int(user_nrep)
        if nrep <= 2:
            return empty
    elif mode == 2:
        nrep = 0
    else:
        return empty

    for out_idx in range(no):
        if out_idx >= len(weights_by_out):
            continue
        w = np.asarray(weights_by_out[out_idx], dtype=float)
        if w.size != n:
            continue
        row_samples: List[np.ndarray] = []

        if mode == 1:
            # C++ restarts mt19937_64 with seed 1234 for every output row.
            # NumPy's MT19937 is not bit-identical to std::mt19937_64, but
            # reseeding here reproduces the same deterministic workflow.
            row_rng = MT19937_64(1234)
            for _ in range(nrep):
                idx = row_rng.indexes(n, n)
                try:
                    row_samples.append(
                        _estimate_Z_row_fixed_weights(ft[idx], p, out_idx, w[idx], coeffs_all, freq)
                    )
                except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                    continue
        else:  # jackknife is retained as an optional pyTRACMT extension
            if n <= 40:
                groups = [np.array([ii], dtype=int) for ii in range(n)]
            else:
                ngroups = min(20, max(8, int(np.sqrt(n))))
                order = np.arange(n)
                rng.shuffle(order)
                groups = np.array_split(order, ngroups)
            all_idx = np.arange(n)
            for group in groups:
                keep = np.setdiff1d(all_idx, group, assume_unique=False)
                try:
                    row_samples.append(
                        _estimate_Z_row_fixed_weights(ft[keep], p, out_idx, w[keep], coeffs_all, freq)
                    )
                except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                    continue

        if len(row_samples) < 3:
            continue
        arr = np.asarray(row_samples, dtype=complex)
        for j in range(min(p.num_input, Z.shape[1])):
            name = zkey.get((out_idx, j))
            if name is None or out_idx >= Z.shape[0]:
                continue
            vals = arr[:, j]
            vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
            if len(vals) < 3:
                continue
            z0 = Z[out_idx, j]
            if mode == 1:
                d_z = _tracmt_complex_bootstrap_std(vals)
                d_rho, d_phase = _tracmt_propagate_impedance_error(z0, freq, d_z)
            else:
                m = len(vals)
                mean_z = np.mean(vals)
                d_z = math.sqrt(max(float((m - 1) / m * np.sum(np.abs(vals - mean_z) ** 2)), 0.0))
                d_rho, d_phase = _tracmt_propagate_impedance_error(z0, freq, d_z)
            result[name] = (float(d_z), float(d_rho), float(d_phase))
    return result

def _resampled_rr_errors(
    samples: List[np.ndarray],
    Z: np.ndarray,
    freq: float,
    mode: int,
) -> Dict[str, Tuple[float, float, float]]:
    """Convert replicate impedance matrices into 1-sigma error bars."""
    labels = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
    empty = {name: (np.nan, np.nan, np.nan) for _i, _j, name in labels}
    if len(samples) < 3:
        return empty
    arr = np.asarray(samples, dtype=complex)
    out = dict(empty)
    for i, j, name in labels:
        if i >= Z.shape[0] or j >= Z.shape[1]:
            continue
        vals = arr[:, i, j]
        vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
        if len(vals) < 3:
            continue
        z0 = Z[i, j]
        rho0, ph0 = rhoa_phase(z0, freq)
        if mode == 2:
            m = len(vals)
            mean_z = np.nanmean(vals)
            dZ = math.sqrt(max(float((m - 1) / m * np.nansum(np.abs(vals - mean_z) ** 2)), 0.0))
            rho_samples = np.array([rhoa_phase(v, freq)[0] for v in vals], dtype=float)
            ph_samples = _unwrap_phase_samples(
                np.array([rhoa_phase(v, freq)[1] for v in vals], dtype=float),
                ph0,
            )
            d_rho = math.sqrt(max(float((m - 1) / m * np.nansum((rho_samples - np.nanmean(rho_samples)) ** 2)), 0.0))
            d_ph = math.sqrt(max(float((m - 1) / m * np.nansum((ph_samples - np.nanmean(ph_samples)) ** 2)), 0.0))
        else:
            dZ = _complex_component_std(vals, z0)
            if mode in (3, 4):
                d_rho, d_ph = _tracmt_propagate_impedance_error(z0, freq, dZ)
            else:
                d_rho, d_ph = _rho_phase_std(vals, freq, ph0)
        if not np.isfinite(dZ) or dZ <= 0:
            dZ = abs(z0) / math.sqrt(max(len(vals), 1.0))
        if not np.isfinite(d_rho) or d_rho <= 0:
            d_rho = abs(rho0) * 2.0 / math.sqrt(max(len(vals), 1.0))
        if not np.isfinite(d_ph) or d_ph <= 0:
            d_ph = math.degrees(1.0 / math.sqrt(max(len(vals), 1.0)))
        out[name] = (float(dZ), float(d_rho), float(d_ph))
    return out


def _estimate_error_bars(
    ft: np.ndarray,
    p: Params,
    Z: np.ndarray,
    freq: float,
    coeffs_all=None,
    weights: Optional[np.ndarray] = None,
    weights_by_out: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Estimate 1-sigma uncertainty for each impedance component.

    TRACMT-compatible modes (see TRACMT.log / TRACMT.cvg):
      0 = parametric covariance (NonRobustRemoteReference)
      1 = fixed-weights bootstrap after robust RR (OrdinaryRobustRemoteReference)
      2 = jackknife with fixed weights
      3 = robust bootstrap (re-run full RR/RRMS on each resample; RRMS default)
      4 = strict bootstrap (re-run the complete selected estimator on each resample)

    Returns {component: (dZ_abs, d_rhoa, d_phase_deg)}.
    """
    labels = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
    empty = {name: (np.nan, np.nan, np.nan) for _i, _j, name in labels}
    n = int(len(ft))
    mode = int(getattr(p, "error_estimation", 0) or 0)
    if mode < 0:
        return empty
    if n < 4:
        return empty

    w_fallback = np.ones(n, dtype=float) if weights is None else np.asarray(weights, float)
    if len(w_fallback) != n:
        w_fallback = np.ones(n, dtype=float)

    if mode == 0:
        if p.procedure == 0:
            return _parametric_rr_errors(
                ft, p, Z, freq, coeffs_all, weights_by_out=weights_by_out, weights=w_fallback
            )
        return empty

    rng = np.random.default_rng(20240604 + int(round(freq * 1e9)) % 1000003)
    user_nrep = int(getattr(p, "bootstrap", 200) or 200)

    if p.procedure == 0 and mode in (1, 2) and weights_by_out:
        result = _bootstrap_rr_errors(
            ft, p, Z, freq, coeffs_all, weights_by_out, mode, rng, user_nrep
        )
        if any(np.isfinite(result.get(name, (np.nan,))[0]) for _i, _j, name in labels):
            return result
        return _parametric_rr_errors(
            ft, p, Z, freq, coeffs_all, weights_by_out=weights_by_out, weights=w_fallback
        )

    # RRMS / robust-bootstrap fallback: resample full block.
    samples: List[np.ndarray] = []
    if mode == 1:
        nrep = max(50, min(user_nrep, 1000))
        for _ in range(nrep):
            idx = rng.integers(0, n, size=n)
            try:
                if p.procedure == 0:
                    samples.append(_estimate_Z_fixed_weights(ft[idx], p, w_fallback[idx], coeffs_all, freq))
                else:
                    samples.append(_estimate_Z_quiet(ft[idx], p, coeffs_all, freq))
            except Exception:
                pass
    elif mode == 2:
        if n <= 40:
            for ii in range(n):
                idx = np.ones(n, dtype=bool)
                idx[ii] = False
                try:
                    if p.procedure == 0:
                        samples.append(_estimate_Z_fixed_weights(ft[idx], p, w_fallback[idx], coeffs_all, freq))
                    else:
                        samples.append(_estimate_Z_quiet(ft[idx], p, coeffs_all, freq))
                except Exception:
                    pass
        else:
            ngroups = min(20, max(8, int(np.sqrt(n))))
            order = np.arange(n)
            rng.shuffle(order)
            groups = np.array_split(order, ngroups)
            all_idx = np.arange(n)
            for g in groups:
                keep = np.setdiff1d(all_idx, g, assume_unique=False)
                try:
                    if p.procedure == 0:
                        samples.append(_estimate_Z_fixed_weights(ft[keep], p, w_fallback[keep], coeffs_all, freq))
                    else:
                        samples.append(_estimate_Z_quiet(ft[keep], p, coeffs_all, freq))
                except Exception:
                    pass
    elif mode in (3, 4):
        # V8: TRACMT-style resampling uses one std::mt19937_64 stream seeded
        # with 1234. Strict bootstrap (4) repeats the complete selected
        # estimator for every resampled Fourier block. Robust bootstrap (3)
        # uses the same full refit in this Python implementation, avoiding the
        # V7 fixed-weight shortcut at difficult/high-frequency bands.
        nrep = max(4, min(user_nrep, 30 if int(getattr(p, "procedure", 0)) != 0 else 200))
        cpp_rng = MT19937_64(1234)
        for _ in range(nrep):
            idx = cpp_rng.indexes(n, n)
            try:
                samples.append(_estimate_Z_quiet(ft[idx], p, coeffs_all, freq))
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                pass
    else:
        return empty

    if len(samples) < 3:
        if p.procedure == 0:
            return _parametric_rr_errors(
                ft, p, Z, freq, coeffs_all, weights_by_out=weights_by_out, weights=w_fallback
            )
        rel = 1.0 / math.sqrt(max(n, 1))
        out = dict(empty)
        for i, j, name in labels:
            if i < Z.shape[0] and j < Z.shape[1] and np.isfinite(Z[i, j].real):
                z0 = Z[i, j]
                rho0, _ph0 = rhoa_phase(z0, freq)
                out[name] = (abs(z0) * rel, abs(rho0) * 2.0 * rel, math.degrees(rel))
        return out

    return _resampled_rr_errors(samples, Z, freq, mode)

def write_edi(path: Path, rows: List[dict], site: str = "pyTRACMT"):
    with path.open("w", encoding="utf-8") as f:
        f.write(">HEAD\n")
        f.write(f"  DATAID={site}\n")
        f.write("  ACQBY=pyTRACMT_v8\n")
        f.write(">INFO\n")
        f.write("  Generated by pyTRACMT Professional V8 Bit-Compatible Edition\n")
        f.write(">=DEFINEMEAS\n")
        f.write(">=MTSECT\n")
        f.write(f"  NFREQ={len(rows)}\n")
        f.write(">FREQ\n")
        f.write(" ".join(f"{r['freq']:.8e}" for r in rows) + "\n")
        for comp in ["ZXX", "ZXY", "ZYX", "ZYY"]:
            f.write(f">{comp}R ROT=0\n")
            f.write(" ".join(f"{r.get(comp.lower(), np.nan).real:.8e}" for r in rows) + "\n")
            f.write(f">{comp}I ROT=0\n")
            f.write(" ".join(f"{r.get(comp.lower(), np.nan).imag:.8e}" for r in rows) + "\n")
        f.write(">END\n")


def process(param_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    param_path = Path(param_path)
    p = parse_param(param_path)
    outdir = Path(output_dir) if output_dir else param_path.parent
    out = OutputFiles(outdir)
    try:
        out.write_log("Read parameters.")
        out.write_log(f"Processing Method type: {p.procedure} (0=RR, 1=RRMS)", elapsed=False)
        out.write_log(f"Number of channels: {p.nchan}", elapsed=False)
        out.write_log(f"Error Estimation mode: {p.error_estimation} (0=Parametric, 1=Fixed-weights bootstrap, 2=Jackknife, 3=Robust bootstrap, 4=Strict bootstrap); bootstrap={p.bootstrap}", elapsed=False)
        sections = load_data(param_path, p, out)
        sections = [apply_filters(sec, p.sampling_freq, p) for sec in sections]
        sections = [apply_robust_filter(sec, p, out) for sec in sections]
        sections, coeffs = apply_prewhitening(sections, p, out)
        cals = load_calibrations(param_path, p, out)
        blocks = collect_frequency_blocks(sections, p)
        blocks.sort(key=lambda x: x[2])

        resp_rows = []
        rho_rows = []
        engine_rows = []
        edi_rows = []
        for L, k, freq, ft0 in blocks:
            if freq <= 0:
                continue
            out.write_both("=" * 80, elapsed=False)
            out.write_both(_tracmt_freq_header(freq), elapsed=False)
            out.write_both("=" * 80, elapsed=False)
            ft = apply_calibration_ft(ft0, freq, cals)
            mask = coherence_mask(ft, p, out)
            ft = ft[mask]
            if len(ft) < 2:
                continue
            if p.procedure == 0:
                z, weights, absr, var, weights_by_out = rr_estimate(ft, p, out, coeffs_all=coeffs, freq=freq)
                istf = np.full((p.num_out+p.num_input, 2), np.nan+1j*np.nan)
                istf[:p.num_out, :] = z
                Z = _dewhiten_local_impedance(z, p, coeffs, freq)
                md = absr / max(madn(absr), EPS)
                scale = madn(absr)
            else:
                istf, weights, scale, var, md = rrms_estimate(ft, p, out, cov_full=(p.procedure == 2))
                Z = _dewhiten_local_impedance(impedance_from_rrms_istf(istf, p), p, coeffs, freq)
                weights_by_out = None
                z = None
            out.write_log(f"Estimate error bars mode={p.error_estimation} for {freq:.6e} Hz using {len(ft)} segments", elapsed=False)
            err_bars = _estimate_error_bars(
                ft, p, Z, freq, coeffs, weights=weights, weights_by_out=weights_by_out
            )
            period = 1.0 / freq
            if p.error_estimation == 0:
                out.write_cvg("-" * 80)
                out.write_cvg("Estimate errors by parametric approximation")
                out.write_cvg("-" * 80)
            elif p.error_estimation == 1:
                out.write_cvg("-" * 80)
                out.write_cvg("Estimate errors by fixed-weights bootstrap")
                out.write_cvg("-" * 80)
            elif p.error_estimation == 2:
                out.write_cvg("-" * 80)
                out.write_cvg("Estimate errors by jackknife")
                out.write_cvg("-" * 80)
            elif p.error_estimation == 3:
                out.write_cvg("-" * 80)
                out.write_cvg("Estimate errors by robust bootstrap")
                out.write_cvg("-" * 80)
            resp_rows.append(_build_tracmt_response_row(freq, period, ft, Z, err_bars, p, Zw=z, weights_by_out=weights_by_out))
            engine_rows.append(_build_engine_response_row(freq, period, istf, Z, err_bars, scale, weights, len(ft)))
            if p.output_rhoa_phs:
                rho_rows.append(_build_tracmt_rho_row(freq, period, ft, Z, err_bars, p, Zw=z, weights_by_out=weights_by_out))
            erow = {"freq": freq}
            for i, j, name in TRACMT_Z_COMPONENTS:
                if i < Z.shape[0] and j < Z.shape[1]:
                    erow[name] = Z[i, j]
            edi_rows.append(erow)
            if p.output_level >= 1:
                pd.DataFrame({"MD": md, "weight": weights}).to_csv(outdir / f"residuals_{freq:.6e}.csv", index=False)

        resp_path = outdir / "response_functions.csv"
        _write_tracmt_csv(resp_path, resp_rows, _tracmt_response_column_names(p))
        paths = {"response_functions": resp_path, "log": out.log_path, "cvg": out.cvg_path}
        if engine_rows:
            ext_path = outdir / "response_functions_extended.csv"
            pd.DataFrame(engine_rows).to_csv(ext_path, index=False, float_format=TRACMT_CSV_FLOAT_FORMAT, encoding="utf-8-sig")
            paths["response_functions_extended"] = ext_path
        if p.output_rhoa_phs:
            rho_path = outdir / "apparent_resistivity_and_phase.csv"
            _write_tracmt_csv(rho_path, rho_rows, _tracmt_rho_column_names(p))
            paths["rhoa_phase"] = rho_path
        if p.output_edi and edi_rows:
            edi_path = outdir / "pyTRACMT_v8.edi"
            write_edi(edi_path, edi_rows)
            paths["edi"] = edi_path
        return paths
    finally:
        out.close()


# -----------------------------------------------------------------------------
# GUI compatibility wrappers: keep the original GUI tables/plots while using v4 engine
# -----------------------------------------------------------------------------
import queue
import threading
import logging

try:
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None


def _read_preview_data(param_file: Path, p: Params) -> Tuple[np.ndarray, List[str]]:
    """Read a small/complete preview for the GUI using the same path and CSV logic as the engine.

    Older GUI code rebuilt relative paths by hand and could accidentally create paths like
    C:/.../ex.txt/ex.txt when the output/param folder was mis-selected.  This wrapper now
    delegates path resolution to _resolve_data_path() and reading to read_channel(), so preview
    behavior matches the actual processing engine.  If preview fails, calculation can still
    continue; the error is shown in the message tab instead of aborting the whole run.
    """
    cols, names = [], []
    if not p.data_sections:
        return np.empty((0, 0)), []
    nread, files = p.data_sections[0]
    for fname, skip in files:
        fpath = _resolve_data_path(Path(param_file), fname)
        # If the user accidentally chose an output folder ending with a channel filename
        # (e.g. .../ex.txt) and DATA_FILES contains ex.txt, avoid .../ex.txt/ex.txt.
        if (not fpath.exists()) and fpath.parent.name.lower() == Path(fname).name.lower():
            alt = fpath.parent
            if alt.exists() and alt.is_file():
                fpath = alt
        arr = read_channel(fpath, skip=skip, n=nread)
        cols.append(arr.astype(float))
        names.append(str(fpath.name))
    if not cols:
        return np.empty((0, 0)), []
    nmin = min(len(c) for c in cols)
    return np.column_stack([c[:nmin] for c in cols]), names


def _response_alias_for_gui_v4(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "frequency" in out.columns and "freq" not in out.columns:
        out["freq"] = out["frequency"]
    elif "freq" in out.columns and "frequency" not in out.columns:
        out["frequency"] = out["freq"]
    if "period" not in out.columns and "freq" in out.columns:
        out["period"] = 1.0 / out["freq"]
    for oi, j, low in TRACMT_Z_COMPONENTS:
        lab = _z_component_label(oi, j)
        rr_new, ri_new, dr_new = f"{lab}_real", f"{lab}_imag", f"d{lab}"
        ij = 2 + j
        rr_old, ri_old = f"resp_real_{oi}_{ij}", f"resp_imag_{oi}_{ij}"
        dr_old = f"dresp_{oi}_{ij}"
        rr = rr_new if rr_new in out.columns else rr_old
        ri = ri_new if ri_new in out.columns else ri_old
        dr = dr_new if dr_new in out.columns else dr_old
        if rr in out.columns:
            out[f"{low}_real"] = out[rr]
            out[f"{lab}_real"] = out[rr]
        if ri in out.columns:
            out[f"{low}_imag"] = out[ri]
            out[f"{lab}_imag"] = out[ri]
        if dr in out.columns:
            out[f"{low}_err_abs"] = out[dr]
            out[f"d{lab}"] = out[dr]
    return out


def _rho_alias_for_gui_v4(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        freq = r.get("frequency", r.get("freq", np.nan))
        period = r.get("period", 1.0 / freq if np.isfinite(freq) and freq != 0 else np.nan)
        for oi, j, low in TRACMT_Z_COMPONENTS:
            comp = _z_component_label(oi, j)
            lab = comp
            ij = 2 + j
            rho_new, ph_new = f"rho_{lab}", f"phase_{lab}"
            drho_new, dph_new = f"drho_{lab}", f"dphase_{lab}"
            rho_old, ph_old = f"app_res_{oi}_{ij}", f"phase_{oi}_{ij}"
            drho_old, dph_old = f"dapp_res_{oi}_{ij}", f"dphase_{oi}_{ij}"
            rho_col = rho_new if rho_new in df.columns else rho_old
            ph_col = ph_new if ph_new in df.columns else ph_old
            drho_col = drho_new if drho_new in df.columns else drho_old
            dph_col = dph_new if dph_new in df.columns else dph_old
            if rho_col not in df.columns or not pd.notna(r.get(rho_col, np.nan)):
                continue
            rows.append({
                "frequency": freq,
                "period": period,
                "component": comp,
                "apparent_resistivity_ohm_m": r.get(rho_col, np.nan),
                "phase_deg": r.get(ph_col, np.nan),
                "d_apparent_resistivity_ohm_m": r.get(drho_col, np.nan),
                "d_phase_deg": r.get(dph_col, np.nan),
            })
    return pd.DataFrame(rows)


def run_pipeline(param_file: Path, outdir: Path) -> Tuple[Params, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Run pyTRACMT V8 engine and return objects expected by the original GUI."""
    outdir.mkdir(parents=True, exist_ok=True)
    p = parse_param(param_file)
    try:
        data, names = _read_preview_data(param_file, p)
    except Exception as preview_error:
        # Preview is optional; do not stop the calculation only because the GUI table cannot load it.
        logging.warning("Time-series preview failed: %s", preview_error)
        data, names = np.empty((0, 0)), []
    paths = process(param_file, outdir)
    rf = pd.read_csv(paths["response_functions"]) if "response_functions" in paths else pd.DataFrame()
    rho = pd.read_csv(paths["rhoa_phase"]) if "rhoa_phase" in paths else pd.DataFrame()
    rf_gui = _response_alias_for_gui_v4(rf)
    rho_gui = _rho_alias_for_gui_v4(rho)
    rf_gui.to_csv(outdir / "response_functions_gui_alias.csv", index=False, encoding="utf-8-sig")
    rho_gui.to_csv(outdir / "apparent_resistivity_and_phase_gui_alias.csv", index=False, encoding="utf-8-sig")
    return p, data, names, rf_gui, rho_gui

# ============================================================
# GUI
# ============================================================
class PyTRACMTGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("pyTRACMT Professional V9 - TRACMT Workflow-Compatible Edition")
        self.root.geometry(self._initial_geometry())
        self.root.minsize(980, 640)
        self.ui_scale = tk.DoubleVar(value=1.0)
        self.base_font_size = max(9, int(round(9 * self.root.tk.call("tk", "scaling") / 1.333)))
        self._setup_responsive_ui()

        self.param_path = tk.StringVar(value="")
        self.outdir_path = tk.StringVar(value=str(Path.cwd()))
        self.status_var = tk.StringVar(value="Ready")
        self.plot_kind = tk.StringVar(value="Apparent resistivity")
        self.component_var = tk.StringVar(value="Zxy")

        # GUI param.dat builder variables
        self.gui_sampling_freq = tk.StringVar(value="32")
        self.gui_num_out = tk.IntVar(value=2)
        self.gui_ex_file = tk.StringVar(value="")
        self.gui_ey_file = tk.StringVar(value="")
        self.gui_hx_file = tk.StringVar(value="")
        self.gui_hy_file = tk.StringVar(value="")
        self.gui_use_rr = tk.BooleanVar(value=True)
        self.gui_hrx_file = tk.StringVar(value="")
        self.gui_hry_file = tk.StringVar(value="")
        self.gui_output_rhoa = tk.BooleanVar(value=True)
        self.gui_unit_mode = tk.StringVar(value="MTH5/TRACMT practical: E=mV/km, B=nT")
        self.gui_estimator = tk.StringVar(value="Huber")
        self.gui_estimator2 = tk.StringVar(value="Tukey")
        self.gui_error = tk.StringVar(value="1 Fixed-weights Bootstrap")
        self.gui_bootstrap = tk.StringVar(value="1000")
        self.gui_seg_text = tk.StringVar(value="AUTO_FAST")
        self.gui_data_count = tk.StringVar(value="")

        # TRACMT v4 advanced parameters; defaults follow Usui/TRACMT manual examples where practical.
        self.gui_procedure = tk.StringVar(value="Ordinary Remote Reference (RR)")
        self.gui_rrms_random = tk.IntVar(value=1)
        self.gui_rrms_ninit = tk.StringVar(value="100")
        self.gui_rrms_iter1 = tk.StringVar(value="3")
        self.gui_rrms_conv1 = tk.StringVar(value="0.05")
        self.gui_rrms_nbest = tk.StringVar(value="10")
        self.gui_rrms_iter2 = tk.StringVar(value="16")
        self.gui_rrms_conv2 = tk.StringVar(value="0.01")
        self.gui_prewhite_enable = tk.BooleanVar(value=False)
        self.gui_prewhite_type = tk.StringVar(value="0: standard AR")
        self.gui_prewhite_degree = tk.StringVar(value="10")
        self.gui_prewhite_Candidates = tk.StringVar(value="5")
        self.gui_robust_filter_enable = tk.BooleanVar(value=False)
        self.gui_robust_filter_replace = tk.IntVar(value=0)
        self.gui_robust_filter_thresholds = tk.StringVar(value="10 12 50")
        self.gui_coherence_enable = tk.BooleanVar(value=False)
        self.gui_coherence_nseg = tk.StringVar(value="10")
        self.gui_coherence_thr = tk.StringVar(value="0.3")
        self.gui_cal_files = tk.StringVar(value="")
        self.gui_high_pass = tk.StringVar(value="")
        self.gui_low_pass = tk.StringVar(value="")
        self.gui_notch = tk.StringVar(value="")
        self.gui_output_edi = tk.BooleanVar(value=True)

        self.params: Optional[Params] = None
        self.data: Optional[np.ndarray] = None
        self.names: List[str] = []
        self.rf = pd.DataFrame()
        self.rho = pd.DataFrame()

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None

        self._build_ui()
        self.root.after(200, self._poll_queue)


    def _initial_geometry(self) -> str:
        """Choose a startup size that fits the current monitor instead of using a fixed 1500x940 window."""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = min(1500, max(980, int(sw * 0.92)))
            h = min(940, max(640, int(sh * 0.88)))
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            return f"{w}x{h}+{x}+{y}"
        except Exception:
            return "1200x760"

    def _setup_responsive_ui(self):
        """Shared fonts, style, and keyboard shortcuts for zooming the GUI."""
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.text_font = tkfont.nametofont("TkTextFont")
        self.fixed_font = tkfont.nametofont("TkFixedFont")
        self.tree_font = tkfont.nametofont("TkDefaultFont")
        self.style = ttk.Style(self.root)
        self._apply_ui_scale()
        for seq in ("<Control-plus>", "<Control-equal>"):
            self.root.bind_all(seq, lambda e: self.change_ui_scale(1.10))
        self.root.bind_all("<Control-minus>", lambda e: self.change_ui_scale(1/1.10))
        self.root.bind_all("<Control-0>", lambda e: self.reset_ui_scale())

    def _apply_ui_scale(self):
        size = max(8, min(18, int(round(self.base_font_size * float(self.ui_scale.get())))))
        for font in (self.default_font, self.text_font, self.fixed_font, self.tree_font):
            font.configure(size=size)
        self.style.configure("Treeview", rowheight=max(22, int(size * 2.4)))
        self.style.configure("Treeview.Heading", font=(self.default_font.actual("family"), size, "bold"))
        if hasattr(self, "status_var"):
            self.status_var.set(f"Ready  |  GUI zoom {int(float(self.ui_scale.get())*100)}%  |  Ctrl+ / Ctrl- to adjust font size")
        if hasattr(self, "fig"):
            self.fig.set_dpi(max(80, int(100 * float(self.ui_scale.get()))))
            self.canvas.draw_idle()

    def change_ui_scale(self, factor: float):
        self.ui_scale.set(max(0.75, min(1.75, float(self.ui_scale.get()) * factor)))
        self._apply_ui_scale()
        self._sync_builder_scrollregion()

    def reset_ui_scale(self):
        self.ui_scale.set(1.0)
        self._apply_ui_scale()
        self._sync_builder_scrollregion()

    def _bind_mousewheel_to_canvas(self, canvas):
        def _on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                delta = int(-1 * (event.delta / 120)) if event.delta else 0
                canvas.yview_scroll(delta, "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

    def _sync_builder_scrollregion(self, event=None):
        if hasattr(self, "builder_canvas"):
            self.builder_canvas.configure(scrollregion=self.builder_canvas.bbox("all"))

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=3)
        top.columnconfigure(4, weight=2)

        ttk.Label(top, text="param.dat").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(top, textvariable=self.param_path).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(top, text="Browse param.dat", command=self.choose_param).grid(row=0, column=2, sticky="w", padx=4, pady=2)

        ttk.Label(top, text="Output Folder").grid(row=0, column=3, sticky="w", padx=(12, 4), pady=2)
        ttk.Entry(top, textvariable=self.outdir_path).grid(row=0, column=4, sticky="ew", padx=4, pady=2)
        ttk.Button(top, text="Browse", command=self.choose_outdir).grid(row=0, column=5, sticky="w", padx=4, pady=2)
        ttk.Button(top, text="Run", command=self.start_run).grid(row=0, column=6, sticky="w", padx=(8, 4), pady=2)
        ttk.Button(top, text="Open Output Folder", command=self.open_outdir).grid(row=0, column=7, sticky="w", padx=4, pady=2)

        zoom = ttk.Frame(self.root, padding=(8, 0))
        zoom.pack(fill=tk.X)
        ttk.Label(zoom, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(zoom, text="A-", width=4, command=lambda: self.change_ui_scale(1/1.10)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(zoom, text="A+", width=4, command=lambda: self.change_ui_scale(1.10)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(zoom, text="100%", width=6, command=self.reset_ui_scale).pack(side=tk.RIGHT, padx=2)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_builder = ttk.Frame(self.nb)
        self.tab_param = ttk.Frame(self.nb)
        self.tab_log = ttk.Frame(self.nb)
        self.tab_rf = ttk.Frame(self.nb)
        self.tab_rho = ttk.Frame(self.nb)
        self.tab_plot = ttk.Frame(self.nb)
        self.tab_data = ttk.Frame(self.nb)

        self.nb.add(self.tab_builder, text="Parameter Builder")
        self.nb.add(self.tab_param, text="param.dat Content")
        self.nb.add(self.tab_log, text="Log")
        self.nb.add(self.tab_rf, text="Response functions")
        self.nb.add(self.tab_rho, text="ρa / Phase")
        self.nb.add(self.tab_plot, text="Plots")
        self.nb.add(self.tab_data, text="Time-series Preview")

        self._build_param_builder_tab()

        self.param_text = tk.Text(self.tab_param, height=20)
        self.param_text.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(self.tab_log, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.rf_tree = self._make_tree(self.tab_rf)
        self.rho_tree = self._make_tree(self.tab_rho)

        self._build_plot_tab()
        self.data_tree = self._make_tree(self.tab_data)

    def _make_tree(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, show="headings")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _build_plot_tab(self):
        controls = ttk.Frame(self.tab_plot, padding=6)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Plot Type").pack(side=tk.LEFT)
        plot_combo = ttk.Combobox(controls, textvariable=self.plot_kind,
                                  values=["Apparent resistivity", "Phase", "Response amplitude", "Response phase"],
                                  state="readonly", width=24)
        plot_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Component").pack(side=tk.LEFT, padx=(12, 2))
        comp_combo = ttk.Combobox(controls, textvariable=self.component_var,
                                  values=["Zxx", "Zxy", "Zyx", "Zyy"], state="readonly", width=8)
        comp_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Update Plot", command=self.update_plot).pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="Save PNG", command=self.save_plot).pack(side=tk.LEFT, padx=4)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.tab_plot)

    def _build_param_builder_tab(self):
        # Use a scrollable canvas so all controls remain reachable on smaller screens.
        container = ttk.Frame(self.tab_builder)
        container.pack(fill=tk.BOTH, expand=True)
        self.builder_canvas = tk.Canvas(container, highlightthickness=0)
        yscroll = ttk.Scrollbar(container, orient="vertical", command=self.builder_canvas.yview)
        xscroll = ttk.Scrollbar(container, orient="horizontal", command=self.builder_canvas.xview)
        self.builder_canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.builder_canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        outer = ttk.Frame(self.builder_canvas, padding=10)
        self.builder_window = self.builder_canvas.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", self._sync_builder_scrollregion)
        self.builder_canvas.bind("<Configure>", lambda e: self.builder_canvas.itemconfigure(self.builder_window, width=max(e.width, outer.winfo_reqwidth())))
        self._bind_mousewheel_to_canvas(self.builder_canvas)

        basic = ttk.LabelFrame(outer, text="Basic Settings")
        basic.pack(fill=tk.X, pady=6)
        ttk.Label(basic, text="Sampling Frequency (Hz)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(basic, textvariable=self.gui_sampling_freq, width=12).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(basic, text="NUM_OUT").grid(row=0, column=2, sticky="w", padx=(18,4), pady=4)
        ttk.Spinbox(basic, from_=1, to=3, textvariable=self.gui_num_out, width=6).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(basic, text="OUTPUT_RHOA_PHS", variable=self.gui_output_rhoa).grid(row=0, column=4, sticky="w", padx=16, pady=4)
        ttk.Label(basic, text="Unit Mode").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(basic, textvariable=self.gui_unit_mode, state="readonly", width=42,
                     values=["MTH5/TRACMT practical: E=mV/km, B=nT",
                             "MTH5 SI-B: E=V/m, B=T",
                             "SI-H: E=V/m, H=A/m"]).grid(row=1, column=1, columnspan=4, sticky="w", padx=4, pady=4)

        proc = ttk.LabelFrame(outer, text="Processing Method / Error Estimation")
        proc.pack(fill=tk.X, pady=6)
        ttk.Label(proc, text="Processing procedure").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.proc_combo = ttk.Combobox(proc, textvariable=self.gui_procedure, state="readonly", width=36,
                     values=["Ordinary Remote Reference (RR)", "Robust Multivariate Regression (RRMS)", "Modified RRMS (MRRMS)"])
        self.proc_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.proc_combo.bind("<<ComboboxSelected>>", self._update_tracmt_control_states)

        ttk.Label(proc, text="1st M-estimator").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.est1_combo = ttk.Combobox(proc, textvariable=self.gui_estimator, state="readonly", width=18,
                     values=["None", "Huber", "Tukey", "Thomson"])
        self.est1_combo.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        self.est1_combo.bind("<<ComboboxSelected>>", self._update_tracmt_control_states)
        ttk.Label(proc, text="2nd M-estimator").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.est2_combo = ttk.Combobox(proc, textvariable=self.gui_estimator2, state="readonly", width=18,
                     values=["None", "Huber", "Tukey", "Thomson"])
        self.est2_combo.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(proc, text="Error estimation").grid(row=0, column=2, sticky="w", padx=(18,4), pady=4)
        self.error_combo = ttk.Combobox(proc, textvariable=self.gui_error, state="readonly", width=30,
                     values=["0 Parametric", "1 Fixed-weights Bootstrap", "2 Fixed-weights Jackknife", "3 Robust Bootstrap", "4 Strict Bootstrap"])
        self.error_combo.grid(row=0, column=3, sticky="w", padx=4, pady=4)
        self.error_combo.bind("<<ComboboxSelected>>", self._update_tracmt_control_states)
        ttk.Label(proc, text="Resampling repetitions").grid(row=1, column=2, sticky="w", padx=(18,4), pady=4)
        self.bootstrap_entry = ttk.Entry(proc, textvariable=self.gui_bootstrap, width=10)
        self.bootstrap_entry.grid(row=1, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(proc, text="TRACMT rule: robust bootstrap requires Tukey as the first and no second M-estimator.",
                  foreground="#555").grid(row=3, column=0, columnspan=5, sticky="w", padx=4, pady=(2,6))

        adv = ttk.LabelFrame(outer, text="Advanced Settings")
        adv.pack(fill=tk.X, pady=6)
        ttk.Label(adv, text="RRMS parameters: random ninit iter1 conv1 nbest iter2 conv2").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.rrms_frame = ttk.Frame(adv)
        rrms_frame = self.rrms_frame
        rrms_frame.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_random, width=4).pack(side=tk.LEFT)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_ninit, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_iter1, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_conv1, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_nbest, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_iter2, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rrms_frame, textvariable=self.gui_rrms_conv2, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(adv, text="PREWHITENING", variable=self.gui_prewhite_enable).grid(row=1, column=0, sticky="w", padx=4, pady=3)
        pwf = ttk.Frame(adv); pwf.grid(row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Combobox(pwf, textvariable=self.gui_prewhite_type, state="readonly", width=18,
                     values=["0: standard AR", "1: robust PARCOR", "-1: user AR"]).pack(side=tk.LEFT)
        ttk.Label(pwf, text="degree").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(pwf, textvariable=self.gui_prewhite_degree, width=6).pack(side=tk.LEFT)
        ttk.Label(pwf, text="Candidates").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(pwf, textvariable=self.gui_prewhite_Candidates, width=6).pack(side=tk.LEFT)

        ttk.Checkbutton(adv, text="ROBUST_FILTER", variable=self.gui_robust_filter_enable).grid(row=2, column=0, sticky="w", padx=4, pady=3)
        rff = ttk.Frame(adv); rff.grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(rff, text="Replace 0/1").pack(side=tk.LEFT)
        ttk.Entry(rff, textvariable=self.gui_robust_filter_replace, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(rff, text="Threshold per channel, e.g. 10 12 50").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(rff, textvariable=self.gui_robust_filter_thresholds, width=20).pack(side=tk.LEFT)

        ttk.Checkbutton(adv, text="COHERENCE_CRITERIA", variable=self.gui_coherence_enable).grid(row=3, column=0, sticky="w", padx=4, pady=3)
        coh = ttk.Frame(adv); coh.grid(row=3, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(coh, text="Nseg").pack(side=tk.LEFT)
        ttk.Entry(coh, textvariable=self.gui_coherence_nseg, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(coh, text="threshold").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(coh, textvariable=self.gui_coherence_thr, width=8).pack(side=tk.LEFT)

        ttk.Label(adv, text="CAL_FILES, comma/semicolon separated").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(adv, textvariable=self.gui_cal_files, width=90).grid(row=4, column=1, sticky="we", padx=4, pady=3)
        ttk.Label(adv, text="Filters: HP / LP / Notch Hz").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        ff = ttk.Frame(adv); ff.grid(row=5, column=1, sticky="w", padx=4, pady=3)
        ttk.Entry(ff, textvariable=self.gui_high_pass, width=10).pack(side=tk.LEFT)
        ttk.Entry(ff, textvariable=self.gui_low_pass, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Entry(ff, textvariable=self.gui_notch, width=30).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(ff, text="Output EDI", variable=self.gui_output_edi).pack(side=tk.LEFT, padx=10)
        adv.columnconfigure(1, weight=1)

        seg = ttk.LabelFrame(outer, text="Frequency Windows / SEGMENT")
        seg.pack(fill=tk.X, pady=6)
        ttk.Label(seg, text="SEGMENT: AUTO_FAST / AUTO_FULL or length:indexes; length:indexes").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(seg, textvariable=self.gui_seg_text, width=70).grid(row=0, column=1, sticky="we", padx=4, pady=4)
        ttk.Label(seg, text="Data Count (blank = all)").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(seg, textvariable=self.gui_data_count, width=18).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        seg.columnconfigure(1, weight=1)

        files = ttk.LabelFrame(outer, text="Input Time-series Files")
        files.pack(fill=tk.X, pady=6)
        self._file_row(files, 0, "Ex", self.gui_ex_file)
        self._file_row(files, 1, "Ey", self.gui_ey_file)
        self._file_row(files, 2, "Hx", self.gui_hx_file)
        self._file_row(files, 3, "Hy", self.gui_hy_file)
        self.rr_check = ttk.Checkbutton(files, text="Use independent remote-reference Hx/Hy (required by TRACMT RR/RRMS)", variable=self.gui_use_rr, command=self._update_tracmt_control_states)
        self.rr_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        self._file_row(files, 5, "RR Hx", self.gui_hrx_file)
        self._file_row(files, 6, "RR Hy", self.gui_hry_file)
        files.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=10)
        ttk.Button(actions, text="Generate param.dat", command=self.generate_param_from_gui).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Generate and Run", command=self.generate_and_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Save Current param.dat", command=self.save_param_text).pack(side=tk.LEFT, padx=4)

        note = ("Channel order is written as Ex, Ey, Hx, Hy. If Remote Reference is enabled, RR Hx and RR Hy are appended. "
                "V9 follows the TRACMT procedure hierarchy: Ordinary RR with two sequential M-estimators, RRMS/MRRMS with dedicated parameters, compatible error-estimation choices, remote-reference requirements, preprocessing, calibration, CVG/log, and EDI output.")
        ttk.Label(outer, text=note, wraplength=900, foreground="#555", justify="left").pack(fill=tk.X, pady=6)
        self._update_tracmt_control_states()

    def _update_tracmt_control_states(self, event=None):
        """Enforce the same option dependencies used by the TRACMT workflow."""
        method = self.gui_procedure.get()
        is_rr = method.startswith("Ordinary")
        rrms_state = "disabled" if is_rr else "normal"
        est_state = "readonly" if is_rr else "disabled"
        if hasattr(self, "est1_combo"):
            self.est1_combo.configure(state=est_state)
            self.est2_combo.configure(state=est_state)
        if hasattr(self, "rrms_frame"):
            for child in self.rrms_frame.winfo_children():
                try:
                    child.configure(state=rrms_state)
                except tk.TclError:
                    pass
        err = self.gui_error.get().strip()
        needs_repetitions = err.startswith(("1 ", "3 ", "4 "))
        if hasattr(self, "bootstrap_entry"):
            self.bootstrap_entry.configure(state="normal" if needs_repetitions else "disabled")
        if err.startswith("3 ") and is_rr:
            self.gui_estimator.set("Tukey")
            self.gui_estimator2.set("None")
        # All currently implemented TRACMT procedures require two remote-reference channels.
        if not self.gui_use_rr.get():
            self.status_var.set("Remote reference is required for RR/RRMS; select RR Hx and RR Hy before running.")

    @staticmethod
    def _mestimator_code(name: str) -> int:
        return {"None": -1, "Huber": 0, "Tukey": 1, "Thomson": 2}.get(name, -1)

    def _validate_tracmt_gui_options(self) -> None:
        method = self.gui_procedure.get()
        err = self.gui_error.get().strip()
        if not self.gui_use_rr.get():
            raise ValueError("TRACMT RR/RRMS processing requires independent RR Hx and RR Hy channels. Enable Remote Reference and select both files.")
        if method.startswith("Ordinary"):
            if self.gui_estimator.get() == "None" and self.gui_estimator2.get() != "None":
                raise ValueError("The second M-estimator cannot be enabled when the first M-estimator is None.")
            if err.startswith("3 ") and not (self.gui_estimator.get() == "Tukey" and self.gui_estimator2.get() == "None"):
                raise ValueError("TRACMT robust bootstrap requires: first M-estimator = Tukey, second M-estimator = None.")
        elif err.startswith("1 ") or err.startswith("2 "):
            raise ValueError("Fixed-weights bootstrap/jackknife belongs to Ordinary RR. For RRMS/MRRMS use Parametric, Robust Bootstrap, or Strict Bootstrap.")
        reps = int(float(self.gui_bootstrap.get() or 0))
        if err.startswith(("1 ", "3 ", "4 ")) and reps < 4:
            raise ValueError("Resampling repetitions must be at least 4.")

    def _file_row(self, parent, row: int, label: str, var: tk.StringVar):
        ttk.Label(parent, text=label, width=10).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=90).grid(row=row, column=1, sticky="we", padx=4, pady=3)
        ttk.Button(parent, text="Browse", command=lambda v=var: self._choose_data_file(v)).grid(row=row, column=2, sticky="w", padx=4, pady=3)

    def _choose_data_file(self, var: tk.StringVar):
        fp = filedialog.askopenfilename(title="Select time-series file", filetypes=[("Data files", "*.txt *.csv *.dat *.*")])
        if fp:
            var.set(fp)

    def _parse_segment_builder(self) -> List[SegmentSpec]:
        txt = self.gui_seg_text.get().strip()
        if not txt:
            raise ValueError("SEGMENT setting is empty")
        txt_upper = txt.upper()
        if txt_upper in ("AUTO", "AUTO_FAST", "AUTO_FULL"):
            # Multi-scale TRACMT-like frequency coverage.
            # AUTO_FAST keeps a moderate number of frequency bins for interactive GUI use.
            # AUTO_FULL adds longer windows and more bins, but it can be much slower.
            lengths = [1024, 2048, 4096, 8192, 16384, 32768]
            if txt_upper == "AUTO_FULL":
                lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
            out = []
            for L in lengths:
                kmax = 96 if txt_upper != "AUTO_FULL" else 160
                kmax = min(kmax, L // 2)
                idxs = sorted(set(np.round(np.logspace(np.log10(2), np.log10(kmax), 28 if txt_upper != "AUTO_FULL" else 42)).astype(int)))
                idxs = [int(i) for i in idxs if 1 < int(i) < L // 2]
                out.append(SegmentSpec(L, idxs))
            return out
        out = []
        for part in txt.split(';'):
            part = part.strip()
            if not part:
                continue
            if ':' not in part:
                raise ValueError(f"SEGMENT item must be length:indexes, got {part}")
            length_s, idx_s = part.split(':', 1)
            idxs = [int(x.strip()) for x in idx_s.replace(' ', ',').split(',') if x.strip()]
            out.append(SegmentSpec(int(length_s), idxs))
        return out

    def _unit_mode_code_from_gui(self) -> str:
        txt = self.gui_unit_mode.get().lower()
        if "si-b" in txt or "b=t" in txt:
            return "si_b_tesla"
        if "si-h" in txt or "a/m" in txt:
            return "si_h_ampere"
        return "mth5_tracmt_practical"

    def _build_param_text_from_gui(self) -> str:
        fs = float(self.gui_sampling_freq.get())
        nout = int(self.gui_num_out.get())
        segs = self._parse_segment_builder()
        self._validate_tracmt_gui_options()
        method = self.gui_procedure.get()
        if method.startswith("Ordinary"):
            procedure = 0
            mest = f"{self._mestimator_code(self.gui_estimator.get())} {self._mestimator_code(self.gui_estimator2.get())}"
        elif method.startswith("Modified"):
            procedure = 2
            mest = "-1 -1"
        else:
            procedure = 1
            mest = "-1 -1"
        err_text = self.gui_error.get().strip()
        err = {
            "0 Parametric": 0, "1 Fixed-weights Bootstrap": 1, "2 Fixed-weights Jackknife": 2, "3 Robust Bootstrap": 3, "4 Strict Bootstrap": 4,
            "0 None": 0, "1 Parametric": 1, "3 Bootstrap": 3,
            "None": 0, "Parametric": 0, "Jackknife": 2, "Bootstrap": 3,
        }.get(err_text, 1)
        boot = int(float(self.gui_bootstrap.get() or 1000))

        file_vars = [self.gui_ex_file, self.gui_ey_file, self.gui_hx_file, self.gui_hy_file]
        if self.gui_use_rr.get():
            file_vars += [self.gui_hrx_file, self.gui_hry_file]
        files = [v.get().strip() for v in file_vars]
        missing = [f"channel #{i+1}" for i, f in enumerate(files) if not f]
        if missing:
            raise ValueError("Missing input file(s): " + ', '.join(missing))

        # DATA count: TRACMT requires a number.  If blank, use 0 here and load_ascii_data will read all.
        # For compatibility with our parser, blank is represented as a very large number.
        data_count_s = self.gui_data_count.get().strip()
        if data_count_s:
            data_count = int(float(data_count_s))
        else:
            data_count = 10**18

        lines = []
        lines += ["NUM_OUT", str(nout)]
        lines += ["NUM_RR", "2"]
        lines += ["SAMPLING_FREQ", f"{fs:g}"]
        lines += ["PROCEDURE", str(procedure)]
        if procedure in (1, 2):
            lines += ["RRMS",
                      str(int(self.gui_rrms_random.get())),
                      str(int(float(self.gui_rrms_ninit.get()))),
                      str(int(float(self.gui_rrms_iter1.get()))),
                      str(float(self.gui_rrms_conv1.get())),
                      str(int(float(self.gui_rrms_nbest.get()))),
                      str(int(float(self.gui_rrms_iter2.get()))),
                      str(float(self.gui_rrms_conv2.get()))]
        lines += ["MESTIMATORS", mest]
        lines += ["ERROR_ESTIMATION", str(err)]
        if err in (1, 3, 4):
            lines += ["BOOTSTRAP", str(boot)]
        if self.gui_output_rhoa.get():
            lines += ["OUTPUT_RHOA_PHS"]
        lines += ["UNIT_MODE", self._unit_mode_code_from_gui()]
        lines += ["APPLY_MTH5_RESPONSE", "1"]
        if self.gui_prewhite_enable.get():
            pw_type = int(self.gui_prewhite_type.get().split(":")[0])
            lines += ["PREWHITENING", str(pw_type), str(int(float(self.gui_prewhite_degree.get()))), str(int(float(self.gui_prewhite_Candidates.get())))]
        if self.gui_robust_filter_enable.get():
            lines += ["ROBUST_FILTER", str(int(self.gui_robust_filter_replace.get()))]
            thr = self.gui_robust_filter_thresholds.get().strip() or "10 12 50"
            for _ in range(4 + (2 if self.gui_use_rr.get() else 0)):
                lines.append(thr)
        if self.gui_coherence_enable.get():
            lines += ["COHERENCE_CRITERIA", str(int(float(self.gui_coherence_nseg.get()))), str(float(self.gui_coherence_thr.get()))]
        if self.gui_high_pass.get().strip():
            lines += ["HIGH_PASS", self.gui_high_pass.get().strip()]
        if self.gui_low_pass.get().strip():
            lines += ["LOW_PASS", self.gui_low_pass.get().strip()]
        if self.gui_notch.get().strip():
            vals = [v for v in re.split(r"[,;\s]+", self.gui_notch.get().strip()) if v]
            lines += ["NOTCH", str(len(vals)), " ".join(vals)]
        cal_text = self.gui_cal_files.get().strip()
        if cal_text:
            cals = [c for c in re.split(r"[,;]+", cal_text) if c.strip()]
            lines += ["CAL_FILES"] + [c.strip() for c in cals]
        if self.gui_output_edi.get():
            lines += ["OUTPUT_EDI"]
        lines += ["SEGMENT", str(len(segs))]
        for sg in segs:
            lines.append(" ".join([str(sg.length), str(len(sg.indexes)), *map(str, sg.indexes)]))
        lines += ["DATA_FILES", str(data_count)]
        for f in files:
            lines += [f, "0"]
        lines += ["END", ""]
        return "\n".join(lines)

    def generate_param_from_gui(self):
        try:
            txt = self._build_param_text_from_gui()
            outdir = Path(self.outdir_path.get().strip() or Path.cwd())
            # If a file path was accidentally put in the output-folder box, use its parent.
            # This prevents param_gui.dat from being created under a pseudo-folder such as ex.txt.
            if outdir.suffix and not outdir.exists():
                outdir = outdir.parent
            elif outdir.exists() and outdir.is_file():
                outdir = outdir.parent
            outdir.mkdir(parents=True, exist_ok=True)
            fp = outdir / "param_gui.dat"
            fp.write_text(txt, encoding="utf-8")
            self.param_path.set(str(fp))
            self.param_text.delete("1.0", tk.END)
            self.param_text.insert(tk.END, txt)
            self._log(f"Generated param.dat: {fp}")
        except Exception as e:
            messagebox.showerror("Generate param.dat error", str(e))
            self._log(traceback.format_exc())

    def generate_and_run(self):
        self.generate_param_from_gui()
        if self.param_path.get().strip():
            self.start_run()

    def save_param_text(self):
        fp = filedialog.asksaveasfilename(defaultextension=".dat", filetypes=[("DAT", "*.dat"), ("Text", "*.txt")])
        if not fp:
            return
        Path(fp).write_text(self.param_text.get("1.0", tk.END), encoding="utf-8")
        self.param_path.set(fp)
        self._log(f"Saved param.dat: {fp}")

    def choose_param(self):
        fp = filedialog.askopenfilename(title="Select param.dat", filetypes=[("param.dat / text", "*.dat *.txt *.*")])
        if not fp:
            return
        self.param_path.set(fp)
        self.outdir_path.set(str(Path(fp).parent / "pytracmt_output"))
        self.load_param_preview()

    def choose_outdir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir_path.set(d)

    def load_param_preview(self):
        self.param_text.delete("1.0", tk.END)
        fp = self.param_path.get().strip()
        if not fp:
            return
        try:
            txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
            self.param_text.insert(tk.END, txt)
            p = parse_param(fp)
            nfiles = sum(len(sec[1]) for sec in getattr(p, "data_sections", []))
            self._log(f"Parsed param.dat successfully: NUM_OUT={p.num_out}, fs={p.sampling_freq}, segments={len(p.segments)}, files={nfiles}")
        except Exception as e:
            self._log(f"param.dat preview/parse warning: {e}")

    def start_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Calculation is still running.")
            return
        fp = self.param_path.get().strip()
        out = self.outdir_path.get().strip()
        if not fp:
            messagebox.showwarning("Missing param.dat", "Please select a param.dat file first.")
            return
        if not Path(fp).exists():
            messagebox.showerror("File not found", fp)
            return
        self.status_var.set("Running...")
        self._log("Start calculation")
        self.worker = threading.Thread(target=self._run_worker, args=(Path(fp), Path(out)), daemon=True)
        self.worker.start()

    def _run_worker(self, fp: Path, out: Path):
        try:
            result = run_pipeline(fp, out)
            self.msg_queue.put(("done", result))
        except Exception:
            self.msg_queue.put(("error", traceback.format_exc()))

    def _poll_queue(self):
        try:
            while True:
                typ, payload = self.msg_queue.get_nowait()
                if typ == "done":
                    self.params, self.data, self.names, self.rf, self.rho = payload
                    self.status_var.set("Done")
                    self._log("Calculation finished.")
                    self._log(f"response_functions.csv rows: {len(self.rf)}")
                    if not self.rho.empty:
                        self._log(f"apparent_resistivity_and_phase.csv rows: {len(self.rho)}")
                    self.refresh_all_views()
                elif typ == "error":
                    self.status_var.set("Error")
                    self._log(payload)
                    messagebox.showerror("Calculation error", payload[-2000:])
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def refresh_all_views(self):
        self.fill_tree(self.rf_tree, self.rf)
        self.fill_tree(self.rho_tree, self.rho)
        if self.data is not None:
            cols = self._channel_names_for_data()
            preview = pd.DataFrame(self.data[:500, :], columns=cols)
            preview.insert(0, "sample_index", np.arange(len(preview)))
            if self.params:
                preview.insert(1, "time_s", preview["sample_index"] / self.params.sampling_freq)
            self.fill_tree(self.data_tree, preview)
        self.update_plot()

    def _channel_names_for_data(self) -> List[str]:
        if self.names:
            return [Path(n).stem for n in self.names]
        if self.data is None:
            return []
        return [f"ch{i}" for i in range(self.data.shape[1])]

    def fill_tree(self, tree: ttk.Treeview, df: pd.DataFrame, max_rows: int = 1000):
        tree.delete(*tree.get_children())
        if df is None or df.empty:
            tree["columns"] = []
            return
        show = df.head(max_rows).copy()
        cols = list(show.columns)
        tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=max(90, min(220, len(str(col)) * 10)), stretch=True)
        for _, row in show.iterrows():
            vals = []
            for v in row.values:
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            tree.insert("", tk.END, values=vals)

    @staticmethod
    def _finite_yerr(yerr_array) -> Optional[np.ndarray]:
        if yerr_array is None:
            return None
        yerr = np.asarray(yerr_array, dtype=float)
        if yerr.size == 0 or not np.isfinite(yerr).any():
            return None
        return yerr

    def _plot_xy_errorbar(self, x, y, yerr=None, *, logx: bool = False, logy: bool = False, **kwargs):
        """Draw error bars after log scales are set so caps remain visible."""
        if logx:
            self.ax.set_xscale("log")
        if logy:
            self.ax.set_yscale("log")
        yerr = self._finite_yerr(yerr)
        if logy and yerr is not None:
            y = np.asarray(y, dtype=float)
            yerr = np.asarray(yerr, dtype=float)
            valid = (y > 0) & np.isfinite(y) & np.isfinite(yerr) & (y > yerr)
            if not valid.all():
                x = np.asarray(x)[valid]
                y = y[valid]
                yerr = yerr[valid]
                if y.size == 0:
                    return
        self.ax.errorbar(x, y, yerr=yerr, capsize=4, elinewidth=1.2, capthick=1.2, **kwargs)

    def update_plot(self):
        self.ax.clear()
        kind = self.plot_kind.get()
        comp = self.component_var.get()

        if kind in ("Apparent resistivity", "Phase"):
            if self.rho.empty:
                self.ax.text(0.5, 0.5, "No ρa/phase results. Add OUTPUT_RHOA_PHS to param.dat.",
                             ha="center", va="center", transform=self.ax.transAxes)
                self.canvas.draw_idle()
                return
            sub = self.rho[self.rho["component"] == comp].sort_values("period")
            if sub.empty:
                self.ax.text(0.5, 0.5, f"No component {comp}", ha="center", va="center", transform=self.ax.transAxes)
            elif kind == "Apparent resistivity":
                y = sub["apparent_resistivity_ohm_m"].to_numpy()
                yerr = sub["d_apparent_resistivity_ohm_m"].to_numpy(dtype=float) if "d_apparent_resistivity_ohm_m" in sub else None
                self._plot_xy_errorbar(sub["period"], y, yerr=yerr, logx=True, logy=True, marker="o", linestyle="-")
                self.ax.set_ylabel("Apparent resistivity (ohm m)")
                self.ax.set_xlabel("Period (s)")
                self.ax.set_title(f"{comp} apparent resistivity")
            else:
                y = sub["phase_deg"].to_numpy()
                yerr = sub["d_phase_deg"].to_numpy(dtype=float) if "d_phase_deg" in sub else None
                self._plot_xy_errorbar(sub["period"], y, yerr=yerr, logx=True, marker="o", linestyle="-")
                self.ax.set_ylabel("Phase (deg)")
                self.ax.set_xlabel("Period (s)")
                self.ax.set_title(f"{comp} phase")
        else:
            if self.rf.empty or self.params is None:
                self.ax.text(0.5, 0.5, "No response function results.", ha="center", va="center", transform=self.ax.transAxes)
                self.canvas.draw_idle()
                return
            # map component to output/input indexes according to standard NUM_OUT=2, Hx index=2, Hy index=3.
            if self.params.num_out < 2:
                self.ax.text(0.5, 0.5, "Response component plot assumes NUM_OUT >= 2.", ha="center", va="center", transform=self.ax.transAxes)
                self.canvas.draw_idle()
                return
            hx_col = self.params.num_out
            hy_col = self.params.num_out + 1
            cmap = {
                "Zxx": (0, hx_col),
                "Zxy": (0, hy_col),
                "Zyx": (1, hx_col),
                "Zyy": (1, hy_col),
            }
            oi, ij = cmap.get(comp, (0, hy_col))
            lab = comp
            rcol = f"{lab}_real"
            icol = f"{lab}_imag"
            if rcol not in self.rf.columns:
                rcol = f"resp_real_{oi}_{ij}"
                icol = f"resp_imag_{oi}_{ij}"
            if rcol not in self.rf or icol not in self.rf:
                self.ax.text(0.5, 0.5, f"Columns {rcol}/{icol} not found.", ha="center", va="center", transform=self.ax.transAxes)
            else:
                z = self.rf[rcol].to_numpy() + 1j * self.rf[icol].to_numpy()
                err_col = f"d{lab}"
                if err_col not in self.rf.columns:
                    err_col = f"{comp.lower()}_err_abs"
                zerr = self.rf[err_col].to_numpy(dtype=float) if err_col in self.rf.columns else None
                if kind == "Response amplitude":
                    self._plot_xy_errorbar(
                        self.rf["period"], np.abs(z), yerr=zerr,
                        logx=True, logy=True, marker="o", linestyle="-",
                    )
                    self.ax.set_ylabel("|Response|")
                    self.ax.set_xlabel("Period (s)")
                    self.ax.set_title(f"{comp} response amplitude")
                else:
                    phase_err = None
                    if not self.rho.empty and "d_phase_deg" in self.rho.columns:
                        rho_sub = self.rho[self.rho["component"] == comp][["period", "d_phase_deg"]]
                        merged = self.rf[["period"]].merge(rho_sub, on="period", how="left")
                        phase_err = merged["d_phase_deg"].to_numpy(dtype=float)
                    self._plot_xy_errorbar(
                        self.rf["period"], np.angle(z, deg=True), yerr=phase_err,
                        logx=True, marker="o", linestyle="-",
                    )
                    self.ax.set_ylabel("Response phase (deg)")
                    self.ax.set_xlabel("Period (s)")
                    self.ax.set_title(f"{comp} response phase")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_plot(self):
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fp:
            self.fig.savefig(fp, dpi=300, bbox_inches="tight")
            self._log(f"Saved plot: {fp}")

    def open_outdir(self):
        out = Path(self.outdir_path.get().strip())
        out.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(out)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{out}"')
            else:
                os.system(f'xdg-open "{out}"')
        except Exception as e:
            messagebox.showinfo("Output folder", str(out))

    def _log(self, msg: str):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)





def main():
    if not TK_OK:
        print("Tkinter is not available. Use CLI: python pyTRACMT_Professional_V9_TRACMT_Workflow.py param.dat [outdir]")
        return
    root = tk.Tk()
    PyTRACMTGUI(root)
    root.title("pyTRACMT Professional V9 - TRACMT Workflow-Compatible Edition")
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python pyTRACMT_Professional_V9_TRACMT_Workflow.py [param.dat] [output_folder]")
        print("Without arguments, the graphical interface is opened.")
    elif len(sys.argv) > 1:
        param = sys.argv[1]
        outdir = sys.argv[2] if len(sys.argv) > 2 else None
        paths = process(param, outdir)
        for k, v in paths.items():
            print(f"{k}: {v}")
    else:
        main()
