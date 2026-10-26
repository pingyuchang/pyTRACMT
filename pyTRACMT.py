#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyTRACMT Professional V1.1 (raw ADC counts / physical-unit routing; No-RR preserved)
==============================================================

STANDARD EDITION:
    - diagnostic/debug CSV generation disabled
    - no channel-mapping / RAW-vs-SCAL diagnostic output
    - standard TRACMT.log and TRACMT.cvg retained
    - numerical processing, MTH5 calibration, unit routing, RR/RRMS and error estimation unchanged

Integrated Python engine + English GUI/CLI for TRACMT-style MT processing.

V11.3.1 update:
    - memory-safe limited-lag Yule-Walker autocorrelation (no full FFT correlate)
    - retains V11.3 MTH5 stored-unit routing and selected-run sampling-rate synchronization
V12.5 V66 calibration-chain changes:
    - fail-closed preflight accepts V66 RAW/unlabeled TD only when a usable response is resolved
    - falls back to /Phoenix_Calibration_JSON when Survey/Filters references are not exposed by the installed MTH5 API
    - FAP frequency response uses log-frequency/log-amplitude interpolation and linear end-slope extrapolation
V12.2 strict calibration-chain changes:
    - reads phoenix_v22.calibration_applied/source_units/source_format directly from channel HDF5 attrs
    - explicit calibration_applied=False overrides stale physical-looking unit labels
    - filter mounting is never treated as proof of numerical calibration
    - logs the calibration-state decision for every MTH5 channel

This MTH5 edition adds:

    - Standard MTH5 time-series input via DATA_FILES specs:
      MTH5::<file.h5>::<station>::<run>::<channel>
      or MTH5::<file.h5>::<survey>::<station>::<run>::<channel>
    - APPLY_MTH5_RESPONSE:
        * induction coil: FAP/ZPK -> TRACMT CAL_FILES + spectral correction
        * fluxgate: MTH5 coefficient/sensitivity + unit conversion in time domain
        * electric: dipole length + unit conversion to match UNIT_MODE
        * Phoenix RXCAL/SCAL FAP chain when mounted with applied=False
    - UNIT_MODE alignment to TRACMT expectations:
        practical: E=mV/km, B=nT
        SI-B: E=V/m, B=T
        SI-H: E=V/m, H=A/m
    - CAL_FILES: TRACMT calibration file format, frequency-domain correction
    - PROCEDURE 0/1, PREWHITENING, ROBUST_FILTER, COHERENCE_CRITERIA
    - TRACMT.log / TRACMT.cvg, response_functions.csv, EDI export

V11.3 rules:
- MTH5 channel stored units are read automatically and select the calibration path:
  counts -> digitizer/receiver/sensor chain; mV/V -> receiver/sensor chain;
  physical mV/km or nT/T -> unit conversion only unless an unapplied response remains.
- Selecting an MTH5 run automatically synchronizes SAMPLING_FREQ; V12 detects stored units and normalizes the internal target to TRACMT practical units.
- Physical stored units (mV/km, V/m, nT, T, A/m) are authoritative and block repeat amplitude calibration.
- Phoenix TD voltage (mV/V) never receives a second receiver analog-gain division.
- filters attached in Survey/Filters does NOT mean the stored
time series is already nT or mV/km.  Deconvolve unapplied FAP/ZPK in the
frequency domain, then convert remaining voltage/field units to UNIT_MODE.
ADC counts-to-volt and receiver-gain stages are used only when present as
MTH5 coefficient filters; otherwise the log records an incomplete scale.

V7+ focuses on numerical compatibility with TRACMT, including corrected robust
scales, staged prior weights, Tukey scale iteration, and TRACMT error propagation.

Important implementation note
-----------------------------
This is a Python reconstruction guided by the TRACMT source structure and
published algorithm. It is intended to be numerically compatible in workflow and
file format. V8 adds the C++ std::mt19937_64 engine, strict full-refit bootstrap, and LAPACK-style complex solves. V10 embeds the complete TRACMT Tukey b/c table and uses the exact C++ interpolation and scale update. V11 additionally matches short-section zero padding, removes unintended segment demeaning, uses the TRACMT seed 1234 mt19937_64 candidate stream, and reproduces the C++ ordinary-RR coherence rejection definition. Bit identity still depends on using the same TRACMT build, compiler standard library, input precision, and preprocessing settings.

Author: generated for Ping-Yu Chang
"""
from __future__ import annotations

import dataclasses as dc
import json
import math
import os
import re
import sys
import time
import traceback
import builtins
import io
import importlib
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
_USUI_NONROBUST_SEGMENT = ";".join(
    f"{L}:2,3,4" for L in (65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128)
)


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
        self.write_log(f"Start {program} Version pyTRACMT-V1.1", elapsed=False)

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
    apply_mth5_response: bool = True
    cts_mode: str = "auto"  # auto | on | off
    cts_file: str = ""  # local full-system CTS
    rr_cts_file: str = ""  # independent remote receiver full-system CTS
    unit_mode: str = "mth5_tracmt_practical"
    input_unit_mode: str = "auto_from_mth5"
    # Optional GUI/engine override: force coil FAP even when filter.applied=True
    mth5_force_coil_cal: bool = False

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
        elif key == "CTS_FILE":
            p.cts_file = lines[i].strip().strip(chr(34)); i += 1
        elif key == "RR_CTS_FILE":
            p.rr_cts_file = lines[i].strip().strip(chr(34)); i += 1
        elif key == "CTS_MODE":
            p.cts_mode = lines[i].strip().lower(); i += 1
            if p.cts_mode not in ("auto", "on", "off"):
                raise ValueError("CTS_MODE must be auto, on, or off")
            p.apply_mth5_response = p.cts_mode != "off"
        elif key == "APPLY_MTH5_RESPONSE":
            p.apply_mth5_response = int(float(lines[i].split()[0])) != 0; i += 1
        elif key == "MTH5_FORCE_COIL_CAL":
            p.mth5_force_coil_cal = int(float(lines[i].split()[0])) != 0; i += 1
        elif key == "INPUT_UNIT_MODE":
            p.input_unit_mode = lines[i].strip().lower(); i += 1
        elif key == "UNIT_MODE":
            p.unit_mode = lines[i].strip().lower(); i += 1
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
def read_channel(path: Path | str, skip: int, n: int) -> np.ndarray:
    """Read one numeric channel robustly.

    Supports whitespace or CSV files, optional headers, multi-column files,
    and MTH5 channel specs (MTH5::<file>::station::run::channel).
    If multiple numeric columns exist, the last numeric column is used, matching
    the earlier pyTRACMT behavior.
    """
    text = str(path).strip().strip('"').strip("'")
    if is_mth5_spec(text):
        arr, _meta = read_mth5_channel_array(text, skip=skip, n=n)
        if arr.size == 0:
            raise ValueError(f"No numeric samples read from MTH5 spec {text}")
        return arr.astype(float)

    path = _candidate_existing_path(Path(text))
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
    For MTH5 specs, returns the underlying .h5/.mth5 file path.
    """
    text = str(fname).strip().strip('"').strip("'")
    text = os.path.expandvars(os.path.expanduser(text))
    if is_mth5_spec(text):
        info = parse_mth5_spec(text)
        mp = Path(info["path"])
        if mp.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(mp)):
            return _candidate_existing_path(mp)
        return _candidate_existing_path(param_path.parent / mp)
    # Windows absolute paths are not considered absolute by pathlib on Linux,
    # but they are absolute when the script is executed on Windows.
    if Path(text).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text):
        return _candidate_existing_path(Path(text))
    return _candidate_existing_path(param_path.parent / text)


def _resolve_channel_source(param_path: Path, fname: str) -> str:
    """Return a readable source string for ascii/CSV path or absolute MTH5 spec."""
    text = str(fname).strip().strip('"').strip("'")
    text = os.path.expandvars(os.path.expanduser(text))
    if is_mth5_spec(text):
        info = parse_mth5_spec(text)
        mp = Path(info["path"])
        if not (mp.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(mp))):
            mp = _candidate_existing_path(param_path.parent / mp)
        else:
            mp = _candidate_existing_path(mp)
        return make_mth5_spec(mp, info["station"], info["run"], info["channel"], info.get("survey"))
    return str(_resolve_data_path(param_path, text))


def load_data(param_path: str | Path, p: Params, out: OutputFiles) -> List[np.ndarray]:
    """Load all DATA_FILES sections.

    This is intentionally TRACMT-compatible rather than split()-based:
    paths may contain spaces, each channel has its own skip count, and
    all channels in a section are trimmed to the same minimum length.
    Supports ASCII/CSV files and MTH5 channel specs.
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
            src = _resolve_channel_source(param_path, fname)
            arr = read_channel(src, skip=skip, n=ndata)
            cols.append(arr)
            names.append(str(src))
            out.write_log(f"  channel {ich}: {src} skip={skip} samples={len(arr)}", elapsed=False)
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
    # V11.3.1 memory-safe Yule-Walker autocorrelation.
    # Only lags 0..max_order are required.  scipy.signal.correlate(..., mode="full")
    # may select FFT convolution and allocate several arrays proportional to the full
    # record length, causing std::bad_alloc for long MTH5 runs.  Direct dot products
    # preserve the same biased normalization used previously (/N) without the FFT workspace.
    n_y = max(len(y), 1)
    ac = np.empty(max_order + 1, dtype=np.float64)
    ac[0] = float(np.dot(y, y)) / n_y
    for lag in range(1, max_order + 1):
        ac[lag] = float(np.dot(y[:-lag], y[lag:])) / n_y
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
# MTH5 input and induction-coil calibration extraction
# -----------------------------------------------------------------------------
MTH5_SPEC_PREFIX = "MTH5::"
_MTH5_EXT_RE = re.compile(r"(?i)(.+?\.(?:h5|hdf5|mth5))(?:::(.+))?$")


def is_mth5_spec(text: str) -> bool:
    s = str(text or "").strip().strip('"').strip("'")
    return s.upper().startswith(MTH5_SPEC_PREFIX) or bool(re.search(r"(?i)\.(?:h5|hdf5|mth5)::", s))


def make_mth5_spec(path: str | Path, station: str, run: str, channel: str,
                   survey: Optional[str] = None) -> str:
    parts = [str(Path(path)), str(station), str(run), str(channel)]
    if survey:
        parts = [str(Path(path)), str(survey), str(station), str(run), str(channel)]
    return MTH5_SPEC_PREFIX + "::".join(parts)


def parse_mth5_spec(spec: str) -> Dict[str, Optional[str]]:
    """Parse MTH5::<file>::[survey::]station::run::channel."""
    s = str(spec).strip().strip('"').strip("'")
    if s.upper().startswith(MTH5_SPEC_PREFIX):
        s = s[len(MTH5_SPEC_PREFIX):]
    m = _MTH5_EXT_RE.match(s)
    if not m:
        raise ValueError(
            "Invalid MTH5 channel spec. Expected "
            "MTH5::<file.h5>::<station>::<run>::<channel> "
            f"(got {spec!r})"
        )
    path = Path(m.group(1).strip().strip('"').strip("'"))
    rest = [p for p in (m.group(2) or "").split("::") if p != ""]
    if len(rest) == 3:
        survey, station, run, channel = None, rest[0], rest[1], rest[2]
    elif len(rest) == 4:
        survey, station, run, channel = rest[0], rest[1], rest[2], rest[3]
    else:
        raise ValueError(
            "MTH5 spec must end with station::run::channel or "
            f"survey::station::run::channel (got {spec!r})"
        )
    return {
        "path": str(path),
        "survey": survey,
        "station": station,
        "run": run,
        "channel": channel,
    }


def _attr_to_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    return str(val)


def _attr_to_bool_list(val) -> List[bool]:
    if val is None:
        return []
    if isinstance(val, (bool, np.bool_)):
        return [bool(val)]
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for v in list(val):
            if isinstance(v, (bytes, str)):
                out.append(str(v).strip().lower() in ("1", "true", "t", "yes", "y"))
            else:
                out.append(bool(v))
        return out
    if isinstance(val, (bytes, str)):
        txt = _attr_to_text(val).strip()
        if txt.startswith("["):
            try:
                import ast
                return _attr_to_bool_list(ast.literal_eval(txt))
            except Exception:
                pass
        return [x.strip().lower() in ("1", "true", "t", "yes", "y")
                for x in re.split(r"[,;\s]+", txt) if x.strip()]
    return [bool(val)]


def _attr_to_name_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [_attr_to_text(v).strip() for v in list(val) if _attr_to_text(v).strip()]
    txt = _attr_to_text(val).strip()
    if not txt:
        return []
    if txt.startswith("["):
        try:
            import ast
            return _attr_to_name_list(ast.literal_eval(txt))
        except Exception:
            pass
    return [x.strip().strip("'\"") for x in re.split(r"[,;]+", txt) if x.strip()]


def _dtype_is_integer(dt) -> bool:
    try:
        return bool(np.issubdtype(dt, np.integer))
    except Exception:
        txt = str(dt or "").lower()
        return ("int" in txt) and ("float" not in txt) and ("complex" not in txt)


def _mt_metadata_utf8_open_patch():
    """Force UTF-8 when mt_metadata opens packaged JSON on CP950 Windows."""
    original_builtin_open = builtins.open
    original_io_open = io.open

    def _needs_utf8(file, mode, kwargs):
        try:
            name = os.fspath(file)
        except Exception:
            return False
        if not isinstance(name, str):
            return False
        norm = name.replace("\\", "/").lower()
        return (
            "mt_metadata" in norm
            and norm.endswith(".json")
            and "b" not in str(mode)
            and "r" in str(mode)
            and kwargs.get("encoding") is None
        )

    def patched_builtin_open(file, mode="r", *args, **kwargs):
        if _needs_utf8(file, mode, kwargs):
            kwargs["encoding"] = "utf-8"
        return original_builtin_open(file, mode, *args, **kwargs)

    def patched_io_open(file, mode="r", *args, **kwargs):
        if _needs_utf8(file, mode, kwargs):
            kwargs["encoding"] = "utf-8"
        return original_io_open(file, mode, *args, **kwargs)

    class _Patch:
        def __enter__(self):
            builtins.open = patched_builtin_open
            io.open = patched_io_open
            return self

        def __exit__(self, exc_type, exc, tb):
            builtins.open = original_builtin_open
            io.open = original_io_open
            return False

    return _Patch()


def _import_mth5_class():
    with _mt_metadata_utf8_open_patch():
        return importlib.import_module("mth5.mth5").MTH5


def _parse_filter_ref_blob(val) -> Tuple[List[str], List[bool]]:
    """Parse AppliedFilter list / JSON / HDF5 attribute into names + applied flags."""
    names: List[str] = []
    applied: List[bool] = []
    if val is None:
        return names, applied
    items = val
    if isinstance(val, (bytes, str)):
        txt = _attr_to_text(val).strip()
        if not txt:
            return names, applied
        for loader in (json.loads,):
            try:
                items = loader(txt)
                break
            except Exception:
                items = txt
        else:
            try:
                import ast
                items = ast.literal_eval(txt)
            except Exception:
                items = txt
    if isinstance(items, dict):
        inner = items.get("filters") or items.get("applied_filter") or items
        if isinstance(inner, list):
            items = inner
        else:
            nm = inner.get("name") if isinstance(inner, dict) else None
            if nm:
                names.append(str(nm))
                applied.append(bool(inner.get("applied", False)) if isinstance(inner, dict) else False)
                return names, applied
    if not isinstance(items, (list, tuple, np.ndarray)):
        return names, applied
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            inner = item.get("applied_filter") if "applied_filter" in item else item
            if not isinstance(inner, dict):
                inner = item
            nm = inner.get("name") or item.get("name")
            if not nm:
                continue
            names.append(str(nm))
            applied.append(bool(inner.get("applied", item.get("applied", False))))
        else:
            nm = getattr(item, "name", None)
            if nm:
                names.append(str(nm))
                applied.append(bool(getattr(item, "applied", False)))
    return names, applied


def _channel_filter_refs(md=None, attrs=None) -> Tuple[List[str], List[bool]]:
    """Read filter names and applied flags from mt_metadata or HDF5 attrs.

    New mt_metadata uses ``filters: list[AppliedFilter]``.
    Older files use ``filter.name`` / ``filter.applied``.
    """
    names: List[str] = []
    applied: List[bool] = []
    if md is not None:
        try:
            if getattr(md, "filters", None):
                names, applied = _parse_filter_ref_blob(md.filters)
        except Exception:
            names, applied = [], []
        if not names:
            try:
                names = [str(x) for x in (getattr(md, "filter_names", None) or []) if str(x).strip()]
            except Exception:
                names = []
            try:
                applied = [bool(getattr(f, "applied", False)) for f in (getattr(md, "filters", None) or [])]
            except Exception:
                applied = []
        if not names:
            filt = getattr(md, "filter", None)
            if filt is not None:
                names = list(getattr(filt, "name", []) or [])
                applied = [bool(x) for x in list(getattr(filt, "applied", []) or [])]
    if (not names) and attrs is not None:
        names = _attr_to_name_list(
            attrs.get("filter.name", attrs.get("channel.filter.name", attrs.get("filter_name", [])))
        )
        applied = _attr_to_bool_list(
            attrs.get("filter.applied", attrs.get("channel.filter.applied", attrs.get("filter_applied", [])))
        )
        if not names:
            blob = attrs.get("filters", attrs.get("channel.filters", None))
            n2, a2 = _parse_filter_ref_blob(blob)
            if n2:
                names, applied = n2, a2
    while len(applied) < len(names):
        applied.append(False)
    return [str(n).strip() for n in names if str(n).strip()], [bool(a) for a in applied[:len(names)]]


def _instrument_fap_still_in_samples(names: Sequence[str], applied: Sequence[bool],
                                     tables: Sequence["MTH5FilterTable"]) -> bool:
    """True when attached FAP/ZPK must still be removed from the stored waveform.

    Filters present in metadata are not the same as a physical nT / mV/km series.
    applied=False (Phoenix v21) means the hardware response is still in the samples.
    applied=True on every FAP, with no unapplied tables, means deconvolution was
    already done (or the writer claims the hardware stage should not be removed).
    """
    fap_tables = [t for t in tables if str(t.kind).lower() in
                  ("fap", "zpk", "frequencyresponsetable", "frequency_response_table")]
    if fap_tables:
        by_name = {_filter_name_key(n): bool(a) for n, a in zip(names, applied)}
        flags = []
        for t in fap_tables:
            flags.append(bool(by_name.get(_filter_name_key(t.name), False)))
        # Any unapplied FAP, or FAP found without applied=True, stays in the data.
        return (not flags) or (not all(flags))
    if names:
        return not (applied and all(bool(a) for a in list(applied)[:len(names)]))
    return False


def _mt_filter_object_to_table(filt) -> Optional["MTH5FilterTable"]:
    if filt is None:
        return None
    name = str(getattr(filt, "name", "") or "filter")
    kind = str(getattr(filt, "type", "") or getattr(filt, "_filter_type", "") or "").lower()
    uin = str(getattr(filt, "units_in", "") or "")
    uout = str(getattr(filt, "units_out", "") or "")
    try:
        gain = complex(float(getattr(filt, "gain", 1.0) or 1.0))
    except Exception:
        gain = 1.0 + 0j
    if kind in ("fap", "frequency response table", "frequency_response_table", "frequencyresponsetable"):
        freqs = np.asarray(getattr(filt, "frequencies", []), dtype=float)
        amps = np.asarray(getattr(filt, "amplitudes", []), dtype=float)
        phases = np.asarray(getattr(filt, "phases", []), dtype=float)
        if freqs.size == 0:
            return None
        cplx = np.asarray(amps, float) * np.exp(1j * _phase_to_rad(phases))
        return MTH5FilterTable(name, "fap", freqs=freqs, complex_resp=cplx, gain=gain,
                               units_in=uin, units_out=uout)
    if kind in ("coefficient", "scalar"):
        return MTH5FilterTable(name, "coefficient", gain=gain, units_in=uin, units_out=uout)
    if kind in ("zpk", "poles_zeros", "polezero", "poleszeros"):
        poles = np.asarray(getattr(filt, "poles", []), dtype=complex)
        zeros = np.asarray(getattr(filt, "zeros", []), dtype=complex)
        return MTH5FilterTable(name, "zpk", poles=poles, zeros=zeros, gain=gain,
                               units_in=uin, units_out=uout)
    if kind in ("time_delay", "time delay", "delay", "timedelay"):
        delay = float(getattr(filt, "delay", getattr(filt, "time_delay", np.real(gain))))
        return MTH5FilterTable(name, "delay", gain=complex(delay), units_in=uin, units_out=uout)
    return None


def _fap_arrays_from_node(node) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Read frequency/amplitude/phase from official or ad-hoc FAP HDF5 nodes."""
    if node is None:
        return None, None, None
    if "fap_table" in getattr(node, "keys", lambda: [])():
        try:
            arr = node["fap_table"][()]
            return (np.asarray(arr["frequency"], float),
                    np.asarray(arr["amplitude"], float),
                    np.asarray(arr["phase"], float))
        except Exception:
            try:
                arr = np.asarray(node["fap_table"][()])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, 0], arr[:, 1], arr[:, 2]
            except Exception:
                pass
    for fk, ak, pk in (
        ("frequencies", "amplitudes", "phases"),
        ("frequency", "amplitude", "phase"),
        ("f", "a", "p"),
    ):
        if fk in node and ak in node and pk in node:
            return (np.asarray(node[fk][()], dtype=float),
                    np.asarray(node[ak][()], dtype=float),
                    np.asarray(node[pk][()], dtype=float))
    for dk in ("fap", "table", "data", "fap_table"):
        if dk in node:
            arr = np.asarray(node[dk][()])
            if getattr(arr, "dtype", None) is not None and arr.dtype.names:
                names = {n.lower(): n for n in arr.dtype.names}
                if "frequency" in names and "amplitude" in names and "phase" in names:
                    return (np.asarray(arr[names["frequency"]], float),
                            np.asarray(arr[names["amplitude"]], float),
                            np.asarray(arr[names["phase"]], float))
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, 0], arr[:, 1], arr[:, 2]
    return None, None, None


def _phase_to_rad(phase: np.ndarray) -> np.ndarray:
    ph = np.asarray(phase, dtype=float)
    if ph.size == 0:
        return ph
    if np.nanmax(np.abs(ph)) > (2.0 * math.pi + 0.5):
        return np.deg2rad(ph)
    return ph


def _is_coil_sensor_text(*parts: object) -> bool:
    blob = " ".join(_attr_to_text(p).lower() for p in parts if p is not None)
    # Keep "phoenix" out of automatic coil tagging: Phoenix systems may use coils or fluxgates.
    keys = ("induction", "coil", "searchcoil", "search coil", "bf-", "bf4", "bf7",
            "mtsensor", "ant/4", "ant4", "geode", "lemisi", "mfs06", "mfs07", "mfs10",
            "cm11", "cm16", "shft", "inductioncoil", "mtc-1", "mtc1", "amtc")
    return any(k in blob for k in keys)


def _is_fluxgate_sensor_text(*parts: object) -> bool:
    blob = " ".join(_attr_to_text(p).lower() for p in parts if p is not None)
    keys = ("fluxgate", "flux gate", "flux-gate", "bartington", "mag-03", "mag03",
            "mag-13", "mag13", "fgm", "ringcore", "ring-core", "tffg", "tfm", "susemo")
    return any(k in blob for k in keys)


def _looks_like_fap_name(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in (
        "fap", "coil", "induction", "sensor", "magnetometer", "hx", "hy", "hz",
        "rxcal", "scal", "lowpass", "phoenix", "mtu-5c", "mtu-5a", "mtu5a", "clb", "clc", "mtc-",
    ))


def tracmt_target_units(unit_mode: str) -> Tuple[str, str]:
    """Return (electric_unit, magnetic_unit) expected by TRACMT formulas for unit_mode."""
    mode = (unit_mode or "mth5_tracmt_practical").strip().lower()
    if mode in ("si_b_tesla", "si-b", "sib", "si_b"):
        return "v/m", "t"
    if mode in ("si_h_ampere", "si-h", "sih", "si_h"):
        return "v/m", "a/m"
    # Default TRACMT/Usui practical units used by rho_a = 0.2 T |Z|^2
    return "mv/km", "nt"


def _normalize_unit_token(units: str) -> str:
    """Canonicalize MTH5/mt_metadata unit strings.

    Handles forms like 'milliVolt per kilometer', 'nanoTesla', 'mV/km'.
    """
    u = (units or "").strip().lower()
    u = u.replace("μ", "u").replace("µ", "u")
    u = re.sub(r"[\s_]+", "", u)
    # Keep slash forms; also accept 'per' spellings before slash rewrite.
    aliases = {
        "mv/km": "mv/km",
        "millivolt/kilometer": "mv/km",
        "millivolts/kilometer": "mv/km",
        "millivoltperkilometer": "mv/km",
        "millivoltsperkilometer": "mv/km",
        "mvolt/km": "mv/km",
        "mvperkm": "mv/km",
        "uv/m": "uv/m",
        "microvolt/meter": "uv/m",
        "microvolts/meter": "uv/m",
        "microvoltpermeter": "uv/m",
        "microvoltspermeter": "uv/m",
        "v/m": "v/m",
        "volt/meter": "v/m",
        "volts/meter": "v/m",
        "voltpermeter": "v/m",
        "voltspermeter": "v/m",
        "v": "v", "volt": "v", "volts": "v",
        "mv": "mv", "millivolt": "mv", "millivolts": "mv",
        "nt": "nt", "nanotesla": "nt", "nanoteslas": "nt",
        "pt": "pt", "picotesla": "pt", "picoteslas": "pt",
        "t": "t", "tesla": "t",
        "a/m": "a/m",
        "ampere/meter": "a/m", "amperes/meter": "a/m",
        "amperepermeter": "a/m", "amperespermeter": "a/m",
        "count": "counts", "counts": "counts", "digitalcounts": "counts",
        "digitalcount": "counts",
    }
    if u in aliases:
        return aliases[u]
    # 'millivoltperkilometer' style after removing spaces, or with 'per' left in.
    u2 = u.replace("per", "/")
    if u2 in aliases:
        return aliases[u2]
    # Collapse repeated separators
    u3 = re.sub(r"/+", "/", u2)
    return aliases.get(u3, u3)


def _electric_field_scale(from_unit: str, to_unit: str, dipole_m: Optional[float] = None) -> Tuple[float, str]:
    """Scale factor: data_to = data_from * scale. Voltage needs dipole length in meters."""
    src = _normalize_unit_token(from_unit)
    dst = _normalize_unit_token(to_unit)
    # Treat mV/km and uV/m as identical (common MT convention).
    if src == "uv/m":
        src = "mv/km"
    if dst == "uv/m":
        dst = "mv/km"

    # Convert source to V/m first.
    if src in ("v", "mv"):
        if dipole_m is None or not np.isfinite(dipole_m) or dipole_m <= 0:
            raise ValueError(
                f"Electric channel unit is {from_unit!r} (voltage); dipole_length [m] is required "
                "to convert to field units for TRACMT."
            )
        # V/m = V / L_m ; if millivolts, /1000 first
        volt_scale = 1.0 if src == "v" else 1e-3
        to_v_per_m = volt_scale / float(dipole_m)
        note = f"voltage {from_unit} / dipole {dipole_m:g} m -> V/m"
    elif src == "v/m":
        to_v_per_m = 1.0
        note = "already V/m"
    elif src == "mv/km":
        # 1 mV/km = 1e-6 V/m
        to_v_per_m = 1e-6
        note = "mV/km -> V/m"
    else:
        raise ValueError(f"Unsupported electric unit {from_unit!r}; cannot convert to {to_unit!r}")

    if dst == "v/m":
        return to_v_per_m, note + f" -> {to_unit}"
    if dst == "mv/km":
        # 1 V/m = 1e6 mV/km
        return to_v_per_m * 1e6, note + f" -> {to_unit}"
    raise ValueError(f"Unsupported target electric unit {to_unit!r}")


def _magnetic_field_scale(from_unit: str, to_unit: str) -> Tuple[float, str]:
    """Scale factor among T / nT / pT / A/m. data_to = data_from * scale."""
    src = _normalize_unit_token(from_unit)
    dst = _normalize_unit_token(to_unit)

    def to_tesla(u: str) -> float:
        if u == "t":
            return 1.0
        if u == "nt":
            return 1e-9
        if u == "pt":
            return 1e-12
        if u == "a/m":
            return MU0  # B = mu0 * H
        raise ValueError(f"Unsupported magnetic unit {u!r}")

    s = to_tesla(src)
    d = to_tesla(dst)
    return s / d, f"{from_unit} -> {to_unit}"


def _coefficient_gain_as_v_per_nt(tab: "MTH5FilterTable") -> Optional[float]:
    """Interpret a coefficient filter as magnetic sensitivity in V/nT when possible."""
    if tab is None or tab.kind not in ("coefficient", "scalar"):
        return None
    gain = float(np.real(tab.gain))
    if not np.isfinite(gain) or abs(gain) <= EPS:
        return None
    u_in = _normalize_unit_token(tab.units_in)
    u_out = _normalize_unit_token(tab.units_out)
    # Preferred mt_metadata convention: units_in=nT, units_out=V, gain=V/nT
    if u_in in ("nt", "t", "pt") and u_out in ("v", "mv"):
        g = gain
        if u_out == "mv":
            g *= 1e-3
        if u_in == "t":
            g *= 1e9  # V/T -> V/nT
        elif u_in == "pt":
            g *= 1e-3  # V/pT -> V/nT
        return g
    # Inverse convention: units_in=V, units_out=nT, gain=nT/V
    if u_out in ("nt", "t", "pt") and u_in in ("v", "mv"):
        g = gain
        if u_in == "mv":
            # (nT/mV) -> treat input as mV: B = V_mV * g => for volts, *1000
            inv = g * 1000.0  # nT/V
        else:
            inv = g  # nT/V or T/V etc
        if u_out == "t":
            inv *= 1e9
        elif u_out == "pt":
            inv *= 1e-3
        if abs(inv) <= EPS:
            return None
        return 1.0 / inv
    # Unknown unit labels: assume V/nT if |gain| is typical (1e-5 .. 1)
    if abs(gain) < 1.0 and abs(gain) > 1e-8:
        return gain
    # Large gain often nT/V
    if abs(gain) >= 1.0:
        return 1.0 / gain
    return None


def classify_mth5_magnetic_sensor(meta: Dict[str, object], tables: Sequence["MTH5FilterTable"]) -> str:
    """Return 'coil', 'fluxgate', or 'unknown'.

    FAP/ZPK tables mean a frequency-dependent coil/receiver chain even when
    channel.units happens to say nT.  Physical-looking units are not enough
    to classify a channel as a fluxgate.
    """
    kinds = {str(getattr(t, "kind", "") or "").lower() for t in tables}
    if any(k in ("fap", "zpk", "frequencyresponsetable", "frequency_response_table") for k in kinds):
        return "coil"
    sensor = str(meta.get("sensor_type", ""))
    model = str(meta.get("sensor_model", ""))
    manuf = str(meta.get("sensor_manufacturer", ""))
    names = " ".join(str(x) for x in (meta.get("filter_names") or []))
    if _is_fluxgate_sensor_text(sensor, model, manuf, names):
        return "fluxgate"
    if _is_coil_sensor_text(sensor, model, manuf, names):
        return "coil"
    if "coefficient" in kinds or "scalar" in kinds:
        return "fluxgate"
    units = _normalize_unit_token(str(meta.get("units", "")))
    if units in ("nt", "t", "pt", "a/m"):
        return "fluxgate"
    return "unknown"


def _interp_extrap_linear(xp, fp, x):
    """1-D linear interpolation with true linear extrapolation at both ends."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(xp) & np.isfinite(fp)
    xp, fp = xp[good], fp[good]
    if xp.size == 0:
        return np.full(x.shape, np.nan, dtype=float)
    order = np.argsort(xp)
    xp, fp = xp[order], fp[order]
    ux, ui = np.unique(xp, return_index=True)
    xp, fp = ux, fp[ui]
    if xp.size == 1:
        return np.full(x.shape, fp[0], dtype=float)
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    slo = (fp[1]-fp[0]) / max(xp[1]-xp[0], EPS)
    shi = (fp[-1]-fp[-2]) / max(xp[-1]-xp[-2], EPS)
    y = np.asarray(y, dtype=float)
    y[lo] = fp[0] + slo*(x[lo]-xp[0])
    y[hi] = fp[-1] + shi*(x[hi]-xp[-1])
    return y


class MTH5FilterTable:
    """One instrument filter that can be turned into a TRACMT calibration table."""

    def __init__(self, name: str, kind: str, freqs: Optional[np.ndarray] = None,
                 complex_resp: Optional[np.ndarray] = None, gain: complex = 1.0 + 0j,
                 poles: Optional[np.ndarray] = None, zeros: Optional[np.ndarray] = None,
                 units_in: str = "", units_out: str = ""):
        self.name = name
        self.kind = kind
        self.freqs = None if freqs is None else np.asarray(freqs, dtype=float)
        self.complex_resp = None if complex_resp is None else np.asarray(complex_resp, dtype=complex)
        self.gain = complex(gain)
        self.poles = np.asarray(poles, dtype=complex) if poles is not None else np.array([], dtype=complex)
        self.zeros = np.asarray(zeros, dtype=complex) if zeros is not None else np.array([], dtype=complex)
        self.units_in = units_in
        self.units_out = units_out

    def response_on(self, freqs: np.ndarray) -> np.ndarray:
        f = np.asarray(freqs, dtype=float)
        if self.kind in ("coefficient", "scalar"):
            return np.full(f.shape, self.gain, dtype=complex)
        if self.kind in ("fap", "frequencyresponsetable", "frequency_response_table"):
            if self.freqs is None or self.complex_resp is None or len(self.freqs) == 0:
                return np.full(f.shape, self.gain, dtype=complex)
            lf = np.log10(np.maximum(self.freqs, EPS))
            x = np.log10(np.maximum(f, EPS))
            amp = np.abs(self.complex_resp)
            ph = np.unwrap(np.angle(self.complex_resp))
            logamp = np.log10(np.maximum(amp, EPS))
            # V12.4: interpolate/extrapolate in log10(frequency).  Amplitude is
            # interpolated in log10(amplitude), phase is unwrapped before linear
            # interpolation/extrapolation.  This avoids endpoint clamping when an
            # FFT frequency has no exact calibration anchor.
            ai = _interp_extrap_linear(lf, logamp, x)
            pi = _interp_extrap_linear(lf, ph, x)
            return self.gain * (10.0 ** ai) * np.exp(1j * pi)
        if self.kind in ("zpk", "polezero", "poleszeros"):
            w = 2.0 * math.pi * f
            s = 1j * w
            num = np.ones(f.shape, dtype=complex)
            den = np.ones(f.shape, dtype=complex)
            for z in self.zeros:
                num *= (s - z)
            for p in self.poles:
                den *= (s - p)
            return self.gain * num / np.where(np.abs(den) > EPS, den, EPS)
        if self.kind in ("delay", "timedelay"):
            # gain holds delay seconds (negative = delay). Phase-only.
            delay = float(np.real(self.gain))
            return np.exp(-1j * 2.0 * math.pi * f * delay)
        return np.ones(f.shape, dtype=complex)


_FAP_KINDS = ("fap", "zpk", "frequencyresponsetable", "frequency_response_table")
_COEFF_KINDS = ("coefficient", "scalar")
_SPECTRAL_KINDS = _FAP_KINDS + _COEFF_KINDS + ("delay", "timedelay", "time_delay")
_ELECTRIC_FIELD_UNITS = ("mv/km", "uv/m", "v/m")
_ELECTRIC_VOLTAGE_UNITS = ("v", "mv")
_MAGNETIC_FIELD_UNITS = ("nt", "t", "pt", "a/m")


def _filter_name_key(name: str) -> str:
    s = str(name or "").strip().lower().replace("/", " per ")
    return re.sub(r"[\s_]+", " ", s).strip()


def _is_fap_table(tab: "MTH5FilterTable") -> bool:
    return str(getattr(tab, "kind", "") or "").lower() in _FAP_KINDS


def _subset_tables_by_applied(tables: Sequence["MTH5FilterTable"],
                              names: Sequence[str],
                              applied: Sequence[bool],
                              force: bool) -> List["MTH5FilterTable"]:
    """Keep filters that still reside in the stored samples.

    applied=True means the writer claims that stage was already removed.
    A mounted filter with applied=False is still in the waveform.
    """
    tables = list(tables)
    if force:
        return tables
    applied_map = {_filter_name_key(n): bool(a) for n, a in zip(names, applied)}
    all_named_applied = bool(names) and bool(applied) and all(
        bool(a) for a in list(applied)[:len(names)]
    )
    kept: List[MTH5FilterTable] = []
    for tab in tables:
        key = _filter_name_key(tab.name)
        if key in applied_map:
            if not applied_map[key]:
                kept.append(tab)
            continue
        if all_named_applied:
            continue
        kept.append(tab)
    return kept


def _merge_filter_tables(*groups: Sequence["MTH5FilterTable"]) -> List["MTH5FilterTable"]:
    out: List[MTH5FilterTable] = []
    seen = set()
    for group in groups:
        for tab in group or []:
            key = _filter_name_key(tab.name)
            if key in seen:
                continue
            seen.add(key)
            out.append(tab)
    return out


def _coefficient_counts_to_volt_gain(tab: "MTH5FilterTable") -> Optional[float]:
    """Return volts-per-count if this coefficient is a digitizer/ADC stage."""
    if str(getattr(tab, "kind", "") or "").lower() not in _COEFF_KINDS:
        return None
    uin = _normalize_unit_token(str(tab.units_in or ""))
    uout = _normalize_unit_token(str(tab.units_out or ""))
    try:
        gain = float(np.real(tab.gain))
    except Exception:
        return None
    if abs(gain) <= EPS:
        return None
    if uin in ("counts",) and uout == "v":
        return gain
    if uin in ("counts",) and uout == "mv":
        return gain * 1e-3
    if uin == "v" and uout in ("counts",):
        return 1.0 / gain
    if uin == "mv" and uout in ("counts",):
        return 1e-3 / gain
    return None


_PHOENIX_E_TAG = {"ex": "E1", "ey": "E2", "ez": "E3"}
_PHOENIX_TAG = {"ex": "E1", "ey": "E2", "ez": "E3", "hx": "H1", "hy": "H2", "hz": "H3"}


def _positive_length_m(val) -> Optional[float]:
    try:
        x = float(val)
    except Exception:
        return None
    if np.isfinite(x) and x > 1e-6:
        return x
    return None


def _dipole_from_length_pair(length1, length2) -> Optional[float]:
    a = _positive_length_m(length1) or 0.0
    b = _positive_length_m(length2) or 0.0
    s = a + b
    return s if s > 1e-6 else None


def _phoenix_sidecar_search_dirs(mth5_path) -> List[Path]:
    """Folders where MTU-5C/5A recmeta.json and config.json may sit."""
    try:
        folder = Path(mth5_path).expanduser().resolve().parent
    except Exception:
        return []
    dirs = [folder, folder / "mth5_cal_files"]
    out: List[Path] = []
    seen = set()
    for d in dirs:
        try:
            key = str(d.resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _phoenix_load_sidecar_jsons(mth5_path) -> List[Tuple[dict, str]]:
    """Load recmeta.json then config.json from the MTH5 folder and mth5_cal_files."""
    found: List[Tuple[dict, str]] = []
    seen = set()
    for folder in _phoenix_sidecar_search_dirs(mth5_path):
        for name in ("recmeta.json", "config.json"):
            p = folder / name
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if key in seen or not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                seen.add(key)
                found.append((obj, str(p)))
    return found


def _phoenix_iter_chan_dicts(obj) -> List[dict]:
    chans: List[dict] = []
    try:
        chans = [c for c in ((obj.get("chconfig") or {}).get("chans") or []) if isinstance(c, dict)]
    except Exception:
        chans = []
    if not chans:
        try:
            chans = [c for c in ((obj.get("config") or [{}])[0].get("chanconfig") or []) if isinstance(c, dict)]
        except Exception:
            chans = []
    return chans


def _phoenix_lookup_chan(mth5_path, channel) -> Tuple[Optional[dict], str]:
    """Return (channel dict, source filename). recmeta.json wins over config.json."""
    tag = _PHOENIX_TAG.get(str(channel or "").strip().lower())
    if not tag:
        return None, ""
    found: Optional[dict] = None
    source = ""
    recmeta_hit: Optional[Tuple[dict, str]] = None
    for obj, src in _phoenix_load_sidecar_jsons(mth5_path):
        for c in _phoenix_iter_chan_dicts(obj):
            if str(c.get("tag") or "").upper() != tag:
                continue
            found, source = c, src
            if Path(src).name.lower() == "recmeta.json":
                recmeta_hit = (c, src)
            break
        if recmeta_hit is not None and (
            recmeta_hit[0].get("ga") is not None
            or recmeta_hit[0].get("pg") is not None
            or recmeta_hit[0].get("length1") is not None
            or recmeta_hit[0].get("dipole_length_m") is not None
        ):
            return recmeta_hit
    return found, source


def _phoenix_linear_gain(rec: Optional[dict]) -> Optional[float]:
    """Phoenix recmeta ga and pg are linear amplifier factors, not dB."""
    if not rec:
        return None
    g = 1.0
    n = 0
    for key in ("ga", "pg"):
        try:
            v = float(rec.get(key))
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            g *= v
            n += 1
    return float(g) if n else None


def _phoenix_dipole_from_json(obj, tag: str) -> Optional[float]:
    tag_u = str(tag or "").upper()
    mt_ch = {"E1": "Ex", "E2": "Ey"}.get(tag_u, tag_u)
    geom = obj.get("geometry") if isinstance(obj, dict) else None
    if isinstance(geom, dict):
        for key in (mt_ch, tag_u, mt_ch.lower(), tag_u.lower()):
            g = geom.get(key)
            if isinstance(g, dict):
                dlen = _positive_length_m(g.get("dipole_length_m"))
                if dlen:
                    return dlen
                dlen = _dipole_from_length_pair(g.get("length1"), g.get("length2"))
                if dlen:
                    return dlen
    esp = obj.get("electrode_spacing_m") if isinstance(obj, dict) else None
    if isinstance(esp, dict):
        if tag_u == "E1":
            dlen = _positive_length_m(esp.get("Ex_EXLN") or esp.get("EXLN"))
            if dlen:
                return dlen
        if tag_u == "E2":
            dlen = _positive_length_m(esp.get("Ey_EYLN") or esp.get("EYLN"))
            if dlen:
                return dlen
    for c in _phoenix_iter_chan_dicts(obj):
        if str(c.get("tag") or "").upper() != tag_u:
            continue
        d = c.get("dipole") or {}
        try:
            de = float(d.get("PosEast", 0) or 0) - float(d.get("NegEast", 0) or 0)
            dn = float(d.get("PosNorth", 0) or 0) - float(d.get("NegNorth", 0) or 0)
            hypot = float(np.hypot(de, dn))
            if hypot > 1e-6:
                return hypot
        except Exception:
            pass
        dlen = _positive_length_m(c.get("dipole_length_m"))
        if dlen:
            return dlen
        return _dipole_from_length_pair(c.get("length1"), c.get("length2"))
    return None


def _phoenix_dipole_from_sidecars(mth5_path, channel) -> Optional[float]:
    """Recover Phoenix dipole length from recmeta.json / config.json beside the MTH5."""
    tag = _PHOENIX_E_TAG.get(str(channel or "").strip().lower())
    if not tag:
        return None
    for obj, _src in _phoenix_load_sidecar_jsons(mth5_path):
        got = _phoenix_dipole_from_json(obj, tag)
        if got:
            return got
    return None


def _finalize_electric_dipole_meta(meta: Dict[str, object], path, info: Dict[str, Optional[str]]) -> None:
    dlen = _positive_length_m(meta.get("dipole_length"))
    if dlen is None:
        dlen = _dipole_from_length_pair(meta.get("length1"), meta.get("length2"))
    ch = info.get("channel") if info else meta.get("channel")
    rec, src = _phoenix_lookup_chan(path or meta.get("path"), ch)
    if dlen is None:
        if rec:
            d = rec.get("dipole") or {}
            try:
                de = float(d.get("PosEast", 0) or 0) - float(d.get("NegEast", 0) or 0)
                dn = float(d.get("PosNorth", 0) or 0) - float(d.get("NegNorth", 0) or 0)
                hypot = float(np.hypot(de, dn))
                if hypot > 1e-6:
                    dlen = hypot
            except Exception:
                pass
            if dlen is None:
                dlen = _positive_length_m(rec.get("dipole_length_m"))
                if dlen is None:
                    dlen = _dipole_from_length_pair(rec.get("length1"), rec.get("length2"))
        if dlen is None:
            dlen = _phoenix_dipole_from_sidecars(path or meta.get("path"), ch)
        if dlen:
            meta["dipole_source"] = f"phoenix {src or 'sidecar'} length1+length2"
    meta["dipole_length"] = dlen
    stored = str((rec or {}).get("stored_units") or (rec or {}).get("units") or "")
    if "count" in stored.lower():
        meta["units"] = "digital counts"
    else:
        for obj, _src in _phoenix_load_sidecar_jsons(path or meta.get("path")):
            top = str(obj.get("stored_units") or "")
            if "count" in top.lower():
                meta["units"] = "digital counts"
                break
    g = _phoenix_linear_gain(rec)
    if g is not None:
        meta["receiver_gain"] = g
        try:
            meta["receiver_ga"] = float(rec.get("ga")) if rec.get("ga") is not None else None
        except Exception:
            meta["receiver_ga"] = rec.get("ga") if rec else None
        try:
            meta["receiver_pg"] = float(rec.get("pg")) if rec.get("pg") is not None else None
        except Exception:
            meta["receiver_pg"] = rec.get("pg") if rec else None
        meta["receiver_gain_source"] = f"phoenix {src}"


def _apply_phoenix_receiver_gain(plan: Dict[str, object], meta: Dict[str, object],
                                 voltage_like: bool) -> None:
    """Remove analog ga*pg. RXCAL is normalized and does not include this scalar.

    Recorded voltage is after the preamp: V_input = V_recorded / (ga*pg).
    This makes E smaller if pg>1; it does not invent an extra 1e6 scale.
    """
    if not voltage_like:
        return
    try:
        g = float(meta.get("receiver_gain"))
    except Exception:
        return
    if not np.isfinite(g) or g <= 0 or abs(g - 1.0) <= 1e-12:
        return
    plan["time_domain_scale"] = float(plan.get("time_domain_scale", 1.0) or 1.0) / g
    plan["notes"].append(
        f"receiver analog gain ga={meta.get('receiver_ga')} pg={meta.get('receiver_pg')} "
        f"(linear {g:g}) removed: V_input=V_recorded/gain; "
        f"RXCAL is normalized and does not include ga/pg "
        f"({meta.get('receiver_gain_source') or 'sidecar'})"
    )


def _infer_electric_working_unit(nu: str, meta: Dict[str, object],
                                 dipole_m: Optional[float]) -> Tuple[str, Optional[str]]:
    """Working unit of stored electric samples, plus an assumption note if needed."""
    if nu in _ELECTRIC_FIELD_UNITS or nu in _ELECTRIC_VOLTAGE_UNITS or nu == "counts":
        return nu, None
    integer_samples = bool(meta.get("integer_samples")) or _dtype_is_integer(meta.get("storage_dtype"))
    if integer_samples:
        return "counts", None
    if dipole_m is not None and np.isfinite(dipole_m) and dipole_m > 0:
        return "mv", (
            "unlabeled float electric samples assumed millivolt (Phoenix TD-style); "
            "dipole_length used to reach field units"
        )
    return nu, "unlabeled electric samples; no voltage/field unit and no dipole"


def _first_fap_units_in(tables: Sequence["MTH5FilterTable"]) -> str:
    for tab in tables:
        if _is_fap_table(tab):
            uin = _normalize_unit_token(str(tab.units_in or ""))
            if uin:
                return uin
    return ""



def _v123_norm_key(k: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k or "").lower())


def _v123_find_first(node, aliases):
    wanted = {_v123_norm_key(x) for x in aliases}
    if isinstance(node, dict):
        for k, v in node.items():
            if _v123_norm_key(k) in wanted:
                return v
        for v in node.values():
            if isinstance(v, (dict, list)):
                got = _v123_find_first(v, aliases)
                if got is not None:
                    return got
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                got = _v123_find_first(v, aliases)
                if got is not None:
                    return got
    return None


def _v123_float_array(v):
    if v is None:
        return None
    try:
        if isinstance(v, dict):
            # Common JSON wrappers: {value:[...]}, {values:[...]}, {data:[...]}
            for key in ("value", "values", "data"):
                if key in v:
                    v = v[key]
                    break
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return a if a.size else None
    except Exception:
        return None


def _v123_json_response_records(obj, source="embedded-json"):
    """Conservatively find explicit FAP/ZPK records in a JSON object.

    V12.3 never interprets Phoenix binary RXCAL word layouts here.  A record is
    accepted only when the JSON already exposes frequency+amplitude (phase is
    optional), or explicit poles/zeros.  This prevents diagnostic JSON/state
    metadata from being mistaken for a transfer function.
    """
    out = []
    def walk(node, path="/"):
        if isinstance(node, dict):
            keys = {_v123_norm_key(k) for k in node}
            has_f = bool(keys & {"frequency","frequencies","freq","freqs","hz"})
            has_a = bool(keys & {"amplitude","amplitudes","magnitude","magnitudes","gain","gains"})
            has_pz = bool(keys & {"poles","zeros"})
            if (has_f and has_a) or has_pz:
                out.append((node, f"{source}:{path}"))
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, (dict, list)):
                    walk(v, f"{path}[{i}]")
    walk(obj)
    return out


def _v123_record_identity(rec):
    def scalar(names):
        v = _v123_find_first(rec, names)
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            return str(v).strip()
        return ""
    return {
        "serial": scalar(("serial","sensorserial","serialnumber","sensorid","id","ser")),
        "model": scalar(("model","sensormodel","modelname")),
        "tag": scalar(("tag","channel","component","channelname","input","axis")),
        "sample_rate": scalar(("samplerate","sample_rate","samplingrate","fs","rate")),
    }


def _v123_record_to_table(rec, name):
    freq = _v123_float_array(_v123_find_first(rec, ("frequency","frequencies","freq","freqs","hz")))
    amp = _v123_float_array(_v123_find_first(rec, ("amplitude","amplitudes","magnitude","magnitudes","gain","gains")))
    phase = _v123_float_array(_v123_find_first(rec, ("phase","phases","phasedeg","phase_deg","phaseangle")))
    if freq is not None and amp is not None:
        n = min(len(freq), len(amp), len(phase) if phase is not None else 10**9)
        if n >= 2:
            if phase is None:
                phase = np.zeros(n, dtype=float)
            # Phoenix/mt_metadata JSON phase is expected in degrees when exposed
            # as phase/phase_deg; convert only here, at the complex-response edge.
            order = np.argsort(freq[:n])
            f = np.asarray(freq[:n], float)[order]
            a = np.asarray(amp[:n], float)[order]
            ph = np.asarray(phase[:n], float)[order]
            good = np.isfinite(f) & np.isfinite(a) & np.isfinite(ph) & (f > 0) & (a > 0)
            if np.count_nonzero(good) >= 2:
                c = a[good] * np.exp(1j * np.radians(ph[good]))
                uin = str(_v123_find_first(rec, ("unitsin","units_in","inputunits")) or "")
                uout = str(_v123_find_first(rec, ("unitsout","units_out","outputunits")) or "")
                return MTH5FilterTable(name, "fap", freqs=f[good], complex_resp=c,
                                       units_in=uin, units_out=uout)
    # Explicit ZPK fallback.
    poles = _v123_find_first(rec, ("poles",))
    zeros = _v123_find_first(rec, ("zeros",))
    if poles is not None or zeros is not None:
        def carr(v):
            vals = v if isinstance(v, list) else ([] if v is None else [v])
            z = []
            for x in vals:
                try:
                    if isinstance(x, dict):
                        z.append(complex(float(x.get("real", x.get("re", 0))),
                                         float(x.get("imag", x.get("im", 0)))))
                    elif isinstance(x, (list, tuple)) and len(x) >= 2:
                        z.append(complex(float(x[0]), float(x[1])))
                    else:
                        z.append(complex(x))
                except Exception:
                    pass
            return np.asarray(z, complex)
        p, z = carr(poles), carr(zeros)
        if p.size or z.size:
            gv = _v123_find_first(rec, ("gain","normalizationfactor","a0","constant"))
            try: g = float(gv) if gv is not None else 1.0
            except Exception: g = 1.0
            return MTH5FilterTable(name, "zpk", poles=p, zeros=z, gain=g)
    return None


def _v123_decode_json_candidate(v):
    try:
        if isinstance(v, bytes):
            txt = v.decode("utf-8", errors="strict")
        elif isinstance(v, str):
            txt = v
        elif isinstance(v, np.ndarray) and v.dtype.kind in ("S", "U"):
            txt = "".join(str(x) for x in v.ravel())
        elif isinstance(v, np.ndarray) and v.dtype.kind in ("u", "i") and v.size <= 5_000_000:
            txt = bytes(np.asarray(v, dtype=np.uint8).ravel()).decode("utf-8", errors="strict")
        else:
            return None
        txt = txt.strip().strip("\x00")
        if not txt or txt[0] not in "[{":
            return None
        obj = json.loads(txt)
        return obj if isinstance(obj, (dict, list)) else None
    except Exception:
        return None


def _v123_embedded_json_tables(mth5_path, info, meta):
    """Recover explicit response tables from JSON blobs stored *inside* MTH5.

    Matching is intentionally strict: channel/tag match is preferred; a serial
    match is accepted when both sides expose a serial. Unidentified response
    records are not auto-attached, avoiding cross-channel RXCAL/SCAL mistakes.
    """
    import h5py
    path = Path(mth5_path)
    ch = str((info or {}).get("channel") or meta.get("channel") or "").lower()
    tags = {ch, ch.upper()}
    tags.update({"ex":"E1","ey":"E2","hx":"H1","hy":"H2","hz":"H3"}.get(ch, "") for _ in [0])
    tags = {str(x).lower() for x in tags if x}
    serials = set()
    for k in ("sensor_serial", "sensor_id", "serial", "receiver_serial", "data_logger_id"):
        v = str(meta.get(k) or "").strip()
        if v: serials.add(v.lower())
    candidates = []
    try:
        with h5py.File(path, "r") as h5:
            def inspect_obj(name, obj):
                # attrs are cheap and commonly used for embedded provenance JSON.
                for ak, av in obj.attrs.items():
                    jo = _v123_decode_json_candidate(av)
                    if jo is not None:
                        candidates.extend(_v123_json_response_records(jo, f"h5:{name}@{ak}"))
                if isinstance(obj, h5py.Dataset):
                    try:
                        if obj.size <= 5_000_000 and (obj.dtype.kind in ("S","U","O") or (obj.dtype.kind in ("u","i") and obj.ndim == 1)):
                            jo = _v123_decode_json_candidate(obj[()])
                            if jo is not None:
                                candidates.extend(_v123_json_response_records(jo, f"h5:{name}"))
                    except Exception:
                        pass
            h5.visititems(inspect_obj)
    except Exception as e:
        return [], [f"V12.3 embedded-JSON scan failed: {e}"]

    tables, notes = [], []
    rejected_unidentified = 0
    for i, (rec, src) in enumerate(candidates):
        ident = _v123_record_identity(rec)
        tag = str(ident.get("tag") or "").lower()
        ser = str(ident.get("serial") or "").lower()
        tag_match = bool(tag and (tag in tags or any(t in tag for t in tags)))
        serial_match = bool(ser and serials and ser in serials)
        if not (tag_match or serial_match):
            rejected_unidentified += 1
            continue
        tab = _v123_record_to_table(rec, f"v123_json_{ch}_{i}")
        if tab is not None:
            tables.append(tab)
            notes.append(f"V12.3 embedded JSON response accepted for {ch}: {src}; identity={ident}")
    if candidates and not tables:
        notes.append(f"V12.3 embedded JSON: {len(candidates)} response-like record(s) found but none safely matched {ch}; unidentified/mismatched={rejected_unidentified}")
    return _merge_filter_tables(tables), notes

def _plan_from_channel_meta(info: Dict[str, Optional[str]],
                            meta: Dict[str, object],
                            tables: Sequence["MTH5FilterTable"],
                            tables_all: Sequence["MTH5FilterTable"],
                            unit_mode: str,
                            force_coil: bool = False) -> Dict[str, object]:
    """Plan TD scale + spectral deconvolution from already-loaded MTH5 metadata.

    Mounted Survey/Filters objects are not a claim that samples are already
    nT or mV/km.  Unapplied FAP/ZPK are deconvolved; dipole/ADC/gain are used
    only when those stages exist in the MTH5 metadata.
    """
    target_e, target_h = tracmt_target_units(unit_mode)
    ch_type = str(meta.get("channel_type", "")).lower()
    ch_name = str(info.get("channel") or meta.get("channel") or "").lower()
    is_magnetic = ch_type.startswith("mag") or ch_name in ("hx", "hy", "hz", "bx", "by", "bz")
    is_electric = ch_type.startswith("elec") or ch_name in ("ex", "ey", "ez")
    units = str(meta.get("units", "") or "")
    nu = _normalize_unit_token(units)
    cal_state = _calibration_state(meta)
    phoenix_source_units = str(meta.get("phoenix_source_units", "") or "")
    phoenix_source_format = str(meta.get("phoenix_source_format", "") or "")
    names = [str(x) for x in (meta.get("filter_names") or [])]
    applied = [bool(x) for x in (meta.get("filter_applied") or [])]
    while len(applied) < len(names):
        applied.append(False)
    dipole_m = _positive_length_m(meta.get("dipole_length"))
    if dipole_m is None:
        dipole_m = _dipole_from_length_pair(meta.get("length1"), meta.get("length2"))

    all_tabs = _merge_filter_tables(tables_all, tables)
    v123_json_notes = []
    # V12.6: V66 direct-writer files may preserve RAW TD with blank units while
    # calibration_applied is absent/unknown.  Do not require the provenance flag
    # before looking for the archived RXCAL/SCAL JSON.  If no usable spectral
    # object was resolved from Survey/Filters and the channel is RAW-like
    # (unapplied OR unlabeled/counts/voltage-like), recover the embedded JSON.
    raw_like_for_json = (
        cal_state == "unapplied"
        or nu in ("", "counts", "v", "mv")
        or "uncalibrated" in phoenix_source_units.lower()
        or "td" in phoenix_source_format.lower()
    )
    if raw_like_for_json:
        # V12.6: always inspect the V66 embedded archive as a supplemental source.
        # Some V66 files have Survey/Filters objects but no usable per-channel
        # filter-reference attributes.  In that case V12.5 saw *some* spectral
        # objects and incorrectly skipped the JSON recovery path.
        jt, v123_json_notes = _v123_embedded_json_tables(meta.get("path"), info, meta)
        if jt:
            all_tabs = _merge_filter_tables(all_tabs, jt)
            meta["v123_embedded_json_tables"] = jt
            if cal_state == "unknown":
                cal_state = "unapplied"
                meta["v126_inferred_calibration_state"] = "unapplied_from_raw_plus_embedded_response"
    active = list(tables) if force_coil else _subset_tables_by_applied(
        all_tabs, names, applied, force=False
    )
    fap_still = bool(force_coil) or _instrument_fap_still_in_samples(names, applied, all_tabs)
    if cal_state == "unapplied":
        # Explicit writer provenance is stronger than a stale/missing per-filter applied list.
        fap_still = True
        active = list(all_tabs)
    spectral = [t for t in active if str(t.kind).lower() in _SPECTRAL_KINDS]
    if not fap_still:
        spectral = [t for t in spectral if str(t.kind).lower() not in _FAP_KINDS]

    plan: Dict[str, object] = {
        "channel": info.get("channel") or meta.get("channel"),
        "sensor_kind": "unknown",
        "input_units": units,
        "target_units": target_e if is_electric else (target_h if is_magnetic else ""),
        "time_domain_scale": 1.0,
        "post_cal_scale": 1.0,
        "spectral_tables": [],
        "notes": [],
        "meta": meta,
        "filters_mounted": bool(names or all_tabs),
        "filters_mounted_not_physical": True,
        "scale_incomplete": False,
        "fap_still_in_samples": bool(fap_still),
        "output_units": units,
        "calibration_state": cal_state,
        "phoenix_source_units": phoenix_source_units,
        "phoenix_source_format": phoenix_source_format,
    }
    plan["notes"].append(
        f"V12.3 calibration state={cal_state}; phoenix_source_units={phoenix_source_units!r}; "
        f"source_format={phoenix_source_format!r}"
    )
    plan["notes"].extend(v123_json_notes)
    if meta.get("v123_embedded_json_tables"):
        plan["notes"].append("V12.3 calibration source: embedded JSON response recovered from MTH5")
    if cal_state == "unapplied":
        plan["notes"].append(
            "V12.3 explicit calibration_applied=False overrides physical-looking channel.units; "
            "mounted response remains eligible for deconvolution"
        )
    if names or all_tabs:
        plan["notes"].append(
            f"MTH5 filters mounted {names or [t.name for t in all_tabs]} "
            f"applied={applied or [False] * len(all_tabs)}; "
            "mounted filters are NOT a claim that samples are already nT or mV/km"
        )

    if is_electric:
        plan["sensor_kind"] = "electric"
        if dipole_m:
            src = str(meta.get("dipole_source") or "MTH5 channel.dipole_length")
            plan["notes"].append(f"electric dipole {dipole_m:g} m ({src})")
        working, assumption = _infer_electric_working_unit(nu, meta, dipole_m)
        if cal_state == "unapplied" and "uncalibrated phoenix td" in phoenix_source_units.lower():
            # Converter explicitly says TD float32 is uncalibrated and its units are unknown.
            # For electric TD, use receiver-level mV only as the existing Phoenix-TD convention,
            # then require dipole length; do not trust a stale physical channel.units label.
            working = "mv"
            assumption = "V12.3 Phoenix TD provenance: uncalibrated electric float treated as receiver-level mV; dipole required"
        elif cal_state == "unapplied" and "digital count" in phoenix_source_units.lower():
            working = "counts"
            assumption = "V12.3 Phoenix provenance: stored electric samples are digital counts"
        if assumption:
            plan["notes"].append(assumption)
        adc_gains = [g for g in (_coefficient_counts_to_volt_gain(t) for t in spectral) if g is not None]
        has_adc = bool(adc_gains)
        fap_tabs = [t for t in spectral if _is_fap_table(t) or str(t.kind).lower() in ("delay", "timedelay", "time_delay")]
        coeff_tabs = [t for t in spectral if str(t.kind).lower() in _COEFF_KINDS]

        # V11.3.2 safety rule: a physical stored field unit is authoritative.
        # Never amplitude-calibrate mV/km, V/m, or uV/m again merely because
        # response metadata is mounted with applied=False.
        if working in _ELECTRIC_FIELD_UNITS and not force_coil and cal_state != "unapplied":
            plan["spectral_tables"] = []
            fap_still = False
            plan["fap_still_in_samples"] = False
            plan["notes"].append(
                f"V11.3.2 physical-unit guard: electric samples are already {units!r}; "
                "ADC/RXCAL amplitude calibration is prohibited; only unit conversion is allowed"
            )
        else:
            plan["spectral_tables"] = fap_tabs + coeff_tabs
            if fap_still and plan["spectral_tables"]:
                plan["notes"].append(
                    "electric unapplied FAP/ZPK/coefficient will be deconvolved in frequency domain"
                )
            elif fap_still and not plan["spectral_tables"]:
                msg = (
                    f"V12.3 STRICT CALIBRATION ERROR: {str(ch_name).upper()} contains uncalibrated "
                    "Phoenix samples (calibration_applied=False), but no usable electric "
                    "FAP/ZPK/coefficient response was found in MTH5 Survey/Filters or safely matched embedded JSON. "
                    "Dipole-length conversion alone is not a complete receiver calibration. "
                    "Rebuild the MTH5 with a real RXCAL/filter chain or provide a calibrated MTH5."
                )
                plan["notes"].append(msg)
                if cal_state == "unapplied":
                    raise RuntimeError(msg)

        dipole_usable = (
            working in _ELECTRIC_VOLTAGE_UNITS
            or (working == "counts" and has_adc)
        )
        try:
            src_for_dipole = working
            if working == "counts" and has_adc:
                src_for_dipole = "v"
            if working in _ELECTRIC_FIELD_UNITS:
                scale, note = _electric_field_scale(working, target_e, None)
            elif dipole_usable:
                scale, note = _electric_field_scale(src_for_dipole, target_e, dipole_m)
            elif working == "counts":
                raise ValueError(
                    "digital counts with no counts-to-volt coefficient; "
                    "dipole_length is not applied (counts are not volts)"
                )
            else:
                raise ValueError(
                    f"electric samples are unlabeled voltage-like data (units={units!r}); "
                    "dipole_length [m] is required to convert to mV/km "
                    "(do not assume the stored series is already a field)"
                )
            if has_adc and working == "counts":
                # ADC stays in spectral_tables; dipole must wait until after deconvolution.
                plan["time_domain_scale"] = 1.0
                plan["post_cal_scale"] = float(scale)
                plan["notes"].append(
                    f"ADC/gain coefficient in spectral cal, then {note} after deconvolution"
                )
            else:
                plan["time_domain_scale"] = float(scale)
                plan["post_cal_scale"] = 1.0
                plan["notes"].append(note)
            plan["output_units"] = target_e
        except Exception as e:
            plan["time_domain_scale"] = 1.0
            plan["post_cal_scale"] = 1.0
            plan["scale_incomplete"] = True
            plan["notes"].append(f"electric absolute scale incomplete: {e}")
            if working == "counts":
                plan["notes"].append(
                    "FAP shape can still be removed; ADC counts-to-volt and receiver "
                    "gain were not present as MTH5 coefficient filters"
                )
            plan["output_units"] = units or working
        # V11.3.2: Phoenix TD stored as mV is already receiver-level voltage.
        # Do not divide analog gain again. Receiver gain removal is permitted only
        # for raw counts when an explicit counts-to-volt stage exists.
        _apply_phoenix_receiver_gain(
            plan, meta,
            voltage_like=(working == "counts" and has_adc),
        )
        if working in _ELECTRIC_VOLTAGE_UNITS:
            plan["notes"].append("V11.3.2 TD-voltage guard: receiver analog gain not removed again")
        return plan

    if not is_magnetic:
        plan["notes"].append("non E/H channel; no unit alignment")
        return plan

    sensor_kind = classify_mth5_magnetic_sensor(meta, all_tabs)
    plan["sensor_kind"] = sensor_kind

    if sensor_kind == "coil":
        # V11.3.2: nT/T/pT stored samples are already physical magnetic field.
        # Their unit label overrides stale/mis-set applied=False filter metadata.
        if nu in _MAGNETIC_FIELD_UNITS and not force_coil and cal_state != "unapplied":
            plan["spectral_tables"] = []
            fap_still = False
            plan["fap_still_in_samples"] = False
            after_unit = nu
            plan["notes"].append(
                f"V11.3.2 physical-unit guard: magnetic samples already {units!r}; "
                "RXCAL/SCAL amplitude deconvolution prohibited; unit conversion only"
            )
        elif fap_still:
            plan["spectral_tables"] = [t for t in spectral if str(t.kind).lower() in _SPECTRAL_KINDS]
            if cal_state == "unapplied" and not plan["spectral_tables"]:
                msg = (
                    f"V12.3 STRICT CALIBRATION ERROR: {str(ch_name).upper()} contains uncalibrated "
                    "Phoenix magnetic TD samples (calibration_applied=False), but no usable "
                    "coil/receiver FAP/ZPK response was found in MTH5 Survey/Filters or safely matched embedded JSON. "
                    "The samples cannot safely be treated as nT with scale=1. "
                    "Rebuild the MTH5 with the correct RXCAL+SCAL filter chain or provide a calibrated MTH5."
                )
                plan["notes"].append(msg)
                raise RuntimeError(msg)
            after_unit = _first_fap_units_in(all_tabs) or "nt"
            if after_unit not in _MAGNETIC_FIELD_UNITS:
                # RXCAL-only chain leaves voltage; SCAL missing.
                if after_unit in _ELECTRIC_VOLTAGE_UNITS:
                    plan["scale_incomplete"] = True
                    plan["notes"].append(
                        f"magnetic FAP chain units_in={after_unit!r} (no SCAL/nT stage); "
                        "frequency shape will be removed but absolute nT scale is incomplete"
                    )
                    after_unit = "nt"
            if nu in _MAGNETIC_FIELD_UNITS:
                plan["notes"].append(
                    f"channel.units={units!r} but unapplied coil/receiver FAP is still in the "
                    "samples; deconvolving FAP (filter-mounted ≠ already nT)"
                )
            if nu == "counts":
                adc_gains = [g for g in (_coefficient_counts_to_volt_gain(t) for t in spectral) if g is not None]
                if not adc_gains:
                    plan["scale_incomplete"] = True
                    plan["notes"].append(
                        "magnetic counts: applying FAP shape (SCAL×RXCAL); "
                        "absolute nT scale is incomplete without a counts-to-volt coefficient"
                    )
            plan["notes"].append(
                f"coil/receiver spectral deconvolution {[t.name for t in plan['spectral_tables']]}; "
                f"after_cal unit treated as {after_unit}"
            )
        else:
            plan["spectral_tables"] = []
            after_unit = nu if nu in _MAGNETIC_FIELD_UNITS else "nt"
            plan["notes"].append(
                f"magnetic FAP applied=True; stored samples treated as physical ({after_unit})"
            )
        try:
            scale, note = _magnetic_field_scale(after_unit, target_h)
            plan["post_cal_scale"] = float(scale)
            plan["time_domain_scale"] = 1.0
            plan["notes"].append(f"coil spectral cal then {note}")
            plan["output_units"] = target_h
        except Exception as e:
            plan["post_cal_scale"] = 1.0
            plan["notes"].append(f"coil unit scale warning: {e}")
            plan["output_units"] = after_unit
        _apply_phoenix_receiver_gain(
            plan, meta,
            voltage_like=bool(fap_still) and nu == "counts",
        )
        if nu in _ELECTRIC_VOLTAGE_UNITS:
            plan["notes"].append("V11.3.2 TD-voltage guard: receiver analog gain not removed again")
        return plan

    # Fluxgate / constant sensitivity path
    v_per_nt = None
    for tab in all_tabs:
        v_per_nt = _coefficient_gain_as_v_per_nt(tab)
        if v_per_nt is not None:
            plan["notes"].append(
                f"fluxgate coefficient {tab.name}: interpreted as {v_per_nt:g} V/nT "
                f"(units_in={tab.units_in!r}, units_out={tab.units_out!r}, gain={tab.gain})"
            )
            break

    if fap_still:
        plan["spectral_tables"] = [t for t in spectral if _is_fap_table(t)]
        if plan["spectral_tables"]:
            plan["notes"].append(
                "unexpected FAP on fluxgate-classified channel; deconvolving unapplied FAP"
            )

    if nu in ("v", "mv"):
        if v_per_nt is None:
            plan["notes"].append(
                "fluxgate/magnetic voltage data found but no usable sensitivity coefficient in MTH5 Filters"
            )
            plan["scale_incomplete"] = True
            plan["output_units"] = units
            return plan
        volt = 1.0 if nu == "v" else 1e-3
        to_nt = volt / float(v_per_nt)
        scale_nt_to_target, note = _magnetic_field_scale("nt", target_h)
        plan["time_domain_scale"] = float(to_nt * scale_nt_to_target)
        plan["notes"].append(f"voltage -> nT using sensitivity, then {note}")
        plan["output_units"] = target_h
        return plan

    if nu in _MAGNETIC_FIELD_UNITS and not fap_still:
        scale, note = _magnetic_field_scale(nu, target_h)
        plan["time_domain_scale"] = float(scale)
        plan["notes"].append(f"fluxgate physical units: {note}")
        plan["output_units"] = target_h
        return plan

    if nu in ("counts",):
        plan["notes"].append("magnetic counts without digitizer+sensor chain cannot be auto-converted")
        plan["scale_incomplete"] = True
        plan["output_units"] = units
        return plan

    if v_per_nt is not None:
        scale_nt_to_target, note = _magnetic_field_scale("nt", target_h)
        plan["time_domain_scale"] = float((1.0 / float(v_per_nt)) * scale_nt_to_target)
        plan["notes"].append(f"assumed voltage data; applied fluxgate sensitivity, then {note}")
        plan["output_units"] = target_h
        return plan

    plan["notes"].append(f"no magnetic unit conversion applied (units={units!r})")
    plan["output_units"] = units
    return plan


def _combine_filter_tables(tables: Sequence[MTH5FilterTable],
                           freqs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    grid = None
    for t in tables:
        if t.freqs is not None and len(t.freqs):
            grid = np.asarray(t.freqs, dtype=float) if grid is None else np.unique(np.concatenate([grid, t.freqs]))
    if freqs is not None and len(freqs):
        grid = np.asarray(freqs, dtype=float) if grid is None else np.unique(np.concatenate([grid, freqs]))
    if grid is None or len(grid) == 0:
        grid = np.logspace(-4, 3, 81)
    resp = np.ones(len(grid), dtype=complex)
    for t in tables:
        resp *= t.response_on(grid)
    return grid, resp


def _h5_find_survey_station_run(h5, survey: Optional[str], station: str, run: str):
    """Return (survey_name, station_group, run_group) for common MTH5 layouts."""
    root_candidates = []
    if "Experiment" in h5:
        exp = h5["Experiment"]
        if "Surveys" in exp:
            root_candidates.append(exp["Surveys"])
        root_candidates.append(exp)
    root_candidates.append(h5)

    survey_groups = []
    for root in root_candidates:
        try:
            keys = list(root.keys())
        except Exception:
            continue
        for key in keys:
            node = root[key]
            if not hasattr(node, "keys"):
                continue
            # Surveys container
            if key.lower() == "surveys":
                for sk in node.keys():
                    survey_groups.append((sk, node[sk]))
            else:
                survey_groups.append((key, node))

    errors = []
    for sname, sgrp in survey_groups:
        if survey and str(sname) != str(survey):
            # still allow exact survey match later; skip non-matching
            continue
        stations = None
        if "Stations" in sgrp:
            stations = sgrp["Stations"]
        elif "stations" in sgrp:
            stations = sgrp["stations"]
        else:
            # survey group itself may contain stations
            stations = sgrp
        if station not in stations:
            # case-insensitive station match
            match = None
            for k in stations.keys():
                if str(k).lower() == str(station).lower():
                    match = k
                    break
            if match is None:
                continue
            station_grp = stations[match]
        else:
            station_grp = stations[station]

        run_grp = None
        if "Runs" in station_grp and run in station_grp["Runs"]:
            run_grp = station_grp["Runs"][run]
        elif "runs" in station_grp and run in station_grp["runs"]:
            run_grp = station_grp["runs"][run]
        elif run in station_grp:
            run_grp = station_grp[run]
        else:
            for k in station_grp.keys():
                if str(k).lower() == str(run).lower():
                    run_grp = station_grp[k]
                    break
        if run_grp is None:
            errors.append(f"survey={sname}: station found but run {run!r} missing")
            continue
        return str(sname), station_grp, run_grp

    # If survey filter excluded everything, retry without survey constraint.
    if survey:
        return _h5_find_survey_station_run(h5, None, station, run)
    raise FileNotFoundError(
        f"MTH5 station/run not found: station={station!r}, run={run!r}. "
        + ("; ".join(errors) if errors else "No Surveys/Stations groups matched.")
    )


def _h5_get_channel_dataset(run_grp, channel: str):
    # Prefer exact, then case-insensitive, then component attribute match.
    if channel in run_grp:
        return run_grp[channel]
    for k in run_grp.keys():
        if str(k).lower() == str(channel).lower():
            return run_grp[k]
    want = str(channel).lower()
    for k in run_grp.keys():
        node = run_grp[k]
        attrs = getattr(node, "attrs", {})
        comp = _attr_to_text(attrs.get("component", attrs.get("channel.component", ""))).lower()
        if comp == want:
            return node
    raise FileNotFoundError(f"Channel {channel!r} not found in run group keys={list(run_grp.keys())}")


def _h5_read_fap_filter(filt_grp, name: str) -> Optional[MTH5FilterTable]:
    # Official FAP groups usually contain frequencies/amplitudes/phases datasets.
    candidates = [filt_grp]
    for key in ("fap", "FAP", "frequency_response_table_filters", "FrequencyResponseTable"):
        if key in filt_grp:
            candidates.append(filt_grp[key])
    for container in candidates:
        node = None
        if name in container:
            node = container[name]
        else:
            for k in container.keys():
                if str(k).lower() == name.lower():
                    node = container[k]
                    break
        if node is None:
            continue
        freqs, amps, phases = _fap_arrays_from_node(node)
        if freqs is None:
            continue
        gain = complex(float(node.attrs.get("gain", node.attrs.get("factor", 1.0))))
        units_in = _attr_to_text(node.attrs.get("units_in", node.attrs.get("input_units", "")))
        units_out = _attr_to_text(node.attrs.get("units_out", node.attrs.get("output_units", "")))
        cplx = np.asarray(amps, float) * np.exp(1j * _phase_to_rad(phases))
        return MTH5FilterTable(name, "fap", freqs=freqs, complex_resp=cplx, gain=gain,
                               units_in=units_in, units_out=units_out)
    return None


def _h5_read_zpk_filter(filt_grp, name: str) -> Optional[MTH5FilterTable]:
    for key in ("zpk", "ZPK", "pole_zero_filters", "PoleZero"):
        container = filt_grp[key] if key in filt_grp else filt_grp
        node = None
        if name in container:
            node = container[name]
        else:
            for k in getattr(container, "keys", lambda: [])():
                if str(k).lower() == name.lower():
                    node = container[k]
                    break
        if node is None:
            continue
        poles = np.asarray(node["poles"][()], dtype=complex) if "poles" in node else np.array([], complex)
        zeros = np.asarray(node["zeros"][()], dtype=complex) if "zeros" in node else np.array([], complex)
        gain = complex(float(node.attrs.get("gain", node.attrs.get("normalization_factor", 1.0))))
        return MTH5FilterTable(name, "zpk", poles=poles, zeros=zeros, gain=gain,
                               units_in=_attr_to_text(node.attrs.get("units_in", "")),
                               units_out=_attr_to_text(node.attrs.get("units_out", "")))
    return None


def _h5_read_coefficient_filter(filt_grp, name: str) -> Optional[MTH5FilterTable]:
    for key in ("coefficient", "Coefficient", "coefficients", "coefficient_filters"):
        container = filt_grp[key] if key in filt_grp else filt_grp
        node = None
        if name in container:
            node = container[name]
        else:
            for k in getattr(container, "keys", lambda: [])():
                if str(k).lower() == name.lower():
                    node = container[k]
                    break
        if node is None:
            continue
        gain = complex(float(node.attrs.get("gain", node.attrs.get("coefficient", 1.0))))
        return MTH5FilterTable(name, "coefficient", gain=gain,
                               units_in=_attr_to_text(node.attrs.get("units_in", "")),
                               units_out=_attr_to_text(node.attrs.get("units_out", "")))
    return None


def _h5_locate_filters_group(survey_grp):
    for key in ("Filters", "filters"):
        if key in survey_grp:
            return survey_grp[key]
    # sometimes filters live under Experiment
    return None


def _h5_load_named_filter(filters_grp, name: str) -> Optional[MTH5FilterTable]:
    if filters_grp is None:
        return None
    for loader in (_h5_read_fap_filter, _h5_read_zpk_filter, _h5_read_coefficient_filter):
        try:
            got = loader(filters_grp, name)
        except Exception:
            got = None
        if got is not None:
            return got
    # Deep search by name
    def _walk(g, depth=0):
        if depth > 6:
            return None
        for k in g.keys():
            if str(k).lower() == name.lower():
                node = g[k]
                # try interpret as FAP node directly
                try:
                    freqs, amps, phases = _fap_arrays_from_node(node)
                    if freqs is not None:
                        cplx = np.asarray(amps, float) * np.exp(1j * _phase_to_rad(phases))
                        return MTH5FilterTable(name, "fap", freqs=freqs, complex_resp=cplx)
                except Exception:
                    pass
            try:
                child = g[k]
                if hasattr(child, "keys"):
                    got = _walk(child, depth + 1)
                    if got is not None:
                        return got
            except Exception:
                continue
        return None
    return _walk(filters_grp)


def list_mth5_inventory(path: str | Path) -> List[Dict[str, object]]:
    """Return channel inventory rows for GUI dropdowns."""
    path = _candidate_existing_path(Path(path))
    if not path.exists():
        raise FileNotFoundError(f"MTH5 file not found: {path}")
    rows: List[Dict[str, object]] = []
    # Prefer official API summary when available.
    try:
        MTH5 = _import_mth5_class()
        with _mt_metadata_utf8_open_patch():
            m = MTH5()
            m.open_mth5(str(path), "r")
            try:
                summary = m.channel_summary.to_dataframe()
                for _, r in summary.iterrows():
                    rows.append({
                        "survey": str(r.get("survey", "")),
                        "station": str(r.get("station", "")),
                        "run": str(r.get("run", "")),
                        "channel": str(r.get("component", r.get("channel", ""))),
                        "sample_rate": float(r.get("sample_rate", np.nan)),
                        "units": str(r.get("units", "")),
                        "n_samples": int(r.get("n_samples", r.get("npts", 0)) or 0),
                    })
            finally:
                try:
                    m.close_mth5()
                except Exception:
                    pass
        if rows:
            return rows
    except Exception:
        pass

    try:
        import h5py
    except ImportError as e:
        raise ImportError("Reading MTH5 requires the 'mth5' or 'h5py' package.") from e

    with h5py.File(path, "r") as h5:
        # Walk Surveys/*/Stations/*/* channels
        survey_roots = []
        if "Experiment" in h5 and "Surveys" in h5["Experiment"]:
            survey_roots.append(h5["Experiment"]["Surveys"])
        elif "Experiment" in h5:
            survey_roots.append(h5["Experiment"])
        else:
            survey_roots.append(h5)
        for sroot in survey_roots:
            for sname, sgrp in sroot.items():
                stations = sgrp["Stations"] if "Stations" in sgrp else sgrp
                if not hasattr(stations, "items"):
                    continue
                for st_name, st_grp in stations.items():
                    run_parent = st_grp["Runs"] if "Runs" in st_grp else st_grp
                    if not hasattr(run_parent, "items"):
                        continue
                    for run_name, run_grp in run_parent.items():
                        if not hasattr(run_grp, "items"):
                            continue
                        # skip nested metadata-only groups
                        for ch_name, ch_node in run_grp.items():
                            if not hasattr(ch_node, "shape"):
                                continue
                            attrs = ch_node.attrs
                            rows.append({
                                "survey": str(sname),
                                "station": str(st_name),
                                "run": str(run_name),
                                "channel": str(attrs.get("component", ch_name)),
                                "sample_rate": float(attrs.get("sample_rate", attrs.get("channel.sample_rate", np.nan))),
                                "units": _attr_to_text(attrs.get("units", attrs.get("channel.units", ""))),
                                "n_samples": int(np.prod(ch_node.shape)) if len(ch_node.shape) else 0,
                            })
    return rows


def _read_dataset_1d(dset, skip: int = 0, n: Optional[int] = None) -> np.ndarray:
    """Read a 1-D HDF5/MTH5 dataset slice without loading the unused tail."""
    skip = max(int(skip or 0), 0)
    try:
        shape = tuple(dset.shape)
    except Exception:
        raw = np.asarray(dset[()] if hasattr(dset, "__getitem__") else dset)
        data = np.asarray(raw, dtype=float).reshape(-1)
        if skip:
            data = data[skip:]
        if n is not None and 0 < int(n) < 10**17:
            data = data[:int(n)]
        return data
    size = int(np.prod(shape)) if shape else 0
    if size <= 0 or skip >= size:
        return np.empty(0, dtype=float)
    take = size - skip
    if n is not None and 0 < int(n) < 10**17:
        take = min(take, int(n))
    stop = skip + take
    try:
        if len(shape) == 1:
            sl = dset[skip:stop]
        elif len(shape) == 2 and shape[1] == 1:
            sl = dset[skip:stop, 0]
        elif len(shape) == 2 and shape[0] == 1:
            sl = dset[0, skip:stop]
        else:
            sl = np.asarray(dset[()], dtype=float).reshape(-1)[skip:stop]
        return np.asarray(sl, dtype=float).reshape(-1)
    except Exception:
        raw = np.asarray(dset[()], dtype=float).reshape(-1)
        return raw[skip:stop]


def _section_channel_audit_stats(sections: Sequence[np.ndarray], ich: int,
                                 max_samples: int = 131072) -> Dict[str, float]:
    """first/std/p95abs for logs without concatenating the full record into RAM."""
    n = 0
    first = np.nan
    parts: List[np.ndarray] = []
    nsec = max(1, len(sections))
    per = max(256, int(max_samples // nsec))
    for sec in sections:
        if ich >= int(getattr(sec, "shape", (0, 0))[1] if getattr(sec, "ndim", 0) > 1 else 0):
            continue
        col = np.asarray(sec[:, ich])
        n += int(col.size)
        if col.size == 0:
            continue
        if not np.isfinite(first):
            for v in col[: min(col.size, 4096)]:
                if np.isfinite(v):
                    first = float(v)
                    break
        step = max(1, int(np.ceil(col.size / per)))
        sub = np.asarray(col[::step], dtype=np.float64)
        sub = sub[np.isfinite(sub)]
        if sub.size:
            parts.append(sub)
    if not parts:
        return {"n": float(n), "first": first, "std": np.nan, "p95abs": np.nan, "stats_n": 0.0}
    sub = np.concatenate(parts)
    if sub.size > max_samples:
        sub = sub[:: max(1, int(np.ceil(sub.size / max_samples)))]
        sub = sub[:max_samples]
    return {
        "n": float(n),
        "first": first,
        "std": float(np.std(sub)) if sub.size else np.nan,
        "p95abs": float(np.percentile(np.abs(sub), 95)) if sub.size else np.nan,
        "stats_n": float(sub.size),
    }


def _boolish_attr(value):
    """Return True/False for HDF5 boolean-like attrs, else None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return bool(int(value))
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    t = str(value).strip().lower()
    if t in ("true", "1", "yes", "y", "on"):
        return True
    if t in ("false", "0", "no", "n", "off"):
        return False
    return None


def _merge_phoenix_state_attrs(meta: Dict[str, object], attrs) -> None:
    """Read Phoenix converter provenance attrs without treating filter mounting as calibration."""
    def get_text(*keys):
        for k in keys:
            try:
                if k in attrs:
                    return _attr_to_text(attrs.get(k))
            except Exception:
                pass
        return ""
    def get_raw(*keys):
        for k in keys:
            try:
                if k in attrs:
                    return attrs.get(k)
            except Exception:
                pass
        return None
    src = get_text("phoenix_source_units", "phoenix_v22.source_units", "phoenix.source_units", "source_units")
    fmt = get_text("phoenix_source_format", "phoenix_v22.source_format", "phoenix.source_format", "source_format")
    cap = _boolish_attr(get_raw("phoenix_calibration_applied", "phoenix_v22.calibration_applied", "phoenix.calibration_applied", "calibration_applied"))
    rfc = _boolish_attr(get_raw("phoenix_real_filter_chain_attached", "phoenix_v22.real_filter_chain_attached", "phoenix.real_filter_chain_attached"))
    if src:
        meta["phoenix_source_units"] = src
    if fmt:
        meta["phoenix_source_format"] = fmt
    if cap is not None:
        meta["phoenix_calibration_applied"] = cap
    if rfc is not None:
        meta["phoenix_real_filter_chain_attached"] = rfc


def _calibration_state(meta: Dict[str, object]) -> str:
    """Return applied / unapplied / unknown, prioritizing explicit Phoenix provenance."""
    cap = meta.get("phoenix_calibration_applied", None)
    if cap is True:
        return "applied"
    if cap is False:
        return "unapplied"
    names = list(meta.get("filter_names") or [])
    applied = list(meta.get("filter_applied") or [])
    if names and applied:
        aa = [bool(x) for x in applied[:len(names)]]
        if aa and all(aa):
            return "applied"
        if aa and not all(aa):
            return "unapplied"
    return "unknown"


def read_mth5_channel_array(spec: str, skip: int = 0, n: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read one channel from an MTH5 spec. Returns (data, metadata)."""
    info = parse_mth5_spec(spec)
    path = _candidate_existing_path(Path(info["path"]))
    if not path.exists():
        # try resolve relative later by caller; still attempt as-is
        path = Path(info["path"])
    if not path.exists():
        raise FileNotFoundError(f"MTH5 file not found: {path}")

    meta: Dict[str, object] = dict(info)
    meta["path"] = str(path)

    # Official mth5 API path
    try:
        MTH5 = _import_mth5_class()
        with _mt_metadata_utf8_open_patch():
            m = MTH5()
            m.open_mth5(str(path), "r")
            try:
                kwargs = {}
                if info.get("survey"):
                    kwargs["survey"] = info["survey"]
                ch = m.get_channel(info["station"], info["run"], info["channel"], **kwargs)
                dset = ch.hdf5_dataset
                integer_samples = _dtype_is_integer(getattr(dset, "dtype", None))
                storage_dtype = str(getattr(dset, "dtype", ""))
                data = _read_dataset_1d(dset, skip=skip, n=n)
                md = ch.metadata if hasattr(ch, "metadata") else getattr(ch, "channel_metadata", None)
                sample_rate = None
                units = ""
                sensor_type = ""
                sensor_model = ""
                sensor_manufacturer = ""
                dipole_length = None
                filter_names: List[str] = []
                filter_applied: List[bool] = []
                if md is not None:
                    sample_rate = getattr(md, "sample_rate", None)
                    units = str(getattr(md, "units", "") or "")
                    sens = getattr(md, "sensor", None)
                    if sens is not None:
                        sensor_type = str(getattr(sens, "type", "") or "")
                        sensor_model = str(getattr(sens, "model", "") or "")
                        sensor_manufacturer = str(getattr(sens, "manufacturer", "") or "")
                    try:
                        dipole_length = float(getattr(md, "dipole_length", None))
                    except Exception:
                        dipole_length = None
                    filter_names, filter_applied = _channel_filter_refs(md=md)
                meta.update({
                    "sample_rate": float(sample_rate) if sample_rate else np.nan,
                    "units": units,
                    "sensor_type": sensor_type or sensor_model,
                    "sensor_model": sensor_model,
                    "sensor_manufacturer": sensor_manufacturer,
                    "dipole_length": dipole_length,
                    "filter_names": filter_names,
                    "filter_applied": filter_applied,
                    "filters_mounted": bool(filter_names),
                    "channel_type": str(getattr(md, "type", "") if md is not None else ""),
                    "integer_samples": integer_samples,
                    "storage_dtype": storage_dtype,
                })
                _merge_phoenix_state_attrs(meta, getattr(dset, "attrs", {}))
                # The waveform dataset's explicit Phoenix provenance wins over
                # mt_metadata's sometimes auto-normalized physical unit labels.
                attrs = getattr(dset, "attrs", {})
                raw_units = _attr_to_text(attrs.get("units", "")) if "units" in attrs else ""
                source_units = str(meta.get("phoenix_source_units", "") or "")
                if "count" in source_units.lower() or "count" in raw_units.lower():
                    if meta.get("phoenix_calibration_applied") is not True:
                        meta["units"] = "digital counts"
                        meta["units_provenance"] = "Phoenix HDF5 dataset / explicit source_units"
                elif raw_units:
                    meta["units"] = raw_units
                    meta["units_provenance"] = "HDF5 dataset"
                api_tables: List[MTH5FilterTable] = []
                try:
                    cr = ch.channel_response
                    for filt in getattr(cr, "filters_list", []) or []:
                        tab = _mt_filter_object_to_table(filt)
                        if tab is not None:
                            api_tables.append(tab)
                except Exception:
                    api_tables = []
                if not api_tables and filter_names:
                    try:
                        fg = m.get_survey(str(info.get("survey") or meta.get("survey") or "")).filters_group
                    except Exception:
                        fg = None
                    if fg is None:
                        try:
                            sid = str(getattr(ch, "survey_id", "") or "")
                            if sid:
                                fg = m.get_survey(sid).filters_group
                        except Exception:
                            fg = None
                    if fg is not None:
                        for nm in filter_names:
                            try:
                                filt = fg.to_filter_object(str(nm).replace("/", " per ").lower())
                                tab = _mt_filter_object_to_table(filt)
                                if tab is not None:
                                    api_tables.append(tab)
                            except Exception:
                                continue
                meta["api_filter_tables"] = api_tables
            finally:
                try:
                    m.close_mth5()
                except Exception:
                    pass
        _finalize_electric_dipole_meta(meta, path, info)
        return np.asarray(data, dtype=float), meta
    except Exception:
        pass

    import h5py
    with h5py.File(path, "r") as h5:
        survey_name, station_grp, run_grp = _h5_find_survey_station_run(
            h5, info.get("survey"), info["station"], info["run"]
        )
        dset = _h5_get_channel_dataset(run_grp, info["channel"])
        storage_dtype = str(getattr(dset, "dtype", ""))
        integer_samples = _dtype_is_integer(getattr(dset, "dtype", None))
        data = _read_dataset_1d(dset, skip=skip, n=n)
        attrs = dset.attrs
        filter_names, filter_applied = _channel_filter_refs(attrs=attrs)
        dipole_raw = attrs.get(
            "dipole_length",
            attrs.get("channel.dipole_length", attrs.get("dipole.length", None)),
        )
        try:
            dipole_length = float(dipole_raw) if dipole_raw is not None else None
        except Exception:
            dipole_length = None
        meta.update({
            "survey": survey_name,
            "sample_rate": float(attrs.get("sample_rate", attrs.get("channel.sample_rate", np.nan))),
            "units": _attr_to_text(attrs.get("units", attrs.get("channel.units", ""))),
            "sensor_type": _attr_to_text(
                attrs.get("sensor.type", attrs.get("channel.sensor.type", ""))
            ),
            "sensor_model": _attr_to_text(
                attrs.get("sensor.model", attrs.get("channel.sensor.model", ""))
            ),
            "sensor_manufacturer": _attr_to_text(
                attrs.get("sensor.manufacturer", attrs.get("channel.sensor.manufacturer", ""))
            ),
            "dipole_length": dipole_length,
            "filter_names": filter_names,
            "filter_applied": filter_applied,
            "filters_mounted": bool(filter_names),
            "channel_type": _attr_to_text(attrs.get("type", attrs.get("channel.type", ""))),
            "integer_samples": integer_samples,
            "storage_dtype": storage_dtype,
            "api_filter_tables": [],
        })
        _merge_phoenix_state_attrs(meta, attrs)
        if not meta["sensor_type"]:
            meta["sensor_type"] = meta["sensor_model"]
    _finalize_electric_dipole_meta(meta, path, info)
    return np.asarray(data, dtype=float), meta


def _load_all_named_filters_for_channel(path: Path, info: Dict[str, Optional[str]],
                                        meta: Dict[str, object],
                                        force: bool = False) -> List[MTH5FilterTable]:
    names = list(meta.get("filter_names") or [])
    applied = list(meta.get("filter_applied") or [])
    while len(applied) < len(names):
        applied.append(False)
    api_tables = list(meta.get("api_filter_tables") or [])
    h5_tables: List[MTH5FilterTable] = []
    import h5py
    with h5py.File(path, "r") as h5:
        survey_name, _station_grp, _run_grp = _h5_find_survey_station_run(
            h5, info.get("survey") or meta.get("survey"), info["station"], info["run"]
        )
        filters_grp = None
        if "Experiment" in h5 and "Surveys" in h5["Experiment"]:
            sgrp = h5["Experiment"]["Surveys"].get(survey_name)
            if sgrp is not None:
                filters_grp = _h5_locate_filters_group(sgrp)
        if filters_grp is None and "Experiment" in h5:
            filters_grp = _h5_locate_filters_group(h5["Experiment"])

        selected = []
        for i, name in enumerate(names):
            is_applied = bool(applied[i]) if i < len(applied) else False
            if (not force) and is_applied:
                continue
            selected.append(name)
        if force:
            selected = list(names)

        ch_name = str(info["channel"]).lower()
        # If every named filter is applied=True, selected is empty unless force=True.
        # Do not rediscover those filters from Survey/Filters.
        allow_fallback = ((not names) and (not api_tables)) or (bool(force) and not names)
        if not selected and allow_fallback:
            cand = []
            if filters_grp is not None:
                for key in ("fap", "FAP", "frequency_response_table_filters",
                            "coefficient", "Coefficient", "coefficients", "zpk", "ZPK"):
                    if key in filters_grp:
                        cand.extend(list(filters_grp[key].keys()))
            for nm in cand:
                nml = str(nm).lower()
                if ch_name in nml or _looks_like_fap_name(nml) or "sensitiv" in nml:
                    selected.append(str(nm))

        loaded_names = {_filter_name_key(t.name) for t in api_tables}
        for name in selected:
            if _filter_name_key(name) in loaded_names:
                continue
            tab = _h5_load_named_filter(filters_grp, name)
            if tab is not None:
                h5_tables.append(tab)
                loaded_names.add(_filter_name_key(tab.name))

    merged = _merge_filter_tables(api_tables, h5_tables)
    if not merged and api_tables:
        merged = list(api_tables)
    return _subset_tables_by_applied(merged, names, applied, force=bool(force))


def extract_mth5_channel_cal_tables(spec: str,
                                    force_coil: bool = False,
                                    only_unapplied: bool = True) -> Tuple[List[MTH5FilterTable], Dict[str, object]]:
    """Extract instrument filters for one MTH5 channel (coil FAP/ZPK or electric RXCAL)."""
    info = parse_mth5_spec(spec)
    path = _candidate_existing_path(Path(info["path"]))
    if not path.exists():
        path = Path(info["path"])
    _, meta = read_mth5_channel_array(spec, skip=0, n=1)
    force = bool(force_coil) or (not only_unapplied)
    tables = _load_all_named_filters_for_channel(path, info, meta, force=force)
    tables_all = _load_all_named_filters_for_channel(path, info, meta, force=True)

    ch_type = str(meta.get("channel_type", "")).lower()
    ch_name = str(info["channel"]).lower()
    is_magnetic = ch_type.startswith("mag") or ch_name in ("hx", "hy", "hz", "bx", "by", "bz")
    is_electric = ch_type.startswith("elec") or ch_name in ("ex", "ey", "ez")
    sensor_kind = classify_mth5_magnetic_sensor(meta, tables_all) if is_magnetic else ("electric" if is_electric else "unknown")

    names = list(meta.get("filter_names") or [])
    applied = list(meta.get("filter_applied") or [])
    while len(applied) < len(names):
        applied.append(False)
    fap_still = bool(force) or _instrument_fap_still_in_samples(names, applied, tables_all)

    kept: List[MTH5FilterTable] = []
    if is_magnetic and sensor_kind == "coil":
        if (not fap_still) and not force:
            meta["calibration_skip_reason"] = (
                "coil FAP applied=True; stored samples treated as already deconvolved "
                "(filter-mounted is not by itself a physical-unit claim)"
            )
            meta["sensor_kind"] = sensor_kind
            return [], meta
        for tab in tables:
            if str(tab.kind).lower() in _SPECTRAL_KINDS:
                kept.append(tab)
    elif is_magnetic and sensor_kind == "fluxgate":
        for tab in tables:
            if str(tab.kind).lower() in _COEFF_KINDS:
                kept.append(tab)
        if (not fap_still) and not force:
            meta["calibration_skip_reason"] = "fluxgate already in physical units; use unit conversion only"
            meta["sensor_kind"] = sensor_kind
            meta["fluxgate_coefficients"] = [t.name for t in kept]
            return [], meta
    elif is_electric:
        if (not fap_still) and not force:
            meta["calibration_skip_reason"] = (
                "electric FAP applied=True; stored samples treated as already deconvolved"
            )
            meta["sensor_kind"] = sensor_kind
            return [], meta
        for tab in tables:
            if str(tab.kind).lower() in _SPECTRAL_KINDS:
                kept.append(tab)
    else:
        kept = list(tables)

    meta["selected_filters"] = [t.name for t in kept]
    meta["sensor_kind"] = sensor_kind
    meta["coil_like"] = sensor_kind == "coil"
    meta["fluxgate_like"] = sensor_kind == "fluxgate"
    meta["fap_still_in_samples"] = bool(fap_still)
    return kept, meta


def plan_mth5_channel_unit_alignment(spec: str, unit_mode: str,
                                     force_coil: bool = False) -> Dict[str, object]:
    """Plan time-domain unit conversion and spectral calibration for one MTH5 channel.

    TRACMT practical target: E=mV/km, B=nT (rho_a = 0.2 T |Z|^2).
    Filter objects mounted in Survey/Filters do not mean the stored samples
    are already in those units. Unapplied FAP/ZPK are deconvolved; dipole,
    ADC, and receiver gain are used only when present in the MTH5 metadata.
    """
    info = parse_mth5_spec(spec)
    path = _candidate_existing_path(Path(info["path"]))
    if not path.exists():
        path = Path(info["path"])
    _, meta = read_mth5_channel_array(spec, skip=0, n=1)
    tables = _load_all_named_filters_for_channel(
        path, info, meta, force=bool(force_coil)
    )
    tables_all = _load_all_named_filters_for_channel(path, info, meta, force=True)
    return _plan_from_channel_meta(info, meta, tables, tables_all, unit_mode, force_coil=bool(force_coil))


def write_tracmt_cal_file(path: Path, freqs: np.ndarray, complex_resp: np.ndarray, factor: float = 1.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = np.asarray(freqs, dtype=float)
    c = np.asarray(complex_resp, dtype=complex)
    order = np.argsort(f)
    f, c = f[order], c[order]
    lines = [f"{factor:.16g}", str(len(f))]
    for fi, ci in zip(f, c):
        lines.append(f"{fi:.16g} {ci.real:.16g} {ci.imag:.16g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve_mth5_spec_abs(param_path: Path, fname: str) -> Tuple[str, Dict[str, Optional[str]]]:
    info = parse_mth5_spec(fname)
    mp = Path(info["path"])
    if not mp.is_absolute():
        mp = _resolve_data_path(param_path, str(mp))
    else:
        mp = _candidate_existing_path(mp)
    return make_mth5_spec(mp, info["station"], info["run"], info["channel"], info.get("survey")), {
        **info, "path": str(mp)
    }


def mth5_calibration_preflight(param_path: Path, p: Params, out: OutputFiles) -> None:
    """V12.4 fail-closed preflight for RAW MTH5 channels with mounted responses."""
    if not p.data_sections:
        return
    _ndata, files = p.data_sections[0]
    out.write_log("MTH5 CALIBRATION PREFLIGHT (V12.6)", elapsed=False)
    out.write_log("=" * 80, elapsed=False)
    failures = []
    for ich, (fname, _skip) in enumerate(files[:p.nchan]):
        if not is_mth5_spec(fname):
            continue
        try:
            fname_abs, info = _resolve_mth5_spec_abs(param_path, fname)
            plan = plan_mth5_channel_unit_alignment(fname_abs, p.unit_mode, force_coil=bool(p.mth5_force_coil_cal))
            tabs = list(plan.get("spectral_tables") or [])
            names = [t.name for t in tabs]
            nu = _normalize_unit_token(str(plan.get("input_units") or ""))
            state = str(plan.get("calibration_state") or "unknown")
            rawish = state == "unapplied" or nu in ("", "counts", "v", "mv")
            mode = "MTH5_RAW_WITH_RESPONSE" if rawish and tabs else ("MTH5_PHYSICAL" if not rawish else "MTH5_RAW_NO_RESPONSE")
            out.write_log(
                f"  ch{ich} {str(info.get('channel')).upper()}: mode={mode}; stored_units={plan.get('input_units')!r}; "
                f"calibration_applied={plan.get('meta',{}).get('phoenix_calibration_applied', None)!r}; "
                f"filters={names}; action={'APPLY ' + ' + '.join(names) if tabs else 'NO SPECTRAL RESPONSE'}",
                elapsed=False,
            )
            if rawish and not tabs:
                failures.append(f"ch{ich} {info.get('channel')}: RAW/unlabeled samples have no usable FAP/ZPK response")
        except Exception as e:
            failures.append(f"ch{ich}: {e}")
    if failures:
        for msg in failures:
            out.warning("MTH5 PREFLIGHT FATAL: " + msg)
        detail = " | ".join(failures[:8])
        raise RuntimeError(
            "MTH5 calibration preflight failed; calculation aborted before preprocessing/FFT. "
            "Details: " + detail
        )
    out.write_log("MTH5 CALIBRATION PREFLIGHT: PASS", elapsed=False)


def align_mth5_sections_to_unit_mode(sections: List[np.ndarray], param_path: Path, p: Params,
                                     out: OutputFiles) -> List[np.ndarray]:
    """Convert MTH5 E/H channels in time domain to TRACMT UNIT_MODE units.

    Coil FAP remains for frequency-domain calibration; fluxgate sensitivity and
    electric dipole/unit conversions are applied here.
    """
    if not p.data_sections:
        return sections
    _ndata, files = p.data_sections[0]
    target_e, target_h = tracmt_target_units(p.unit_mode)
    out.write_log(
        f"MTH5 unit alignment target (UNIT_MODE={p.unit_mode}): E={target_e}, H/B={target_h}",
        elapsed=False,
    )

    # Adopt MTH5 sample_rate when it disagrees with SAMPLING_FREQ.
    # Wrong fs shifts all periods and scales rho_a = 0.2 T |Z|^2 incorrectly.
    mth5_rates: List[float] = []
    for fname, _skip in files[:p.nchan]:
        if not is_mth5_spec(fname):
            continue
        try:
            fname_abs, _info = _resolve_mth5_spec_abs(param_path, fname)
            _arr, meta = read_mth5_channel_array(fname_abs, skip=0, n=1)
            rate = float(meta.get("sample_rate", np.nan))
            if np.isfinite(rate) and rate > 0:
                mth5_rates.append(rate)
        except Exception:
            continue
    if mth5_rates:
        fs_mth5 = float(np.median(np.asarray(mth5_rates, dtype=float)))
        if abs(fs_mth5 - float(p.sampling_freq)) / max(fs_mth5, EPS) > 0.01:
            out.warning(
                f"SAMPLING_FREQ={p.sampling_freq:g} Hz differs from MTH5 sample_rate="
                f"{fs_mth5:g} Hz; adopting MTH5 sample_rate for frequency/period labels "
                f"and rho_a = 0.2*T*|Z|^2."
            )
            p.sampling_freq = fs_mth5
            p.sampling_freq_org = fs_mth5
        else:
            out.write_log(f"MTH5 sample_rate OK: {fs_mth5:g} Hz", elapsed=False)

    out_sections = []
    for sec in sections:
        a = np.asarray(sec)
        if a.dtype == np.float64 and a.flags.writeable:
            out_sections.append(a)
        else:
            out_sections.append(np.asarray(a, dtype=float))
    for ich, (fname, _skip) in enumerate(files[:p.nchan]):
        if not is_mth5_spec(fname):
            continue
        try:
            fname_abs, info = _resolve_mth5_spec_abs(param_path, fname)
            plan = plan_mth5_channel_unit_alignment(
                fname_abs, p.unit_mode, force_coil=bool(p.mth5_force_coil_cal)
            )
        except Exception as e:
            out.warning(f"MTH5 unit alignment channel {ich} failed: {e}")
            continue
        scale = float(plan.get("time_domain_scale", 1.0) or 1.0)
        try:
            st = _section_channel_audit_stats(out_sections, ich)
            out.write_log(
                f"  ch{ich} ({info['channel']}): MTH5 READBACK "
                f"units={plan.get('input_units')!r}, n={int(st['n'])}, "
                f"first={st['first']:.9g}, std={st['std']:.9g}, "
                f"p95abs={st['p95abs']:.9g}, "
                f"filters={plan.get('meta',{}).get('filter_names',[])}, "
                f"applied={plan.get('meta',{}).get('filter_applied',[])}, "
                f"mounted_not_physical={plan.get('filters_mounted_not_physical')}, "
                f"cal_state={plan.get('calibration_state')}, "
                f"source_units={plan.get('phoenix_source_units')!r}, "
                f"source_format={plan.get('phoenix_source_format')!r}, "
                f"fap_still={plan.get('fap_still_in_samples')}",
                elapsed=False,
            )
        except Exception as e:
            out.warning(f"MTH5 READBACK stats skipped for ch{ich}: {e}")
        for note in plan.get("notes", []):
            out.write_log(f"  ch{ich} ({info['channel']}): {note}", elapsed=False)
        if abs(scale - 1.0) > 1e-15:
            for i, sec in enumerate(out_sections):
                if ich < sec.shape[1]:
                    if not sec.flags.writeable or sec.dtype != np.float64:
                        sec = np.array(sec, dtype=float, copy=True)
                        out_sections[i] = sec
                    sec[:, ich] *= scale
            out.write_log(
                f"  ch{ich} ({info['channel']}): time-domain scale x{scale:.8g} "
                f"-> {plan.get('output_units', '')}",
                elapsed=False,
            )
            try:
                st2 = _section_channel_audit_stats(out_sections, ich)
                out.write_log(
                    f"  ch{ich} ({info['channel']}): ALIGNED first={st2['first']:.9g}, "
                    f"std={st2['std']:.9g}, p95abs={st2['p95abs']:.9g}",
                    elapsed=False,
                )
            except Exception as e:
                out.warning(f"MTH5 ALIGNED stats skipped for ch{ich}: {e}")
        else:
            out.write_log(
                f"  ch{ich} ({info['channel']}): sensor={plan.get('sensor_kind')} "
                f"units {plan.get('input_units')!r} -> {plan.get('output_units')!r} (scale=1)",
                elapsed=False,
            )
        for note in plan.get("notes", []):
            if "warning" in str(note).lower() or "cannot" in str(note).lower() or "required" in str(note).lower() or "incomplete" in str(note).lower():
                out.warning(f"MTH5 unit ch{ich} ({info['channel']}): {note}")
    return out_sections


def _band_safe_mth5_spectral_tables(tables: Sequence[MTH5FilterTable], fs: float, out: Optional[OutputFiles] = None, channel: str = "") -> List[MTH5FilterTable]:
    """legacy build: skip Phoenix RXCAL when its calibrated band is disjoint from the sampled band."""
    nyq = 0.5 * float(fs)
    kept: List[MTH5FilterTable] = []
    for t in tables:
        name = str(getattr(t, "name", "") or "")
        tf0 = getattr(t, "freqs", None)
        tf = np.asarray(tf0, dtype=float) if tf0 is not None else np.array([])
        tf = tf[np.isfinite(tf) & (tf > 0)]
        if "rxcal" in name.lower() and tf.size:
            fmin, fmax = float(np.min(tf)), float(np.max(tf))
            if fmin > nyq:
                if out is not None:
                    out.write_log(f"MTH5 band-safe calibration: channel {channel} SKIP {name} [{fmin:.6g}..{fmax:.6g} Hz]; processing Nyquist={nyq:.6g} Hz; no out-of-band RXCAL extrapolation.", elapsed=False)
                continue
        kept.append(t)
    return kept


def auto_calibrations_from_mth5(param_path: Path, p: Params, out: OutputFiles,
                                cal_outdir: Optional[Path] = None) -> List[Optional["CalibrationFunction"]]:
    """Build TRACMT CalibrationFunction list from MTH5 coil FAP/ZPK (and rare constant cals)."""
    cal_outdir = Path(cal_outdir) if cal_outdir else Path(param_path).parent / "mth5_cal_files"
    cal_outdir.mkdir(parents=True, exist_ok=True)
    cals: List[Optional[CalibrationFunction]] = [None] * p.nchan
    if not p.data_sections:
        return cals
    _ndata, files = p.data_sections[0]
    for ich, (fname, _skip) in enumerate(files[:p.nchan]):
        if not is_mth5_spec(fname):
            continue
        try:
            fname_abs, info = _resolve_mth5_spec_abs(param_path, fname)
            plan = plan_mth5_channel_unit_alignment(
                fname_abs, p.unit_mode, force_coil=bool(p.mth5_force_coil_cal)
            )
        except Exception as e:
            out.warning(f"MTH5 cal: cannot plan channel {ich}: {e}")
            continue

        sensor_kind = str(plan.get("sensor_kind", ""))
        tables: List[MTH5FilterTable] = list(plan.get("spectral_tables") or [])
        tables = _band_safe_mth5_spectral_tables(tables, p.sampling_freq, out=out, channel=str(info.get("channel", ich)))
        # Fluxgate sensitivities are handled in time-domain alignment; skip spectral cal.
        if sensor_kind == "fluxgate":
            out.write_log(
                f"MTH5 cal: channel {ich} ({info['channel']}) fluxgate -> time-domain sensitivity/units only",
                elapsed=False,
            )
            continue
        if not tables:
            reason = "no spectral filters to apply"
            if plan.get("notes"):
                reason = "; ".join(str(x) for x in plan["notes"])
            out.write_log(f"MTH5 cal: channel {ich} ({info['channel']}) skipped ({reason})", elapsed=False)
            continue

        freqs, resp = _combine_filter_tables(tables)
        post = float(plan.get("post_cal_scale", 1.0) or 1.0)
        # y_target = (y/resp)*post = y/(resp/post) => CalibrationFunction.factor = 1/post
        factor = 1.0 / post if abs(post) > EPS else 1.0
        ch_tag = re.sub(r"[^A-Za-z0-9_]+", "_", f"{info['station']}_{info['run']}_{info['channel']}")
        cal_path = cal_outdir / f"mth5_cal_{ich:02d}_{ch_tag}.cal"
        write_tracmt_cal_file(cal_path, freqs, resp, factor=factor)
        ranges = []
        for t in tables:
            if t.freqs is not None and len(t.freqs):
                ranges.append(f"{t.name}[{float(np.min(t.freqs)):.6g}..{float(np.max(t.freqs)):.6g} Hz]")
            else:
                ranges.append(t.name)
        out.write_log(
            f"MTH5 cal: channel {ich} ({info['channel']}) sensor={sensor_kind} "
            f"filters={[t.name for t in tables]} ranges={ranges} interpolation=log-f/log-amp+unwrapped-phase; "
            f"extrapolation=linear-end-slope; post_scale={post:g} factor={factor:g} "
            f"fap_still={plan.get('fap_still_in_samples')} "
            f"scale_incomplete={plan.get('scale_incomplete')} -> {cal_path}",
            elapsed=False,
        )
        cals[ich] = CalibrationFunction(cal_path)
    return cals


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
class CalibrationFunction:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else Path("")
        self.factor = 1.0
        self.freq = np.array([], dtype=float)
        self.amp = np.array([], dtype=float)
        self.phase = np.array([], dtype=float)
        if path is not None:
            self._read()

    @classmethod
    def from_arrays(cls, freqs: np.ndarray, complex_resp: np.ndarray, factor: float = 1.0,
                    path: Optional[Path] = None) -> "CalibrationFunction":
        obj = cls(path=None)
        obj.path = Path(path) if path is not None else Path("memory://mth5")
        obj.factor = float(factor)
        f = np.asarray(freqs, dtype=float)
        c = np.asarray(complex_resp, dtype=complex)
        idx = np.argsort(f)
        obj.freq = f[idx]
        obj.amp = np.abs(c[idx])
        obj.phase = np.unwrap(np.angle(c[idx]))
        return obj

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
        lf = np.log10(np.maximum(self.freq, EPS))
        x = np.asarray([np.log10(max(float(freq), EPS))], dtype=float)
        logamp = np.log10(np.maximum(self.amp, EPS))
        # V12.4 uses the same deterministic log-frequency interpolation/extrapolation
        # for JSON/MTH5-derived and ordinary TRACMT calibration tables.
        ai = _interp_extrap_linear(lf, logamp, x)[0]
        pi = _interp_extrap_linear(lf, self.phase, x)[0]
        return self.factor * (10.0 ** float(ai)) * np.exp(1j * float(pi))


def load_calibrations(param_file: Path, p: Params, out: OutputFiles,
                      cal_outdir: Optional[Path] = None) -> List[Optional[CalibrationFunction]]:
    cals: List[Optional[CalibrationFunction]] = []
    base = param_file.parent
    for i, name in enumerate(p.cal_files):
        if not name or name.lower() in ("none", "null", "-", "auto", "mth5"):
            cals.append(None); continue
        fp = Path(name)
        if not fp.is_absolute():
            fp = base / fp
        if not fp.exists():
            out.warning(f"Calibration file not found for channel {i}: {fp}")
            cals.append(None)
        else:
            out.write_log(f"Read calibration function from {fp}")
            cals.append(CalibrationFunction(fp))
    while len(cals) < p.nchan:
        cals.append(None)

    if p.apply_mth5_response:
        out.write_log(
            "APPLY_MTH5_RESPONSE enabled: unapplied MTH5 FAP/ZPK -> spectral CAL; "
            "filter-mounted does not mean samples are already nT or mV/km; "
            "dipole/ADC/gain used only when present in MTH5 metadata",
            elapsed=False,
        )
        auto = auto_calibrations_from_mth5(param_file, p, out, cal_outdir=cal_outdir)
        for i, cal in enumerate(auto):
            if i >= len(cals):
                cals.append(cal)
            elif cals[i] is None and cal is not None:
                cals[i] = cal
                out.write_log(f"Using MTH5-derived calibration for channel {i}", elapsed=False)
            elif cals[i] is not None and cal is not None:
                out.write_log(f"Channel {i}: explicit CAL_FILES overrides MTH5 auto calibration", elapsed=False)
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
# TRACMT Tukey biweight table: external numeric CSV, verified before loading.
# Source and licensing: THIRD_PARTY_NOTICES.md (BSD-3-Clause attribution).
# -----------------------------------------------------------------------------
_TUKEY_TABLE_SHA256 = "8a5c6b494a700138bc7610e9afab97220d92251286c4bc20f663343c106d68bb"
_TUKEY_TABLE_CACHE = None

def _load_tracmt_tukey_table() -> Tuple[np.ndarray, np.ndarray]:
    """Read verified external TRACMT Tukey table (c, b_dim_1 ... b_dim_101)."""
    global _TUKEY_TABLE_CACHE
    if _TUKEY_TABLE_CACHE is None:
        import hashlib
        path = Path(__file__).resolve().parent / "data" / "tukey_biweight_parameters.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Tukey parameter table not found: {path}. "
                "Keep the data/ directory alongside pyTRACMT.py."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != _TUKEY_TABLE_SHA256:
            raise RuntimeError(
                f"Tukey parameter CSV SHA-256 mismatch: {path}; "
                f"expected {_TUKEY_TABLE_SHA256}, got {actual}. "
                "Restore the original released CSV; do not disable this check."
            )
        arr = np.loadtxt(path, delimiter=",", comments="#", skiprows=5, dtype=np.float64)
        if arr.shape != (1001, 102):
            raise RuntimeError(f"Invalid Tukey parameter table shape: {arr.shape} != (1001, 102)")
        if not np.isfinite(arr).all() or not np.all(np.diff(arr[:, 0]) > 0):
            raise RuntimeError("Invalid Tukey parameter CSV values or c-axis order")
        cs = arr[:, 0].copy()
        bs = arr[:, 1:].T.copy()
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


def _ols_solve_row(y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Direct weighted least squares on local magnetic channels (no remote reference)."""
    w = np.asarray(weights, float)
    sw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sw[:, None]
    yw = y * sw
    try:
        return np.asarray(linalg.lstsq(Xw, yw, rcond=1e-14, lapack_driver="gelsd")[0]).ravel()
    except Exception:
        A = (X.conj().T * w) @ X
        b = (X.conj().T * w) @ y
        return np.asarray(linalg.solve(A, b, assume_a="gen", check_finite=False)).ravel()


def _mt_solve_row(y: np.ndarray, X: np.ndarray, Rr: Optional[np.ndarray], weights: np.ndarray, num_rr: int) -> np.ndarray:
    if int(num_rr) <= 0 or Rr is None:
        return _ols_solve_row(y, X, weights)
    return _rr_solve_row(y, X, Rr, weights)


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
    """Ordinary remote reference / No-RR OLS following TRACMT's staged IRLS workflow."""
    no, ni, nr = p.num_out, p.num_input, p.num_rr
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + nr] if nr > 0 else None
    resp = np.zeros((no, ni), dtype=complex)
    weights = np.ones(ft.shape[0], dtype=float)
    weights_by_out: List[np.ndarray] = []
    absr = np.ones(ft.shape[0], dtype=float)
    solve = lambda y, w: _mt_solve_row(y, X, Rr, w, nr)

    for out_idx in range(no):
        y = Y[:, out_idx]
        weights = np.ones(ft.shape[0], dtype=float)
        z_row = solve(y, weights)
        out.write_cvg("-" * 80)
        out.write_cvg(f"Calculate response functions for output variable {out_idx}")
        out.write_cvg("-" * 80)
        if nr <= 0:
            out.write_cvg("Calculate response functions by the ordinary least square method (no remote reference)")
        else:
            out.write_cvg("Calculate response functions by the ordinary least square method")
        _rr_log_response_block(out, _rr_report_row(z_row, out_idx, p, coeffs_all, freq),
                               z_row, y, X, weights, out_idx, p, ft)

        for mtype in p.m_estimators:
            if mtype < 0:
                continue
            est_name = _rr_estimator_name(mtype)
            if nr <= 0:
                out.write_cvg(f"Calculate response functions by iteratively reweighted least squares using {est_name} weight")
            else:
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
                z_new = solve(y, wnew)
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
    no, ni, nr = p.num_out, p.num_input, p.num_rr
    X = block[:, no:no + ni]
    Rr = block[:, no + ni:no + ni + nr] if nr > 0 else None
    coherence = 0.0
    weights = np.ones(block.shape[0], dtype=float)
    for out_idx in range(no):
        y = block[:, out_idx]
        z = _mt_solve_row(y, X, Rr, weights, nr)
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


def mt_phase_pm90(phase_deg):
    """Unwrap MT phase with its 180-degree ambiguity and fold to [-90, +90).

    Doubling the phase converts the MT 180-degree ambiguity to a standard
    360-degree unwrap problem.  After unwrapping, divide by two and choose
    the equivalent +/-90-degree branch.  This is intended for reported /
    plotted phase; impedance real/imaginary values are never changed.
    """
    ph = np.asarray(phase_deg, dtype=float)
    scalar = ph.ndim == 0
    x = np.atleast_1d(ph).copy()
    good = np.isfinite(x)
    if np.any(good):
        u = np.rad2deg(np.unwrap(np.deg2rad(2.0 * x[good]))) / 2.0
        u = (u + 90.0) % 180.0 - 90.0
        # Prefer +90 rather than -90 for values numerically on the boundary.
        u[np.isclose(u, -90.0, atol=1e-10)] = 90.0
        x[good] = u
    return float(x[0]) if scalar else x


def _normalize_rho_rows_phase_pm90(rows: List[Dict[str, float]], p: Params) -> None:
    """Normalize all reported rho/phase row series to continuous MT +/-90 convention."""
    if not rows:
        return
    # Rows are normally generated in spectral order. Unwrap in period order so
    # the operation is deterministic even when segment/frequency order changes.
    order = np.argsort([float(r.get("period", np.nan)) if np.isfinite(float(r.get("period", np.nan))) else np.inf for r in rows])
    for oi in range(p.num_out):
        for j in range(p.num_input):
            key = f"phase_{_z_component_label(oi, j)}"
            vals = np.array([rows[k].get(key, np.nan) for k in order], dtype=float)
            vals = mt_phase_pm90(vals)
            for kk, vv in zip(order, vals):
                rows[int(kk)][key] = float(vv) if np.isfinite(vv) else np.nan


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
    """Solve all RR/No-RR output rows once with fixed segment weights (whitened domain)."""
    no, ni, nr = p.num_out, p.num_input, p.num_rr
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + nr] if nr > 0 else None
    w = np.asarray(weights, float)
    resp = np.zeros((no, ni), dtype=complex)
    for out_idx in range(no):
        resp[out_idx] = _mt_solve_row(Y[:, out_idx], X, Rr, w, nr)
    return resp


def _estimate_Z_row_fixed_weights(
    ft: np.ndarray,
    p: Params,
    out_idx: int,
    weights: np.ndarray,
    coeffs_all=None,
    freq: float = 0.0,
) -> np.ndarray:
    """Re-estimate one RR/No-RR output row with fixed segment weights (physical units)."""
    if len(ft) < 3 or out_idx >= p.num_out:
        return np.full(p.num_input, np.nan + 1j * np.nan)
    no, ni, nr = p.num_out, p.num_input, p.num_rr
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + nr] if nr > 0 else None
    z_row = _mt_solve_row(Y[:, out_idx], X, Rr, weights, nr)
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
    """TRACMT parametric 1-sigma errors from weighted RR / No-RR covariance."""
    labels = [(0, 0, "zxx"), (0, 1, "zxy"), (1, 0, "zyx"), (1, 1, "zyy")]
    empty = {name: (np.nan, np.nan, np.nan) for _i, _j, name in labels}
    no, ni, nr = p.num_out, p.num_input, p.num_rr
    Y = ft[:, :no]
    X = ft[:, no:no + ni]
    Rr = ft[:, no + ni:no + ni + nr] if nr > 0 else None
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
        z_row = _mt_solve_row(y, X, Rr, w, nr)
        resid = y - X @ z_row
        wsum = float(np.sum(w))
        dof = max(wsum - float(ni), 1.0)
        sigma2 = float(np.sum(w * np.abs(resid) ** 2) / dof)
        if nr <= 0:
            try:
                A = (X.conj().T * w) @ X
                Pinv = np.linalg.inv(A)
            except Exception:
                Pinv = np.linalg.pinv((X.conj().T * w) @ X)
        else:
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

def _coherence_value_from_row(row: Dict[str, float], out_idx: int, p: Optional[Params] = None) -> float:
    keys = [_coherence_column_name(out_idx)]
    if p is not None:
        keys.append(f"coherence_{_output_field_label(out_idx)}")
    for key in keys:
        val = row.get(key, np.nan)
        if np.isfinite(val):
            return float(val)
    return np.nan


STITCH_LOG10_BIN_WIDTH = 0.075
STITCH_PREFER_MIN_FREQ_INDEX = 4
STITCH_ABSOLUTE_MIN_FREQ_INDEX = 2


def _row_relative_error_score(row: Dict[str, float]) -> float:
    """Relative uncertainty score used to pick the best candidate in one frequency bin."""
    if "Zxy_real" in row and "dZxy" in row:
        z = abs(complex(row.get("Zxy_real", np.nan), row.get("Zxy_imag", np.nan)))
        dz = float(row.get("dZxy", np.nan))
    elif "rho_Zxy" in row and "drho_Zxy" in row:
        z = abs(float(row.get("rho_Zxy", np.nan)))
        dz = float(row.get("drho_Zxy", np.nan))
    else:
        return 1.0
    if np.isfinite(z) and z > 0 and np.isfinite(dz):
        return dz / max(z, EPS)
    return 1.0


def _stitch_score_row(row: Dict[str, float], p: Params) -> float:
    coh = _coherence_value_from_row(row, 0, p)
    if not np.isfinite(coh):
        coh = 0.0
    k = int(row.get("freq_index", 0) or 0)
    nseg = int(row.get("n_segments", 0) or 0)
    relerr = _row_relative_error_score(row)
    index_penalty = 0.08 * abs(math.log(max(k, 1.0) / 10.0, 2.0))
    err_term = math.log10(1.0 + 100.0 * max(relerr, 1.0e-8))
    short_data_penalty = 0.5 if nseg < 3 else 0.0
    return 2.25 * float(coh) - 0.85 * err_term + 0.22 * math.log10(max(nseg, 1)) - index_penalty - short_data_penalty


def _stitch_result_rows(
    rows: List[Dict[str, float]],
    p: Params,
    bin_width: float = STITCH_LOG10_BIN_WIDTH,
) -> List[Dict[str, float]]:
    """Keep one stable estimate per log-frequency bin across many SEGMENT lengths."""
    if not rows:
        return []
    groups: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        freq = float(row.get("frequency", row.get("freq", np.nan)))
        if not np.isfinite(freq) or freq <= 0:
            continue
        key = int(round(math.log10(freq) / max(float(bin_width), 1.0e-6)))
        groups.setdefault(key, []).append(row)

    stitched: List[Dict[str, float]] = []
    for key in sorted(groups):
        candidates = groups[key]
        valid = [r for r in candidates if int(r.get("freq_index", 0) or 0) >= STITCH_ABSOLUTE_MIN_FREQ_INDEX]
        if not valid:
            valid = candidates
        preferred = [r for r in valid if int(r.get("freq_index", 0) or 0) >= STITCH_PREFER_MIN_FREQ_INDEX]
        if preferred:
            valid = preferred
        best = max(valid, key=lambda r: _stitch_score_row(r, p))
        stitched.append(dict(best))
    stitched.sort(key=lambda r: float(r.get("frequency", r.get("freq", 0.0))))
    return stitched


def _annotate_multisegment_row(row: Dict[str, float], segment_length: int, freq_index: int, n_segments: int) -> Dict[str, float]:
    row["segment_length"] = int(segment_length)
    row["freq_index"] = int(freq_index)
    row["n_segments"] = int(n_segments)
    return row


def write_edi(path: Path, rows: List[dict], site: str = "pyTRACMT"):
    with path.open("w", encoding="utf-8") as f:
        f.write(">HEAD\n")
        f.write(f"  DATAID={site}\n")
        f.write("  ACQBY=pyTRACMT_v12\n")
        f.write("  Z_UNIT=MV/KM/NT\n")
        f.write("  IMPEDANCE_UNITS=MV/KM/NT\n")
        f.write(">INFO\n")
        f.write("  Generated by pyTRACMT; impedance in (mV/km)/nT\n")
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


def normalize_manual_input_units_to_tracmt(sections: List[np.ndarray], param_path: Path, p: Params, out: OutputFiles) -> List[np.ndarray]:
    """Normalize explicitly selected non-Auto input units to TRACMT practical units.

    All downstream TRACMT estimation always receives E=mV/km and B=nT.
    MTH5 channels remain metadata/calibration driven; the manual selector is applied
    to ordinary ASCII/CSV/TXT channels only, preventing double conversion of MTH5.
    Channel order is Ex,Ey,Hx,Hy[,Hxr,Hyr].
    """
    mode = str(getattr(p, "input_unit_mode", "auto_from_mth5") or "auto_from_mth5").lower()
    if mode.startswith("auto"):
        out.write_log("INPUT_UNIT_MODE=auto_from_mth5: MTH5 metadata/SCAL/dipole conversion controls physical units.", elapsed=False)
        return sections
    if mode in ("mth5_tracmt_practical", "practical", "mv/km_nt"):
        e_scale, h_scale, desc = 1.0, 1.0, "E=mV/km, B=nT -> no scale"
    elif mode in ("si_b_tesla", "si-b", "sib", "si_b"):
        e_scale, h_scale, desc = 1e6, 1e9, "E=V/m x1e6 -> mV/km; B=T x1e9 -> nT"
    elif mode in ("si_h_ampere", "si-h", "sih", "si_h"):
        e_scale, h_scale, desc = 1e6, MU0*1e9, "E=V/m x1e6 -> mV/km; H=A/m x(mu0*1e9) -> B=nT"
    else:
        raise ValueError(f"Unsupported INPUT_UNIT_MODE={mode!r}")
    out.write_log(f"Manual input normalization BEFORE TRACMT: {desc}", elapsed=False)
    files = p.data_sections[0][1] if p.data_sections else []
    out_sections=[np.array(sec,dtype=float,copy=True) for sec in sections]
    for ich in range(p.nchan):
        # MTH5 must use its own metadata/dipole/SCAL chain; never double-scale it.
        if ich < len(files) and is_mth5_spec(str(files[ich][0])):
            out.write_log(f"  ch{ich}: MTH5 source -> manual scale skipped; MTH5 calibration path retained", elapsed=False)
            continue
        scale = e_scale if ich < p.num_out else h_scale
        for sec in out_sections:
            if ich < sec.shape[1]: sec[:,ich] *= scale
        out.write_log(f"  ch{ich}: manual input scale x{scale:.12g}", elapsed=False)
    return out_sections



def _rms_finite(x: np.ndarray) -> float:
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a*a))) if a.size else np.nan


def write_mt_input_diagnostic(outdir: Path, raw_sections: List[np.ndarray], physical_sections: List[np.ndarray],
                              param_path: Path, p: Params, out: OutputFiles) -> Path:
    """Write raw -> physical-unit amplitude audit immediately before TRACMT preprocessing.

    This is intentionally BEFORE high/low/notch, robust filtering and prewhitening.  It exposes
    unit mistakes and double calibration without changing the inversion/estimation itself.
    """
    outdir=Path(outdir); rows=[]
    labels=['Ex','Ey'] + [f'H{i+1}' for i in range(max(p.nchan-p.num_out,0))]
    if p.nchan>=4: labels[:4]=['Ex','Ey','Hx','Hy']
    if p.nchan>=6: labels[4:6]=['Hxr','Hyr']
    files=p.data_sections[0][1] if p.data_sections else []
    for ich in range(p.nchan):
        raw=np.concatenate([np.asarray(x[:,ich],float) for x in raw_sections if ich<x.shape[1]]) if raw_sections else np.array([])
        phy=np.concatenate([np.asarray(x[:,ich],float) for x in physical_sections if ich<x.shape[1]]) if physical_sections else np.array([])
        src=str(files[ich][0]) if ich<len(files) else ''
        is_e=ich<p.num_out
        rows.append({
            'channel': labels[ich] if ich<len(labels) else f'ch{ich}',
            'source_type':'MTH5' if is_mth5_spec(src) else 'ASCII/CSV',
            'raw_rms':_rms_finite(raw),'raw_median':float(np.nanmedian(raw)) if raw.size else np.nan,
            'raw_p01':float(np.nanpercentile(raw,1)) if raw.size else np.nan,'raw_p99':float(np.nanpercentile(raw,99)) if raw.size else np.nan,
            'physical_rms':_rms_finite(phy),'physical_median':float(np.nanmedian(phy)) if phy.size else np.nan,
            'physical_p01':float(np.nanpercentile(phy,1)) if phy.size else np.nan,'physical_p99':float(np.nanpercentile(phy,99)) if phy.size else np.nan,
            'effective_rms_scale':(_rms_finite(phy)/_rms_finite(raw)) if np.isfinite(_rms_finite(raw)) and _rms_finite(raw)>0 else np.nan,
            'pre_spectral_target_unit':'mV/km' if is_e else 'nT',
            'spectral_calibration_pending': bool(is_mth5_spec(src) and not is_e),
            'input_unit_mode':str(getattr(p,'input_unit_mode','auto_from_mth5')),
            'source':src,
        })
    path=outdir/'MT_input_diagnostic.csv'; pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
    out.write_log(f'legacy build diagnostic written: {path.name} (raw vs physical units BEFORE TRACMT filters)',elapsed=False)
    return path


def write_mt_response_diagnostic(outdir: Path, resp_rows: List[dict], p: Params, out: OutputFiles) -> Optional[Path]:
    """Write frequency-by-frequency Z/rho/phase/coherence audit from final response rows."""
    if not resp_rows: return None
    rows=[]
    for r in resp_rows:
        f=float(r.get('frequency',r.get('freq',np.nan))); T=1/f if np.isfinite(f) and f>0 else np.nan
        row={'frequency_Hz':f,'period_s':T,'segment_length':r.get('segment_length',np.nan),'fft_index':r.get('fft_index',np.nan),'n_segments':r.get('n_segments',np.nan)}
        for oi,j,name in TRACMT_Z_COMPONENTS:
            lab=_z_component_label(oi,j); rr=r.get(f'{lab}_real',np.nan); ii=r.get(f'{lab}_imag',np.nan)
            if np.isfinite(rr) and np.isfinite(ii):
                z=complex(rr,ii); rho,ph=rhoa_phase(z,f)
                row[f'{lab}_abs_practical']=abs(z); row[f'{lab}_phase_deg']=ph; row[f'{lab}_rho_ohm_m']=rho
        # Preserve whatever coherence fields the response writer generated.
        for k,v in r.items():
            if 'coher' in str(k).lower(): row[str(k)]=v
        rows.append(row)
    path=Path(outdir)/'MT_response_diagnostic.csv'; pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
    out.write_log(f'legacy build diagnostic written: {path.name} (Z magnitude -> rho_a=0.2*T*|Z|^2)',elapsed=False)
    return path

class ExternalPhoenixCTS:
    """Full-system Phoenix SysCal FLDTYPE=1: frequency, TS, 5 complex channels.

    CTS ordinates use the ADC full scale +/-2**23 per Phoenix SysCal manual. This class
    returns the tabulated complex response without applying 2**23 itself.
    """
    CHANNELS = ("ex", "ey", "hx", "hy", "hz")

    def __init__(self, filename, sampling_rate):
        self.filename = Path(filename)
        if not self.filename.is_file():
            raise FileNotFoundError(f"External CTS not found: {self.filename}")
        header = self.filename.open(encoding="utf-8-sig", errors="replace").readline()
        fields = [x.strip() for x in header.split(",")]
        if len(fields) < 4:
            raise ValueError("CTS header is not a Phoenix SysCal header")
        try:
            self.fieldtype = int(fields[2])
            self.n_channels = int(fields[3])
        except (ValueError, IndexError) as exc:
            raise ValueError("Cannot identify SysCal CTS FLDTYPE and channel count") from exc
        if self.fieldtype != 1:
            raise ValueError("SysCal CTS FLDTYPE=0 is a BOX-only response (1/V). "
                             "Full-system physical calibration requires FLDTYPE=1.")
        if self.n_channels != 5:
            raise ValueError(f"CTS specifies {self.n_channels} channels, expected five")
        try:
            import csv
            with self.filename.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                next(stream)
                rows = []
                for row in csv.reader(stream):
                    try:
                        v = [float(x.strip()) for x in row[:12]]
                        if len(v) == 12 and v[0] > 0 and all(np.isfinite(v)):
                            rows.append(v)
                    except (ValueError, TypeError):
                        continue
        except OSError as exc:
            raise ValueError(f"Cannot read CTS: {exc}") from exc
        data = np.asarray(rows, dtype=float)
        if data.ndim != 2 or data.shape[1] != 12:
            raise ValueError("CTS must have frequency, TS, and five complex responses")
        self.level = min((3, 4, 5), key=lambda x: abs(np.log(float(sampling_rate) / {3:2400,4:150,5:15}[x])))
        expected_fs = {3:2400., 4:150., 5:15.}[self.level]
        if abs(float(sampling_rate)-expected_fs)/expected_fs > .05:
            raise ValueError(f"Sampling rate {sampling_rate} Hz cannot identify CTS TS3/4/5 safely")
        data = data[data[:, 1] == self.level]
        if len(data) < 2:
            raise ValueError(f"No CTS rows for TS{self.level}")
        data = data[np.argsort(data[:, 0])]
        freqs, indices = np.unique(data[:, 0], return_index=True)
        self.freqs = freqs
        data = data[indices]
        self.logamp = {}
        self.phase = {}
        self.cts_nulls = {}
        self.valid_response = {}
        for j, ch in enumerate(self.CHANNELS):
            h = data[:, 2+2*j] + 1j*data[:, 3+2*j]
            if np.any(np.abs(h) < 1e-100):
                raise ValueError(f"CTS contains zero response for {ch}")
            amp = np.abs(h)
            # Exact/near-zero SysCal notches make phase undefined. Do not
            # include them in unwrap or bridge across them in interpolation.
            typical = float(np.median(amp[amp > 0]))
            valid = np.isfinite(amp) & (amp > typical * 1e-6)
            self.cts_nulls[ch] = self.freqs[~valid]
            self.valid_response[ch] = valid
            self.logamp[ch] = np.log(np.maximum(amp, np.finfo(float).tiny))
            self.phase[ch] = np.angle(h)
            if np.count_nonzero(valid) < 2:
                raise ValueError(f"CTS TS{self.level} {ch}: fewer than two valid response points")

    def response(self, f, ch):
        """Complex calibration at f, refusing interpolation through response nulls.

        For dense phase-rotating CTS, interpolation in complex Cartesian space
        between adjacent samples avoids choosing an arbitrary global unwrap
        branch. Interpolation is local, and is never allowed across a null.
        """
        if not np.isfinite(f) or f <= 0 or not self.freqs[0] <= f <= self.freqs[-1]:
            raise ValueError(f"CTS TS{self.level} {ch}: {f:g} Hz outside calibrated "
                             f"range [{self.freqs[0]:g}, {self.freqs[-1]:g}] Hz")
        ix = int(np.searchsorted(self.freqs, f, side="left"))
        good = self.valid_response[ch]
        if ix < len(self.freqs) and np.isclose(self.freqs[ix], f, rtol=1e-10, atol=0):
            if not good[ix]:
                raise ValueError(f"CTS_NULL TS{self.level} {ch} at {f:.9g} Hz")
            return complex(np.exp(self.logamp[ch][ix] + 1j*self.phase[ch][ix]))
        if ix == 0 or ix == len(self.freqs):
            raise ValueError(f"CTS TS{self.level} {ch}: no bracketing points at {f:g} Hz")
        lo, hi = ix - 1, ix
        if not (good[lo] and good[hi]):
            raise ValueError(f"CTS_NULL TS{self.level} {ch}: interpolation at {f:.9g} Hz "
                             f"would cross null in [{self.freqs[lo]:.9g}, {self.freqs[hi]:.9g}] Hz")
        # Local polar interpolation with shortest phase increment; unlike
        # global unwrap this cannot propagate a null-induced phase error.
        t = (np.log(f) - np.log(self.freqs[lo])) / (np.log(self.freqs[hi]) - np.log(self.freqs[lo]))
        delta = np.angle(np.exp(1j*(self.phase[ch][hi] - self.phase[ch][lo])))
        amp = np.exp((1-t)*self.logamp[ch][lo] + t*self.logamp[ch][hi])
        return complex(amp * np.exp(1j*(self.phase[ch][lo] + t*delta)))

    def validate_frequency(self, f, channels):
        for ch in channels:
            self.response(f, ch)


def _resolve_group_cts(param_path, p, out, group, channel_indices):
    """Resolve a receiver-specific CTS, never share a local CTS with remote data."""
    explicit = p.cts_file if group == "local" else p.rr_cts_file
    sources = [p.data_sections[0][1][i][0] for i in channel_indices]
    folders = []
    for source in sources:
        if is_mth5_spec(source):
            _, info = _resolve_mth5_spec_abs(param_path, source)
            folders.append(Path(info["path"]).parent)
    folders = list(dict.fromkeys(folders + [param_path.parent]))
    if explicit:
        fp = Path(os.path.expandvars(os.path.expanduser(explicit)))
        if not fp.is_absolute():
            fp = param_path.parent / fp
        obj = ExternalPhoenixCTS(fp, p.sampling_freq)
        out.write_log(f"{group.upper()} CTS selected explicitly: {fp}", elapsed=False)
        return obj
    candidates = []
    for folder in folders:
        for pattern in ("*.cts", "*.CTS"):
            for candidate in sorted(folder.glob(pattern)):
                if candidate not in candidates:
                    candidates.append(candidate)
    valid = []
    for candidate in candidates:
        try:
            valid.append(ExternalPhoenixCTS(candidate, p.sampling_freq))
        except (ValueError, OSError) as exc:
            out.write_log(f"{group.upper()} CTS candidate rejected: {candidate}: {exc}", elapsed=False)
    if len(valid) != 1:
        raise ValueError(f"{group.upper()} raw ADC counts require one unambiguous CTS; "
                         f"found {len(valid)}: {[str(x.filename) for x in valid]}. "
                         f"Specify {'CTS_FILE' if group == 'local' else 'RR_CTS_FILE'}.")
    out.write_log(f"{group.upper()} CTS auto-selected: {valid[0].filename}", elapsed=False)
    return valid[0]


def prepare_external_cts(param_path, p, out):
    """Independently route local and remote MTH5 channels by their stored units.

    Returns (per-channel CTS map, ADC scaling map, physical-unit scaling map).
    Remote reference counts MUST be calibrated by the remote receiver's CTS.
    """
    n = p.nchan
    files = p.data_sections[0][1]
    channel_cts = {}
    count_scales = {}
    physical_scales = {}
    group_specs = [("local", list(range(min(4, n))))]
    if p.num_rr:
        group_specs.append(("remote", list(range(p.num_out + p.num_input, n))))
    # MTU-5C TD is stored as voltage and must use its embedded RXCAL/SCAL
    # calibration chain. Never attempt to interpret a voltage waveform as
    # MTU-5A raw ADC counts or force a SysCal CTS onto it.
    voltage_channels = []
    counts_channels = []
    for group, indices in group_specs:
        for i in indices:
            fname = files[i][0]
            if not is_mth5_spec(fname):
                continue
            source, _ = _resolve_mth5_spec_abs(param_path, fname)
            _, meta = read_mth5_channel_array(source, skip=0, n=1)
            unit = _normalize_unit_token(str(meta.get("units") or ""))
            if unit in ("v", "mv"):
                voltage_channels.append(i)
            elif unit == "counts":
                counts_channels.append(i)
    if voltage_channels:
        if counts_channels:
            raise ValueError(
                "Mixed voltage and raw ADC-count MTH5 inputs are not supported "
                "in the same processing pass: use separate calibrated runs. "
                f"voltage channels={voltage_channels}; counts channels={counts_channels}"
            )
        out.write_log(
            "MTU-5C VOLTAGE MTH5: route all voltage channels through embedded "
            "RXCAL/SCAL calibration and dipole geometry; external CTS disabled. "
            f"voltage channels={voltage_channels}", elapsed=False)
        # Existing mth5_calibration_preflight and
        # align_mth5_sections_to_unit_mode perform the embedded response chain.
        # Do not bypass those routines merely because the metadata has filters.
        return {}, {}, {}
    for group, indices in group_specs:
        raw = []
        for i in indices:
            fname = files[i][0]
            if not is_mth5_spec(fname):
                out.write_log(f"{group.upper()} ch{i}: non-MTH5 source; legacy CAL_FILES handling", elapsed=False)
                continue
            source, info = _resolve_mth5_spec_abs(param_path, fname)
            _, meta = read_mth5_channel_array(source, skip=0, n=1)
            unit = _normalize_unit_token(str(meta.get("units") or ""))
            state = _calibration_state(meta)
            out.write_log(f"{group.upper()} CTS input ch{i} {info['channel']}: "
                          f"unit={unit}, state={state}", elapsed=False)
            if unit == "counts" and state != "applied":
                raw.append(i)
            elif unit in ("mv/km", "v/m", "nt", "t", "a/m", "pt", "uv/m"):
                if unit == "v/m": scale = 1e6
                elif unit == "uv/m": scale = 1e3
                elif unit == "t": scale = 1e9
                elif unit == "pt": scale = 1e-3
                elif unit == "a/m": scale = MU0 * 1e9
                else: scale = 1.0
                physical_scales[i] = scale
            else:
                raise ValueError(f"{group.upper()} ch{i}: ambiguous MTH5 unit {unit!r}, "
                                 f"calibration state {state}; refusing unsafe calibration")
        if raw:
            cts = _resolve_group_cts(param_path, p, out, group, raw)
            for i in raw:
                channel_cts[i] = cts
                count_scales[i] = 1.0 / (2**23)
            out.write_log(f"{group.upper()} CTS ACTIVE: channels={raw}, "
                          f"TS{cts.level}, file={cts.filename}", elapsed=False)
        else:
            out.write_log(f"{group.upper()} CTS BYPASSED: no raw ADC-count channels", elapsed=False)
    return channel_cts, count_scales, physical_scales


def rotate_calibrated_spectra(ft: np.ndarray, degrees: float, p: Params) -> np.ndarray:
    """Passive clockwise rotation of calibrated horizontal N/E components.

    Positive angle: new X axis points clockwise from North toward East.
    Columns: Ex, Ey, Hx, Hy, optionally Hxr, Hyr. Hz stays unchanged.
    """
    if not np.isfinite(degrees):
        raise ValueError("ROTATION angle must be finite")
    if abs(degrees) < 1e-12:
        return ft
    if p.num_out != 2 or p.num_input != 2:
        raise ValueError("ROTATION requires Ex,Ey,Hx,Hy (NUM_OUT=2 and two local magnetic inputs)")
    out = np.array(ft, copy=True)
    c, sn = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    pairs = [(0, 1), (2, 3)]
    if p.num_rr >= 2:
        pairs.append((p.num_out + p.num_input, p.num_out + p.num_input + 1))
    for a, b in pairs:
        xa, xb = ft[:, a].copy(), ft[:, b].copy()
        out[:, a] = c * xa + sn * xb
        out[:, b] = -sn * xa + c * xb
    return out


def process(param_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    param_path = Path(param_path)
    p = parse_param(param_path)
    # Legacy CTS_MODE=off/on is ignored for MTH5 inputs: the waveform units
    # alone determine whether CTS is required. This avoids unsafe GUI state.
    if any(is_mth5_spec(fn) for _, files in p.data_sections for fn, _ in files):
        p.cts_mode = "auto"
    outdir = Path(output_dir) if output_dir else param_path.parent
    out = OutputFiles(outdir)
    try:
        out.write_log("Read parameters.")
        out.write_log(f"Processing Method type: {p.procedure} (0=RR, 1=RRMS)", elapsed=False)
        out.write_log(f"Number of channels: {p.nchan}", elapsed=False)
        out.write_log(f"Error Estimation mode: {p.error_estimation} (0=Parametric, 1=Fixed-weights bootstrap, 2=Jackknife, 3=Robust bootstrap, 4=Strict bootstrap); bootstrap={p.bootstrap}", elapsed=False)
        out.write_log(f"ROTATION={p.rotation_deg:g} degrees (positive clockwise axis, North toward East); applied to calibrated spectra before estimation", elapsed=False)
        sections = load_data(param_path, p, out)
        out.write_log(f"CTS_MODE={p.cts_mode}; No-RR procedure and channel mapping unchanged", elapsed=False)
        channel_cts, external_scales, physical_scales = prepare_external_cts(param_path, p, out)
        # MTH5 waveforms are either calibrated by their own CTS or already physical.
        # Never run the embedded response chain again on these channels.
        if channel_cts or physical_scales:
            p.apply_mth5_response = False
        else:
            if p.apply_mth5_response:
                mth5_calibration_preflight(param_path, p, out)
            sections = align_mth5_sections_to_unit_mode(sections, param_path, p, out)
        for sec in sections:
            for ci, scale in external_scales.items():
                sec[:, ci] *= scale
            for ci, scale in physical_scales.items():
                sec[:, ci] *= scale
        sections = normalize_manual_input_units_to_tracmt(sections, param_path, p, out)
        sections = [apply_filters(sec, p.sampling_freq, p) for sec in sections]
        sections = [apply_robust_filter(sec, p, out) for sec in sections]
        sections, coeffs = apply_prewhitening(sections, p, out)
        cals = load_calibrations(param_path, p, out, cal_outdir=outdir / "mth5_cal_files")
        for ci in set(channel_cts) | set(physical_scales):
            cals[ci] = None
        blocks = collect_frequency_blocks(sections, p)
        blocks.sort(key=lambda x: x[2])

        resp_rows = []
        rho_rows = []
        engine_rows = []
        edi_rows = []
        cts_skipped = 0
        for group, indices in (("LOCAL", range(min(4, p.nchan))),
                               ("REMOTE", range(p.num_out+p.num_input, p.nchan))):
            objs = [channel_cts[i] for i in indices if i in channel_cts]
            if objs:
                obj = objs[0]
                out.write_log(f"{group} CTS VALID FREQUENCY BAND: TS{obj.level}, "
                              f"{obj.freqs[0]:g} to {obj.freqs[-1]:g} Hz", elapsed=False)
        for L, k, freq, ft0 in blocks:
            if freq <= 0:
                continue
            invalid = [(i, obj) for i, obj in channel_cts.items()
                       if not (obj.freqs[0] <= freq <= obj.freqs[-1])]
            if invalid:
                cts_skipped += 1
                out.write_log(f"CTS SKIP out-of-band: f={freq:.9g} Hz; "
                              + ", ".join(f"ch{i} TS{obj.level} "
                                          f"[{obj.freqs[0]:g},{obj.freqs[-1]:g}]"
                                          for i, obj in invalid), elapsed=False)
                continue
            # Fail closed on CTS response nulls. Never divide by a near-zero
            # calibration and never silently create a corrupted EDI row.
            try:
                for ci, obj in channel_cts.items():
                    ch = ("ex", "ey", "hx", "hy")[ci] if ci < 4 else ("hx", "hy")[(ci-4) % 2]
                    obj.response(freq, ch)
            except ValueError as exc:
                cts_skipped += 1
                out.write_log(f"CTS SKIP invalid complex response at f={freq:.9g} Hz: {exc}", elapsed=False)
                continue
            out.write_both("=" * 80, elapsed=False)
            out.write_both(_tracmt_freq_header(freq), elapsed=False)
            out.write_both("=" * 80, elapsed=False)
            ft = apply_calibration_ft(ft0, freq, cals)
            for ci, obj in channel_cts.items():
                ch = ("ex", "ey", "hx", "hy")[ci] if ci < 4 else ("hx", "hy")[(ci-4) % 2]
                practical_scale = 1e6 if ci < p.num_out else 1e9
                ft[:, ci] = ft[:, ci] / obj.response(freq, ch) * practical_scale
            ft = rotate_calibrated_spectra(ft, p.rotation_deg, p)
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
            resp_rows.append(_annotate_multisegment_row(
                _build_tracmt_response_row(freq, period, ft, Z, err_bars, p, Zw=z, weights_by_out=weights_by_out),
                L, k, len(ft),
            ))
            engine_rows.append(_annotate_multisegment_row(
                _build_engine_response_row(freq, period, istf, Z, err_bars, scale, weights, len(ft)),
                L, k, len(ft),
            ))
            if p.output_rhoa_phs:
                rho_rows.append(_annotate_multisegment_row(
                    _build_tracmt_rho_row(freq, period, ft, Z, err_bars, p, Zw=z, weights_by_out=weights_by_out),
                    L, k, len(ft),
                ))
            erow = {"freq": freq}
            for i, j, name in TRACMT_Z_COMPONENTS:
                if i < Z.shape[0] and j < Z.shape[1]:
                    erow[name] = Z[i, j]
            edi_rows.append(erow)
            if p.output_level >= 1:
                pd.DataFrame({"MD": md, "weight": weights}).to_csv(outdir / f"residuals_{freq:.6e}.csv", index=False)

        if len(resp_rows) > 1 and len(p.segments) > 1:
            out.write_log(
                f"Multi-segment raw estimates: {len(resp_rows)} rows from {len(p.segments)} SEGMENT lengths",
                elapsed=False,
            )
            if engine_rows:
                pd.DataFrame(engine_rows).to_csv(
                    outdir / "response_functions_multisegment.csv",
                    index=False, float_format=TRACMT_CSV_FLOAT_FORMAT, encoding="utf-8-sig",
                )
            if rho_rows:
                _normalize_rho_rows_phase_pm90(rho_rows, p)
                pd.DataFrame(rho_rows).to_csv(
                    outdir / "apparent_resistivity_and_phase_multisegment.csv",
                    index=False, float_format=TRACMT_CSV_FLOAT_FORMAT, encoding="utf-8-sig",
                )
            resp_rows = _stitch_result_rows(resp_rows, p)
            rho_rows = _stitch_result_rows(rho_rows, p) if rho_rows else rho_rows
            engine_rows = _stitch_result_rows(engine_rows, p) if engine_rows else engine_rows
            out.write_log(f"Stitched to {len(resp_rows)} frequency bins", elapsed=False)
            # Rebuild EDI from stitched Z rows
            edi_rows = []
            for r in resp_rows:
                freq = float(r.get("frequency", r.get("freq", np.nan)))
                erow = {"freq": freq}
                for _oi, _j, name in TRACMT_Z_COMPONENTS:
                    lab = _z_component_label(_oi, _j)
                    rr, ri = r.get(f"{lab}_real", np.nan), r.get(f"{lab}_imag", np.nan)
                    if np.isfinite(rr) and np.isfinite(ri):
                        erow[name] = complex(rr, ri)
                edi_rows.append(erow)

        # legacy build: final reported MT phases use the equivalent +/-90-degree branch.
        # Z real/imaginary estimates and EDI impedance are intentionally unchanged.
        _normalize_rho_rows_phase_pm90(rho_rows, p)

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


def _preview_apply_scal(y_volt: np.ndarray, fs: float, tables: Sequence["MTH5FilterTable"]) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Convert induction-coil Volt samples to delta-B [nT] using SCAL FAP.

    The SCAL response is inspected instead of assuming that SCAL contains a DC/local
    geomagnetic-field offset.  Induction coils measure the varying field (delta B), not
    the absolute geomagnetic field.  When units identify a physical sensor response,
    the conversion direction is selected from units_in/units_out.  If units are absent,
    the legacy Phoenix convention is used only as an explicit fallback and is reported.
    """
    yy = np.asarray(y_volt, dtype=float)
    scal_tabs = [t for t in tables if "scal" in _filter_name_key(getattr(t, "name", ""))]
    details = []
    if not scal_tabs or yy.size < 2 or not np.isfinite(fs) or fs <= 0:
        return yy.copy(), [], details
    n = yy.size
    f = np.fft.rfftfreq(n, d=1.0 / float(fs))
    Y = np.fft.rfft(yy)
    resp = np.ones(f.shape, dtype=complex)
    direction = "unknown"
    scale_to_nt = 1.0
    for tab in scal_tabs:
        uin = _normalize_unit_token(getattr(tab, "units_in", ""))
        uout = _normalize_unit_token(getattr(tab, "units_out", ""))
        r = np.asarray(tab.response_on(np.maximum(f, EPS)), dtype=complex)
        resp *= r
        details.append({"name": str(tab.name), "kind": str(tab.kind),
                        "units_in": str(getattr(tab, "units_in", "")),
                        "units_out": str(getattr(tab, "units_out", "")),
                        "freqs": getattr(tab, "freqs", None),
                        "complex_resp": getattr(tab, "complex_resp", None),
                        "gain": getattr(tab, "gain", 1.0)})
        # mt_metadata filter convention: response maps units_in -> units_out.
        if uin in ("nt", "t") and uout in ("v", "mv"):
            direction = "divide_sensor_response"   # stored V / (V/nT) -> nT
            # convert response output-voltage unit to V because waveform is stored V
            if uout == "mv": rfac = 1e-3
            else: rfac = 1.0
            # convert input field unit to nT
            if uin == "t": rfac *= 1e9
            resp = resp / np.where(np.abs(r) > EPS, r, 1.0) * (r * rfac)
        elif uin in ("v", "mv") and uout in ("nt", "t"):
            direction = "multiply_conversion_response"
            # response maps voltage -> field; normalize V input and nT output.
            rfac = (1000.0 if uin == "mv" else 1.0)
            if uout == "t": rfac *= 1e9
            resp = resp / np.where(np.abs(r) > EPS, r, 1.0) * (r * rfac)
    good = np.isfinite(resp.real) & np.isfinite(resp.imag) & (np.abs(resp) > EPS)
    good[0] = False
    Yc = Y.copy()
    if direction == "multiply_conversion_response":
        Yc[good] = Y[good] * resp[good]
    elif direction == "divide_sensor_response":
        Yc[good] = Y[good] / resp[good]
    else:
        # Explicit fallback requested in V12.7: 1 mV = 1 nT, then SCAL correction.
        # Keep it visible in the log/inspector because units were insufficient to prove it.
        Y0 = Y * 1000.0
        Yc = Y0.copy()
        Yc[good] = Y0[good] / resp[good]
        direction = "fallback_Vx1000_then_divide_SCAL_UNITS_UNKNOWN"
    Yc[~good] = Y[~good] if direction != "fallback_Vx1000_then_divide_SCAL_UNITS_UNKNOWN" else (Y * 1000.0)[~good]
    return np.fft.irfft(Yc, n=n).real, [str(t.name) for t in scal_tabs], details + [{"direction": direction}]


def _write_scal_inspector(outdir: Path, channel: str, details: List[dict]) -> None:
    """Write human-readable SCAL metadata plus FAP frequency/amplitude/phase CSV."""
    if not details:
        return
    rows = []
    summary = [f"SCAL Inspector V1.1 - {channel}",
               "Induction-coil preview represents delta-B; no local absolute geomagnetic field is added."]
    for d in details:
        if "direction" in d:
            summary.append(f"conversion_direction={d['direction']}")
            continue
        summary += [f"name={d.get('name','')}", f"kind={d.get('kind','')}",
                    f"units_in={d.get('units_in','')}", f"units_out={d.get('units_out','')}",
                    f"gain={d.get('gain',1.0)}"]
        ff=d.get("freqs"); cc=d.get("complex_resp")
        if ff is not None and cc is not None:
            ff=np.asarray(ff,float); cc=np.asarray(cc,complex)
            for fq,z in zip(ff,cc):
                rows.append({"channel":channel,"filter_name":d.get("name",""),"frequency_Hz":fq,
                             "amplitude":abs(z),"phase_deg":np.degrees(np.angle(z)),
                             "units_in":d.get("units_in",""),"units_out":d.get("units_out","")})
    (Path(outdir)/f"SCAL_inspector_{channel}.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_csv(Path(outdir)/f"SCAL_FAP_{channel}.csv",index=False,encoding="utf-8-sig")


def _read_preview_data(param_file: Path, p: Params, outdir: Optional[Path] = None) -> Tuple[np.ndarray, List[str]]:
    """Read preview: Ex/Ey Volt -> mV/km using MTH5 dipole length; coil Volt -> delta-B nT via SCAL."""
    cols, names = [], []
    if not p.data_sections:
        return np.empty((0, 0)), []
    nread, files = p.data_sections[0]
    fs_candidates = []
    conversion_notes = []
    inspector_cache = []
    for ich, (fname, skip) in enumerate(files):
        src = _resolve_channel_source(Path(param_file), fname)
        if is_mth5_spec(src):
            arr, meta = read_mth5_channel_array(src, skip=skip, n=nread)
            info = parse_mth5_spec(src); ch = str(info["channel"] or "").lower(); names.append(ch)
            fs0 = float(meta.get("sample_rate", np.nan))
            if np.isfinite(fs0) and fs0 > 0: fs_candidates.append(fs0)
            if ch in ("ex", "ey", "ez"):
                dipole_m = _positive_length_m(meta.get("dipole_length"))
                if dipole_m is None: dipole_m = _dipole_from_length_pair(meta.get("length1"), meta.get("length2"))
                if dipole_m is None or not np.isfinite(dipole_m) or dipole_m <= 0:
                    raise ValueError(f"Preview conversion for {ch.upper()} requires the actual MTH5 dipole length; none was resolved from {info['path']}.")
                arr = np.asarray(arr,float)*(1e6/float(dipole_m))
                conversion_notes.append(f"{ch}: stored Volt / {dipole_m:g} m x 1e6 -> mV/km")
            elif ch in ("hx","hy","hz","bx","by","bz"):
                path=_candidate_existing_path(Path(info["path"]))
                tables=_load_all_named_filters_for_channel(path,info,meta,force=True)
                arr, scal_names, details = _preview_apply_scal(np.asarray(arr,float), fs0 if np.isfinite(fs0) else p.sampling_freq, tables)
                direction=next((d.get("direction") for d in details if "direction" in d),"unknown")
                conversion_notes.append(f"{ch}: stored Volt -> SCAL -> delta-B nT; SCAL="+(" + ".join(scal_names) if scal_names else "not found")+f"; mode={direction}; absolute field NOT added")
                inspector_cache.append((ch,details))
        else:
            fpath=Path(src); arr=read_channel(fpath,skip=skip,n=nread); names.append(str(fpath.name))
            mode=str(getattr(p,"input_unit_mode","auto_from_mth5") or "auto_from_mth5").lower()
            if not mode.startswith("auto"):
                if mode in ("si_b_tesla","si-b","sib","si_b"):
                    scale=1e6 if ich < p.num_out else 1e9
                elif mode in ("si_h_ampere","si-h","sih","si_h"):
                    scale=1e6 if ich < p.num_out else MU0*1e9
                else:
                    scale=1.0
                arr=np.asarray(arr,float)*scale
                conversion_notes.append(f"ch{ich} {fpath.name}: manual {mode} scale x{scale:.12g} -> TRACMT practical")
        cols.append(np.asarray(arr,dtype=float))
    if not cols: return np.empty((0,0)),[]
    nmin=min(len(c) for c in cols); data=np.column_stack([c[:nmin] for c in cols])
    if fs_candidates:
        p.sampling_freq=float(np.median(np.asarray(fs_candidates,float))); p.sampling_freq_org=p.sampling_freq
    if outdir is not None:
        outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
        labels=[]
        for nm in names:
            ch=str(nm).lower()
            labels.append(f"{ch}_mV_per_km" if ch in ("ex","ey","ez") else (f"{ch}_deltaB_nT" if ch in ("hx","hy","hz","bx","by","bz") else Path(str(nm)).stem))
        df=pd.DataFrame(data,columns=labels); df.insert(0,"sample_index",np.arange(len(df),dtype=np.int64))
        if np.isfinite(p.sampling_freq) and p.sampling_freq>0: df.insert(1,"time_s",df["sample_index"].to_numpy(float)/float(p.sampling_freq))
        df.to_csv(outdir/"time_series_preview.csv",index=False,encoding="utf-8-sig")
        (outdir/"time_series_preview_conversion.txt").write_text("V1.1 preview conversion\n"+"\n".join(conversion_notes)+"\n",encoding="utf-8")
        for ch,details in inspector_cache: _write_scal_inspector(outdir,ch,details)
    return data,names

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
        data, names = _read_preview_data(param_file, p, outdir=outdir)
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
        self.root.title("pyTRACMT Professional V1.1 - MT Input Diagnostic + SCAL calibrated")
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
        self.gui_unit_mode = tk.StringVar(value="Auto from MTH5")
        self.gui_rotation_deg = tk.StringVar(value="0")
        self.gui_estimator = tk.StringVar(value="Huber")
        self.gui_estimator2 = tk.StringVar(value="Tukey")
        self.gui_error = tk.StringVar(value="1 Fixed-weights Bootstrap")
        self.gui_bootstrap = tk.StringVar(value="1000")
        self.gui_seg_text = tk.StringVar(value="AUTO_FAST")
        self.gui_segment_mode = tk.StringVar(value="Auto")
        self.gui_manual_segment = tk.StringVar(value="1024:40,50,60,70,80,90,100,120,140,160,180,200")
        self.gui_data_count = tk.StringVar(value="")

        # MTH5 helpers
        self.gui_mth5_file = tk.StringVar(value="")
        self.gui_mth5_survey = tk.StringVar(value="")
        self.gui_mth5_station = tk.StringVar(value="")
        self.gui_mth5_run = tk.StringVar(value="")
        self.gui_apply_mth5_response = tk.BooleanVar(value=True)
        self.gui_cts_mode = tk.StringVar(value="Auto (unapplied MTH5 filters)")
        self.gui_cts_file = tk.StringVar(value="")
        self.gui_rr_cts_file = tk.StringVar(value="")
        self.gui_mth5_force_coil_cal = tk.BooleanVar(value=False)
        self.gui_rr_mth5_file = tk.StringVar(value="")
        self.gui_rr_mth5_station = tk.StringVar(value="")
        self.gui_rr_mth5_run = tk.StringVar(value="")
        self._mth5_inventory: List[Dict[str, object]] = []
        self._rr_mth5_inventory: List[Dict[str, object]] = []

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
                     values=["Auto from MTH5",
                             "TRACMT practical: E=mV/km, B=nT",
                             "SI-B: E=V/m, B=T",
                             "SI-H: E=V/m, H=A/m"]).grid(row=1, column=1, columnspan=4, sticky="w", padx=4, pady=4)

        ttk.Label(basic, text="Rotation (degrees; + clockwise N to E)").grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Entry(basic, textvariable=self.gui_rotation_deg, width=12).grid(row=2, column=2, sticky="w", padx=4, pady=4)

        proc = ttk.LabelFrame(outer, text="Processing Method / Error Estimation")
        proc.pack(fill=tk.X, pady=6)
        ttk.Label(proc, text="Processing procedure").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.proc_combo = ttk.Combobox(proc, textvariable=self.gui_procedure, state="readonly", width=44,
                     values=["No-RR (Non-robust, no remote reference)",
                             "Ordinary Remote Reference (RR)",
                             "Robust Multivariate Regression (RRMS)"])
        self.proc_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.proc_combo.bind("<<ComboboxSelected>>", self._on_procedure_selected)

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

        ttk.Label(adv, text="CAL_FILES, comma/semicolon separated (blank = auto from MTH5 coils)").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(adv, textvariable=self.gui_cal_files, width=90).grid(row=4, column=1, sticky="we", padx=4, pady=3)
        ttk.Label(adv, text="CTS spectral calibration (No-RR / RR / RRMS)").grid(row=4, column=2, sticky="w", padx=4, pady=(2, 0))
        ttk.Label(adv, text="Automatic by MTH5 units: counts -> CTS; physical -> bypass").grid(
            row=5, column=2, sticky="w", padx=4, pady=3)
        ctsrow = ttk.Frame(adv); ctsrow.grid(row=7, column=2, sticky="we", padx=4, pady=3)
        ttk.Entry(ctsrow, textvariable=self.gui_cts_file, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ctsrow, text="Browse Local CTS", command=lambda: self.gui_cts_file.set(
            filedialog.askopenfilename(filetypes=[("Phoenix CTS", "*.cts"), ("All files", "*.*")])
            or self.gui_cts_file.get())).pack(side=tk.LEFT, padx=3)
        rrctsrow = ttk.Frame(adv); rrctsrow.grid(row=8, column=2, sticky="we", padx=4, pady=3)
        ttk.Label(rrctsrow, text="Remote CTS").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(rrctsrow, textvariable=self.gui_rr_cts_file, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rrctsrow, text="Browse Remote CTS", command=lambda: self.gui_rr_cts_file.set(
            filedialog.askopenfilename(filetypes=[("Phoenix CTS", "*.cts"), ("All files", "*.*")])
            or self.gui_rr_cts_file.get())).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(adv, text="Force FAP cal even if filter.applied=True (expert only)",
                        variable=self.gui_mth5_force_coil_cal).grid(row=6, column=2, sticky="w", padx=4, pady=3)
        ttk.Label(adv, text="Filters: HP / LP / Notch Hz").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        ff = ttk.Frame(adv); ff.grid(row=5, column=1, sticky="w", padx=4, pady=3)
        ttk.Entry(ff, textvariable=self.gui_high_pass, width=10).pack(side=tk.LEFT)
        ttk.Entry(ff, textvariable=self.gui_low_pass, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Entry(ff, textvariable=self.gui_notch, width=30).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(ff, text="Output EDI", variable=self.gui_output_edi).pack(side=tk.LEFT, padx=10)
        adv.columnconfigure(1, weight=1)

        seg = ttk.LabelFrame(outer, text="Frequency Windows / SEGMENT")
        seg.pack(fill=tk.X, pady=6)
        ttk.Label(seg, text="Frequency windows / SEGMENT mode").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.segment_mode_combo = ttk.Combobox(seg, textvariable=self.gui_segment_mode, state="readonly", width=18,
                                               values=["Auto", "Full", "Manual"])
        self.segment_mode_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.segment_mode_combo.bind("<<ComboboxSelected>>", self._on_segment_mode_changed)
        ttk.Label(seg, text="Manual (length:indexes; length:indexes)").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.segment_manual_entry = ttk.Entry(seg, textvariable=self.gui_manual_segment, width=70)
        self.segment_manual_entry.grid(row=1, column=1, sticky="we", padx=4, pady=4)
        ttk.Label(seg, text="Data Count (blank = all)").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(seg, textvariable=self.gui_data_count, width=18).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        self._on_segment_mode_changed()
        seg.columnconfigure(1, weight=1)

        mth5 = ttk.LabelFrame(outer, text="MTH5 Input (standard MT archive)")
        mth5.pack(fill=tk.X, pady=6)
        ttk.Label(mth5, text="Local MTH5").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(mth5, textvariable=self.gui_mth5_file, width=70).grid(row=0, column=1, sticky="we", padx=4, pady=3)
        ttk.Button(mth5, text="Browse MTH5", command=self._choose_mth5_file).grid(row=0, column=2, sticky="w", padx=4, pady=3)
        ttk.Button(mth5, text="Scan", command=self._scan_mth5_file).grid(row=0, column=3, sticky="w", padx=4, pady=3)
        ttk.Label(mth5, text="Survey").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.mth5_survey_combo = ttk.Combobox(mth5, textvariable=self.gui_mth5_survey, width=18, state="readonly")
        self.mth5_survey_combo.grid(row=1, column=1, sticky="w", padx=4, pady=3)
        self.mth5_survey_combo.bind("<<ComboboxSelected>>", self._refresh_mth5_station_run_choices)
        ttk.Label(mth5, text="Station").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self.mth5_station_combo = ttk.Combobox(mth5, textvariable=self.gui_mth5_station, width=18, state="readonly")
        self.mth5_station_combo.grid(row=2, column=1, sticky="w", padx=4, pady=3)
        self.mth5_station_combo.bind("<<ComboboxSelected>>", self._refresh_mth5_run_choices)
        ttk.Label(mth5, text="Run").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self.mth5_run_combo = ttk.Combobox(mth5, textvariable=self.gui_mth5_run, width=18, state="readonly")
        self.mth5_run_combo.grid(row=3, column=1, sticky="w", padx=4, pady=3)
        self.mth5_run_combo.bind("<<ComboboxSelected>>", self._on_mth5_run_selected)
        ttk.Button(mth5, text="Fill Ex/Ey/Hx/Hy from MTH5", command=self._fill_channels_from_mth5).grid(row=2, column=2, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Label(mth5, text="RR MTH5 (optional)").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(mth5, textvariable=self.gui_rr_mth5_file, width=70).grid(row=4, column=1, sticky="we", padx=4, pady=3)
        ttk.Button(mth5, text="Browse RR MTH5", command=self._choose_rr_mth5_file).grid(row=4, column=2, sticky="w", padx=4, pady=3)
        ttk.Button(mth5, text="Scan RR", command=self._scan_rr_mth5_file).grid(row=4, column=3, sticky="w", padx=4, pady=3)
        ttk.Label(mth5, text="RR Station/Run").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        rrbox = ttk.Frame(mth5); rrbox.grid(row=5, column=1, sticky="w", padx=4, pady=3)
        self.rr_mth5_station_combo = ttk.Combobox(rrbox, textvariable=self.gui_rr_mth5_station, width=14, state="readonly")
        self.rr_mth5_station_combo.pack(side=tk.LEFT)
        self.rr_mth5_station_combo.bind("<<ComboboxSelected>>", self._refresh_rr_mth5_run_choices)
        self.rr_mth5_run_combo = ttk.Combobox(rrbox, textvariable=self.gui_rr_mth5_run, width=14, state="readonly")
        self.rr_mth5_run_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(mth5, text="Fill RR Hx/Hy from RR MTH5", command=self._fill_rr_channels_from_mth5).grid(row=5, column=2, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Label(mth5, text="Unit rules: TRACMT practical needs E=mV/km and B=nT. Fluxgate uses MTH5 coefficient (V/nT) + unit metadata; coils use FAP spectral CAL; electric uses units and dipole_length. SI modes convert accordingly.",
                  wraplength=900, foreground="#555").grid(row=6, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        mth5.columnconfigure(1, weight=1)

        files = ttk.LabelFrame(outer, text="Input Time-series Files / MTH5 channel specs")
        files.pack(fill=tk.X, pady=6)
        self._file_row(files, 0, "Ex", self.gui_ex_file)
        self._file_row(files, 1, "Ey", self.gui_ey_file)
        self._file_row(files, 2, "Hx", self.gui_hx_file)
        self._file_row(files, 3, "Hy", self.gui_hy_file)
        self.rr_check = ttk.Checkbutton(
            files,
            text="Use independent remote-reference Hx/Hy (required by Ordinary RR / RRMS; unused for No-RR)",
            variable=self.gui_use_rr,
            command=self._update_tracmt_control_states,
        )
        self.rr_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        self._file_row(files, 5, "RR Hx", self.gui_hrx_file)
        self._file_row(files, 6, "RR Hy", self.gui_hry_file)
        files.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=10)
        ttk.Button(actions, text="Generate param.dat", command=self.generate_param_from_gui).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Generate and Run", command=self.generate_and_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Save Current param.dat", command=self.save_param_text).pack(side=tk.LEFT, padx=4)

        note = ("Channel order is Ex, Ey, Hx, Hy. RR Hx/Hy are appended only when remote reference is enabled. "
                "No-RR uses local Hx/Hy only (4 channels) and does not require an RR MTH5. "
                "MTH5: coil -> FAP spectral calibration; fluxgate -> coefficient/units in time domain; "
                "electric channels are converted to the selected UNIT_MODE (default mV/km).")
        ttk.Label(outer, text=note, wraplength=900, foreground="#555", justify="left").pack(fill=tk.X, pady=6)
        self._update_tracmt_control_states()

    def _on_procedure_selected(self, event=None):
        method = self.gui_procedure.get()
        if method.startswith("No-RR"):
            self._apply_no_rr_defaults()
        elif method.startswith(("Ordinary", "Robust")):
            self.gui_use_rr.set(True)
        self._update_tracmt_control_states()

    def _on_segment_mode_changed(self, event=None):
        """Keep the manual input visible, editable only in Manual mode."""
        mode = self.gui_segment_mode.get()
        self.segment_manual_entry.configure(state="normal" if mode == "Manual" else "disabled")
        if mode == "Auto":
            self.gui_seg_text.set("AUTO_FAST")
        elif mode == "Full":
            self.gui_seg_text.set("AUTO_FULL")
        else:
            self.gui_seg_text.set(self.gui_manual_segment.get())

    def _apply_no_rr_defaults(self):
        """USUI NonRobust-style OLS defaults, without remote-reference channels."""
        self.gui_estimator.set("None")
        self.gui_estimator2.set("None")
        self.gui_error.set("0 Parametric")
        self.gui_prewhite_enable.set(True)
        self.gui_prewhite_type.set("0: standard AR")
        self.gui_prewhite_degree.set("10")
        self.gui_prewhite_Candidates.set("5")
        # CTS-aware automatic selection must not be overwritten by No-RR preset.
        # Preserve the user's Auto / Full / Manual frequency-window selection.
        self._on_segment_mode_changed()
        self.gui_use_rr.set(False)
        self.gui_hrx_file.set("")
        self.gui_hry_file.set("")
        self.gui_rr_mth5_file.set("")
        self.gui_rr_mth5_station.set("")
        self.gui_rr_mth5_run.set("")

    def _update_tracmt_control_states(self, event=None):
        """Enforce the same option dependencies used by the TRACMT workflow."""
        method = self.gui_procedure.get()
        is_no_rr = method.startswith("No-RR")
        is_ordinary_rr = method.startswith("Ordinary")
        is_rr_proc = is_no_rr or is_ordinary_rr
        rrms_state = "disabled" if is_rr_proc else "normal"
        if is_no_rr:
            est_state = "disabled"
        elif is_ordinary_rr:
            est_state = "readonly"
        else:
            est_state = "disabled"
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
        if err.startswith("3 ") and is_ordinary_rr:
            self.gui_estimator.set("Tukey")
            self.gui_estimator2.set("None")
        if is_no_rr and hasattr(self, "error_combo"):
            self.error_combo.configure(state="readonly")
            if not err.startswith("0 "):
                self.gui_error.set("0 Parametric")
        elif hasattr(self, "error_combo"):
            self.error_combo.configure(state="readonly")
        rr_ui_state = "disabled" if is_no_rr else "normal"
        if hasattr(self, "rr_check"):
            self.rr_check.configure(state=rr_ui_state)
        if is_no_rr and self.gui_use_rr.get():
            self.gui_use_rr.set(False)
        if not is_no_rr and not self.gui_use_rr.get() and not is_rr_proc:
            self.status_var.set("RRMS requires remote reference; enable RR and select RR Hx/Hy (or RR MTH5).")
        elif not is_no_rr and is_ordinary_rr and not self.gui_use_rr.get():
            self.status_var.set("Ordinary RR requires remote reference; enable RR and select RR Hx/Hy (or RR MTH5).")
        elif is_no_rr:
            self.status_var.set("No-RR: local Ex/Ey/Hx/Hy only; RR MTH5 is not required.")

    @staticmethod
    def _mestimator_code(name: str) -> int:
        return {"None": -1, "Huber": 0, "Tukey": 1, "Thomson": 2}.get(name, -1)

    def _validate_tracmt_gui_options(self) -> None:
        method = self.gui_procedure.get()
        err = self.gui_error.get().strip()
        is_no_rr = method.startswith("No-RR")
        if not is_no_rr and not self.gui_use_rr.get():
            raise ValueError(
                "TRACMT RR/RRMS processing requires independent RR Hx and RR Hy channels. "
                "Enable Remote Reference and select both files (or use Fill RR from RR MTH5). "
                "Or choose No-RR."
            )
        is_ordinary_rr = method.startswith("Ordinary")
        if is_no_rr:
            if not err.startswith("0 "):
                raise ValueError("No-RR uses parametric error estimation (0 Parametric).")
        elif is_ordinary_rr:
            if self.gui_estimator.get() == "None" and self.gui_estimator2.get() != "None":
                raise ValueError("The second M-estimator cannot be enabled when the first M-estimator is None.")
            if err.startswith("3 ") and not (self.gui_estimator.get() == "Tukey" and self.gui_estimator2.get() == "None"):
                raise ValueError("TRACMT robust bootstrap requires: first M-estimator = Tukey, second M-estimator = None.")
        elif err.startswith("1 ") or err.startswith("2 "):
            raise ValueError("Fixed-weights bootstrap/jackknife belongs to Ordinary RR. For RRMS use Parametric, Robust Bootstrap, or Strict Bootstrap.")
        reps = int(float(self.gui_bootstrap.get() or 0))
        if err.startswith(("1 ", "3 ", "4 ")) and reps < 4:
            raise ValueError("Resampling repetitions must be at least 4.")

    def _file_row(self, parent, row: int, label: str, var: tk.StringVar):
        ttk.Label(parent, text=label, width=10).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=90).grid(row=row, column=1, sticky="we", padx=4, pady=3)
        ttk.Button(parent, text="Browse", command=lambda v=var: self._choose_data_file(v)).grid(row=row, column=2, sticky="w", padx=4, pady=3)

    def _choose_data_file(self, var: tk.StringVar):
        fp = filedialog.askopenfilename(
            title="Select time-series file or MTH5",
            filetypes=[
                ("MTH5 / HDF5", "*.h5 *.hdf5 *.mth5"),
                ("Data files", "*.txt *.csv *.dat *.*"),
                ("All files", "*.*"),
            ],
        )
        if fp:
            var.set(fp)

    def _choose_mth5_file(self):
        fp = filedialog.askopenfilename(
            title="Select MTH5 file",
            filetypes=[("MTH5 / HDF5", "*.h5 *.hdf5 *.mth5"), ("All files", "*.*")],
        )
        if fp:
            self.gui_mth5_file.set(fp)
            self._scan_mth5_file()

    def _choose_rr_mth5_file(self):
        fp = filedialog.askopenfilename(
            title="Select remote-reference MTH5 file",
            filetypes=[("MTH5 / HDF5", "*.h5 *.hdf5 *.mth5"), ("All files", "*.*")],
        )
        if fp:
            self.gui_rr_mth5_file.set(fp)
            self._scan_rr_mth5_file()

    def _unique_sorted(self, values: Iterable[object]) -> List[str]:
        out = sorted({str(v) for v in values if str(v).strip()})
        return out

    def _scan_mth5_file(self):
        path = self.gui_mth5_file.get().strip()
        if not path:
            messagebox.showwarning("MTH5", "Select an MTH5 file first.")
            return
        try:
            self._mth5_inventory = list_mth5_inventory(path)
            surveys = self._unique_sorted(r.get("survey", "") for r in self._mth5_inventory)
            self.mth5_survey_combo.configure(values=surveys)
            if surveys:
                self.gui_mth5_survey.set(surveys[0])
            self._refresh_mth5_station_run_choices()
            # SAMPLING_FREQ is synchronized from the selected station/run, not the first file channel.
            self._sync_sampling_freq_from_selected_mth5_run(warn=False)
            self._log(f"MTH5 scan OK: {len(self._mth5_inventory)} channels in {path}")
            self.status_var.set(f"MTH5 scanned: {len(self._mth5_inventory)} channels")
        except Exception as e:
            messagebox.showerror("MTH5 scan error", str(e))
            self._log(traceback.format_exc())

    def _scan_rr_mth5_file(self):
        path = self.gui_rr_mth5_file.get().strip()
        if not path:
            messagebox.showwarning("RR MTH5", "Select a remote-reference MTH5 file first.")
            return
        try:
            self._rr_mth5_inventory = list_mth5_inventory(path)
            stations = self._unique_sorted(r.get("station", "") for r in self._rr_mth5_inventory)
            self.rr_mth5_station_combo.configure(values=stations)
            if stations:
                self.gui_rr_mth5_station.set(stations[0])
            self._refresh_rr_mth5_run_choices()
            self._log(f"RR MTH5 scan OK: {len(self._rr_mth5_inventory)} channels in {path}")
        except Exception as e:
            messagebox.showerror("RR MTH5 scan error", str(e))
            self._log(traceback.format_exc())

    def _refresh_mth5_station_run_choices(self, event=None):
        survey = self.gui_mth5_survey.get().strip()
        rows = self._mth5_inventory
        if survey:
            rows = [r for r in rows if str(r.get("survey", "")) == survey or not str(r.get("survey", "")).strip()]
        stations = self._unique_sorted(r.get("station", "") for r in rows)
        self.mth5_station_combo.configure(values=stations)
        if stations and self.gui_mth5_station.get() not in stations:
            self.gui_mth5_station.set(stations[0])
        self._refresh_mth5_run_choices()

    def _refresh_mth5_run_choices(self, event=None):
        survey = self.gui_mth5_survey.get().strip()
        station = self.gui_mth5_station.get().strip()
        rows = self._mth5_inventory
        if survey:
            rows = [r for r in rows if str(r.get("survey", "")) == survey or not str(r.get("survey", "")).strip()]
        if station:
            rows = [r for r in rows if str(r.get("station", "")) == station]
        runs = self._unique_sorted(r.get("run", "") for r in rows)
        self.mth5_run_combo.configure(values=runs)
        if runs and self.gui_mth5_run.get() not in runs:
            self.gui_mth5_run.set(runs[0])
        if runs and self.gui_mth5_run.get():
            self._sync_sampling_freq_from_selected_mth5_run(warn=False)

    def _sync_sampling_freq_from_selected_mth5_run(self, warn: bool = True) -> Optional[float]:
        """Set GUI SAMPLING_FREQ from the currently selected local MTH5 run.

        All usable channels in one run should share one sampling rate.  A mixed-rate
        run is not silently accepted because TRACMT processing requires one fs.
        """
        survey = self.gui_mth5_survey.get().strip()
        station = self.gui_mth5_station.get().strip()
        run = self.gui_mth5_run.get().strip()
        if not station or not run or not self._mth5_inventory:
            return None
        rows = [r for r in self._mth5_inventory
                if str(r.get("station", "")) == station and str(r.get("run", "")) == run]
        if survey:
            rows = [r for r in rows if (not str(r.get("survey", "")).strip())
                    or str(r.get("survey", "")) == survey]
        rates = []
        for r in rows:
            try:
                fs = float(r.get("sample_rate", np.nan))
            except Exception:
                continue
            if np.isfinite(fs) and fs > 0:
                rates.append(fs)
        uniq = sorted(set(round(x, 9) for x in rates))
        if not uniq:
            if warn:
                self._log(f"MTH5 run {run}: no valid sample_rate metadata; SAMPLING_FREQ unchanged")
            return None
        if len(uniq) > 1:
            msg = f"Selected MTH5 run {run!r} contains mixed sample rates: {uniq}. Select a single-rate run."
            if warn:
                messagebox.showerror("MTH5 sample rate", msg)
            self._log(msg)
            return None
        fs = float(uniq[0])
        self.gui_sampling_freq.set(f"{fs:g}")
        self._log(f"MTH5 run selected: station={station} run={run}; SAMPLING_FREQ auto -> {fs:g} Hz")
        return fs

    def _on_mth5_run_selected(self, event=None):
        self._sync_sampling_freq_from_selected_mth5_run(warn=True)
        if self.gui_unit_mode.get().lower().startswith("auto"):
            self._auto_unit_mode_from_selected_mth5_run(warn=True)

    def _refresh_rr_mth5_run_choices(self, event=None):
        station = self.gui_rr_mth5_station.get().strip()
        rows = self._rr_mth5_inventory
        if station:
            rows = [r for r in rows if str(r.get("station", "")) == station]
        runs = self._unique_sorted(r.get("run", "") for r in rows)
        self.rr_mth5_run_combo.configure(values=runs)
        if runs and self.gui_rr_mth5_run.get() not in runs:
            self.gui_rr_mth5_run.set(runs[0])

    def _mth5_channel_map(self, inventory: Sequence[Dict[str, object]], station: str, run: str,
                          survey: str = "") -> Dict[str, str]:
        rows = [r for r in inventory if str(r.get("station", "")) == station and str(r.get("run", "")) == run]
        if survey:
            rows = [r for r in rows if (not str(r.get("survey", "")).strip()) or str(r.get("survey", "")) == survey]
        available = {str(r.get("channel", "")).lower(): str(r.get("channel", "")) for r in rows}
        out: Dict[str, str] = {}
        for want, cands in (
            ("ex", ("ex",)),
            ("ey", ("ey",)),
            ("hx", ("hx", "bx")),
            ("hy", ("hy", "by")),
            ("hz", ("hz", "bz")),
        ):
            for c in cands:
                if c in available:
                    out[want] = available[c]
                    break
        return out

    def _fill_channels_from_mth5(self):
        path = self.gui_mth5_file.get().strip()
        station = self.gui_mth5_station.get().strip()
        run = self.gui_mth5_run.get().strip()
        survey = self.gui_mth5_survey.get().strip() or None
        if not path or not station or not run:
            messagebox.showwarning("MTH5", "Scan MTH5 and select station/run first.")
            return
        if not self._mth5_inventory:
            self._scan_mth5_file()
        chmap = self._mth5_channel_map(self._mth5_inventory, station, run, survey or "")
        missing = [k for k in ("ex", "ey", "hx", "hy") if k not in chmap]
        if missing:
            messagebox.showerror("MTH5", f"Missing channels in selected run: {', '.join(missing)}")
            return
        self.gui_ex_file.set(make_mth5_spec(path, station, run, chmap["ex"], survey))
        self.gui_ey_file.set(make_mth5_spec(path, station, run, chmap["ey"], survey))
        self.gui_hx_file.set(make_mth5_spec(path, station, run, chmap["hx"], survey))
        self.gui_hy_file.set(make_mth5_spec(path, station, run, chmap["hy"], survey))
        self._sync_sampling_freq_from_selected_mth5_run(warn=True)
        unit_rows = [r for r in self._mth5_inventory
                     if str(r.get("station", "")) == station and str(r.get("run", "")) == run]
        unit_desc = ", ".join(f"{r.get('channel')}={r.get('units') or '[unlabeled]'}" for r in unit_rows)
        self._log(f"Filled local channels from MTH5 station={station} run={run}; stored units: {unit_desc}")
        if self.gui_unit_mode.get().lower().startswith("auto"):
            self._auto_unit_mode_from_selected_mth5_run(warn=True)

    def _fill_rr_channels_from_mth5(self):
        if self.gui_procedure.get().startswith("No-RR"):
            messagebox.showinfo("No-RR", "No-RR mode does not use remote-reference channels. Switch to Ordinary RR / RRMS first.")
            return
        path = self.gui_rr_mth5_file.get().strip() or self.gui_mth5_file.get().strip()
        station = self.gui_rr_mth5_station.get().strip() or self.gui_mth5_station.get().strip()
        run = self.gui_rr_mth5_run.get().strip() or self.gui_mth5_run.get().strip()
        if not path or not station or not run:
            messagebox.showwarning("RR MTH5", "Select RR MTH5 station/run (or local MTH5) first.")
            return
        inv = self._rr_mth5_inventory if self.gui_rr_mth5_file.get().strip() else self._mth5_inventory
        if not inv:
            if self.gui_rr_mth5_file.get().strip():
                self._scan_rr_mth5_file()
                inv = self._rr_mth5_inventory
            else:
                self._scan_mth5_file()
                inv = self._mth5_inventory
        chmap = self._mth5_channel_map(inv, station, run)
        missing = [k for k in ("hx", "hy") if k not in chmap]
        if missing:
            messagebox.showerror("RR MTH5", f"Missing RR channels: {', '.join(missing)}")
            return
        self.gui_hrx_file.set(make_mth5_spec(path, station, run, chmap["hx"]))
        self.gui_hry_file.set(make_mth5_spec(path, station, run, chmap["hy"]))
        self.gui_use_rr.set(True)
        self._update_tracmt_control_states()
        self._log(f"Filled RR channels from MTH5 station={station} run={run}")

    def _parse_segment_builder(self) -> List[SegmentSpec]:
        mode = self.gui_segment_mode.get()
        if mode == "Manual":
            txt = self.gui_manual_segment.get().strip()
        elif mode == "Full":
            txt = "AUTO_FULL"
        else:
            txt = "AUTO_FAST"
        self.gui_seg_text.set(txt)
        if not txt:
            raise ValueError("SEGMENT setting is empty")
        if txt == _USUI_NONROBUST_SEGMENT and self.gui_cts_file.get().strip():
            self._log("Legacy No-RR SEGMENT preset detected with CTS; switching to CTS-aware AUTO_FAST")
            txt = "AUTO_FAST"
            self.gui_seg_text.set(txt)
        txt_upper = txt.upper()
        if txt_upper in ("AUTO", "AUTO_FAST", "AUTO_FULL"):
            # CTS-aware AUTO selection.  The former generic logspace k=2..96
            # missed the calibrated TS4 band at 150 Hz for short FFT windows.
            # Keep the independently verified 150 Hz / 1024-point setup intact.
            fs = float(self.gui_sampling_freq.get())
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError("AUTO SEGMENT requires a positive sampling frequency")
            bands = {15: (0.000013113, 4.6875),
                     150: (5.625, 32.5),
                     2400: (39.70588, 317.6471)}
            nominal = min(bands, key=lambda x: abs(np.log(fs / x)))
            if abs(fs - nominal) / nominal > 0.05:
                # Non-Phoenix MTH5 runs (e.g. MTU5C at 24000 Hz) have no
                # corresponding Phoenix CTS TS3/4/5 calibration band.
                # Generate generic FFT windows rather than rejecting the run.
                # Do not silently reuse TS3 at a different sampling rate.
                if self.gui_cts_file.get().strip() and self.gui_cts_mode.get().strip().lower() != "off":
                    raise ValueError(
                        f"AUTO SEGMENT: {fs:g} Hz has no matching Phoenix CTS TS3/4/5. "
                        "Disable CTS for this run or select a compatible calibration."
                    )
                lengths = ([1024, 2048, 4096, 8192, 16384, 32768]
                           if txt_upper == "AUTO_FULL" else [1024, 4096, 16384])
                result = []
                for L in lengths:
                    kmin, kmax = 2, L // 2 - 1
                    npoints = 60 if txt_upper == "AUTO_FULL" else 30
                    indexes = sorted(set(int(k) for k in np.rint(
                        np.geomspace(kmin, kmax, npoints))))
                    if indexes:
                        result.append(SegmentSpec(L, indexes))
                self._log(f"AUTO SEGMENT generic MTH5: fs={fs:g} Hz, "
                          f"mode={txt_upper}, {sum(len(x.indexes) for x in result)} FFT bins; "
                          "no Phoenix CTS band restriction")
                return result
            fmin, fmax = bands[nominal]
            # Prefer the *selected* CTS band, not hard-coded nominal limits.
            cts_path = self.gui_cts_file.get().strip()
            if cts_path:
                try:
                    import csv
                    with open(cts_path, encoding="utf-8-sig", errors="replace", newline="") as fh:
                        next(fh)
                        pts = [(float(row[0]), int(float(row[1]))) for row in csv.reader(fh)
                               if len(row) >= 2]
                    level = {15: 5, 150: 4, 2400: 3}[nominal]
                    ff = [f for f, ts in pts if ts == level and np.isfinite(f) and f > 0]
                    if len(ff) >= 2:
                        fmin, fmax = min(ff), max(ff)
                except Exception as exc:
                    self._log(f"AUTO SEGMENT: could not inspect CTS ({exc}); using TS{ {15:5,150:4,2400:3}[nominal] } default band")
            # Avoid exactly-on-boundary bins affected by rounded CTS frequency labels.
            low = fmin * (1.0 + 1e-5)
            high = fmax * (1.0 - 1e-5)
            if nominal == 150:
                # Reproduce the user-verified working configuration exactly.
                lengths = [1024]
                verified = [40,50,60,70,80,90,100,120,140,160,180,200]
                idx_by_length = {1024: verified}
                if txt_upper == "AUTO_FULL":
                    lengths += [2048, 4096]
            elif nominal == 2400:
                lengths = [1024, 2048] if txt_upper == "AUTO_FULL" else [1024]
                idx_by_length = {}
            else:
                lengths = [1024, 4096, 16384] if txt_upper == "AUTO_FULL" else [1024, 4096]
                idx_by_length = {}
            result = []
            for L in lengths:
                kmin = max(2, int(np.ceil(low * L / fs)))
                kmax = min(L // 2 - 1, int(np.floor(high * L / fs)))
                if kmax < kmin:
                    continue
                if L in idx_by_length:
                    indexes = [k for k in idx_by_length[L] if kmin <= k <= kmax]
                else:
                    npoints = 40 if txt_upper == "AUTO_FULL" else 30
                    # Include endpoints and spread sampling in log frequency.
                    indexes = sorted(set(int(k) for k in np.rint(np.geomspace(kmin, kmax, npoints))))
                    indexes = [k for k in indexes if kmin <= k <= kmax]
                if indexes:
                    result.append(SegmentSpec(L, indexes))
            if not result:
                raise ValueError(f"AUTO SEGMENT: no FFT indexes inside TS band {fmin:g}–{fmax:g} Hz")
            summary = "; ".join(f"{seg.length}:{','.join(map(str,seg.indexes))}" for seg in result)
            self._log(f"AUTO SEGMENT CTS-aware: fs={fs:g} Hz, TS{ {15:5,150:4,2400:3}[nominal] }, "
                      f"CTS band={fmin:.7g}–{fmax:.7g} Hz, {sum(len(x.indexes) for x in result)} bins; {summary}")
            return result
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

    def _auto_unit_mode_from_selected_mth5_run(self, warn: bool = True) -> str:
        """Infer UNIT_MODE from all E/H stored units in the selected MTH5 run.

        Physical SI pairs are preserved as SI-B or SI-H. Raw counts or receiver-level
        voltage (mV/V) require calibration to the TRACMT practical target. Mixed or
        incomplete units fall back conservatively to practical and are logged.
        """
        survey = self.gui_mth5_survey.get().strip()
        station = self.gui_mth5_station.get().strip()
        run = self.gui_mth5_run.get().strip()
        rows = [r for r in self._mth5_inventory
                if str(r.get("station", "")) == station and str(r.get("run", "")) == run]
        if survey:
            rows = [r for r in rows if (not str(r.get("survey", "")).strip())
                    or str(r.get("survey", "")) == survey]
        by_ch = {str(r.get("channel", "")).lower(): _normalize_unit_token(str(r.get("units", "") or ""))
                 for r in rows}
        eu = [by_ch.get(k, "") for k in ("ex", "ey") if by_ch.get(k, "")]
        hu = [by_ch.get(k, "") for k in ("hx", "hy", "bx", "by") if by_ch.get(k, "")]
        e_set, h_set = set(eu), set(hu)
        # V12: detect the stored source units, but normalize ALL processing to the
        # native TRACMT practical convention before spectral estimation.  The
        # rho_a formula used by TRACMT (0.2*T*|Z|^2) is valid for
        # E=mV/km and B=nT only.  Keeping SI-B/SI-H internally while using that
        # formula creates a deterministic 10^6-scale rho_a error.
        if e_set and e_set.issubset({"v/m"}) and h_set and h_set.issubset({"t"}):
            source_label = "stored SI-B: E=V/m, B=T"
        elif e_set and e_set.issubset({"v/m"}) and h_set and h_set.issubset({"a/m"}):
            source_label = "stored SI-H: E=V/m, H=A/m"
        elif e_set and e_set.issubset({"mv/km"}) and h_set and h_set.issubset({"nt"}):
            source_label = "stored TRACMT practical: E=mV/km, B=nT"
        else:
            source_label = "stored raw/mixed units"
        mode = "mth5_tracmt_practical"
        label = source_label + "; internal target E=mV/km, B=nT"
        desc = ", ".join(f"{k}={v or '[unlabeled]'}" for k, v in sorted(by_ch.items()) if k in {"ex","ey","hx","hy","bx","by"})
        self._resolved_auto_unit_mode = mode
        self._log(f"MTH5 UNIT_MODE auto: {desc} -> {mode}")
        if warn and ((len(e_set) > 1) or (len(h_set) > 1)):
            self._log("MTH5 UNIT CHECK WARNING: mixed stored units within E or H channels; calibration path will be resolved per channel")
        return mode

    def _unit_mode_code_from_gui(self) -> str:
        txt = self.gui_unit_mode.get().lower()
        if "auto" in txt:
            return self._auto_unit_mode_from_selected_mth5_run(warn=True)
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
        is_no_rr = method.startswith("No-RR")
        if is_no_rr:
            procedure = 0
            mest = "-1 -1"
            num_rr = 0
            self.gui_use_rr.set(False)
        elif method.startswith("Ordinary"):
            procedure = 0
            mest = f"{self._mestimator_code(self.gui_estimator.get())} {self._mestimator_code(self.gui_estimator2.get())}"
            num_rr = 2
        else:
            procedure = 1
            mest = "-1 -1"
            num_rr = 2
        err_text = self.gui_error.get().strip()
        err = {
            "0 Parametric": 0, "1 Fixed-weights Bootstrap": 1, "2 Fixed-weights Jackknife": 2, "3 Robust Bootstrap": 3, "4 Strict Bootstrap": 4,
            "0 None": 0, "1 Parametric": 1, "3 Bootstrap": 3,
            "None": 0, "Parametric": 0, "Jackknife": 2, "Bootstrap": 3,
        }.get(err_text, 1)
        if is_no_rr:
            err = 0
        boot = int(float(self.gui_bootstrap.get() or 1000))

        use_rr = (not is_no_rr) and self.gui_use_rr.get() and num_rr > 0
        file_vars = [self.gui_ex_file, self.gui_ey_file, self.gui_hx_file, self.gui_hy_file]
        if use_rr:
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
        lines += ["NUM_RR", str(num_rr)]
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
        lines += ["INPUT_UNIT_MODE", self._unit_mode_code_from_gui()]
        # TRACMT core always receives practical units after the input-normalization stage.
        lines += ["UNIT_MODE", "mth5_tracmt_practical"]
        rotation = float(self.gui_rotation_deg.get())
        if not np.isfinite(rotation):
            raise ValueError("Rotation must be a finite angle in degrees")
        lines += ["ROTATION", f"{rotation:g}"]
        # CTS mode is not user-selectable: actual MTH5 stored units decide.
        lines += ["APPLY_MTH5_RESPONSE", "1"]
        lines += ["CTS_MODE", "auto"]
        if self.gui_cts_file.get().strip():
            lines += ["CTS_FILE", self.gui_cts_file.get().strip()]
        if self.gui_rr_cts_file.get().strip() and num_rr > 0:
            lines += ["RR_CTS_FILE", self.gui_rr_cts_file.get().strip()]
        if self.gui_mth5_force_coil_cal.get():
            lines += ["MTH5_FORCE_COIL_CAL", "1"]
        if self.gui_prewhite_enable.get():
            pw_type = int(self.gui_prewhite_type.get().split(":")[0])
            lines += ["PREWHITENING", str(pw_type), str(int(float(self.gui_prewhite_degree.get()))), str(int(float(self.gui_prewhite_Candidates.get())))]
        if self.gui_robust_filter_enable.get():
            lines += ["ROBUST_FILTER", str(int(self.gui_robust_filter_replace.get()))]
            thr = self.gui_robust_filter_thresholds.get().strip() or "10 12 50"
            for _ in range(4 + (2 if use_rr else 0)):
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
            out = []
            for n in self.names:
                ch = Path(str(n)).stem.lower()
                if ch in ("ex", "ey", "ez"):
                    out.append(f"{ch}_mV_per_km")
                elif ch in ("hx", "hy", "hz", "bx", "by", "bz"):
                    out.append(f"{ch}_nT")
                else:
                    out.append(Path(str(n)).stem)
            return out
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

    @staticmethod
    def _continuous_mt_phase_for_plot(phases_deg, component):
        """Display MT phase on the same unwrapped +/-90-degree branch as CSV output."""
        return mt_phase_pm90(phases_deg)

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
                y = self._continuous_mt_phase_for_plot(sub["phase_deg"].to_numpy(dtype=float), comp)
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
                        self.rf["period"], self._continuous_mt_phase_for_plot(np.angle(z, deg=True), comp), yerr=phase_err,
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
    root.title("pyTRACMT Professional V12.6 - V66 FAP/JSON Calibration Edition")
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
