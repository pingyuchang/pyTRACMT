#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phoenix .TBL + Multiple .TSn Core GUI v22 MTH5
Simplified GUI: folder auto-discovery, multi-TBL/TSn loading, continuity merge,
Raw/Volt/Physical display, CLB/CLC/CLE inspection, optional experimental response
removal, EMpower rxcal.json/scal.json, and CSV and MTH5 export.
Experimental reader for older Phoenix MTU-5A .TS2/.TS3/.TS4/.TS5 files.

Functions
---------
1. Select a survey folder. The program searches that folder (and shallow subfolders)
   for related MTU-5A files: .TBL, .TSn, .CLB, .CLC, .CLE.
2. Match each .TSn file to a .TBL by filename stem when possible.
3. Auto-match receiver .CLB and coil/electrode .CLC/.CLE using TBL serials.
4. Write native TSn ADC-count time series into a Multi-Run MTH5. Do not convert
   the stored waveform to mV/km or nT in this program.
5. Write EMpower-style recmeta.json / config.json (electrode spacing, gain, ADC)
   plus rxcal.json (CLB) / scal.json (CLC) / electric CLE JSON. Mount the same
   FAP and ADC filters in Survey/Filters with applied=False. pyTRACMT later
   converts counts to mV/km and nT.
6. CLI: python Phoenix_MTU5A_to_MTH5.py --folder PATH

Notes
-----
This GUI can display raw ADC counts or estimated input voltage. Voltage conversion
uses: ADC_volts = counts / 2**24 * full_scale_range_Vpp; input_volts = ADC_volts / gain.
The gain is read from TBL entries EGN/HGN when available and can be edited in the GUI.
Because Phoenix calibration binary format is proprietary, the CLB/CLC/CLE decoder is
heuristic and should be validated against SSMT/EMpower output before publication use.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog

import matplotlib
# Do not force TkAgg too early in some Spyder environments; only use if possible.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


DEFAULT_CHANNELS = ["Ex", "Ey", "Hx", "Hy", "Hz"]
TS_DEFAULT_SAMPLE_RATES = {
    "TS2": 24000.0,
    "TS3": 2400.0,
    "TS4": 150.0,
    "TS5": 15.0,
}
TS_EXTENSIONS = {".ts2", ".ts3", ".ts4", ".ts5", ".ts6", ".ts7"}
TBL_EXTENSIONS = {".tbl"}
CLB_EXTENSIONS = {".clb"}
CLC_EXTENSIONS = {".clc"}
CLE_EXTENSIONS = {".cle"}
CAL_SUBFOLDER_NAMES = (
    "cal", "calib", "calibration", "calibrations",
    "coil", "coils", "clc", "cle", "clb", "sensor", "sensors",
)
CHANNEL_TO_EMPOWER_TAG = {"Ex": "E1", "Ey": "E2", "Hx": "H1", "Hy": "H2", "Hz": "H3"}
EMPOWER_RXCAL_CHANNELS = ["E1", "E2", "H1", "H2", "H3"]
EMPOWER_LP_HZ_TO_STATE = {10000: "T0_10kHz", 1000: "T1_1kHz", 100: "T2_100Hz", 10: "T3_10Hz"}


def _unique_sorted_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(Path(p).expanduser().resolve()).lower()
        except Exception:
            key = str(Path(p)).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(p))
    return sorted(out, key=lambda x: str(x).lower())


def _iter_files_limited_depth(root: Path, max_depth: int) -> List[Path]:
    """List files under root, including root itself, up to max_depth subfolders."""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    found: List[Path] = []
    root_parts = len(root.parts)
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            depth = len(p.parent.parts) - root_parts
            if depth <= max_depth:
                found.append(p)
    except Exception:
        try:
            found.extend([p for p in root.iterdir() if p.is_file()])
        except Exception:
            pass
    return found


def _phoenix_recording_family(stem: str) -> str:
    """Group sequential Phoenix files such as 1878601A / 1878601B into one family."""
    s = str(stem or "").strip()
    if len(s) >= 2 and s[-1].isalpha() and s[-2].isdigit():
        return s[:-1]
    return s


def _safe_station_token(value: object) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"[^A-Za-z0-9_\-]", "_", str(value).strip()).strip("_")
    if len(text) < 3:
        return None
    if len(re.findall(r"[A-Za-z0-9]", text)) < 3:
        return None
    return text


@dataclass
class PhoenixFolderInventory:
    folder: Path
    tbl_files: List[Path] = field(default_factory=list)
    ts_files: List[Path] = field(default_factory=list)
    clb_files: List[Path] = field(default_factory=list)
    clc_files: List[Path] = field(default_factory=list)
    cle_files: List[Path] = field(default_factory=list)

    def all_calibration_files(self) -> List[Path]:
        return _unique_sorted_paths(self.clb_files + self.clc_files + self.cle_files)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Folder: {self.folder}",
            (
                f"Found TBL={len(self.tbl_files)}, TSn={len(self.ts_files)}, "
                f"CLB={len(self.clb_files)}, CLC={len(self.clc_files)}, CLE={len(self.cle_files)}"
            ),
        ]
        for label, files in (
            ("TBL", self.tbl_files),
            ("TSn", self.ts_files),
            ("CLB", self.clb_files),
            ("CLC", self.clc_files),
            ("CLE", self.cle_files),
        ):
            for p in files:
                try:
                    rel = p.relative_to(self.folder)
                except Exception:
                    rel = p
                lines.append(f"  {label}: {rel}")
        return lines


def discover_mtu5a_folder(
    folder: str | Path,
    recursive: bool = True,
    max_depth: int = 2,
    extra_roots: Optional[Iterable[Path]] = None,
) -> PhoenixFolderInventory:
    """Search one MTU-5A survey folder for TBL / TSn / CLB / CLC / CLE files."""
    folder = Path(folder).expanduser()
    try:
        folder = folder.resolve()
    except Exception:
        pass
    max_depth = max(0, min(8, int(max_depth)))
    roots = [folder]
    extra_names = list(CAL_SUBFOLDER_NAMES)
    for name in extra_names:
        q = folder / name
        if q.exists() and q.is_dir():
            roots.append(q)
    if extra_roots:
        roots.extend(Path(x) for x in extra_roots)
    roots = _unique_sorted_paths(roots)

    tbls: List[Path] = []
    tss: List[Path] = []
    clbs: List[Path] = []
    clcs: List[Path] = []
    cles: List[Path] = []
    for root in roots:
        files = (
            _iter_files_limited_depth(root, max_depth)
            if recursive
            else [p for p in root.iterdir() if p.is_file()] if root.is_dir() else []
        )
        for p in files:
            suf = p.suffix.lower()
            if suf in TBL_EXTENSIONS:
                tbls.append(p)
            elif suf in TS_EXTENSIONS:
                tss.append(p)
            elif suf in CLB_EXTENSIONS:
                clbs.append(p)
            elif suf in CLC_EXTENSIONS:
                clcs.append(p)
            elif suf in CLE_EXTENSIONS:
                cles.append(p)
    return PhoenixFolderInventory(
        folder=folder,
        tbl_files=_unique_sorted_paths(tbls),
        ts_files=_unique_sorted_paths(tss),
        clb_files=_unique_sorted_paths(clbs),
        clc_files=_unique_sorted_paths(clcs),
        cle_files=_unique_sorted_paths(cles),
    )


def match_tbl_for_ts(ts_path: str | Path, tbls: List["PhoenixTable"]) -> Optional["PhoenixTable"]:
    """Match a TSn file to a TBL: exact stem, then prefix, then sole TBL."""
    if not tbls:
        return None
    ts_stem = Path(ts_path).stem.lower()
    for tbl in tbls:
        if tbl.path.stem.lower() == ts_stem:
            return tbl
    for tbl in tbls:
        tbl_stem = tbl.path.stem.lower()
        if ts_stem.startswith(tbl_stem) or tbl_stem.startswith(ts_stem):
            return tbl
        fam_ts = _phoenix_recording_family(ts_stem)
        fam_tbl = _phoenix_recording_family(tbl_stem)
        if fam_ts and fam_ts == fam_tbl:
            return tbl
    same_dir: List["PhoenixTable"] = []
    try:
        ts_parent = Path(ts_path).expanduser().resolve().parent
        same_dir = [t for t in tbls if t.path.parent.resolve() == ts_parent]
    except Exception:
        same_dir = []
    if len(same_dir) == 1:
        return same_dir[0]
    if len(tbls) == 1:
        return tbls[0]
    return None


def tbl_receiver_tokens(tbl: Optional["PhoenixTable"]) -> List[str]:
    if tbl is None:
        return []
    toks: List[str] = []
    for key in ("BOX", "BOXN", "UNIT", "SNUM"):
        v = tbl.entries.get(key)
        if v is None:
            rec_val = tbl_get_best_value(tbl, key) if "tbl_get_best_value" in globals() else None
            v = rec_val
        if v is None:
            continue
        s = str(v).strip().lower()
        for t in re.findall(r"[a-z0-9]+", s):
            if len(t) >= 2 and t not in {"none", "null"}:
                toks.append(t)
    out: List[str] = []
    for t in toks:
        if t not in out:
            out.append(t)
    return out


def pick_best_clb(
    clb_files: List[Path],
    tbl: Optional["PhoenixTable"],
) -> Tuple[Optional[Path], str]:
    """Choose a receiver .CLB using TBL BOX/UNIT serials, or the sole file in the folder."""
    files = _unique_sorted_paths(clb_files)
    if not files:
        return None, "no CLB files found"
    if len(files) == 1:
        return files[0], f"only CLB in search path: {files[0].name}"
    tokens = tbl_receiver_tokens(tbl)
    scored: List[Tuple[float, Path, str]] = []
    for p in files:
        name = p.stem.lower()
        score = 0.0
        reasons: List[str] = []
        for tok in tokens:
            if tok and tok in name:
                add = 50.0 if len(tok) >= 4 else 15.0
                score += add
                reasons.append(f"filename contains {tok}")
        scored.append((score, p, "; ".join(reasons) if reasons else "no serial token"))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path, reason = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    if best_score >= 15.0 and best_score - second >= 8.0:
        return best_path, f"matched TBL receiver serial: {reason}"
    return files[0], f"multiple CLB files; using first by name: {files[0].name}"


def response_table_to_fap_dict(table: Optional[pd.DataFrame]) -> Optional[Dict[str, object]]:
    if table is None or len(table) < 1:
        return None
    df = table.copy()
    freq = pd.to_numeric(df.get("frequency_Hz"), errors="coerce").to_numpy(float)
    mag = pd.to_numeric(df.get("amplitude"), errors="coerce").to_numpy(float)
    phase = pd.to_numeric(df.get("phase_deg"), errors="coerce").to_numpy(float)
    good = np.isfinite(freq) & np.isfinite(mag) & np.isfinite(phase) & (freq > 0)
    if int(np.sum(good)) < 1:
        return None
    return {
        "frequency_hz": freq[good].tolist(),
        "magnitude": mag[good].tolist(),
        "phase_deg": phase[good].tolist(),
        "n_points": int(np.sum(good)),
    }


def _response_table_arrays(table: Optional[pd.DataFrame]) -> Tuple[List[float], List[float], List[float]]:
    d = response_table_to_fap_dict(table)
    if not d:
        return [], [], []
    return list(d["frequency_hz"]), list(d["magnitude"]), list(d["phase_deg"])


def _safe_cal_token(value: object, fallback: str = "unknown") -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]", "_", str(value or "").strip()).strip("_")
    return text or fallback


def _jsonable(obj):
    """Convert numpy scalars/arrays so json.dumps stays readable."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json_file(path: Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(data), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _empower_state_name(lp_hz: Optional[int]) -> str:
    if lp_hz in EMPOWER_LP_HZ_TO_STATE:
        return EMPOWER_LP_HZ_TO_STATE[int(lp_hz)]
    if lp_hz is None:
        return "legacy_clb"
    return f"lp{int(lp_hz)}Hz"


def make_empower_fap_filter(name: str, freq, mag, phase_deg, units_in: str, units_out: str, comments: str):
    """Build an mt_metadata FrequencyResponseTableFilter in the MTU-5C/EMpower style."""
    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
    f = np.asarray(freq, dtype=float)
    a = np.asarray(mag, dtype=float)
    p = np.asarray(phase_deg, dtype=float)
    good = np.isfinite(f) & np.isfinite(a) & np.isfinite(p) & (f > 0) & (a > 0)
    if int(np.count_nonzero(good)) < 3:
        raise ValueError(f"Calibration table {name} has too few valid FAP points")
    f, a, p = f[good], a[good], p[good]
    order = np.argsort(f)
    f, a, p = f[order], a[order], p[order]
    filt = FrequencyResponseTableFilter(
        name=str(name).replace("/", " per ").lower(),
        units_in=units_in,
        units_out=units_out,
    )
    filt.frequencies = f
    filt.amplitudes = a
    filt.phases = np.deg2rad(p)
    try:
        filt.comments.value = str(comments)
    except Exception:
        try:
            filt.comments = str(comments)
        except Exception:
            pass
    return filt


@dataclass
class PhoenixTable:
    path: Path
    entries: Dict[str, int]
    raw_records: List[Tuple[str, bytes]]

    def sample_rate_for_ts(self, ts_path: Path) -> Optional[float]:
        m = re.search(r"\.TS(\d+)$", ts_path.name, re.IGNORECASE)
        if not m:
            return None
        key = f"SRL{m.group(1)}"
        val = self.entries.get(key)
        if val is None or val <= 0:
            return None
        return float(val)

    def get_record(self, key: str) -> Optional[bytes]:
        key = key.upper()
        for k, rec in self.raw_records:
            if k.upper() == key:
                return rec
        return None

    def get_float64(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Read Phoenix TBL floating-point entries such as EXLN/EYLN.

        In the supplied MTU-5A .TBL, each record is 25 bytes.  Numeric double
        values are stored as little-endian float64 at bytes 12:20.  For example,
        EXLN/EYLN decode to 10.0 m.
        """
        rec = self.get_record(key)
        if rec is None or len(rec) < 20:
            return default
        try:
            import struct
            value = struct.unpack("<d", rec[12:20])[0]
            if np.isfinite(value):
                return float(value)
        except Exception:
            pass
        return default

    def get_int32(self, key: str, default: Optional[int] = None) -> Optional[int]:
        rec = self.get_record(key)
        if rec is None or len(rec) < 16:
            return default
        try:
            return int.from_bytes(rec[12:16], byteorder="little", signed=True)
        except Exception:
            return default


def read_tbl(tbl_path: str | Path) -> PhoenixTable:
    path = Path(tbl_path)
    data = path.read_bytes()
    if data and data[-1] == 0x03:
        data = data[:-1]

    records: List[Tuple[str, bytes]] = []
    entries: Dict[str, int] = {}
    rec_size = 25
    for i in range(0, len(data) - rec_size + 1, rec_size):
        rec = data[i : i + rec_size]
        try:
            key = rec[:4].decode("ascii", errors="ignore").rstrip("\x00")
        except Exception:
            continue
        if not key or not re.match(r"^[A-Z0-9]{2,4}$", key):
            continue
        records.append((key, rec))
        entries[key] = int.from_bytes(rec[12:16], byteorder="little", signed=True)
    return PhoenixTable(path=path, entries=entries, raw_records=records)




# =========================
# TBL metadata helpers for JSON/LOG export
# =========================
# Phoenix legacy .TBL records in this program are treated as 25-byte records:
#   bytes 0:4   = keyword
#   bytes 12:20 = value area for many numeric records
# Important: only a small subset of keys should be interpreted as float64.
# If every value is blindly unpacked as a double, integer/string records become
# denormal IEEE754 numbers such as 2e-323 or 1e-319.
TBL_FLOAT64_KEYS = {
    "EXLN", "EYLN", "EAZM", "HAZM", "ELEV", "ALT", "DECL",
    "LAT", "LON", "LONG", "FSCV", "HNOM", "HATT",
}
TBL_INT32_KEYS = {
    "EGN", "HGN", "EGNC", "HGNC",
    "SRL2", "SRL3", "SRL4", "SRL5", "SRL6", "SRL7",
    "SITE", "UNIT", "BOX", "BOXN", "HNUM", "LPFR",
}
TBL_STRING_KEYS = {
    "LATG", "LNGG", "STNM", "SNUM", "HXSN", "HYSN", "HZSN",
    "HSN", "HSN1", "HSN2", "HSN3", "HTYP", "HTYP1", "HTYP2", "HTYP3",
}
TBL_TIME_KEYS = {"STIM", "ETIM"}


def _clean_printable_ascii(raw: bytes) -> Optional[str]:
    s = ''.join(chr(b) if 32 <= b <= 126 else ' ' for b in raw)
    s = re.sub(r'\s+', ' ', s).strip(' \x00')
    return s or None


def _ascii_from_record_value(rec: bytes) -> Optional[str]:
    """Return printable ASCII content stored in the value area of a TBL record.

    Legacy Phoenix TBL string records commonly store the value after byte 12,
    sometimes with a binary type marker just before the text.  This helper is
    deliberately conservative: it only returns visible text and strips controls.
    """
    if rec is None:
        return None
    s = _clean_printable_ascii(rec[12:] if len(rec) > 12 else rec)
    if not s:
        s = _clean_printable_ascii(rec[4:])
    return s


def _tbl_record_hex(rec: Optional[bytes]) -> Optional[str]:
    if rec is None:
        return None
    return rec.hex(' ')


def _is_denormal_like(x: object) -> bool:
    try:
        v = abs(float(x))
    except Exception:
        return False
    return 0 < v < 1e-250


def tbl_get_string(tbl: Optional[PhoenixTable], key: str) -> Optional[str]:
    if tbl is None:
        return None
    rec = tbl.get_record(key)
    if rec is None:
        return None
    return _ascii_from_record_value(rec)


def tbl_get_uint32(tbl: Optional[PhoenixTable], key: str, default: Optional[int] = None) -> Optional[int]:
    if tbl is None:
        return default
    rec = tbl.get_record(key)
    if rec is None or len(rec) < 16:
        return default
    try:
        return int.from_bytes(rec[12:16], byteorder="little", signed=False)
    except Exception:
        return default


def tbl_get_number_by_key(tbl: Optional[PhoenixTable], key: str):
    """Decode numeric TBL records with a key whitelist.

    This avoids the old failure mode where SRL/EGN/HGN/STIM/ETIM were unpacked
    as float64 and appeared as extremely small scientific notation values.
    """
    if tbl is None:
        return None
    key = key.upper()
    if key in TBL_FLOAT64_KEYS:
        v = tbl.get_float64(key)
        if v is not None and np.isfinite(v) and not _is_denormal_like(v) and abs(float(v)) < 1e12:
            return float(v)
    if key in TBL_INT32_KEYS:
        v = tbl.get_int32(key)
        if v is not None:
            return int(v)
    return None


def decode_tbl_time_candidates(tbl: Optional[PhoenixTable], key: str) -> Dict[str, object]:
    """Return safe STIM/ETIM decoding candidates without forcing one format.

    Phoenix TBL variants are not uniform.  Some timing information is better
    recovered from TS block headers, while TBL STIM/ETIM may be binary counters,
    ASCII, or padding depending on version.  Therefore this function exposes
    raw bytes plus plausible interpretations and explicitly rejects denormal
    float64 values.
    """
    out: Dict[str, object] = {
        "raw_ascii": None,
        "raw_hex": None,
        "int32_le": None,
        "uint32_le": None,
        "int64_le": None,
        "float64_le": None,
        "float64_note": None,
        "unix_time_candidate_utc": None,
        "excel_serial_candidate": None,
        "yyyymmdd_hhmmss_candidate": None,
    }
    if tbl is None:
        return out
    rec = tbl.get_record(key)
    if rec is None:
        return out
    out["raw_ascii"] = _ascii_from_record_value(rec)
    out["raw_hex"] = _tbl_record_hex(rec)
    raw8 = rec[12:20] if len(rec) >= 20 else b""
    try:
        out["int32_le"] = int.from_bytes(raw8[:4], "little", signed=True)
        out["uint32_le"] = int.from_bytes(raw8[:4], "little", signed=False)
        out["int64_le"] = int.from_bytes(raw8[:8], "little", signed=True)
    except Exception:
        pass
    try:
        f = tbl.get_float64(key)
        if f is not None and np.isfinite(f):
            if _is_denormal_like(f):
                out["float64_note"] = f"ignored denormal-like float64 {f}; usually means this field is not a double"
            else:
                out["float64_le"] = float(f)
    except Exception:
        pass

    # Plausible Unix timestamp from 2000-01-01 to 2100-01-01.
    u = out.get("uint32_le")
    if isinstance(u, int) and 946684800 <= u <= 4102444800:
        try:
            out["unix_time_candidate_utc"] = datetime.utcfromtimestamp(u).isoformat(sep=" ") + "Z"
        except Exception:
            pass

    # Plausible Excel day serial, if a float64 happens to be a normal value.
    f = out.get("float64_le")
    if isinstance(f, float) and 20000 <= f <= 80000:
        try:
            excel0 = datetime(1899, 12, 30)
            out["excel_serial_candidate"] = (excel0 + timedelta(days=f)).isoformat(sep=" ")
        except Exception:
            pass

    # ASCII packed candidate such as 20260519123045.
    s = out.get("raw_ascii")
    if isinstance(s, str):
        digits = re.sub(r"\D", "", s)
        for fmt, n in [("%Y%m%d%H%M%S", 14), ("%y%m%d%H%M%S", 12)]:
            if len(digits) >= n:
                try:
                    out["yyyymmdd_hhmmss_candidate"] = datetime.strptime(digits[:n], fmt).isoformat(sep=" ")
                    break
                except Exception:
                    pass
    return out


def tbl_get_best_value(tbl: Optional[PhoenixTable], key: str):
    """Decode one TBL key as string/float/int without producing false denormals."""
    if tbl is None:
        return None
    key = key.upper()
    rec = tbl.get_record(key)
    if rec is None:
        return None

    if key in TBL_TIME_KEYS:
        return decode_tbl_time_candidates(tbl, key)

    s = _ascii_from_record_value(rec)

    # Known string fields, or fields whose value clearly contains text/suffixes.
    if key in TBL_STRING_KEYS:
        return s
    if s and re.search(r'[A-Za-z,/_\\-]', s):
        return s

    n = tbl_get_number_by_key(tbl, key)
    if n is not None:
        return n

    # ASCII numeric fallback, but only if it looks intentionally printable.
    if s and re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', s):
        try:
            return float(s) if ('.' in s or 'e' in s.lower()) else int(s)
        except Exception:
            pass

    # Last fallback: int32 is safer than float64 for unknown Phoenix records.
    i = tbl.get_int32(key)
    if i is not None:
        return int(i)
    return s


def parse_phoenix_ddmm(value: Optional[str], is_lon: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """Parse Phoenix GPS coordinate strings like '2235.183,N' or '12026.718,E'.

    Phoenix TBL LATG/LNGG are normally degrees + decimal minutes:
        LATG 2235.183,N -> 22 deg 35.183 min N -> 22.586383 deg
        LNGG 12026.718,E -> 120 deg 26.718 min E -> 120.445300 deg
    """
    if value is None:
        return None, None
    s = str(value).strip().replace(' ', '')
    m = re.search(r'([+-]?\d+(?:\.\d+)?)(?:,?([NSEWnsew]))?', s)
    if not m:
        return None, s
    num = float(m.group(1))
    hemi = (m.group(2) or '').upper()
    deg = int(abs(num) // 100)
    minute = abs(num) - deg * 100
    dec = deg + minute / 60.0
    if num < 0:
        dec = -dec
    if hemi in ('S', 'W'):
        dec = -abs(dec)
    elif hemi in ('N', 'E'):
        dec = abs(dec)
    return dec, s


def safe_iso_datetime(x) -> Optional[str]:
    if x is None or x == '':
        return None
    try:
        t = pd.to_datetime(x, errors='coerce')
        if pd.isna(t):
            return None
        return t.isoformat()
    except Exception:
        return None


def dataset_start_end_from_meta_or_df(item: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    meta = item.get('meta') or {}
    df = item.get('df')
    start = meta.get('absolute_start_time') or meta.get('start_time')
    end = meta.get('absolute_end_time_estimated') or meta.get('absolute_end_time')
    start = safe_iso_datetime(start)
    end = safe_iso_datetime(end)
    if (start is None or end is None) and isinstance(df, pd.DataFrame) and 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'], errors='coerce').dropna()
        if len(dt):
            start = start or dt.iloc[0].isoformat()
            end = end or dt.iloc[-1].isoformat()
    return start, end


def build_tbl_metadata(tbl: Optional[PhoenixTable]) -> Dict[str, object]:
    """Extract key Phoenix TBL survey metadata for LOG/JSON."""
    if tbl is None:
        return {}
    lat_raw = tbl_get_string(tbl, 'LATG')
    lon_raw = tbl_get_string(tbl, 'LNGG')
    lat_dec, lat_ddmm = parse_phoenix_ddmm(lat_raw, is_lon=False)
    lon_dec, lon_ddmm = parse_phoenix_ddmm(lon_raw, is_lon=True)

    keys_of_interest = [
        'LATG','LNGG','EXLN','EYLN','EAZM','HAZM','EGN','HGN','FSCV','HNOM',
        'HXSN','HYSN','HZSN','HSN','HSN1','HSN2','HSN3','HNUM','HTYP','HTYP1','HTYP2','HTYP3',
        'SRL2','SRL3','SRL4','SRL5','STIM','ETIM','SITE','STNM','SNUM','UNIT','BOX','BOXN','LPFR'
    ]
    decoded = {}
    raw_record_preview = {}
    for k in keys_of_interest:
        rec = tbl.get_record(k)
        if rec is not None:
            decoded[k] = tbl_get_best_value(tbl, k)
            raw_record_preview[k] = {
                'ascii': _ascii_from_record_value(rec),
                'hex': _tbl_record_hex(rec),
            }

    all_keys = sorted(set(k for k, _ in tbl.raw_records))
    return {
        'tbl_file': str(tbl.path),
        'tbl_name': tbl.path.name,
        'latitude_raw_ddmm': lat_ddmm,
        'longitude_raw_ddmm': lon_ddmm,
        'latitude_decimal_deg': lat_dec,
        'longitude_decimal_deg': lon_dec,
        'electrode_spacing_m': {
            'Ex_EXLN': tbl_get_best_value(tbl, 'EXLN'),
            'Ey_EYLN': tbl_get_best_value(tbl, 'EYLN'),
        },
        'orientation_deg_candidates': {
            'EAZM': tbl_get_best_value(tbl, 'EAZM'),
            'HAZM': tbl_get_best_value(tbl, 'HAZM'),
        },
        'coil_serial_or_type_candidates': {
            'HXSN': tbl_get_best_value(tbl, 'HXSN'),
            'HYSN': tbl_get_best_value(tbl, 'HYSN'),
            'HZSN': tbl_get_best_value(tbl, 'HZSN'),
            'HSN': tbl_get_best_value(tbl, 'HSN'),
            'HSN1': tbl_get_best_value(tbl, 'HSN1'),
            'HSN2': tbl_get_best_value(tbl, 'HSN2'),
            'HSN3': tbl_get_best_value(tbl, 'HSN3'),
            'HNUM': tbl_get_best_value(tbl, 'HNUM'),
            'HTYP': tbl_get_best_value(tbl, 'HTYP'),
            'HTYP1': tbl_get_best_value(tbl, 'HTYP1'),
            'HTYP2': tbl_get_best_value(tbl, 'HTYP2'),
            'HTYP3': tbl_get_best_value(tbl, 'HTYP3'),
        },
        'sample_rates_from_tbl': {
            'TS2_SRL2': tbl_get_best_value(tbl, 'SRL2'),
            'TS3_SRL3': tbl_get_best_value(tbl, 'SRL3'),
            'TS4_SRL4': tbl_get_best_value(tbl, 'SRL4'),
            'TS5_SRL5': tbl_get_best_value(tbl, 'SRL5'),
        },
        'gain_from_tbl': {
            'EGN': tbl_get_best_value(tbl, 'EGN'),
            'HGN': tbl_get_best_value(tbl, 'HGN'),
        },
        'stim_etim_time_candidates': {
            'STIM': decode_tbl_time_candidates(tbl, 'STIM') if tbl.get_record('STIM') is not None else None,
            'ETIM': decode_tbl_time_candidates(tbl, 'ETIM') if tbl.get_record('ETIM') is not None else None,
        },
        'decoded_key_values': decoded,
        'raw_record_preview_for_debug': raw_record_preview,
        'available_tbl_keys': all_keys,
        'coordinate_note': 'LATG/LNGG decoded as Phoenix GPS ddmm.mmm,H / dddmm.mmm,H, not decimal degrees.',
        'time_note': 'STIM/ETIM are reported as candidates only. Very small float64 values are ignored as mis-decoded non-double records. Use TS block-header time as primary absolute start/end when available.',
    }

def read_int24_le_from_bytes(data: bytes, offset: int, count: Optional[int] = None) -> np.ndarray:
    b = np.frombuffer(data[offset:], dtype=np.uint8)
    n = len(b) // 3
    if count is not None:
        n = min(n, count)
    if n <= 0:
        return np.array([], dtype=np.int32)
    b = b[: n * 3].reshape(-1, 3).astype(np.int32)
    vals = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
    return vals.astype(np.int32)


def infer_data_offset(ts_path: str | Path, n_channels: int = 5) -> int:
    path = Path(ts_path)
    data = path.read_bytes()[:200000]
    best_offset = 32
    best_score = math.inf
    for off in range(0, 65):
        vals = read_int24_le_from_bytes(data, off)
        n_complete = (len(vals) // n_channels) * n_channels
        if n_complete < n_channels * 100:
            continue
        arr = vals[:n_complete].reshape(-1, n_channels)
        arr2 = arr[5 : min(len(arr), 5000)]
        if arr2.size == 0:
            continue
        dif = np.diff(arr2, axis=0).astype(float)
        score = float(np.nanmedian(np.abs(dif)))
        if off == 32:
            score *= 0.92
        if score < best_score:
            best_score = score
            best_offset = off
    return best_offset


def guess_sample_rate(ts_path: str | Path, tbl: Optional[PhoenixTable], header_int24: np.ndarray) -> Tuple[float, str]:
    path = Path(ts_path)
    if tbl is not None:
        sr = tbl.sample_rate_for_ts(path)
        if sr is not None:
            return float(sr), "TBL SRLn"
    if len(header_int24) > 6 and 0 < header_int24[6] < 100000:
        return float(header_int24[6]), "TS header int24[6]"
    suffix = path.suffix.upper().replace(".", "")
    if suffix in TS_DEFAULT_SAMPLE_RATES:
        return TS_DEFAULT_SAMPLE_RATES[suffix], "TS suffix default"
    return 1.0, "fallback 1 Hz"



def parse_phoenix_block_datetime(header: bytes) -> Optional[datetime]:
    """Decode the timestamp in one legacy Phoenix MTU TSn 32-byte block header.

    For the verified user-supplied MTU files, byte 0 is the actual second value
    (0..59).  In particular 0x1E and 0x1F mean second 30 and 31; they are NOT
    half-minute marker codes.  Bytes 1/2 are minute/hour and bytes 5/6/7 are
    day/month/year-offset-from-2000.

    Timestamp decoding is used only for absolute timing and gap detection.  It
    must NEVER be used to infer or overwrite the sampling rate; SRLn from the
    matching TBL is authoritative.
    """
    if len(header) < 8:
        return None
    second, minute, hour = int(header[0]), int(header[1]), int(header[2])
    day, month, yy = int(header[5]), int(header[6]), int(header[7])
    if not (0 <= second <= 59 and 0 <= minute <= 59 and 0 <= hour <= 23
            and 1 <= day <= 31 and 1 <= month <= 12):
        return None
    year = 2000 + yy if yy < 100 else yy
    try:
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None


def summarize_block_timing(headers: List[bytes], samples_per_block: int,
                           sample_rate: float) -> Dict[str, object]:
    """Summarize block timing without changing the authoritative sample rate."""
    times = [parse_phoenix_block_datetime(h) for h in headers]
    valid_pairs = [(i, t) for i, t in enumerate(times) if t is not None]
    expected_block_duration = (
        float(samples_per_block) / float(sample_rate)
        if sample_rate > 0 and samples_per_block > 0 else np.nan
    )
    summary: Dict[str, object] = {
        "timing_decoder": (
            "Phoenix MTU 32-byte header: byte0=actual second (0..59), "
            "byte1=minute, byte2=hour, byte5=day, byte6=month, byte7=year-2000"
        ),
        "timing_valid_blocks": len(valid_pairs),
        "timing_total_blocks": len(headers),
        "authoritative_sample_rate_Hz": float(sample_rate),
        "samples_per_block": int(samples_per_block),
        "expected_block_duration_s": float(expected_block_duration) if np.isfinite(expected_block_duration) else None,
    }
    if not valid_pairs:
        return summary

    valid = [t for _, t in valid_pairs]
    summary["absolute_start_time"] = valid[0].isoformat(sep=" ")
    summary["absolute_last_block_time"] = valid[-1].isoformat(sep=" ")

    if len(valid) >= 2:
        deltas = np.array([(valid[i+1] - valid[i]).total_seconds()
                           for i in range(len(valid)-1)], dtype=float)
        tol = max(0.01, 0.05 * expected_block_duration) if np.isfinite(expected_block_duration) else 0.05
        gap_idx = np.where(np.abs(deltas - expected_block_duration) > tol)[0].tolist() if np.isfinite(expected_block_duration) else []
        summary.update({
            "timing_delta_min_s": float(np.nanmin(deltas)),
            "timing_delta_median_s": float(np.nanmedian(deltas)),
            "timing_delta_max_s": float(np.nanmax(deltas)),
            "timing_gap_count": int(len(gap_idx)),
            "timing_gap_indices_first20": gap_idx[:20],
        })
    else:
        summary["timing_gap_count"] = 0
        summary["timing_gap_indices_first20"] = []

    if np.isfinite(expected_block_duration):
        summary["absolute_end_time_estimated"] = (
            valid[-1] + timedelta(seconds=expected_block_duration)
        ).isoformat(sep=" ")
        summary["active_recording_duration_s"] = float(len(headers) * expected_block_duration)
        summary["elapsed_span_s"] = float(
            (valid[-1] + timedelta(seconds=expected_block_duration) - valid[0]).total_seconds()
        )
    return summary


def read_tsn(
    ts_path: str | Path,
    tbl: Optional[PhoenixTable] = None,
    channels: Optional[List[str]] = None,
    offset: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read legacy Phoenix MTU TSn data with TBL-authoritative sampling rate.

    The matching TBL SRLn value is authoritative (TS3=SRL3, TS4=SRL4,
    TS5=SRL5).  Header timestamps describe when each block was acquired and may
    contain gaps/bursts, but they never redefine the intra-block sample rate.
    """
    path = Path(ts_path)
    data = path.read_bytes()
    channels = channels or DEFAULT_CHANNELS
    n_channels = len(channels)

    header_int24 = read_int24_le_from_bytes(data, 0, count=16)
    sample_rate, sr_source = guess_sample_rate(path, tbl, header_int24)
    # Hard guard: if matching TBL contains SRLn, it wins unconditionally.
    if tbl is not None:
        tbl_sr = tbl.sample_rate_for_ts(path)
        if tbl_sr is not None and np.isfinite(tbl_sr) and tbl_sr > 0:
            sample_rate = float(tbl_sr)
            sr_source = "TBL SRLn (authoritative; header cannot override)"
    sr_i = int(round(sample_rate))

    read_mode = "continuous int24 fallback"
    block_header_bytes = 32
    samples_per_block = sr_i if sr_i > 0 and abs(sample_rate - sr_i) < 1e-6 else 0
    payload_bytes = samples_per_block * n_channels * 3
    block_bytes = block_header_bytes + payload_bytes

    arr = None
    n_blocks = 0
    block_headers: List[bytes] = []
    if samples_per_block > 0 and payload_bytes > 0 and block_bytes > 0 and len(data) >= block_bytes:
        # Phoenix block layout: 32-byte header + one second of samples at SRLn.
        if len(data) % block_bytes == 0:
            chunks = []
            for start in range(0, len(data), block_bytes):
                block_headers.append(data[start:start + block_header_bytes])
                payload = data[start + block_header_bytes:start + block_bytes]
                vals = read_int24_le_from_bytes(payload, 0)
                n_complete = (len(vals) // n_channels) * n_channels
                if n_complete:
                    chunks.append(vals[:n_complete].reshape(-1, n_channels))
            if chunks:
                arr = np.vstack(chunks)
                n_blocks = len(chunks)
                read_mode = (
                    f"Phoenix blocks: 32-byte header + {samples_per_block} samples/block "
                    f"at authoritative {sample_rate:g} Hz"
                )

    if arr is None:
        if offset is None:
            offset = infer_data_offset(path, n_channels=n_channels)
        vals = read_int24_le_from_bytes(data, offset)
        n_complete = (len(vals) // n_channels) * n_channels
        vals = vals[:n_complete]
        arr = vals.reshape(-1, n_channels)
        read_mode = f"continuous int24 fallback, offset={offset}; fs={sample_rate:g} Hz"
    else:
        offset = block_header_bytes

    if max_samples is not None and max_samples > 0:
        arr = arr[:max_samples]

    timing_meta: Dict[str, object] = {}
    abs_times: Optional[List[str]] = None
    relative_t: Optional[np.ndarray] = None
    if n_blocks and block_headers:
        timing_meta = summarize_block_timing(block_headers, samples_per_block, sample_rate)
        times = [parse_phoenix_block_datetime(h) for h in block_headers]
        step = 1.0 / float(sample_rate)
        abs_dt: List[datetime] = []
        prev_end: Optional[datetime] = None
        for t0 in times:
            if t0 is None:
                t0 = prev_end if prev_end is not None else datetime(1970, 1, 1)
            for j in range(samples_per_block):
                abs_dt.append(t0 + timedelta(seconds=j * step))
            prev_end = t0 + timedelta(seconds=samples_per_block * step)
        abs_dt = abs_dt[:arr.shape[0]]
        abs_times = [x.isoformat(sep=" ", timespec="milliseconds") for x in abs_dt]
        if abs_dt:
            relative_t = np.array([(x - abs_dt[0]).total_seconds() for x in abs_dt], dtype=float)

    if relative_t is None:
        relative_t = np.arange(arr.shape[0], dtype=float) / float(sample_rate)

    df = pd.DataFrame(arr, columns=channels)
    if abs_times is not None:
        df.insert(0, "datetime", abs_times)
    df.insert(0, "time_s", relative_t)

    meta = {
        "ts_file": str(path),
        "tbl_file": str(tbl.path) if tbl else None,
        "file_size_bytes": path.stat().st_size,
        "data_offset_bytes": int(offset),
        "sample_rate_Hz": float(sample_rate),
        "sample_rate_source": sr_source,
        "sample_rate_locked_to_tbl": bool(tbl is not None and tbl.sample_rate_for_ts(path) is not None),
        "read_mode": read_mode,
        "block_header_bytes": block_header_bytes if n_blocks else None,
        "samples_per_block": samples_per_block if n_blocks else None,
        "n_blocks": n_blocks if n_blocks else None,
        "n_channels": n_channels,
        "channels": channels,
        "n_samples": int(arr.shape[0]),
        "duration_s": float(arr.shape[0] / float(sample_rate)),
        "output_units": "raw ADC counts",
        "voltage_conversion_note": (
            "GUI Volt mode uses volts = raw_counts / 2**24 * full_scale_pp_volts."
        ),
        "ts_header_int24_first16": header_int24.tolist(),
    }
    meta.update(timing_meta)
    # Re-assert after timing metadata update so timing diagnostics can never overwrite fs.
    meta["sample_rate_Hz"] = float(sample_rate)
    meta["sample_rate_source"] = sr_source
    if tbl is not None:
        meta["tbl_entries"] = tbl.entries
    return df, meta

@dataclass
class PhoenixCalibration:
    """Container for a Phoenix calibration file.

    Phoenix .CLB/.CLC files are proprietary binary files.  This parser is intentionally
    conservative: it extracts readable header strings, gives a hex preview, and tries
    to recover a plausible numeric response table only when the file contains an
    ASCII/CSV-style table.  For binary Phoenix factory calibration files, this GUI
    will load and inspect the file, but will not silently apply a guessed response.
    """
    path: Path
    kind: str
    size_bytes: int
    header_strings: List[str]
    hex_preview: str
    response_table: Optional[pd.DataFrame]
    note: str


def _hex_preview(data: bytes, n: int = 256) -> str:
    rows = []
    b = data[:n]
    for i in range(0, len(b), 16):
        chunk = b[i:i+16]
        hx = " ".join(f"{x:02X}" for x in chunk)
        asc = "".join(chr(x) if 32 <= x <= 126 else "." for x in chunk)
        rows.append(f"{i:04X}: {hx:<47}  {asc}")
    return "\n".join(rows)


def _printable_strings(data: bytes, min_len: int = 4) -> List[str]:
    out = []
    cur = []
    for x in data:
        if 32 <= x <= 126:
            cur.append(chr(x))
        else:
            if len(cur) >= min_len:
                out.append("".join(cur))
            cur = []
    if len(cur) >= min_len:
        out.append("".join(cur))
    # Keep unique while preserving order, remove pure garbage long fragments sparingly.
    uniq = []
    for t in out:
        if t not in uniq:
            uniq.append(t)
    return uniq[:80]


def _try_parse_ascii_response(data: bytes) -> Optional[pd.DataFrame]:
    """Try ASCII calibration tables: freq amp phase or freq real imag.

    This works for exported text calibration files, not for the Phoenix binary files
    uploaded here.  It is included so the GUI can immediately use converted calibration
    tables later.
    """
    text = data.decode("latin1", errors="ignore")
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 3:
            try:
                vals = [float(x) for x in nums[:3]]
            except Exception:
                continue
            f = vals[0]
            if np.isfinite(f) and f > 0:
                rows.append(vals)
    if len(rows) < 5:
        return None
    df = pd.DataFrame(rows, columns=["frequency_Hz", "amplitude", "phase_deg"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["frequency_Hz"] > 0]
    if len(df) < 5:
        return None
    df = df.sort_values("frequency_Hz").drop_duplicates("frequency_Hz")
    # Heuristic: frequency should cover at least a small positive range.
    if df["frequency_Hz"].max() <= df["frequency_Hz"].min():
        return None
    return df.reset_index(drop=True)



def _table_from_complex(freq: np.ndarray, z: np.ndarray, source: str, channel: str = "") -> pd.DataFrame:
    """Make a normalized response table with amplitude/phase and real/imag columns."""
    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=np.complex128)
    m = (freq > 0) & np.isfinite(freq) & np.isfinite(z.real) & np.isfinite(z.imag) & (np.abs(z) > 0)
    freq = freq[m]
    z = z[m]
    if len(freq) == 0:
        return pd.DataFrame(columns=["frequency_Hz", "real", "imag", "amplitude", "phase_deg", "source", "channel"])
    order = np.argsort(freq)
    freq, z = freq[order], z[order]
    # merge duplicate frequency by median complex value
    rows = []
    for f in np.unique(freq):
        zz = z[freq == f]
        zr = np.nanmedian(zz.real)
        zi = np.nanmedian(zz.imag)
        c = complex(zr, zi)
        rows.append([f, c.real, c.imag, abs(c), np.rad2deg(np.angle(c)), source, channel])
    return pd.DataFrame(rows, columns=["frequency_Hz", "real", "imag", "amplitude", "phase_deg", "source", "channel"])


def _try_parse_phoenix_binary_response(data: bytes, kind: str) -> Optional[pd.DataFrame]:
    """Experimental decoder for legacy Phoenix MTU-5A CLB/CLC binary tables.

    Observed legacy files contain a 128-byte header followed by little-endian 32-bit
    words.  Some words are uint32 fixed-point frequency values; f_Hz = word / 65536.
    Adjacent float32 words form the complex response.  This layout is consistent
    with the MTU-5A .CLB/.CLC examples supplied here, but Phoenix has not published
    this as an open stable format, so the result must be validated against Phoenix
    SSMT/EMpower export.

    CLC heuristic:
      0 + repeated 8-word records: [magic=5, freq_u32, real, imag, extra, 0, real2, imag2]
      real/imag are interpreted as V/nT complex coil response.

    CLB heuristic:
      repeated 32-word records beginning at word 1, 33, 65, ...:
      [magic, freq_u32, response pairs ...].  Multiple complex pairs are averaged
      to obtain an approximate receiver response at each frequency.
    """
    if len(data) <= 160:
        return None
    body = data[128:]
    n = (len(body) // 4) * 4
    if n <= 0:
        return None
    u = np.frombuffer(body[:n], dtype="<u4")
    f = np.frombuffer(body[:n], dtype="<f4")
    kind_u = kind.upper()

    if "CLC" in kind_u or "CLE" in kind_u or "COIL" in kind_u or "SENSOR" in kind_u or "ELEC" in kind_u:
        freqs = []
        zz = []
        # Search all alignments, prefer the one with most plausible magic/frequency records.
        best_records = []
        for start0 in range(0, 8):
            recs = []
            for start in range(start0, len(u) - 7, 8):
                magic = int(u[start])
                fu = int(u[start + 1])
                if magic in (4, 5) and 1000 <= fu <= int(1e9):
                    freq = fu / 65536.0
                    if 1e-5 <= freq <= 1e6:
                        re1, im1 = float(f[start + 2]), float(f[start + 3])
                        re2, im2 = float(f[start + 6]), float(f[start + 7])
                        # Prefer duplicated pair when present; otherwise first pair.
                        if np.isfinite(re2) and np.isfinite(im2) and (abs(re2) + abs(im2) > 0):
                            c = complex(re2, im2)
                        else:
                            c = complex(re1, im1)
                        if np.isfinite(c.real) and np.isfinite(c.imag) and abs(c) > 0:
                            recs.append((freq, c))
            if len(recs) > len(best_records):
                best_records = recs
        if len(best_records) >= 5:
            freqs = np.array([r[0] for r in best_records], dtype=float)
            zz = np.array([r[1] for r in best_records], dtype=np.complex128)
            df = _table_from_complex(freqs, zz, source="Phoenix binary heuristic CLC", channel="coil")
            if len(df) >= 5:
                return df

    # CLB / receiver box: groups of 32 32-bit words were observed in supplied CLB.
    if "CLB" in kind_u or "RECEIVER" in kind_u or "BOX" in kind_u:
        records = []
        for group_start in range(0, len(u) - 31, 32):
            # sometimes a previous group's last float is at group_start; magic at +1
            magic = int(u[group_start + 1])
            fu = int(u[group_start + 2])
            if magic not in (4, 5) or not (1000 <= fu <= int(1e9)):
                continue
            freq = fu / 65536.0
            if not (1e-5 <= freq <= 1e6):
                continue
            comps = []
            # observed response-like pairs begin at +3,+4 then +6,+7, +9,+10...
            for k in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]:
                if group_start + k + 1 >= len(f):
                    continue
                re1 = float(f[group_start + k])
                im1 = float(f[group_start + k + 1])
                c = complex(re1, im1)
                if np.isfinite(c.real) and np.isfinite(c.imag) and 1e-12 < abs(c) < 1e6:
                    comps.append(c)
            if comps:
                # Robust central response; suppress obvious channel/gain outliers.
                mags = np.array([abs(c) for c in comps])
                med = np.median(mags)
                good = [c for c in comps if 0.05 * med <= abs(c) <= 20.0 * med]
                if not good:
                    good = comps
                cmed = complex(float(np.median([c.real for c in good])), float(np.median([c.imag for c in good])))
                records.append((freq, cmed))
        if len(records) >= 5:
            freqs = np.array([r[0] for r in records], dtype=float)
            zz = np.array([r[1] for r in records], dtype=np.complex128)
            df = _table_from_complex(freqs, zz, source="Phoenix binary heuristic CLB", channel="receiver")
            if len(df) >= 5:
                return df
    return None

def read_calibration_file(path: str | Path, kind: str) -> PhoenixCalibration:
    p = Path(path)
    data = p.read_bytes()
    strings = _printable_strings(data)
    # For Phoenix .CLB/.CLC, try the binary decoder first.  A binary header can
    # contain incidental ASCII digits, so a permissive text parser would otherwise
    # produce false response tables.
    if p.suffix.lower() in (".clb", ".clc", ".cle"):
        table = _try_parse_phoenix_binary_response(data, kind)
        if table is not None:
            note = (
                "Experimental Phoenix binary response table decoded. "
                "Use for preliminary frequency-domain response removal only; validate against Phoenix SSMT/EMpower output."
            )
        else:
            table = _try_parse_ascii_response(data)
            if table is not None:
                note = "ASCII-like response table detected. Columns interpreted as frequency_Hz, amplitude, phase_deg."
            else:
                note = (
                    "Loaded as Phoenix binary calibration file. Header strings and hex preview are shown. "
                    "No response table was decoded, so this file will not be applied."
                )
    else:
        table = _try_parse_ascii_response(data)
        if table is not None:
            note = "ASCII-like response table detected. Columns interpreted as frequency_Hz, amplitude, phase_deg."
        else:
            table = _try_parse_phoenix_binary_response(data, kind)
            if table is not None:
                note = "Experimental Phoenix binary response table decoded. Validate before use."
            else:
                note = "No usable response table decoded."
    return PhoenixCalibration(
        path=p,
        kind=kind,
        size_bytes=p.stat().st_size,
        header_strings=strings,
        hex_preview=_hex_preview(data),
        response_table=table,
        note=note,
    )


def interpolate_complex_response(freq: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Interpolate amplitude/phase response table onto FFT frequencies.

    table columns: frequency_Hz, amplitude, phase_deg.  Interpolation is done in
    log-frequency for amplitude and phase.  Returned complex response is amp*exp(i*phase).
    """
    ftab = table["frequency_Hz"].to_numpy(float)
    if "real" in table.columns and "imag" in table.columns:
        ztab = table["real"].to_numpy(float) + 1j * table["imag"].to_numpy(float)
    else:
        amp = table["amplitude"].to_numpy(float)
        ph = table["phase_deg"].to_numpy(float)
        ztab = amp * np.exp(1j * np.deg2rad(ph))
    m = (ftab > 0) & np.isfinite(ftab) & np.isfinite(ztab.real) & np.isfinite(ztab.imag) & (np.abs(ztab) > 0)
    ftab, ztab = ftab[m], ztab[m]
    order = np.argsort(ftab)
    ftab, ztab = ftab[order], ztab[order]
    amp = np.maximum(np.abs(ztab), 1e-300)
    ph = np.unwrap(np.angle(ztab))
    freq_safe = np.maximum(freq, max(ftab[0], 1e-300))
    out_amp = np.exp(np.interp(np.log(freq_safe), np.log(ftab), np.log(amp), left=np.log(amp[0]), right=np.log(amp[-1])))
    out_ph = np.interp(np.log(freq_safe), np.log(ftab), ph, left=ph[0], right=ph[-1])
    return out_amp * np.exp(1j * out_ph)


def response_table_range(table: pd.DataFrame) -> Tuple[float, float]:
    ftab = table["frequency_Hz"].to_numpy(float)
    ftab = ftab[np.isfinite(ftab) & (ftab > 0)]
    if len(ftab) == 0:
        return (np.nan, np.nan)
    return float(np.nanmin(ftab)), float(np.nanmax(ftab))


def is_experimental_binary_calibration(cal: Optional[PhoenixCalibration]) -> bool:
    return bool(cal is not None and isinstance(cal.note, str) and "Experimental Phoenix binary" in cal.note)


def safe_interpolated_response(freq: np.ndarray, cal: Optional[PhoenixCalibration], use_experimental: bool, bandlimit: bool) -> Tuple[Optional[np.ndarray], str]:
    """Return complex response or None, with safety rules for heuristic Phoenix binary files."""
    if cal is None or cal.response_table is None or len(cal.response_table) < 5:
        return None, "no response table"
    if is_experimental_binary_calibration(cal) and not use_experimental:
        return None, "experimental binary response not applied"
    table = cal.response_table
    resp = interpolate_complex_response(np.maximum(freq, 1e-300), table)
    fmin, fmax = response_table_range(table)
    if bandlimit and np.isfinite(fmin) and np.isfinite(fmax):
        mask = (freq >= fmin) & (freq <= fmax)
        # Outside the calibrated frequency band, leave the spectrum unchanged instead of
        # extrapolating a proprietary response. This prevents artificial flat lines.
        resp2 = np.ones_like(resp, dtype=complex)
        good = mask & np.isfinite(resp.real) & np.isfinite(resp.imag) & (np.abs(resp) > 1e-20)
        resp2[good] = resp[good]
        resp = resp2
        return resp, f"response applied only within {fmin:g}-{fmax:g} Hz"
    resp = np.where(np.isfinite(resp.real) & np.isfinite(resp.imag) & (np.abs(resp) > 1e-20), resp, 1.0 + 0j)
    return resp, "response applied"


class PhoenixGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phoenix Legacy MTU Professional - Calibrated CLB/CLC MTH5 v22")
        self.geometry("1420x940")

        self.folder_path_var = tk.StringVar()
        self.survey_id_var = tk.StringVar(value="phoenix_survey")
        self.folder_inventory: Optional[PhoenixFolderInventory] = None
        self._empower_cal_bundle: Dict[str, object] = {}
        self._empower_cal_json_paths: List[str] = []
        self.tbl_path_var = tk.StringVar()
        self.ts_path_var = tk.StringVar()
        self.selected_dataset_var = tk.StringVar()
        self.max_samples_var = tk.StringVar(value="200000")
        self.offset_var = tk.StringVar(value="auto")
        self.info_var = tk.StringVar(value="Select an MTU-5A folder, or pick .TBL/.TSn files, then load or auto-convert to MTH5.")
        self.display_mode_var = tk.StringVar(value="Raw counts")
        # Phoenix 24-bit signed ADC LSB model used here:
        # ADC volts = raw_counts / 2**24 * full_scale_pp_volts.
        # For +/-2.5 V full scale, full_scale_pp_volts = 5.0 V.
        self.ad_range_pp_var = tk.StringVar(value="5.0")
        self.ex_gain_var = tk.StringVar(value="1")
        self.ey_gain_var = tk.StringVar(value="1")
        self.hx_gain_var = tk.StringVar(value="1")
        self.hy_gain_var = tk.StringVar(value="1")
        self.hz_gain_var = tk.StringVar(value="1")
        # Preliminary physical-unit conversion settings.
        # Electric: E[mV/km] = input_volt[V] * 1e6 / dipole_length_m.
        # Magnetic: B[nT] = input_volt[V] / sensor_sensitivity_V_per_nT.
        self.ex_length_m_var = tk.StringVar(value="100")
        self.ey_length_m_var = tk.StringVar(value="100")
        self.hx_sens_mv_nt_var = tk.StringVar(value="1")
        self.hy_sens_mv_nt_var = tk.StringVar(value="1")
        self.hz_sens_mv_nt_var = tk.StringVar(value="1")
        self.clb_path_var = tk.StringVar()
        self.clc_path_var = tk.StringVar()
        self.calibration_status_var = tk.StringVar(value="No calibration loaded")
        self.cal_window_sec_var = tk.StringVar(value="256")
        self.cal_overlap_var = tk.StringVar(value="0.5")
        # Binary CLB/CLC decoding is heuristic. Keep it off by default to avoid false flat-line calibration.
        self.use_experimental_binary_response_var = tk.BooleanVar(value=False)
        self.bandlimit_response_var = tk.BooleanVar(value=True)
        # V18 Professional automatic calibration discovery/matching.
        self.auto_detect_calibration_var = tk.BooleanVar(value=True)
        self.recursive_calibration_search_var = tk.BooleanVar(value=True)
        self.calibration_search_depth_var = tk.StringVar(value="2")
        self.coil_calibration_map: Dict[str, Optional[PhoenixCalibration]] = {"Hx": None, "Hy": None, "Hz": None}
        self.coil_calibration_scores: Dict[str, float] = {"Hx": 0.0, "Hy": 0.0, "Hz": 0.0}
        self.coil_calibration_reasons: Dict[str, str] = {"Hx": "", "Hy": "", "Hz": ""}
        self.calibration_candidates: List[Dict[str, object]] = []

        self.tbl: Optional[PhoenixTable] = None
        self.tbls: List[PhoenixTable] = []
        self.tbl_paths: List[str] = []
        self.df: Optional[pd.DataFrame] = None
        self.meta: Optional[Dict[str, object]] = None
        self.ts_paths: List[str] = []
        self.datasets: List[Dict[str, object]] = []
        self.clb: Optional[PhoenixCalibration] = None
        self.clcs: List[PhoenixCalibration] = []
        self.calibrated_df: Optional[pd.DataFrame] = None
        self.calibrated_units: Dict[str, str] = {}
        self.calibrated_note: str = ""
        self.raw_df_before_apply: Optional[pd.DataFrame] = None
        self.raw_meta_before_apply: Optional[Dict[str, object]] = None
        self.channel_vars: Dict[str, tk.BooleanVar] = {ch: tk.BooleanVar(value=True) for ch in DEFAULT_CHANNELS}

        self._build_ui()
        self._build_plot()

    def _build_ui(self) -> None:
        folder_bar = ttk.LabelFrame(self, text="MTU-5A folder auto conversion / 資料夾自動搜尋並轉 MTH5", padding=6)
        folder_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(folder_bar, text="Survey folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(folder_bar, textvariable=self.folder_path_var, width=88).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(folder_bar, text="Select folder", command=self.select_survey_folder).grid(row=0, column=2, padx=4)
        ttk.Label(folder_bar, text="Survey ID").grid(row=1, column=0, sticky="w")
        ttk.Entry(folder_bar, textvariable=self.survey_id_var, width=28).grid(row=1, column=1, sticky="w", padx=4)
        btns = ttk.Frame(folder_bar)
        btns.grid(row=1, column=2, sticky="e", padx=4)
        ttk.Button(btns, text="Scan folder and load", command=self.scan_survey_folder_and_load).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Auto convert folder → MTH5", command=self.convert_folder_to_mth5_interactive).pack(side=tk.LEFT, padx=3)
        ttk.Label(
            folder_bar,
            text="Like MTU-5C: store TSn as digital counts; write recmeta.json/config.json (electrode spacing, gain) plus rxcal.json/scal.json. pyTRACMT converts to mV/km and nT.",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        folder_bar.columnconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="TBL file(s)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.tbl_path_var, width=95).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Select TBL(s)", command=self.select_tbl).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="TSn files").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.ts_path_var, width=95).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Select TSn files", command=self.select_ts).grid(row=1, column=2, padx=4)

        options = ttk.Frame(top)
        options.grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(options, text="Max samples").pack(side=tk.LEFT)
        ttk.Entry(options, textvariable=self.max_samples_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(options, text="Offset").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(options, textvariable=self.offset_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(options, text="Display").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(
            options,
            textvariable=self.display_mode_var,
            values=["Raw counts", "ADC Volt", "Input Volt", "Physical approx"],
            width=24,
            state="readonly",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(options, text="A/D range Vpp").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(options, textvariable=self.ad_range_pp_var, width=7).pack(side=tk.LEFT, padx=4)
        self.display_mode_var.trace_add("write", lambda *_: self.update_plot())
        self.ad_range_pp_var.trace_add("write", lambda *_: self.update_plot())

        gain_frame = ttk.LabelFrame(top, text="Amplifier gain for Input Volt mode")
        gain_frame.grid(row=4, column=1, sticky="w", pady=4)
        for label, var in [("Ex", self.ex_gain_var), ("Ey", self.ey_gain_var), ("Hx", self.hx_gain_var), ("Hy", self.hy_gain_var), ("Hz", self.hz_gain_var)]:
            ttk.Label(gain_frame, text=label).pack(side=tk.LEFT, padx=(6, 1))
            ttk.Entry(gain_frame, textvariable=var, width=5).pack(side=tk.LEFT, padx=(0, 6))
            var.trace_add("write", lambda *_: self.update_plot())

        phys_frame = ttk.LabelFrame(top, text="Physical unit scaling / 物理量換算")
        phys_frame.grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(phys_frame, text="Ex length m").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Entry(phys_frame, textvariable=self.ex_length_m_var, width=7).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(phys_frame, text="Ey length m").pack(side=tk.LEFT, padx=(6, 1))
        ttk.Entry(phys_frame, textvariable=self.ey_length_m_var, width=7).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(phys_frame, text="Hx/Hy/Hz sens mV/nT (preview only; NOT MTH5)").pack(side=tk.LEFT, padx=(6, 1))
        for var in [self.hx_sens_mv_nt_var, self.hy_sens_mv_nt_var, self.hz_sens_mv_nt_var]:
            ttk.Entry(phys_frame, textvariable=var, width=7).pack(side=tk.LEFT, padx=(0, 4))
            var.trace_add("write", lambda *_: self.update_plot())
        self.ex_length_m_var.trace_add("write", lambda *_: self.update_plot())
        self.ey_length_m_var.trace_add("write", lambda *_: self.update_plot())

        cal_frame = ttk.LabelFrame(top, text="Calibration files / 校正檔")
        cal_frame.grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(cal_frame, text="CLB主機").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(cal_frame, textvariable=self.clb_path_var, width=58).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(cal_frame, text="Select CLB", command=self.select_clb).grid(row=0, column=2, padx=3)
        ttk.Label(cal_frame, text="CLC/CLE校正").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(cal_frame, textvariable=self.clc_path_var, width=58).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(cal_frame, text="Select CLC/CLE(s)", command=self.select_clc).grid(row=1, column=2, padx=3)
        ttk.Button(cal_frame, text="Plot calibration", command=self.plot_calibration_response).grid(row=0, column=3, rowspan=2, padx=4, sticky="ns")
        ttk.Label(cal_frame, textvariable=self.calibration_status_var, foreground="purple").grid(row=2, column=0, columnspan=4, sticky="w", padx=4)
        ttk.Label(cal_frame, text="Time calib window s").grid(row=3, column=0, sticky="w", padx=4)
        ttk.Entry(cal_frame, textvariable=self.cal_window_sec_var, width=8).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(cal_frame, text="overlap 0-0.9").grid(row=3, column=1, sticky="w", padx=(80,4))
        ttk.Entry(cal_frame, textvariable=self.cal_overlap_var, width=6).grid(row=3, column=1, sticky="w", padx=(170,4))
        ttk.Checkbutton(cal_frame, text="Use experimental binary CLB/CLC", variable=self.use_experimental_binary_response_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(cal_frame, text="Band-limit response to calibration table range", variable=self.bandlimit_response_var).grid(row=4, column=1, columnspan=3, sticky="w", padx=(220,4))
        ttk.Checkbutton(cal_frame, text="Auto-detect coil calibration after TBL/TS selection", variable=self.auto_detect_calibration_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(cal_frame, text="Recursive search", variable=self.recursive_calibration_search_var).grid(row=5, column=1, sticky="w", padx=(280,4))
        ttk.Label(cal_frame, text="depth").grid(row=5, column=2, sticky="e")
        ttk.Entry(cal_frame, textvariable=self.calibration_search_depth_var, width=4).grid(row=5, column=3, sticky="w", padx=3)
        ttk.Button(cal_frame, text="Auto Detect Coil Calibration", command=self.auto_detect_coil_calibrations).grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=3)
        ttk.Button(cal_frame, text="Show Match Report", command=self.show_calibration_match_report).grid(row=6, column=2, columnspan=2, sticky="ew", padx=4, pady=3)
        cal_frame.columnconfigure(1, weight=1)

        ttk.Button(top, text="Load selected TS data", command=self.load_ts).grid(row=2, column=2, padx=4, sticky="ew")
        ttk.Button(top, text="Export current view CSV", command=self.export_csv).grid(row=3, column=2, padx=4, sticky="ew")
        ttk.Button(top, text="Export merged absolute-time CSV", command=self.export_merged_absolute_csv).grid(row=4, column=2, padx=4, sticky="ew")
        ttk.Button(top, text="Export JSON/LOG", command=self.export_json_log).grid(row=5, column=2, padx=4, sticky="ew")
        ttk.Button(top, text="Export MTH5 (Auto Multi-Run)", command=self.export_mth5_for_aurora).grid(row=6, column=2, padx=4, sticky="ew")

        ch_frame = ttk.LabelFrame(top, text="Channels to display")
        ch_frame.grid(row=3, column=1, sticky="w", pady=4)
        for ch in DEFAULT_CHANNELS:
            ttk.Checkbutton(ch_frame, text=ch, variable=self.channel_vars[ch], command=self.update_plot).pack(side=tk.LEFT, padx=8)

        dataset_frame = ttk.Frame(top)
        dataset_frame.grid(row=8, column=1, sticky="w", pady=4)
        ttk.Label(dataset_frame, text="Loaded file to view").pack(side=tk.LEFT)
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.selected_dataset_var, values=[], width=55, state="readonly")
        self.dataset_combo.pack(side=tk.LEFT, padx=4)
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda *_: self.select_loaded_dataset())

        ttk.Label(top, textvariable=self.info_var, foreground="blue").grid(row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))
        top.columnconfigure(1, weight=1)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=5)
        right = ttk.Frame(main, padding=5)
        main.add(left, weight=4)
        main.add(right, weight=1)

        self.plot_frame = left
        ttk.Label(right, text="Metadata / Log").pack(anchor="w")
        self.log_text = tk.Text(right, height=35, width=38, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_plot(self) -> None:
        # Use matplotlib.figure.Figure directly, not pyplot.subplots(), to avoid
        # Spyder/IPython opening an extra standalone "Figure 1" window.
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.axes = self.fig.subplots(5, 1, sharex=True)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)


    def _gain_vars(self) -> Dict[str, tk.StringVar]:
        return {
            "Ex": self.ex_gain_var,
            "Ey": self.ey_gain_var,
            "Hx": self.hx_gain_var,
            "Hy": self.hy_gain_var,
            "Hz": self.hz_gain_var,
        }

    def _float_from_var(self, var: tk.StringVar, default: float = 1.0) -> float:
        try:
            value = float(var.get())
            if not np.isfinite(value) or value == 0:
                return default
            return value
        except Exception:
            return default

    def apply_tbl_gain_defaults(self) -> None:
        """Use common Phoenix TBL gain entries if present.

        In the uploaded MTU-5A-style TBL, EGN and HGN appear as electric and
        magnetic gain settings.  If absent, keep current GUI values.
        """
        if self.tbl is None:
            return
        e_gain = self.tbl.entries.get("EGN", self.tbl.entries.get("EGNC"))
        h_gain = self.tbl.entries.get("HGN", self.tbl.entries.get("HGNC"))
        if e_gain and e_gain > 0:
            self.ex_gain_var.set(str(e_gain))
            self.ey_gain_var.set(str(e_gain))
        if h_gain and h_gain > 0:
            self.hx_gain_var.set(str(h_gain))
            self.hy_gain_var.set(str(h_gain))
            self.hz_gain_var.set(str(h_gain))

    def apply_tbl_electrode_defaults(self) -> None:
        """Auto-fill Ex/Ey dipole lengths from Phoenix TBL when present.

        Common Phoenix tags:
          EXLN = Ex dipole/electrode line length, usually meters
          EYLN = Ey dipole/electrode line length, usually meters
          EAZM = E-line azimuth
          HAZM = H-line azimuth

        The uploaded TBL contains EXLN=10.0 and EYLN=10.0.
        """
        if self.tbl is None:
            return
        ex_len = self.tbl.get_float64("EXLN")
        ey_len = self.tbl.get_float64("EYLN")
        if ex_len is not None and 0 < ex_len < 100000:
            self.ex_length_m_var.set(f"{ex_len:g}")
        if ey_len is not None and 0 < ey_len < 100000:
            self.ey_length_m_var.set(f"{ey_len:g}")

    def apply_tbl_adc_defaults(self) -> None:
        """Use TBL FSCV as the 24-bit ADC full-scale when present.

        MTU-5C stores native BIN as integer counts and keeps ADC scale in
        recmeta/config.  MTU-5A TBL FSCV is typically a peak voltage; values
        around 2.5 V are treated as +/-FSCV, so Vpp = 2*FSCV.
        """
        if self.tbl is None:
            return
        fscv = self.tbl.get_float64("FSCV")
        if fscv is None or not np.isfinite(fscv) or fscv <= 0 or fscv > 50:
            return
        vpp = float(2.0 * fscv) if fscv <= 3.5 else float(fscv)
        self.ad_range_pp_var.set(f"{vpp:g}")
        self.log(f"ADC full-scale from TBL FSCV={fscv:g} V -> {vpp:g} Vpp")

    def select_tbl(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select one or more Phoenix TBL files",
            filetypes=[("Phoenix TBL", "*.TBL *.tbl"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.tbl_paths = list(paths)
        self.tbl_path_var.set("; ".join(self.tbl_paths))
        try:
            self.tbls = [read_tbl(p) for p in self.tbl_paths]
            self.tbl = self.tbls[0] if self.tbls else None
            self.log(f"Loaded {len(self.tbls)} TBL file(s):")
            for t in self.tbls:
                self.log(f"  - {t.path.name}: SRL2={t.entries.get('SRL2')}, SRL3={t.entries.get('SRL3')}, SRL4={t.entries.get('SRL4')}, SRL5={t.entries.get('SRL5')}, EXLN={t.get_float64('EXLN')}, EYLN={t.get_float64('EYLN')}")
            if self.tbl is not None:
                self.apply_tbl_gain_defaults()
                self.apply_tbl_electrode_defaults()
                self.apply_tbl_adc_defaults()
                self.log(f"Default GUI settings from first TBL: {self.tbl.path.name}")
                self.log(f"Gain defaults: Ex/Ey={self.ex_gain_var.get()}/{self.ey_gain_var.get()}, Hx/Hy/Hz={self.hx_gain_var.get()}/{self.hy_gain_var.get()}/{self.hz_gain_var.get()}")
                self.log(f"Electrode length defaults from TBL: EXLN={self.ex_length_m_var.get()} m, EYLN={self.ey_length_m_var.get()} m")
                if self.auto_detect_calibration_var.get():
                    self.auto_detect_coil_calibrations(quiet=True)
        except Exception as e:
            messagebox.showerror("TBL read error", str(e))
            self.log(traceback.format_exc())

    def _ensure_tbls_from_entry(self) -> None:
        """Read semicolon-separated TBL paths from the entry if they were typed manually."""
        if self.tbls:
            return
        text = self.tbl_path_var.get().strip()
        if not text:
            return
        paths = [x.strip().strip('"') for x in text.split(';') if x.strip()]
        if not paths:
            return
        self.tbl_paths = paths
        self.tbls = [read_tbl(p) for p in paths]
        self.tbl = self.tbls[0] if self.tbls else None

    def _match_tbl_for_ts(self, ts_path: str | Path) -> Optional[PhoenixTable]:
        """Match a TS file to its own TBL by stem, recording family, or sole TBL."""
        return match_tbl_for_ts(ts_path, self.tbls)

    def _apply_tbl_defaults_to_gui(self, tbl: Optional[PhoenixTable]) -> None:
        """Apply gain and electrode length from a specific matched TBL to GUI variables."""
        if tbl is None:
            return
        old_tbl = self.tbl
        self.tbl = tbl
        self.apply_tbl_gain_defaults()
        self.apply_tbl_electrode_defaults()
        self.apply_tbl_adc_defaults()
        self.tbl = old_tbl

    def select_ts(self) -> None:
        paths = filedialog.askopenfilenames(title="Select one or more Phoenix TSn files", filetypes=[("Phoenix TS", "*.TS2 *.TS3 *.TS4 *.TS5 *.ts2 *.ts3 *.ts4 *.ts5"), ("All files", "*.*")])
        if not paths:
            return
        self.ts_paths = list(paths)
        self.ts_path_var.set("; ".join(self.ts_paths))
        if self.auto_detect_calibration_var.get():
            self.auto_detect_coil_calibrations(quiet=True)

    def _station_key_from_tbl(self, tbl: Optional[PhoenixTable]) -> str:
        if tbl is None:
            folder = self.folder_path_var.get().strip()
            return _safe_station_token(Path(folder).name if folder else None) or "PX001"
        md = build_tbl_metadata(tbl)
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}
        for key in ("STNM", "SITE", "SNUM"):
            token = _safe_station_token(decoded.get(key))
            if token:
                return token
        fam = _phoenix_recording_family(tbl.path.stem)
        return _safe_station_token(fam) or _safe_station_token(tbl.path.stem) or "PX001"

    def _group_inventory_by_station(
        self,
        inventory: PhoenixFolderInventory,
        tbls: List[PhoenixTable],
    ) -> List[Dict[str, object]]:
        groups: Dict[str, Dict[str, object]] = {}
        for ts in inventory.ts_files:
            tbl = match_tbl_for_ts(ts, tbls)
            key = self._station_key_from_tbl(tbl)
            g = groups.setdefault(key, {"station_id": key, "tbls": [], "ts_files": []})
            g["ts_files"].append(ts)
            if tbl is not None and tbl not in g["tbls"]:
                g["tbls"].append(tbl)
        if not groups and tbls:
            key = self._station_key_from_tbl(tbls[0])
            groups[key] = {"station_id": key, "tbls": list(tbls), "ts_files": []}
        return list(groups.values())

    def _apply_inventory_to_gui(self, inventory: PhoenixFolderInventory) -> None:
        self.folder_inventory = inventory
        self.folder_path_var.set(str(inventory.folder))
        self.tbl_paths = [str(p) for p in inventory.tbl_files]
        self.ts_paths = [str(p) for p in inventory.ts_files]
        self.tbl_path_var.set("; ".join(self.tbl_paths))
        self.ts_path_var.set("; ".join(self.ts_paths))
        self.tbls = [read_tbl(p) for p in inventory.tbl_files]
        self.tbl = self.tbls[0] if self.tbls else None
        if self.tbl is not None:
            self.apply_tbl_gain_defaults()
            self.apply_tbl_electrode_defaults()
            self.apply_tbl_adc_defaults()
        survey_guess = _safe_station_token(inventory.folder.name)
        if survey_guess and self.survey_id_var.get().strip() in ("", "phoenix_survey"):
            self.survey_id_var.set(survey_guess)

    def select_survey_folder(self) -> None:
        p = filedialog.askdirectory(title="Select Phoenix MTU-5A station/data folder")
        if not p:
            return
        try:
            self.scan_survey_folder(p, load_ts_data=False, interactive=True)
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Folder scan failed", str(e))

    def scan_survey_folder(
        self,
        folder: Optional[str | Path] = None,
        *,
        load_ts_data: bool = False,
        interactive: bool = True,
        load_all_samples: bool = False,
    ) -> PhoenixFolderInventory:
        folder_text = str(folder or self.folder_path_var.get()).strip()
        if not folder_text:
            raise ValueError("Please select an MTU-5A survey folder first.")
        base = Path(folder_text).expanduser()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Folder not found: {base}")

        recursive = bool(self.recursive_calibration_search_var.get())
        inventory = discover_mtu5a_folder(
            base,
            recursive=recursive,
            max_depth=self._folder_search_depth(),
        )
        self._apply_inventory_to_gui(inventory)
        for line in inventory.summary_lines():
            self.log(line)

        if not inventory.ts_files:
            raise FileNotFoundError(
                f"No Phoenix .TS2/.TS3/.TS4/.TS5 files found in {base}"
            )
        if not inventory.tbl_files:
            self.log("WARNING: No .TBL files found; sample rates will use TS suffix defaults.")

        self.auto_detect_receiver_calibration(quiet=True)
        if self.auto_detect_calibration_var.get():
            self.auto_detect_coil_calibrations(quiet=True)

        msg = (
            f"Scanned {base.name}: TBL={len(inventory.tbl_files)}, "
            f"TSn={len(inventory.ts_files)}, CLB={len(inventory.clb_files)}, "
            f"CLC={len(inventory.clc_files)}, CLE={len(inventory.cle_files)}"
        )
        self.info_var.set(msg)
        if load_ts_data:
            old_max = self.max_samples_var.get()
            if load_all_samples:
                self.max_samples_var.set("")
            try:
                n = self.load_ts(quiet=not interactive, update_display=interactive)
            finally:
                if load_all_samples:
                    self.max_samples_var.set(old_max)
            self.log(f"Loaded {n} TSn dataset(s) from folder.")
        elif interactive:
            messagebox.showinfo(
                "Folder scanned",
                msg + "\n\nClick 'Scan folder and load' to view waveforms, "
                "or 'Auto convert folder → MTH5' to write the MTH5 file."
            )
        return inventory

    def scan_survey_folder_and_load(self) -> None:
        folder = self.folder_path_var.get().strip()
        if not folder:
            folder = filedialog.askdirectory(title="Select Phoenix MTU-5A station/data folder") or ""
            if not folder:
                return
            self.folder_path_var.set(folder)
        try:
            self.scan_survey_folder(folder, load_ts_data=True, interactive=True, load_all_samples=False)
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Folder load failed", str(e))

    def convert_folder_to_mth5_interactive(self) -> None:
        folder = self.folder_path_var.get().strip()
        if not folder:
            folder = filedialog.askdirectory(title="Select Phoenix MTU-5A station/data folder") or ""
            if not folder:
                return
            self.folder_path_var.set(folder)
        try:
            written = self.convert_folder_to_mth5(folder, interactive=True)
            if written:
                messagebox.showinfo(
                    "MTH5 conversion finished",
                    "Wrote:\n" + "\n".join(str(p) for p in written),
                )
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Folder MTH5 conversion failed", str(e))

    def convert_folder_to_mth5(
        self,
        folder: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        survey_id: Optional[str] = None,
        *,
        interactive: bool = False,
    ) -> List[str]:
        """Discover related MTU-5A files in one folder and write calibrated MTH5 file(s)."""
        try:
            from mth5.mth5 import MTH5  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "MTH5 export needs the official mth5 package.\n"
                "Install with: python -m pip install mth5 mt-metadata\n"
                f"Import error: {e}"
            ) from e

        inventory = self.scan_survey_folder(
            folder,
            load_ts_data=False,
            interactive=False,
            load_all_samples=True,
        )
        self.use_experimental_binary_response_var.set(True)
        self.log("Folder auto-convert enabled experimental binary CLB/CLC/CLE decoder.")

        self.auto_detect_receiver_calibration(quiet=True)
        self.auto_detect_coil_calibrations(quiet=True)

        survey = (survey_id or self.survey_id_var.get() or "phoenix_survey").strip()
        survey = re.sub(r"[^A-Za-z0-9_\-]", "_", survey) or "phoenix_survey"
        self.survey_id_var.set(survey)

        groups = self._group_inventory_by_station(inventory, self.tbls)
        if not groups:
            raise RuntimeError("No TSn/TBL groups found to convert.")

        written: List[str] = []
        old_max = self.max_samples_var.get()
        self.max_samples_var.set("")
        try:
            for gi, group in enumerate(groups, start=1):
                tbls = list(group["tbls"])
                ts_files = list(group["ts_files"])
                station_id = str(group["station_id"])
                self.log(f"\n=== Converting station group {gi}/{len(groups)}: {station_id} ===")
                self.tbls = tbls
                self.tbl = tbls[0] if tbls else None
                self.tbl_paths = [str(t.path) for t in tbls]
                self.ts_paths = [str(p) for p in ts_files]
                self.tbl_path_var.set("; ".join(self.tbl_paths))
                self.ts_path_var.set("; ".join(self.ts_paths))
                if self.tbl is not None:
                    self.apply_tbl_gain_defaults()
                    self.apply_tbl_electrode_defaults()
                    self.apply_tbl_adc_defaults()
                self.auto_detect_receiver_calibration(quiet=True)
                self.auto_detect_coil_calibrations(quiet=True)
                n = self.load_ts(quiet=True, update_display=interactive and gi == 1)
                if n <= 0:
                    raise RuntimeError(f"No TSn data loaded for station {station_id}")

                if output_path and len(groups) == 1:
                    out = Path(output_path)
                else:
                    out = inventory.folder / f"{station_id}_Phoenix.mth5"
                self._export_mth5_core(str(out), survey_id=survey, autodetect=False)
                written.append(str(out))
                written.extend(list(self._empower_cal_json_paths))
        finally:
            self.max_samples_var.set(old_max)
        return written



    def _folder_search_depth(self) -> int:
        try:
            return max(0, min(8, int(self.calibration_search_depth_var.get().strip() or "2")))
        except Exception:
            return 2

    def _calibration_search_roots(self) -> List[Path]:
        """Build conservative search roots from the survey folder and selected TBL/TS files."""
        roots: List[Path] = []
        folder_text = self.folder_path_var.get().strip()
        candidates = []
        if folder_text:
            candidates.append(folder_text)
        candidates.extend(list(self.tbl_paths) + list(self.ts_paths))
        for p in candidates:
            try:
                path = Path(p).expanduser().resolve()
            except Exception:
                path = Path(p).expanduser()
            parent = path if path.is_dir() else path.parent
            if parent not in roots:
                roots.append(parent)
        extra: List[Path] = []
        for root in roots:
            for name in CAL_SUBFOLDER_NAMES:
                q = root / name
                if q.exists() and q.is_dir() and q not in extra and q not in roots:
                    extra.append(q)
        return roots + extra

    def _tbl_sensor_tokens(self, channel: str) -> List[str]:
        """Return normalized Phoenix TBL sensor/model tokens for Hx/Hy/Hz."""
        channel = channel.upper()
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        if tbl is None:
            return [channel.lower()]
        key_map = {
            "HX": ["HXSN", "HSN1", "HSN", "HTYP1", "HTYP", "HNUM"],
            "HY": ["HYSN", "HSN2", "HSN", "HTYP2", "HTYP", "HNUM"],
            "HZ": ["HZSN", "HSN3", "HSN", "HTYP3", "HTYP", "HNUM"],
        }
        toks: List[str] = []
        for key in key_map.get(channel, []):
            v = tbl_get_best_value(tbl, key)
            if isinstance(v, dict):
                continue
            if v is None:
                continue
            s = str(v).strip().lower()
            for t in re.findall(r"[a-z0-9]+", s):
                if len(t) >= 2 and t not in {"none", "null", "coil", "sensor"}:
                    toks.append(t)
        # channel aliases are weak but useful fallback hints.
        toks.extend([channel.lower(), channel[-1].lower()])
        out: List[str] = []
        for t in toks:
            if t not in out:
                out.append(t)
        return out

    def _candidate_calibration_files(self) -> List[Path]:
        """Find Phoenix coil/electrode calibration files near the selected survey data."""
        found: List[Path] = []
        if self.folder_inventory is not None:
            found.extend(self.folder_inventory.clc_files)
            found.extend(self.folder_inventory.cle_files)
        roots = self._calibration_search_roots()
        max_depth = self._folder_search_depth()
        recursive = bool(self.recursive_calibration_search_var.get())
        allowed = CLC_EXTENSIONS | CLE_EXTENSIONS

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            if not recursive:
                it = root.iterdir()
                for p in it:
                    if p.is_file() and p.suffix.lower() in allowed and p not in found:
                        found.append(p)
                continue

            root_parts = len(root.parts)
            try:
                for p in root.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in allowed:
                        continue
                    depth = len(p.parent.parts) - root_parts
                    if depth <= max_depth and p not in found:
                        found.append(p)
            except Exception:
                pass
        return _unique_sorted_paths(found)

    def _score_calibration_candidate(self, channel: str, path: Path, cal: PhoenixCalibration) -> Tuple[float, str]:
        """Score one CLC/CLE against a Phoenix TBL channel using serial/type/header evidence."""
        tokens = self._tbl_sensor_tokens(channel)
        filename = path.stem.lower()
        header_text = " ".join(cal.header_strings).lower()
        score = 0.0
        reasons: List[str] = []

        # Strong evidence: explicit sensor serial/model token in filename or header.
        for tok in tokens:
            if len(tok) < 2:
                continue
            if tok in filename:
                add = 55.0 if len(tok) >= 4 else 15.0
                score += add
                reasons.append(f"filename contains {tok} (+{add:g})")
            if tok in header_text:
                add = 45.0 if len(tok) >= 4 else 12.0
                score += add
                reasons.append(f"header contains {tok} (+{add:g})")

        # Channel-specific aliases. Avoid over-weighting a bare x/y/z.
        ch = channel.lower()
        aliases = {
            "hx": ["hx", "h1", "chanx", "channelx", "magx"],
            "hy": ["hy", "h2", "chany", "channely", "magy"],
            "hz": ["hz", "h3", "chanz", "channelz", "magz"],
        }[ch]
        for a in aliases:
            if a in filename:
                score += 20.0
                reasons.append(f"filename channel hint {a} (+20)")
            if a in header_text:
                score += 15.0
                reasons.append(f"header channel hint {a} (+15)")

        # A successfully decoded response table is better evidence that this is a real calibration file.
        if cal.response_table is not None and len(cal.response_table) >= 5:
            score += 10.0
            reasons.append("response table decoded (+10)")
        if is_experimental_binary_calibration(cal):
            reasons.append("binary response decoder is experimental")

        return score, "; ".join(reasons) if reasons else "no identifying token matched"

    def auto_detect_coil_calibrations(self, quiet: bool = False) -> None:
        """Auto-discover and conservatively match Phoenix CLC/CLE files to Hx/Hy/Hz."""
        try:
            self._ensure_tbls_from_entry()
        except Exception:
            pass
        candidates = self._candidate_calibration_files()
        self.calibration_candidates = []
        self.coil_calibration_map = {"Hx": None, "Hy": None, "Hz": None}
        self.coil_calibration_scores = {"Hx": 0.0, "Hy": 0.0, "Hz": 0.0}
        self.coil_calibration_reasons = {"Hx": "", "Hy": "", "Hz": ""}

        if not candidates:
            self.calibration_status_var.set("Auto detect: no CLC/CLE files found near TBL/TS data")
            if not quiet:
                messagebox.showinfo("Auto calibration", "No .CLC/.CLE calibration files were found in the searched folders.")
            return

        loaded: Dict[Path, PhoenixCalibration] = {}
        for p in candidates:
            try:
                cal = read_calibration_file(p, kind="CLC coil/sensor")
                loaded[p] = cal
            except Exception as e:
                self.log(f"Calibration candidate skipped: {p.name}: {e}")

        # Score every channel/candidate pair.
        matrix: Dict[str, List[Tuple[float, Path, PhoenixCalibration, str]]] = {"Hx": [], "Hy": [], "Hz": []}
        for ch in ("Hx", "Hy", "Hz"):
            for p, cal in loaded.items():
                score, reason = self._score_calibration_candidate(ch, p, cal)
                matrix[ch].append((score, p, cal, reason))
            matrix[ch].sort(key=lambda x: x[0], reverse=True)

        # Conservative unique assignment: a file is used for only one channel unless it is the sole candidate.
        used: set[Path] = set()
        confidence_threshold = 35.0
        for ch in ("Hx", "Hy", "Hz"):
            choices = [x for x in matrix[ch] if x[1] not in used]
            if not choices:
                continue
            best = choices[0]
            second_score = choices[1][0] if len(choices) > 1 else -1.0
            score, p, cal, reason = best
            # Require either threshold or a clearly unique single candidate with a decoded response table.
            confident = score >= confidence_threshold and (score - second_score >= 8.0 or len(choices) == 1)
            if confident:
                self.coil_calibration_map[ch] = cal
                self.coil_calibration_scores[ch] = float(score)
                self.coil_calibration_reasons[ch] = reason
                used.add(p)

        unmatched = [ch for ch in ("Hx", "Hy", "Hz") if self.coil_calibration_map[ch] is None]
        leftover = [
            (p, cal) for p, cal in loaded.items()
            if p not in used and cal.response_table is not None and len(cal.response_table) >= 5
        ]
        leftover.sort(key=lambda x: str(x[0]).lower())
        if unmatched and len(leftover) == len(unmatched):
            self.log("CLC/CLE fallback: assigning leftover decoded files to unmatched Hx/Hy/Hz by filename order.")
            for ch, (p, cal) in zip(unmatched, leftover):
                self.coil_calibration_map[ch] = cal
                self.coil_calibration_scores[ch] = max(1.0, self.coil_calibration_scores.get(ch, 0.0))
                self.coil_calibration_reasons[ch] = "fallback leftover unique assignment by filename order"
                used.add(p)

        # If only one calibration file exists, keep it as a candidate rather than silently assigning to all axes.
        self.clcs = list(loaded.values())
        self.clc_path_var.set("; ".join(str(p) for p in loaded))

        report = []
        for ch in ("Hx", "Hy", "Hz"):
            cal = self.coil_calibration_map[ch]
            if cal is not None:
                report.append(f"{ch} -> {cal.path.name} [score={self.coil_calibration_scores[ch]:.0f}]")
            else:
                top = matrix[ch][0] if matrix[ch] else None
                if top is None:
                    report.append(f"{ch} -> no candidate")
                else:
                    report.append(f"{ch} -> UNCONFIRMED; best={top[1].name} [score={top[0]:.0f}]")

        matched = sum(self.coil_calibration_map[ch] is not None for ch in ("Hx", "Hy", "Hz"))
        self.calibration_status_var.set(f"Auto calibration: {matched}/3 coil channels confidently matched")
        self.log("\n=== V18 automatic coil calibration matching ===")
        self.log("Search roots: " + "; ".join(str(x) for x in self._calibration_search_roots()))
        self.log(f"Found {len(loaded)} CLC/CLE candidate(s)")
        for line in report:
            self.log(line)
        for ch in ("Hx", "Hy", "Hz"):
            if self.coil_calibration_reasons[ch]:
                self.log(f"  {ch} evidence: {self.coil_calibration_reasons[ch]}")

        if not quiet:
            messagebox.showinfo("Auto calibration result", "\n".join(report))

    def auto_detect_receiver_calibration(self, quiet: bool = False) -> None:
        """Auto-discover a Phoenix .CLB receiver calibration near the survey folder."""
        clb_files: List[Path] = []
        if self.folder_inventory is not None:
            clb_files.extend(self.folder_inventory.clb_files)
        roots = self._calibration_search_roots()
        recursive = bool(self.recursive_calibration_search_var.get())
        max_depth = self._folder_search_depth()
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            files = _iter_files_limited_depth(root, max_depth) if recursive else [p for p in root.iterdir() if p.is_file()]
            for p in files:
                if p.suffix.lower() in CLB_EXTENSIONS:
                    clb_files.append(p)
        clb_files = _unique_sorted_paths(clb_files)
        if not clb_files:
            if not quiet:
                messagebox.showinfo("Auto CLB", "No .CLB receiver calibration files were found in the searched folders.")
            self.log("Auto CLB: no .CLB files found")
            return
        chosen, reason = pick_best_clb(clb_files, self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None))
        if chosen is None:
            return
        try:
            self.clb_path_var.set(str(chosen))
            self.clb = read_calibration_file(chosen, kind="CLB receiver/box")
            self.log(f"Auto CLB: {chosen.name} ({reason})")
            self.log_calibration(self.clb)
            status = self.calibration_status_var.get().strip()
            extra = f"CLB={chosen.name}"
            self.calibration_status_var.set(f"{status}; {extra}" if status else extra)
        except Exception as e:
            self.log(f"Auto CLB failed for {chosen}: {e}")
            if not quiet:
                messagebox.showerror("CLB read error", str(e))


    def show_calibration_match_report(self) -> None:
        lines = ["Phoenix V18 Professional coil calibration match report", ""]
        for ch in ("Hx", "Hy", "Hz"):
            cal = self.coil_calibration_map.get(ch)
            if cal is None:
                lines.append(f"{ch}: not confidently assigned")
            else:
                lines.append(f"{ch}: {cal.path}")
                lines.append(f"    score: {self.coil_calibration_scores.get(ch, 0):.1f}")
                lines.append(f"    evidence: {self.coil_calibration_reasons.get(ch, '')}")
                lines.append(f"    response rows: {0 if cal.response_table is None else len(cal.response_table)}")
                lines.append(f"    note: {cal.note}")
        messagebox.showinfo("Calibration Match Report", "\n".join(lines))

    def select_clb(self) -> None:
        p = filedialog.askopenfilename(title="Select Phoenix CLB receiver calibration", filetypes=[("Phoenix CLB", "*.CLB *.clb"), ("All files", "*.*")])
        if not p:
            return
        self.clb_path_var.set(p)
        try:
            self.clb = read_calibration_file(p, kind="CLB receiver/box")
            self.calibration_status_var.set(f"CLB loaded: {Path(p).name} ({self.clb.size_bytes} bytes)")
            self.log_calibration(self.clb)
        except Exception as e:
            messagebox.showerror("CLB read error", str(e))
            self.log(traceback.format_exc())

    def select_clc(self) -> None:
        paths = filedialog.askopenfilenames(title="Select Phoenix CLC/CLE coil/electrode calibration file(s)", filetypes=[("Phoenix CLC/CLE", "*.CLC *.clc *.CLE *.cle"), ("All files", "*.*")])
        if not paths:
            return
        self.clc_path_var.set("; ".join(paths))
        try:
            self.clcs = [read_calibration_file(p, kind="CLC coil/sensor") for p in paths]
            self.calibration_status_var.set(f"CLC/CLE loaded manually: {len(self.clcs)} file(s)")
            for cal in self.clcs:
                self.log_calibration(cal)
            # Manual files remain authoritative; V18 reports them but does not guess axis assignment here.
            self.log("Manual CLC/CLE selection loaded. Use Auto Detect Coil Calibration for TBL-based Hx/Hy/Hz matching.")
        except Exception as e:
            messagebox.showerror("CLC read error", str(e))
            self.log(traceback.format_exc())

    def log_calibration(self, cal: PhoenixCalibration) -> None:
        self.log("\n=== Calibration file ===")
        self.log(f"Type: {cal.kind}")
        self.log(f"File: {cal.path}")
        self.log(f"Size: {cal.size_bytes} bytes")
        self.log(f"Readable strings: {cal.header_strings[:20]}")
        self.log(cal.note)
        if cal.response_table is not None:
            self.log(f"Response table rows: {len(cal.response_table)}")
            self.log(str(cal.response_table.head()))
        else:
            self.log("Hex preview:\n" + cal.hex_preview)

    def plot_calibration_response(self) -> None:
        candidates = []
        if self.clb is not None:
            candidates.append(self.clb)
        candidates.extend(self.clcs)
        with_table = [c for c in candidates if c.response_table is not None]
        self.fig.clear()
        if with_table:
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212, sharex=ax1)
            for cal in with_table:
                df = cal.response_table
                assert df is not None
                ax1.loglog(df["frequency_Hz"], np.abs(df["amplitude"]), marker=".", lw=0.8, label=cal.path.name)
                phase = np.rad2deg(np.unwrap(np.deg2rad(df["phase_deg"].to_numpy(float))))
                ax2.semilogx(df["frequency_Hz"], phase, marker=".", lw=0.8, label=cal.path.name)
            ax1.set_ylabel("Amplitude")
            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax1.grid(True, alpha=0.25, which="both")
            ax2.grid(True, alpha=0.25, which="both")
            ax1.legend(fontsize=8)
            self.fig.suptitle("Calibration response table preview")
            self.axes = [ax1, ax2]
        else:
            ax = self.fig.add_subplot(111)
            txt = "No ASCII response table decoded.\n\n"
            if candidates:
                txt += "Loaded binary Phoenix calibration file(s):\n"
                for cal in candidates:
                    txt += f"- {cal.path.name}: {cal.size_bytes} bytes; strings={cal.header_strings[:5]}\n"
                txt += "\nUse this file for metadata inspection, but do not apply response correction until the binary table is decoded."
            else:
                txt += "Please select CLB/CLC file(s) first."
            ax.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10, transform=ax.transAxes)
            ax.axis("off")
            self.axes = [ax]
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def load_ts(self, quiet: bool = False, update_display: bool = True) -> int:
        raw_paths = self.ts_paths[:]
        if not raw_paths:
            text = self.ts_path_var.get().strip()
            raw_paths = [p.strip().strip('"') for p in text.split(";") if p.strip()]
        if not raw_paths:
            if quiet:
                raise ValueError("Please select one or more .TSn files first.")
            messagebox.showwarning("Missing TS file", "Please select one or more .TSn files first.")
            return 0

        tbl_path = self.tbl_path_var.get().strip()
        try:
            if tbl_path and not self.tbls:
                self._ensure_tbls_from_entry()

            max_samples = None
            s = self.max_samples_var.get().strip()
            if s:
                max_samples = int(s)

            offset = None
            off_s = self.offset_var.get().strip().lower()
            if off_s and off_s != "auto":
                offset = int(off_s)

            self.info_var.set(f"Reading {len(raw_paths)} Phoenix raw data file(s) ...")
            try:
                self.update_idletasks()
            except Exception:
                pass
            self.datasets = []
            for i, ts_path in enumerate(raw_paths, start=1):
                self.info_var.set(f"Reading {i}/{len(raw_paths)}: {Path(ts_path).name}")
                try:
                    self.update_idletasks()
                except Exception:
                    pass
                matched_tbl = self._match_tbl_for_ts(ts_path)
                if matched_tbl is None and self.tbls:
                    self.log(f"WARNING: No matching TBL found for {Path(ts_path).name}; using TS suffix/default sample rate and current GUI settings.")
                elif matched_tbl is not None:
                    self.log(f"Matched {Path(ts_path).name} -> {matched_tbl.path.name}")
                    self._apply_tbl_defaults_to_gui(matched_tbl)
                df, meta = read_tsn(ts_path, tbl=matched_tbl, offset=offset, max_samples=max_samples)
                meta["matched_tbl_file"] = str(matched_tbl.path) if matched_tbl is not None else None
                meta["matched_tbl_entries"] = matched_tbl.entries if matched_tbl is not None else None
                meta["gui_voltage_settings"] = {
                    "adc_full_scale_range_vpp": self.ad_range_pp_var.get(),
                    "gain_Ex": self.ex_gain_var.get(),
                    "gain_Ey": self.ey_gain_var.get(),
                    "gain_Hx": self.hx_gain_var.get(),
                    "gain_Hy": self.hy_gain_var.get(),
                    "gain_Hz": self.hz_gain_var.get(),
                    "ex_dipole_length_m": self.ex_length_m_var.get(),
                    "ey_dipole_length_m": self.ey_length_m_var.get(),
                    "hx_sensor_sensitivity_mV_per_nT": self.hx_sens_mv_nt_var.get(),
                    "hy_sensor_sensitivity_mV_per_nT": self.hy_sens_mv_nt_var.get(),
                    "hz_sensor_sensitivity_mV_per_nT": self.hz_sens_mv_nt_var.get(),
                    "formula_adc_volt": "raw_count / 2**24 * full_scale_range_vpp",
                    "formula_input_volt": "adc_volt / amplifier_gain",
                    "formula_E_mV_per_km": "input_volt * 1e6 / dipole_length_m",
                    "formula_B_nT": "input_volt / (sensor_sensitivity_mV_per_nT * 1e-3)",
                    "clb_file": self.clb_path_var.get(),
                    "clc_files": self.clc_path_var.get(),
                    "calibration_note": "Binary Phoenix CLB/CLC files are decoded using an experimental heuristic when possible; validate response removal against Phoenix software.",
                }
                tbl_name = Path(str(meta.get("matched_tbl_file"))).name if meta.get("matched_tbl_file") else "no TBL match"
                label = f"{Path(ts_path).name} | {meta['sample_rate_Hz']} Hz | {meta['n_samples']} samples | TBL={tbl_name}"
                self.datasets.append({"label": label, "df": df, "meta": meta})
                self.log(f"Loaded {label}")

            labels = [str(d["label"]) for d in self.datasets]
            self.dataset_combo.configure(values=labels)
            if self.datasets:
                self.selected_dataset_var.set(labels[0])
                if update_display:
                    self.select_loaded_dataset()
                else:
                    item0 = self.datasets[0]
                    self.df = item0["df"]  # type: ignore[assignment]
                    self.meta = item0["meta"]  # type: ignore[assignment]

            msg = f"Loaded {len(self.datasets)} TS file(s). Use 'Loaded file to view' to switch display."
            self.info_var.set(msg)
            self.log(msg)
            try:
                merge_info = self._merged_time_report()
                if merge_info.get("overall_start_time") or merge_info.get("overall_end_time"):
                    self.log("Merged absolute time: " + str(merge_info.get("overall_start_time")) + "  →  " + str(merge_info.get("overall_end_time")))
                    self.info_var.set(msg + " | " + str(merge_info.get("overall_start_time")) + " → " + str(merge_info.get("overall_end_time")))
            except Exception as _e:
                self.log(f"Merged time report skipped: {_e}")
            return len(self.datasets)
        except Exception as e:
            self.info_var.set("Read failed.")
            self.log(traceback.format_exc())
            if quiet:
                raise
            messagebox.showerror("TS read error", str(e))
            return 0

    def select_loaded_dataset(self) -> None:
        label = self.selected_dataset_var.get()
        for item in self.datasets:
            if item.get("label") == label:
                self.df = item["df"]  # type: ignore[assignment]
                self.meta = item["meta"]  # type: ignore[assignment]
                settings = dict(self.meta.get("gui_voltage_settings", {})) if isinstance(self.meta, dict) else {}
                # Restore per-file TBL-derived parameters, so each file uses its own gain and electrode lengths.
                var_map = {
                    "adc_full_scale_range_vpp": self.ad_range_pp_var,
                    "gain_Ex": self.ex_gain_var, "gain_Ey": self.ey_gain_var,
                    "gain_Hx": self.hx_gain_var, "gain_Hy": self.hy_gain_var, "gain_Hz": self.hz_gain_var,
                    "ex_dipole_length_m": self.ex_length_m_var,
                    "ey_dipole_length_m": self.ey_length_m_var,
                    "hx_sensor_sensitivity_mV_per_nT": self.hx_sens_mv_nt_var,
                    "hy_sensor_sensitivity_mV_per_nT": self.hy_sens_mv_nt_var,
                    "hz_sensor_sensitivity_mV_per_nT": self.hz_sens_mv_nt_var,
                }
                for key, var in var_map.items():
                    if key in settings and settings[key] is not None:
                        var.set(str(settings[key]))
                self.update_plot()
                return

    def _input_volt_dataframe(self) -> Tuple[pd.DataFrame, str]:
        """Return instrument input volts using GUI A/D range and gain settings."""
        if self.df is None:
            raise ValueError("No data loaded")
        try:
            range_pp = float(self.ad_range_pp_var.get())
            if not np.isfinite(range_pp) or range_pp <= 0:
                range_pp = 5.0
        except Exception:
            range_pp = 5.0
        d = self.df.copy()
        gain_map = {ch: self._float_from_var(var, 1.0) for ch, var in self._gain_vars().items()}
        for ch in DEFAULT_CHANNELS:
            if ch in d.columns:
                v = d[ch].astype(float) / float(2**24) * range_pp
                v = v / gain_map.get(ch, 1.0)
                d[ch] = v
        return d, f"input volts = counts/2^24*{range_pp:g} Vpp/gain"

    def _physical_dataframe(self, rename_columns: bool = False) -> Tuple[pd.DataFrame, Dict[str, str], str]:
        """Preliminary physical units from input volts.

        Ex/Ey are converted to electric field using electrode dipole length:
            E_mV_per_km = input_volt * 1e6 / length_m
        because 1 V / 1 m = 1,000,000 mV/km.

        Hx/Hy/Hz are converted using a constant coil sensitivity entered in mV/nT:
            B_nT = input_volt / (sensitivity_mV_per_nT * 1e-3)

        This does not replace full frequency-domain CLB/CLC response removal.
        """
        d, _ = self._input_volt_dataframe()
        units: Dict[str, str] = {}
        ex_len = self._float_from_var(self.ex_length_m_var, 100.0)
        ey_len = self._float_from_var(self.ey_length_m_var, 100.0)
        hx_s = self._float_from_var(self.hx_sens_mv_nt_var, 1.0) * 1e-3
        hy_s = self._float_from_var(self.hy_sens_mv_nt_var, 1.0) * 1e-3
        hz_s = self._float_from_var(self.hz_sens_mv_nt_var, 1.0) * 1e-3
        if "Ex" in d.columns:
            d["Ex"] = d["Ex"].astype(float) * 1e6 / ex_len
            units["Ex"] = "mV/km"
        if "Ey" in d.columns:
            d["Ey"] = d["Ey"].astype(float) * 1e6 / ey_len
            units["Ey"] = "mV/km"
        for ch, sens in [("Hx", hx_s), ("Hy", hy_s), ("Hz", hz_s)]:
            if ch in d.columns:
                d[ch] = d[ch].astype(float) / sens
                units[ch] = "nT"
        note = f"Physical approx: Ex/Ey mV/km using Lx={ex_len:g} m, Ly={ey_len:g} m; H nT using mV/nT sensitivities Hx/Hy/Hz={hx_s*1e3:g}/{hy_s*1e3:g}/{hz_s*1e3:g}. Full CLB/CLC frequency response not removed."
        if rename_columns:
            rename = {}
            for ch, unit in units.items():
                safe = unit.replace("/", "_per_")
                rename[ch] = f"{ch}_{safe}"
            d = d.rename(columns=rename)
        return d, units, note


    def _best_clc_table(self) -> Optional[pd.DataFrame]:
        for cal in self.clcs:
            if cal.response_table is not None and len(cal.response_table) >= 5:
                return cal.response_table
        return None

    def _response_removed_spectrum_dataframe(self) -> Tuple[pd.DataFrame, str]:
        """Frequency-domain response removal and physical-unit conversion.

        Output is a one-sided FFT spectrum, not a time series.  It uses input volts,
        then divides by the decoded receiver response (.CLB) and coil response (.CLC).
        Electric channels are converted to mV/km using dipole length. Magnetic channels
        are interpreted as nT if the CLC response is in V/nT.
        """
        if self.df is None or self.meta is None:
            raise ValueError("No data loaded")
        d, _ = self._input_volt_dataframe()
        fs = float(self.meta.get("sample_rate_Hz", 1.0))
        n = len(d)
        if n < 16:
            raise ValueError("Too few samples for FFT calibration")
        # Limit FFT cost for GUI responsiveness.
        max_n = 262144
        if n > max_n:
            d = d.iloc[:max_n].copy()
            n = len(d)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        out = pd.DataFrame({"frequency_Hz": freq})
        clb_resp = None
        if self.clb is not None and self.clb.response_table is not None and len(self.clb.response_table) >= 5:
            clb_resp = interpolate_complex_response(np.maximum(freq, 1e-300), self.clb.response_table)
            clb_resp[0] = clb_resp[1] if len(clb_resp) > 1 else 1.0 + 0j
        clc_table = self._best_clc_table()
        clc_resp = None
        if clc_table is not None:
            clc_resp = interpolate_complex_response(np.maximum(freq, 1e-300), clc_table)
            clc_resp[0] = clc_resp[1] if len(clc_resp) > 1 else 1.0 + 0j

        ex_len = self._float_from_var(self.ex_length_m_var, 100.0)
        ey_len = self._float_from_var(self.ey_length_m_var, 100.0)
        # Fallback constant sensitivities when no CLC table is available.
        sens_const = {
            "Hx": self._float_from_var(self.hx_sens_mv_nt_var, 1.0) * 1e-3,
            "Hy": self._float_from_var(self.hy_sens_mv_nt_var, 1.0) * 1e-3,
            "Hz": self._float_from_var(self.hz_sens_mv_nt_var, 1.0) * 1e-3,
        }
        for ch in DEFAULT_CHANNELS:
            if ch not in d.columns:
                continue
            x = d[ch].to_numpy(float)
            x = x - np.nanmean(x)
            X = np.fft.rfft(np.nan_to_num(x))
            if clb_resp is not None:
                X = X / np.where(np.abs(clb_resp) > 1e-30, clb_resp, 1.0 + 0j)
            if ch in ("Ex", "Ey"):
                L = ex_len if ch == "Ex" else ey_len
                Y = X * 1e6 / L  # V/m -> mV/km
                unit = "mV_per_km"
            else:
                if clc_resp is not None:
                    Y = X / np.where(np.abs(clc_resp) > 1e-30, clc_resp, np.nan + 1j*np.nan)
                else:
                    Y = X / sens_const[ch]
                unit = "nT"
            out[f"{ch}_{unit}_real"] = Y.real
            out[f"{ch}_{unit}_imag"] = Y.imag
            out[f"{ch}_{unit}_abs"] = np.abs(Y)
            out[f"{ch}_{unit}_phase_deg"] = np.rad2deg(np.angle(Y))
        note = "FFT response removed spectrum: input volt / CLB / CLC; Ex/Ey=mV/km, H=nT. Experimental binary parser; validate externally."
        return out, note

    def _response_removed_time_dataframe(self) -> Tuple[pd.DataFrame, Dict[str, str], str]:
        """Return inverse-FFT response-removed time series for preview."""
        if self.df is None or self.meta is None:
            raise ValueError("No data loaded")
        d, _ = self._input_volt_dataframe()
        fs = float(self.meta.get("sample_rate_Hz", 1.0))
        n = len(d)
        max_n = 262144
        clipped = False
        if n > max_n:
            d = d.iloc[:max_n].copy()
            n = len(d)
            clipped = True
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        clb_resp = None
        if self.clb is not None and self.clb.response_table is not None and len(self.clb.response_table) >= 5:
            clb_resp = interpolate_complex_response(np.maximum(freq, 1e-300), self.clb.response_table)
            clb_resp[0] = clb_resp[1] if len(clb_resp) > 1 else 1.0 + 0j
        clc_table = self._best_clc_table()
        clc_resp = None
        if clc_table is not None:
            clc_resp = interpolate_complex_response(np.maximum(freq, 1e-300), clc_table)
            clc_resp[0] = clc_resp[1] if len(clc_resp) > 1 else 1.0 + 0j
        out = pd.DataFrame({"time_s": np.arange(n, dtype=float) / fs})
        units = {}
        ex_len = self._float_from_var(self.ex_length_m_var, 100.0)
        ey_len = self._float_from_var(self.ey_length_m_var, 100.0)
        sens_const = {"Hx": self._float_from_var(self.hx_sens_mv_nt_var, 1.0) * 1e-3,
                      "Hy": self._float_from_var(self.hy_sens_mv_nt_var, 1.0) * 1e-3,
                      "Hz": self._float_from_var(self.hz_sens_mv_nt_var, 1.0) * 1e-3}
        for ch in DEFAULT_CHANNELS:
            if ch not in d.columns:
                continue
            x = d[ch].to_numpy(float)
            mean = np.nanmean(x)
            X = np.fft.rfft(np.nan_to_num(x - mean))
            if clb_resp is not None:
                X = X / np.where(np.abs(clb_resp) > 1e-30, clb_resp, 1.0 + 0j)
            if ch in ("Ex", "Ey"):
                L = ex_len if ch == "Ex" else ey_len
                X = X * 1e6 / L
                units[ch] = "mV/km"
            else:
                if clc_resp is not None:
                    X = X / np.where(np.abs(clc_resp) > 1e-30, clc_resp, np.nan + 1j*np.nan)
                else:
                    X = X / sens_const[ch]
                units[ch] = "nT"
            y = np.fft.irfft(X, n=n)
            out[ch] = y
        note = "Response removed physical preview" + ("; first 262144 samples only" if clipped else "")
        if self.clb is None or self.clb.response_table is None:
            note += "; no decoded CLB applied"
        if clc_resp is None:
            note += "; no decoded CLC applied, constant H sensitivity used"
        return out, units, note


    def _validate_official_calibration_chain(self) -> Dict[str, object]:
        """Validate the calibration chain required for scientific MTH5 export.

        Official calibrated MTH5 export is intentionally fail-closed:
        - a decoded CLB receiver response is required;
        - Hx, Hy and Hz each require their own confidently matched CLC/CLE response;
        - experimental Phoenix binary response tables are only allowed when the
          user explicitly enables the experimental-response checkbox;
        - no constant mV/nT sensitivity fallback is ever used for MTH5 export.
        """
        use_exp = bool(self.use_experimental_binary_response_var.get())
        problems = []

        if self.clb is None or self.clb.response_table is None or len(self.clb.response_table) < 5:
            problems.append("CLB receiver response is missing or was not decoded")
        elif is_experimental_binary_calibration(self.clb) and not use_exp:
            problems.append("CLB is an experimental binary decode; enable 'Use experimental binary response' only after validation")

        channel_cals: Dict[str, PhoenixCalibration] = {}
        for ch in ("Hx", "Hy", "Hz"):
            cal = self.coil_calibration_map.get(ch)
            if cal is None:
                problems.append(f"{ch}: no confidently matched CLC/CLE calibration")
                continue
            if cal.response_table is None or len(cal.response_table) < 5:
                problems.append(f"{ch}: matched CLC/CLE has no decoded response table")
                continue
            if is_experimental_binary_calibration(cal) and not use_exp:
                problems.append(f"{ch}: CLC/CLE is an experimental binary decode; explicit enable/validation required")
                continue
            channel_cals[ch] = cal

        if problems:
            raise RuntimeError(
                "Scientific MTH5 export requires a complete CLB + Hx/Hy/Hz CLC calibration chain.\n"
                "No constant 1 mV/nT fallback is permitted.\n\n- " + "\n- ".join(problems)
            )
        return {"clb": self.clb, "clc": channel_cals, "use_experimental": use_exp}

    def _response_removed_time_dataframe_ola(self, require_complete_calibration: bool = False) -> Tuple[pd.DataFrame, Dict[str, str], str]:
        """Calibrate Phoenix time series with CLB + channel-specific CLC responses.

        Scientific path used by MTH5 v21:
            TSn counts -> ADC volts -> divide EGN/HGN -> CLB(f)
            -> CLC_Hx/Hy/Hz(f) -> IFFT calibrated time series -> MTH5.

        Electric channels are divided by the receiver response and converted to
        mV/km using electrode length. Magnetic channels are divided by the same
        receiver response and then by their *own* Hx/Hy/Hz CLC/CLE complex response.

        When ``require_complete_calibration`` is True this routine is fail-closed:
        missing/invalid CLB or CLC responses raise an error and there is absolutely
        no constant-sensitivity fallback.
        """
        if self.df is None or self.meta is None:
            raise ValueError("No data loaded")

        if require_complete_calibration:
            chain = self._validate_official_calibration_chain()
        else:
            chain = None

        d, input_note = self._input_volt_dataframe()  # includes EGN/HGN division
        fs = float(self.meta.get("sample_rate_Hz", 1.0))
        n_total = len(d)
        if n_total < 16:
            raise ValueError("Too few samples for calibrated time-series export")

        try:
            win_sec = float(self.cal_window_sec_var.get())
        except Exception:
            win_sec = 256.0
        if not np.isfinite(win_sec) or win_sec <= 0:
            win_sec = 256.0
        try:
            overlap = float(self.cal_overlap_var.get())
        except Exception:
            overlap = 0.5
        if not np.isfinite(overlap):
            overlap = 0.5
        overlap = min(max(overlap, 0.0), 0.9)

        nperseg = max(256, int(round(win_sec * fs)))
        nfft = max(256, 1 << int(np.ceil(np.log2(nperseg))))
        if nfft > n_total and n_total >= 256:
            nfft = max(256, 1 << int(np.floor(np.log2(n_total))))
        hop = max(1, int(round(nfft * (1.0 - overlap))))
        window = np.hanning(nfft)
        if np.all(window == 0):
            window = np.ones(nfft)

        freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
        use_exp = bool(self.use_experimental_binary_response_var.get())
        bandlimit = bool(self.bandlimit_response_var.get())
        clb_resp, clb_status = safe_interpolated_response(freq, self.clb, use_exp, bandlimit)
        if require_complete_calibration and clb_resp is None:
            raise RuntimeError(f"CLB response unavailable for MTH5 calibration: {clb_status}")

        clc_responses: Dict[str, np.ndarray] = {}
        clc_statuses: Dict[str, str] = {}
        for ch in ("Hx", "Hy", "Hz"):
            cal = self.coil_calibration_map.get(ch)
            resp, status = safe_interpolated_response(freq, cal, use_exp, bandlimit)
            clc_statuses[ch] = status
            if resp is not None:
                clc_responses[ch] = resp
            elif require_complete_calibration:
                raise RuntimeError(f"{ch} CLC response unavailable for MTH5 calibration: {status}")

        ex_len = self._float_from_var(self.ex_length_m_var, 100.0)
        ey_len = self._float_from_var(self.ey_length_m_var, 100.0)
        if ex_len <= 0 or ey_len <= 0:
            raise RuntimeError("Electric dipole lengths must be positive for calibrated MTH5 export")

        out = pd.DataFrame({"time_s": np.arange(n_total, dtype=float) / fs})
        if "datetime" in d.columns:
            out["datetime"] = d["datetime"].values
        units: Dict[str, str] = {}

        starts = list(range(0, max(1, n_total - nfft + 1), hop))
        last_start = max(0, n_total - nfft)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        starts = sorted(set(starts))

        weight = np.zeros(n_total, dtype=float)
        for st in starts:
            en = min(st + nfft, n_total)
            weight[st:en] += window[:en-st] ** 2
        weight_safe = np.where(weight > 1e-12, weight, 1.0)

        diagnostics = []
        for ch in DEFAULT_CHANNELS:
            if ch not in d.columns:
                continue
            x_all = pd.to_numeric(d[ch], errors="coerce").to_numpy(float)
            y_sum = np.zeros(n_total, dtype=float)

            for st in starts:
                en = min(st + nfft, n_total)
                seg0 = x_all[st:en]
                finite = np.isfinite(seg0)
                if not np.any(finite):
                    continue
                seg0 = np.nan_to_num(seg0 - np.nanmean(seg0[finite]))
                seg = np.zeros(nfft, dtype=float)
                seg[:len(seg0)] = seg0
                X = np.fft.rfft(seg * window, n=nfft)
                X[0] = 0.0 + 0.0j

                if clb_resp is not None:
                    den = np.where(np.abs(clb_resp) > 1e-30, clb_resp, np.nan + 1j*np.nan)
                    X = X / den
                elif require_complete_calibration:
                    raise RuntimeError("CLB response unexpectedly missing during calibration")

                if ch in ("Ex", "Ey"):
                    L = ex_len if ch == "Ex" else ey_len
                    X = X * 1e6 / L
                    units[ch] = "mV/km"
                else:
                    resp = clc_responses.get(ch)
                    if resp is None:
                        if require_complete_calibration:
                            raise RuntimeError(f"{ch} CLC response unexpectedly missing during calibration")
                        # Preview-only mode: do not pretend an uncalibrated magnetic channel is nT.
                        units[ch] = "V (uncalibrated magnetic preview)"
                    else:
                        den = np.where(np.abs(resp) > 1e-30, resp, np.nan + 1j*np.nan)
                        X = X / den
                        units[ch] = "nT"

                yy = np.fft.irfft(X, n=nfft)
                take = en - st
                y_sum[st:en] += yy[:take] * window[:take]

            vals = y_sum / weight_safe
            out[ch] = vals
            finite = np.isfinite(vals)
            if require_complete_calibration:
                if not np.any(finite):
                    raise RuntimeError(f"{ch}: calibrated output is entirely non-finite")
                sd = float(np.nanstd(vals))
                if not np.isfinite(sd) or sd <= 1e-30:
                    raise RuntimeError(f"{ch}: calibrated output is flat/invalid (std={sd})")
            if np.any(finite):
                diagnostics.append(f"{ch}: std={np.nanstd(vals):.6g}, p95={np.nanpercentile(np.abs(vals),95):.6g}")

        note = (
            f"V21 calibrated OLA: {input_note}; window={nfft/fs:g}s ({nfft} samples), overlap={overlap:g}; "
            f"CLB={clb_status}; "
            + "; ".join(f"CLC_{ch}={clc_statuses.get(ch, 'missing')}" for ch in ("Hx", "Hy", "Hz"))
            + "; constant H sensitivity fallback DISABLED for MTH5"
        )
        if diagnostics:
            note += "; diagnostics: " + " | ".join(diagnostics)
        return out, units, note


    def _merged_raw_dataframe_for_calibration(self) -> Tuple[pd.DataFrame, Dict[str, object], str]:
        """Return raw dataframe/meta to calibrate.

        If multiple loaded files are continuous and have the same sampling rate, this
        creates one continuous raw dataframe before response removal.  Otherwise it
        returns the currently selected dataframe.  This avoids calibrating each file
        independently when A/B/C files are true consecutive blocks.
        """
        if self.df is None or self.meta is None:
            raise ValueError("No data loaded")
        if len(self.datasets) < 2:
            return self.df.copy(), dict(self.meta), "current file"
        ok, report, items = self.check_continuity_report()
        if not ok:
            self.log(report)
            return self.df.copy(), dict(self.meta), "current file only; loaded files are not continuous"
        fs = float(items[0]["fs"])
        start_dt = items[0]["start"]
        frames = []
        source_names = []
        for x in items:
            item = x["item"]
            d0 = item["df"].copy()  # raw counts dataframe
            name = Path(str(item.get("meta", {}).get("ts_file", ""))).name if isinstance(item.get("meta"), dict) else str(item.get("label", ""))
            source_names.extend([name] * len(d0))
            frames.append(d0[[c for c in ["time_s"] + DEFAULT_CHANNELS if c in d0.columns]])
        merged = pd.concat(frames, ignore_index=True)
        merged["time_s"] = np.arange(len(merged), dtype=float) / fs
        if start_dt is not None:
            merged.insert(0, "datetime", [(start_dt + timedelta(seconds=float(t))).isoformat(sep=" ") for t in merged["time_s"].to_numpy(float)])
        merged.insert(0, "source_file", source_names)
        meta = dict(items[0]["item"].get("meta", {}))
        meta["sample_rate_Hz"] = fs
        meta["ts_file"] = "MERGED_CONTINUOUS_TSn"
        if start_dt is not None:
            meta["absolute_start_time"] = start_dt.isoformat(sep=" ")
            meta["absolute_end_time_estimated"] = (start_dt + timedelta(seconds=len(merged)/fs)).isoformat(sep=" ")
        return merged, meta, "continuous merged files"

    def run_calibration_update(self) -> None:
        """Apply CLB/CLC response removal and update GUI to calibrated time series."""
        if self.df is None or self.meta is None:
            messagebox.showwarning("No data", "Please load TS data first.")
            return
        try:
            self.raw_df_before_apply = self.df.copy()
            self.raw_meta_before_apply = dict(self.meta)
            use_df, use_meta, scope = self._merged_raw_dataframe_for_calibration()
            old_df, old_meta = self.df, self.meta
            self.df, self.meta = use_df, use_meta
            out, units, note = self._response_removed_time_dataframe_ola()
            # Preserve absolute datetime/source_file columns when available.
            if "datetime" in use_df.columns and "datetime" not in out.columns:
                out.insert(0, "datetime", use_df["datetime"].values[:len(out)])
            if "source_file" in use_df.columns and "source_file" not in out.columns:
                out.insert(0, "source_file", use_df["source_file"].values[:len(out)])
            self.calibrated_df = out
            self.calibrated_units = units
            self.calibrated_note = note + f"; applied scope={scope}"
            # Keep self.df/self.meta as the raw source used for calibration, so rerun/export remains consistent.
            self.df, self.meta = use_df, use_meta
            self.display_mode_var.set("Physical approx")
            self.info_var.set("Response removal applied. Display is now response-removed physical time series.")
            self.log("Response removal applied and GUI updated to calibrated physical time series.")
            self.log(self.calibrated_note)
            self.update_plot()
            messagebox.showinfo("Calibration done", "Calibration applied. The plot now shows the calibrated time series.\n\n" + self.calibrated_note)
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Calibration failed", str(e))

    def export_applied_calibrated_csv(self) -> None:
        """Export the calibrated dataframe already generated by Run calibration."""
        if self.calibrated_df is None:
            messagebox.showwarning("No calibrated data", "Please click Run calibration first.")
            return
        p = filedialog.asksaveasfilename(
            title="Save applied calibrated time-series CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="phoenix_applied_calibrated_timeseries.csv",
        )
        if not p:
            return
        out = self.calibrated_df.copy()
        rename = {ch: f"{ch}_{self.calibrated_units[ch].replace('/', '_per_')}" for ch in self.calibrated_units if ch in out.columns}
        out = out.rename(columns=rename)
        out.to_csv(p, index=False)
        self.log(f"Applied calibrated time-series CSV saved: {p} ({self.calibrated_note})")
        messagebox.showinfo("Saved", f"Applied calibrated time-series CSV saved:\n{p}")

    def export_calibrated_time_csv(self) -> None:
        if self.df is None:
            messagebox.showwarning("No data", "Please load a TS file first.")
            return
        p = filedialog.asksaveasfilename(title="Save calibrated time-series CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        try:
            out, units, note = self._response_removed_time_dataframe_ola()
            rename = {ch: f"{ch}_{units[ch].replace('/', '_per_')}" for ch in units if ch in out.columns}
            out = out.rename(columns=rename)
            out.to_csv(p, index=False)
            self.log(f"Calibrated time-series CSV saved: {p} ({note})")
            messagebox.showinfo("Saved", f"Calibrated time-series CSV saved:\n{p}\n\n{note}")
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Calibration export failed", str(e))

    def export_calibrated_spectrum_csv(self) -> None:
        if self.df is None:
            messagebox.showwarning("No data", "Please load a TS file first.")
            return
        p = filedialog.asksaveasfilename(title="Save response-removed spectrum CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        out, note = self._response_removed_spectrum_dataframe()
        out.to_csv(p, index=False)
        self.log(f"Response-removed spectrum CSV saved: {p} ({note})")
        messagebox.showinfo("Saved", f"Response-removed spectrum CSV saved:\n{p}")

    def _current_display_dataframe(self) -> Tuple[pd.DataFrame, str, str]:
        """Return a plotting/export view according to selected display mode.

        Raw counts: original signed 24-bit integer samples.
        ADC Volt: voltage at ADC chip, using LSB = full_scale_Vpp / 2**24.
        Input Volt: estimated instrument input voltage after dividing by preamp gain.
        Physical approx: Ex/Ey in mV/km and Hx/Hy/Hz in nT using user settings.
        """
        if self.df is None:
            raise ValueError("No data loaded")
        mode = self.display_mode_var.get()
        if mode == "Physical approx":
            if self.calibrated_df is not None:
                return self.calibrated_df, "mixed", self.calibrated_note
            d, units, note = self._response_removed_time_dataframe()
            return d, "mixed", note
        if mode == "Physical approx":
            d, _, note = self._physical_dataframe(rename_columns=True)
            return d, "mixed", note
        if mode in ("ADC Volt", "Input Volt"):
            if mode == "Input Volt":
                d, note = self._input_volt_dataframe()
                return d, "V", note
            try:
                range_pp = float(self.ad_range_pp_var.get())
                if not np.isfinite(range_pp) or range_pp <= 0:
                    range_pp = 5.0
            except Exception:
                range_pp = 5.0
            d = self.df.copy()
            for ch in DEFAULT_CHANNELS:
                if ch in d.columns:
                    d[ch] = d[ch].astype(float) / float(2**24) * range_pp
            return d, "V", f"ADC volts = counts/2^24*{range_pp:g} Vpp"
        return self.df, "count", "raw ADC counts"

    def update_plot(self) -> None:
        for ax in np.ravel(self.axes):
            ax.clear()
        if self.df is None:
            for ax, ch in zip(np.ravel(self.axes), DEFAULT_CHANNELS):
                ax.set_ylabel(f"{ch}\n(count)")
                ax.grid(True, alpha=0.25)
            self.axes[-1].set_xlabel("Time (s)")
            self.canvas.draw_idle()
            return

        mode = self.display_mode_var.get()
        if mode == "Physical approx":
            display_df, unit_map, unit_note = self._physical_dataframe(rename_columns=False)
            yunit_default = ""
        elif mode == "Physical approx":
            if self.calibrated_df is not None:
                display_df, unit_map, unit_note = self.calibrated_df, self.calibrated_units, self.calibrated_note
            else:
                display_df, unit_map, unit_note = self._response_removed_time_dataframe()
            yunit_default = ""
        else:
            display_df, yunit_default, unit_note = self._current_display_dataframe()
            unit_map = {ch: yunit_default for ch in DEFAULT_CHANNELS}
        channels = [ch for ch in DEFAULT_CHANNELS if self.channel_vars[ch].get() and ch in display_df.columns]
        if not channels:
            self.canvas.draw_idle()
            return

        max_plot_points = 50000
        step = max(1, len(display_df) // max_plot_points)
        d = display_df.iloc[::step]

        self.fig.clear()
        self.axes = self.fig.subplots(len(channels), 1, sharex=True)
        if len(channels) == 1:
            self.axes = [self.axes]
        for ax, ch in zip(self.axes, channels):
            ax.plot(d["time_s"], d[ch], lw=0.6)
            unit = unit_map.get(ch, yunit_default)
            ax.set_ylabel(f"{ch}\n({unit})")
            ax.grid(True, alpha=0.25)
        self.axes[-1].set_xlabel("Time (s)")
        title = Path(str(self.meta.get("ts_file", "Phoenix TS"))).name if self.meta else "Phoenix TS"
        self.fig.suptitle(title + "   " + unit_note)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def export_csv(self) -> None:
        if self.df is None:
            messagebox.showwarning("No data", "Please load a TS file first.")
            return
        p = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        out_df, _, unit_note = self._current_display_dataframe()
        out_df.to_csv(p, index=False)
        self.log(f"CSV saved: {p} ({unit_note})")
        messagebox.showinfo("Saved", f"CSV saved:\n{p}")

    def export_all_csv(self) -> None:
        if not self.datasets:
            messagebox.showwarning("No data", "Please load TS files first.")
            return
        folder = filedialog.askdirectory(title="Select folder for all CSV exports")
        if not folder:
            return
        old_df, old_meta = self.df, self.meta
        saved = 0
        for item in self.datasets:
            self.df = item["df"]  # type: ignore[assignment]
            self.meta = item["meta"]  # type: ignore[assignment]
            out_df, _, unit_note = self._current_display_dataframe()
            ts_file = Path(str(self.meta.get("ts_file", "Phoenix_TS")))
            mode = self.display_mode_var.get().replace(" ", "_")
            out = Path(folder) / f"{ts_file.stem}_{mode}.csv"
            out_df.to_csv(out, index=False)
            saved += 1
            self.log(f"CSV saved: {out} ({unit_note})")
        self.df, self.meta = old_df, old_meta
        self.update_plot()
        messagebox.showinfo("Saved", f"Saved {saved} CSV file(s) to:\n{folder}")


    def _merged_time_report(self) -> Dict[str, object]:
        """Summarize absolute start/end time and continuity across loaded datasets."""
        files = []
        starts = []
        ends = []
        for item in self.datasets:
            meta = item.get('meta', {}) if isinstance(item, dict) else {}
            label = item.get('label', '') if isinstance(item, dict) else ''
            s, e = dataset_start_end_from_meta_or_df(item)
            if s:
                starts.append(pd.to_datetime(s, errors='coerce'))
            if e:
                ends.append(pd.to_datetime(e, errors='coerce'))
            files.append({
                'label': label,
                'ts_file': meta.get('ts_file'),
                'matched_tbl_file': meta.get('matched_tbl_file') or meta.get('tbl_file'),
                'sample_rate_Hz': meta.get('sample_rate_Hz'),
                'n_samples': meta.get('n_samples'),
                'read_mode': meta.get('read_mode'),
                'start_time': s,
                'end_time': e,
                'timing_gap_count': meta.get('timing_gap_count'),
                'duration_s': meta.get('duration_s'),
            })
        starts2 = [x for x in starts if not pd.isna(x)]
        ends2 = [x for x in ends if not pd.isna(x)]
        overall_start = min(starts2).isoformat() if starts2 else None
        overall_end = max(ends2).isoformat() if ends2 else None
        continuity = []
        ordered = []
        for f in files:
            ts = pd.to_datetime(f.get('start_time'), errors='coerce') if f.get('start_time') else pd.NaT
            ordered.append((ts, f))
        ordered = [x for x in ordered if not pd.isna(x[0])]
        ordered.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(ordered[:-1], ordered[1:]):
            ea = pd.to_datetime(a.get('end_time'), errors='coerce') if a.get('end_time') else pd.NaT
            sb = pd.to_datetime(b.get('start_time'), errors='coerce') if b.get('start_time') else pd.NaT
            gap = None
            status = 'unknown'
            if not pd.isna(ea) and not pd.isna(sb):
                gap = float((sb - ea).total_seconds())
                if abs(gap) <= 1.0:
                    status = 'continuous'
                elif gap > 1.0:
                    status = 'gap'
                else:
                    status = 'overlap'
            continuity.append({
                'from': a.get('ts_file'),
                'to': b.get('ts_file'),
                'previous_end': a.get('end_time'),
                'next_start': b.get('start_time'),
                'gap_seconds': gap,
                'status': status,
            })
        total_seconds = None
        if overall_start and overall_end:
            total_seconds = float((pd.to_datetime(overall_end) - pd.to_datetime(overall_start)).total_seconds())
        return {
            'overall_start_time': overall_start,
            'overall_end_time': overall_end,
            'overall_duration_s': total_seconds,
            'files': files,
            'continuity_between_files': continuity,
            'is_continuous': all(x.get('status') == 'continuous' for x in continuity) if continuity else None,
        }

    def _build_json_log_metadata(self) -> Dict[str, object]:
        """Build a corrected Phoenix survey/process LOG dictionary from TBL + loaded TS data."""
        tbl_meta = []
        for tbl in self.tbls:
            tbl_meta.append(build_tbl_metadata(tbl))
        current_tbl_meta = build_tbl_metadata(self.tbl) if self.tbl is not None else {}
        merged_time = self._merged_time_report()
        cal_files = {
            'CLB_receiver_file': self.clb_path_var.get() or None,
            'CLC_sensor_files': [x.strip() for x in self.clc_path_var.get().split(';') if x.strip()],
        }
        cal_decoded = []
        if self.clb is not None:
            cal_decoded.append({'file': str(self.clb.path), 'kind': self.clb.kind, 'size_bytes': self.clb.size_bytes, 'note': self.clb.note, 'response_rows': None if self.clb.response_table is None else int(len(self.clb.response_table))})
        for c in self.clcs:
            cal_decoded.append({'file': str(c.path), 'kind': c.kind, 'size_bytes': c.size_bytes, 'note': c.note, 'response_rows': None if c.response_table is None else int(len(c.response_table))})
        return {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'program': 'Phoenix TBL/TSn GUI corrected JSON/LOG exporter',
            'important_notes': [
                'LATG/LNGG are decoded as degrees + decimal minutes: ddmm.mmm,H and dddmm.mmm,H.',
                'Absolute start/end time is taken from decoded TS block headers when available; TBL STIM/ETIM are kept as raw candidates if present.',
                'Coil type/serial/orientation fields are stored as candidates because Phoenix TBL variants use different key names.'
            ],
            'current_view_dataset': self.selected_dataset_var.get(),
            'all_selected_TBL_metadata': tbl_meta,
            'current_or_first_TBL_metadata': current_tbl_meta,
            'calibration_files': cal_files,
            'decoded_calibration_file_info': cal_decoded,
            'merged_absolute_time': merged_time,
            'gui_scaling_parameters': {
                'adc_full_scale_range_vpp': self.ad_range_pp_var.get(),
                'gain_Ex': self.ex_gain_var.get(),
                'gain_Ey': self.ey_gain_var.get(),
                'gain_Hx': self.hx_gain_var.get(),
                'gain_Hy': self.hy_gain_var.get(),
                'gain_Hz': self.hz_gain_var.get(),
                'ex_dipole_length_m': self.ex_length_m_var.get(),
                'ey_dipole_length_m': self.ey_length_m_var.get(),
                'hx_sensor_sensitivity_mV_per_nT': self.hx_sens_mv_nt_var.get(),
                'hy_sensor_sensitivity_mV_per_nT': self.hy_sens_mv_nt_var.get(),
                'hz_sensor_sensitivity_mV_per_nT': self.hz_sens_mv_nt_var.get(),
            },
            'conversion_formulas': {
                'adc_volt': 'raw_count / 2^24 * ADC_full_scale_range_Vpp',
                'input_volt': 'adc_volt / amplifier_gain',
                'electric_field_mV_per_km': 'input_volt * 1e6 / electrode_length_m',
                'magnetic_field_nT_constant_sensitivity': 'input_volt / (sensor_sensitivity_mV_per_nT * 1e-3)',
            },
        }

    def export_json_log(self) -> None:
        """Export corrected metadata JSON and human-readable LOG side by side."""
        if not self.datasets and self.df is None:
            messagebox.showwarning('No data', 'Please load TBL/TSn data first.')
            return
        p = filedialog.asksaveasfilename(
            title='Save corrected Phoenix metadata JSON/LOG',
            defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
            initialfile='phoenix_metadata_log.json',
        )
        if not p:
            return
        try:
            meta = self._build_json_log_metadata()
            json_path = Path(p)
            json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            log_path = json_path.with_suffix('.log')
            lines = []
            lines.append('Phoenix TBL/TSn corrected metadata LOG')
            lines.append('=' * 60)
            mt = meta.get('merged_absolute_time', {})
            lines.append(f"Overall absolute start : {mt.get('overall_start_time')}")
            lines.append(f"Overall absolute end   : {mt.get('overall_end_time')}")
            lines.append(f"Overall duration (s)   : {mt.get('overall_duration_s')}")
            lines.append(f"Continuous             : {mt.get('is_continuous')}")
            lines.append('')
            for i, t in enumerate(meta.get('all_selected_TBL_metadata', []), 1):
                lines.append(f"[TBL {i}] {t.get('tbl_name')}")
                lines.append(f"  Latitude  raw/decimal : {t.get('latitude_raw_ddmm')} -> {t.get('latitude_decimal_deg')}")
                lines.append(f"  Longitude raw/decimal : {t.get('longitude_raw_ddmm')} -> {t.get('longitude_decimal_deg')}")
                lines.append(f"  EXLN/EYLN (m)         : {t.get('electrode_spacing_m')}")
                lines.append(f"  EAZM/HAZM candidates  : {t.get('orientation_deg_candidates')}")
                lines.append(f"  Coil candidates       : {t.get('coil_serial_or_type_candidates')}")
                lines.append(f"  Sample rates          : {t.get('sample_rates_from_tbl')}")
                lines.append(f"  Gain                  : {t.get('gain_from_tbl')}")
                lines.append('')
            lines.append('[Loaded / merged files]')
            for f in mt.get('files', []):
                lines.append(f"  TS={f.get('ts_file')} | TBL={f.get('matched_tbl_file')} | fs={f.get('sample_rate_Hz')} | start={f.get('start_time')} | end={f.get('end_time')}")
            lines.append('')
            lines.append('[Continuity]')
            for c in mt.get('continuity_between_files', []):
                lines.append(f"  {c.get('status')}: gap={c.get('gap_seconds')} s | {c.get('from')} -> {c.get('to')}")
            lines.append('')
            lines.append('[Calibration files]')
            lines.append(json.dumps(meta.get('calibration_files'), indent=2, ensure_ascii=False))
            lines.append('')
            lines.append('[GUI scaling parameters]')
            lines.append(json.dumps(meta.get('gui_scaling_parameters'), indent=2, ensure_ascii=False))
            log_path.write_text('\n'.join(lines), encoding='utf-8')
            self.log(f'Corrected JSON saved: {json_path}')
            self.log(f'Corrected LOG saved : {log_path}')
            messagebox.showinfo('Saved', f'JSON saved:\n{json_path}\n\nLOG saved:\n{log_path}')
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror('Export JSON/LOG failed', str(e))


    def export_metadata(self) -> None:
        if self.meta is None:
            messagebox.showwarning("No metadata", "Please load a TS file first.")
            return
        p = filedialog.asksaveasfilename(title="Save metadata JSON", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        Path(p).write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Metadata saved: {p}")
        messagebox.showinfo("Saved", f"Metadata saved:\n{p}")


    def _dataset_start_end(self, item: Dict[str, object]) -> Tuple[Optional[datetime], Optional[datetime], Optional[float]]:
        meta = item.get("meta", {})
        if not isinstance(meta, dict):
            return None, None, None
        start_s = meta.get("absolute_start_time")
        end_s = meta.get("absolute_end_time_estimated")
        fs = meta.get("sample_rate_Hz")
        try:
            start = datetime.fromisoformat(str(start_s)) if start_s else None
            end = datetime.fromisoformat(str(end_s)) if end_s else None
            fsf = float(fs) if fs is not None else None
            return start, end, fsf
        except Exception:
            return None, None, None

    def check_continuity_report(self) -> Tuple[bool, str, List[Dict[str, object]]]:
        if not self.datasets:
            return False, "No TS files loaded.", []
        items = []
        for item in self.datasets:
            st, en, fs = self._dataset_start_end(item)
            meta = item.get("meta", {})
            items.append({"item": item, "start": st, "end": en, "fs": fs, "meta": meta})
        if any(x["start"] is None or x["end"] is None or x["fs"] is None for x in items):
            return False, "Some files do not have decodable absolute Phoenix header times.", items
        items.sort(key=lambda x: x["start"])
        fs0 = float(items[0]["fs"])
        ok = True
        lines = ["Phoenix timing continuity report"]
        for x in items:
            label = x["item"].get("label", "")
            mtbl = meta.get("matched_tbl_file") if isinstance(meta, dict) else None
            tbl_note = f" ; TBL={Path(str(mtbl)).name}" if mtbl else " ; TBL=none"
            lines.append(f"- {label}: {x['start']}  ->  {x['end']} ; fs={x['fs']:.6g} Hz{tbl_note}")
            if abs(float(x["fs"]) - fs0) > max(1e-6, 1e-6 * fs0):
                ok = False
                lines.append("  ! sample rate differs from first file")
        tol = max(1.0 / fs0 * 2.0, 0.05)
        for a, b in zip(items[:-1], items[1:]):
            gap = (b["start"] - a["end"]).total_seconds()
            lines.append(f"gap/overlap between files = {gap:.6f} s")
            if abs(gap) > tol:
                ok = False
                lines.append("  ! not continuous within tolerance")
        return ok, "\n".join(lines), items

    def export_merged_absolute_csv(self) -> None:
        """Merge loaded TSn files only when header timing is continuous, then export absolute-time amplitudes."""
        if not self.datasets:
            messagebox.showwarning("No data", "Please load two or more TS files first.")
            return
        ok, report, items = self.check_continuity_report()
        self.log(report)
        if not ok:
            messagebox.showwarning("Not continuous", report)
            return
        p = filedialog.asksaveasfilename(
            title="Save merged absolute-time CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="phoenix_merged_absolute_time.csv",
        )
        if not p:
            return
        old_df, old_meta = self.df, self.meta
        frames = []
        for x in items:
            item = x["item"]
            self.df = item["df"]  # type: ignore[assignment]
            self.meta = item["meta"]  # type: ignore[assignment]
            out_df, _, unit_note = self._current_display_dataframe()
            # Ensure absolute datetime is present even after display conversion.
            if "datetime" not in out_df.columns and "datetime" in self.df.columns:
                out_df.insert(0, "datetime", self.df["datetime"].values)
            out_df.insert(0, "source_file", Path(str(self.meta.get("ts_file", ""))).name)
            frames.append(out_df)
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(p, index=False)
        self.df, self.meta = old_df, old_meta
        self.update_plot()
        self.log(f"Merged absolute-time CSV saved: {p}")
        self.log(f"Rows: {len(merged)}; display mode: {self.display_mode_var.get()}")
        messagebox.showinfo("Saved", f"Files are continuous and were merged.\nSaved:\n{p}\n\n{report}")

    def _physical_export_dataframe_for_mth5(self) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, str], str]:
        """Build a physical-unit dataframe for Aurora/MTH5 export.

        MTH5/Aurora should receive time series in physical units when we do not
        have a formally verified Phoenix response/filter chain.  Therefore this
        exporter intentionally uses the stable time-domain scaling only:

            Ex/Ey = input volts * 1e6 / dipole length in m  [mV/km]
            Hx/Hy/Hz = input volts / sensor sensitivity      [nT]

        If multiple TSn files are loaded and their Phoenix block timestamps are
        continuous with the same sampling rate, they are first merged into one
        run.  Otherwise the currently selected file is exported.
        """
        if self.df is None or self.meta is None:
            raise ValueError("No data loaded")

        # Prefer a continuous merged run when possible.
        if len(self.datasets) >= 2:
            ok, report, _items = self.check_continuity_report()
            self.log(report)
            if ok:
                raw_df, meta, scope = self._merged_raw_dataframe_for_calibration()
            else:
                raw_df, meta, scope = self.df.copy(), dict(self.meta), "current file only; loaded files are not continuous"
        else:
            raw_df, meta, scope = self.df.copy(), dict(self.meta), "current file"

        old_df, old_meta = self.df, self.meta
        try:
            self.df, self.meta = raw_df, meta
            phys_df, unit_map, note = self._physical_dataframe(rename_columns=False)
        finally:
            self.df, self.meta = old_df, old_meta
        return phys_df, meta, unit_map, f"{scope}; {note}"

    def _set_metadata_attr_safe(self, obj, dotted_name: str, value) -> None:
        """Safely assign a nested mt_metadata attribute if it exists."""
        try:
            target = obj.metadata
            parts = dotted_name.split(".")
            for p in parts[:-1]:
                target = getattr(target, p)
            setattr(target, parts[-1], value)
        except Exception:
            pass


    def _refresh_mth5_summary_tables(self, mth5_obj) -> None:
        """Force mth5 to create/update summary tables before closing.

        Aurora relies on the MTH5 channel summary table to discover available
        stations/runs/channels. Some mth5 versions do not materialize this
        table until it is accessed or summarized. This helper tries several
        public/common method names because the API changed among mth5 releases.
        """
        for attr in ("channel_summary", "tf_summary", "fc_summary"):
            try:
                table = getattr(mth5_obj, attr)
            except Exception as e:
                self.log(f"Could not access {attr}: {e}")
                continue
            for method_name in ("summarize", "update", "refresh", "write", "to_dataframe"):
                try:
                    method = getattr(table, method_name, None)
                    if callable(method):
                        method()
                        self.log(f"MTH5 {attr}.{method_name}() OK")
                        break
                except Exception as e:
                    self.log(f"MTH5 {attr}.{method_name}() skipped: {e}")

    def _ensure_channel_summary_with_h5py(self, p: str, survey_id: str, station_id: str, run_id: str, channels: List[str], phys_df: pd.DataFrame, fs: float, start: str, end: str) -> None:
        """Last-resort creation of channel summary datasets if mth5 did not make them.

        It creates both common names used by different mth5 versions:
        `channel_summary` and `channel summary` under `/Experiment/Reports`.
        Official mth5 will normally overwrite/update these when reopened.
        """
        try:
            import h5py
            import numpy as _np
        except Exception as e:
            self.log(f"h5py not available for summary-table check: {e}")
            return
        try:
            with h5py.File(p, "a") as h5:
                reports = h5.require_group("/Experiment/Reports")
                dtype = _np.dtype([
                    ("survey", "S30"), ("station", "S30"), ("run", "S20"),
                    ("latitude", "<f8"), ("longitude", "<f8"), ("elevation", "<f8"),
                    ("component", "S20"), ("start", "S36"), ("end", "S36"),
                    ("n_samples", "<i8"), ("sample_rate", "<f8"),
                    ("measurement_type", "S30"), ("azimuth", "<f8"), ("tilt", "<f8"),
                    ("units", "S60"), ("hdf5_reference", h5py.ref_dtype),
                    ("run_hdf5_reference", h5py.ref_dtype), ("station_hdf5_reference", h5py.ref_dtype),
                ])
                rows = _np.zeros(len(channels), dtype=dtype)
                station_path = f"/Experiment/Surveys/{survey_id}/Stations/{station_id}"
                run_path = f"{station_path}/Runs/{run_id}"
                station_ref = h5[station_path].ref if station_path in h5 else h5.ref
                run_ref = h5[run_path].ref if run_path in h5 else h5.ref
                for i, ch in enumerate(channels):
                    comp = ch.lower()
                    ch_type = "electric" if ch in ("Ex", "Ey") else "magnetic"
                    units = "digital counts"
                    ch_path = f"{run_path}/{comp}"
                    if ch_path not in h5:
                        for alt in (f"{run_path}/Electric/{comp}", f"{run_path}/Magnetic/{comp}", f"{run_path}/electric/{comp}", f"{run_path}/magnetic/{comp}"):
                            if alt in h5:
                                ch_path = alt
                                break
                    rows[i]["survey"] = survey_id.encode()
                    rows[i]["station"] = station_id.encode()
                    rows[i]["run"] = run_id.encode()
                    rows[i]["latitude"] = 0.0
                    rows[i]["longitude"] = 0.0
                    rows[i]["elevation"] = 0.0
                    rows[i]["component"] = comp.encode()
                    rows[i]["start"] = str(start).encode()[:36]
                    rows[i]["end"] = str(end).encode()[:36]
                    rows[i]["n_samples"] = int(len(phys_df[ch]))
                    rows[i]["sample_rate"] = float(fs)
                    rows[i]["measurement_type"] = ch_type.encode()
                    rows[i]["azimuth"] = 0.0 if ch in ("Ex", "Hx", "Hz") else 90.0
                    rows[i]["tilt"] = 0.0
                    rows[i]["units"] = units.encode()
                    rows[i]["hdf5_reference"] = h5[ch_path].ref if ch_path in h5 else h5.ref
                    rows[i]["run_hdf5_reference"] = run_ref
                    rows[i]["station_hdf5_reference"] = station_ref
                for name in ("channel_summary", "channel summary"):
                    if name in reports:
                        del reports[name]
                    reports.create_dataset(name, data=rows, maxshape=(None,))
                self.log("Verified/created /Experiment/Reports/channel_summary table")
        except Exception as e:
            self.log(f"Last-resort channel summary creation skipped: {e}")

    def _write_official_mth5_robust(self, p: str, phys_df: pd.DataFrame, meta: Dict[str, object], channels: List[str], station_id: str, run_id: str, survey_id: str, fs: float, start: str, end: str) -> None:
        """Write MTH5 using official mth5 package with two API styles.

        Some mth5 versions are sensitive to whether children are created via the
        root MTH5 object or via the already-created parent group.  The common
        symptom is an HDF5 error such as "unable to open object".  This helper
        first tries the parent-group style, then falls back to root-object style.
        """
        from mth5.mth5 import MTH5

        def _write_by_parent_groups() -> None:
            with MTH5(file_version="0.2.0") as mth5_obj:
                mth5_obj.open_mth5(p, "w")
                survey_group = mth5_obj.add_survey(survey_id)
                try:
                    station_group = survey_group.stations_group.add_station(station_id)
                except Exception:
                    station_group = mth5_obj.add_station(station_id, survey=survey_id)

                self._set_metadata_attr_safe(station_group, "time_period.start", start)
                self._set_metadata_attr_safe(station_group, "time_period.end", end)
                self._set_metadata_attr_safe(station_group, "orientation.reference_frame", "geomagnetic")
                try:
                    station_group.write_metadata()
                except Exception:
                    pass

                try:
                    run_group = station_group.add_run(run_id)
                except Exception:
                    run_group = mth5_obj.add_run(station_id, run_id, survey=survey_id)
                self._set_metadata_attr_safe(run_group, "sample_rate", fs)
                self._set_metadata_attr_safe(run_group, "time_period.start", start)
                self._set_metadata_attr_safe(run_group, "time_period.end", end)
                try:
                    run_group.write_metadata()
                except Exception:
                    pass

                for ch in channels:
                    comp = ch.lower()
                    ch_type = "electric" if ch in ("Ex", "Ey") else "magnetic"
                    data = phys_df[ch].to_numpy(dtype=np.float64)
                    data = np.where(np.isfinite(data), data, np.nan).astype(np.float64)
                    try:
                        dset = run_group.add_channel(comp, ch_type, data)
                    except Exception:
                        dset = mth5_obj.add_channel(station_id, run_id, comp, ch_type, data, survey=survey_id)
                    self._set_metadata_attr_safe(dset, "component", comp)
                    self._set_metadata_attr_safe(dset, "sample_rate", fs)
                    self._set_metadata_attr_safe(dset, "time_period.start", start)
                    self._set_metadata_attr_safe(dset, "time_period.end", end)
                    self._set_metadata_attr_safe(dset, "units", "digital counts")
                    if ch_type == "electric":
                        length = float(self.ex_length_m_var.get() if ch == "Ex" else self.ey_length_m_var.get())
                        self._set_metadata_attr_safe(dset, "dipole_length", length)
                    try:
                        dset.write_metadata()
                    except Exception:
                        pass

                self._refresh_mth5_summary_tables(mth5_obj)

        def _write_by_root_object() -> None:
            with MTH5(file_version="0.2.0") as mth5_obj:
                mth5_obj.open_mth5(p, "w")
                mth5_obj.add_survey(survey_id)
                station_group = mth5_obj.add_station(station_id, survey=survey_id)
                self._set_metadata_attr_safe(station_group, "time_period.start", start)
                self._set_metadata_attr_safe(station_group, "time_period.end", end)
                try:
                    station_group.write_metadata()
                except Exception:
                    pass
                run_group = mth5_obj.add_run(station_id, run_id, survey=survey_id)
                self._set_metadata_attr_safe(run_group, "sample_rate", fs)
                self._set_metadata_attr_safe(run_group, "time_period.start", start)
                self._set_metadata_attr_safe(run_group, "time_period.end", end)
                try:
                    run_group.write_metadata()
                except Exception:
                    pass
                for ch in channels:
                    comp = ch.lower()
                    ch_type = "electric" if ch in ("Ex", "Ey") else "magnetic"
                    data = phys_df[ch].to_numpy(dtype=np.float64)
                    data = np.where(np.isfinite(data), data, np.nan).astype(np.float64)
                    dset = mth5_obj.add_channel(station_id, run_id, comp, ch_type, data, survey=survey_id)
                    self._set_metadata_attr_safe(dset, "component", comp)
                    self._set_metadata_attr_safe(dset, "sample_rate", fs)
                    self._set_metadata_attr_safe(dset, "time_period.start", start)
                    self._set_metadata_attr_safe(dset, "time_period.end", end)
                    self._set_metadata_attr_safe(dset, "units", "digital counts")
                    try:
                        dset.write_metadata()
                    except Exception:
                        pass

                self._refresh_mth5_summary_tables(mth5_obj)

        first_error = None
        try:
            _write_by_parent_groups()
            return
        except Exception as e:
            first_error = traceback.format_exc()
            self.log("Parent-group MTH5 write failed; trying root-object API...")
            self.log(first_error)
        try:
            _write_by_root_object()
            return
        except Exception:
            self.log("Root-object MTH5 write also failed.")
            self.log(traceback.format_exc())
            raise RuntimeError(
                "Official mth5 write failed. First error was:\n"
                + str(first_error)
                + "\nCheck mth5_error.log in the working directory."
            )

    def _write_simple_hdf5_for_debug(self, p: str, phys_df: pd.DataFrame, meta: Dict[str, object], channels: List[str], station_id: str, run_id: str, survey_id: str, fs: float, start: str, end: str) -> str:
        """Fallback HDF5 writer for debugging data content when official mth5 fails.

        This is NOT a full official MTH5 file for Aurora, but it lets the user
        verify that the time series arrays and metadata are written correctly.
        """
        import h5py
        debug_path = str(Path(p).with_suffix(".debug_h5"))
        with h5py.File(debug_path, "w") as h5:
            h5.attrs["note"] = "Debug HDF5 only; not official MTH5/Aurora input"
            h5.attrs["survey_id"] = survey_id
            h5.attrs["station_id"] = station_id
            h5.attrs["run_id"] = run_id
            h5.attrs["sample_rate_Hz"] = fs
            h5.attrs["start"] = start
            h5.attrs["end"] = end
            h5.attrs["source"] = "Phoenix TSn/TBL GUI physical export"
            grp = h5.create_group(f"Experiment/Surveys/{survey_id}/Stations/{station_id}/Runs/{run_id}")
            for ch in channels:
                data = phys_df[ch].to_numpy(dtype=np.float64)
                ds = grp.create_dataset(ch.lower(), data=data, compression="gzip", shuffle=True)
                ds.attrs["component"] = ch.lower()
                ds.attrs["units"] = "digital counts"
                ds.attrs["sample_rate_Hz"] = fs
        return debug_path


    def _calibrated_dataframe_for_raw_run(self, raw_df: pd.DataFrame, meta: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, str], str]:
        """Calibrate one already-merged raw run for official MTH5 export."""
        old_df, old_meta = self.df, self.meta
        try:
            self.df = raw_df
            self.meta = meta
            cal_df, unit_map, note = self._response_removed_time_dataframe_ola(require_complete_calibration=True)
            required = {"Ex": "mV/km", "Ey": "mV/km", "Hx": "nT", "Hy": "nT", "Hz": "nT"}
            for ch, expected in required.items():
                if ch in raw_df.columns and unit_map.get(ch) != expected:
                    raise RuntimeError(f"{ch}: official MTH5 requires {expected}, got {unit_map.get(ch)!r}")
            return cal_df, unit_map, note
        finally:
            self.df, self.meta = old_df, old_meta


    def _group_phoenix_datasets_for_mth5(self) -> List[Dict[str, object]]:
        """Group loaded TS2/TS3/TS4/TS5 datasets into MTH5 runs.

        ELOG-like behavior:
        - different sample rates become different runs automatically;
        - files with the same sample rate and continuous absolute Phoenix times
          are merged into one run;
        - discontinuous files at the same sample rate become separate numbered runs.

        Typical legacy Phoenix mapping is:
            TS2 -> 24000 Hz
            TS3 -> 2400 Hz
            TS4 -> 150 Hz
            TS5 -> 15 Hz
        but the actual sample rate detected from TBL/header is used.
        """
        if not self.datasets:
            raise ValueError("No Phoenix datasets loaded")

        prepared = []
        for item in self.datasets:
            meta = dict(item.get("meta") or {})
            df = item.get("df")
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            fs = float(meta.get("sample_rate_Hz", 0) or 0)
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"Invalid sample rate in {meta.get('ts_file', 'dataset')}: {fs}")
            st_s, en_s = dataset_start_end_from_meta_or_df(item)
            st = pd.to_datetime(st_s, errors="coerce") if st_s else pd.NaT
            en = pd.to_datetime(en_s, errors="coerce") if en_s else pd.NaT
            prepared.append({
                "item": item, "fs": fs, "start": st, "end": en,
                "label": str(item.get("label") or Path(str(meta.get("ts_file", "TS"))).name)
            })

        if not prepared:
            raise ValueError("No valid loaded Phoenix datasets")

        # Group first by sample rate, using a rounded key only for dictionary stability.
        by_fs: Dict[float, List[Dict[str, object]]] = {}
        for x in prepared:
            key = round(float(x["fs"]), 9)
            by_fs.setdefault(key, []).append(x)

        runs: List[Dict[str, object]] = []
        for fs_key in sorted(by_fs):
            entries = by_fs[fs_key]
            entries.sort(key=lambda x: (
                pd.isna(x["start"]),
                x["start"] if not pd.isna(x["start"]) else pd.Timestamp.max
            ))
            tol = max(2.0 / float(fs_key), 0.05)
            segments: List[List[Dict[str, object]]] = []

            for x in entries:
                if not segments:
                    segments.append([x])
                    continue
                prev = segments[-1][-1]
                can_merge = False
                if not pd.isna(prev["end"]) and not pd.isna(x["start"]):
                    gap = (x["start"] - prev["end"]).total_seconds()
                    can_merge = abs(gap) <= tol
                if can_merge:
                    segments[-1].append(x)
                else:
                    segments.append([x])

            for seg_no, seg in enumerate(segments, start=1):
                raw_frames = []
                merged_meta = dict(seg[0]["item"].get("meta") or {})
                for x in seg:
                    raw_piece = x["item"].get("df")
                    if not isinstance(raw_piece, pd.DataFrame):
                        raise RuntimeError(f"Invalid raw dataframe for {x['label']}")
                    raw_frames.append(raw_piece.copy())

                merged_raw = pd.concat(raw_frames, ignore_index=True)
                if "datetime" in merged_raw.columns:
                    dt = pd.to_datetime(merged_raw["datetime"], errors="coerce")
                    order = np.argsort(dt.fillna(pd.Timestamp.max).to_numpy())
                    merged_raw = merged_raw.iloc[order].reset_index(drop=True)

                # Store native TSn ADC counts. Conversion to mV/km and nT is left to
                # processing, using RXCAL/SCAL + gain + dipole metadata (MTU-5C style).
                unit_map = {ch: "digital counts" for ch in DEFAULT_CHANNELS if ch in merged_raw.columns}
                note = (
                    "Raw Phoenix TSn ADC counts stored unchanged. "
                    "RXCAL/SCAL, EGN/HGN and ADC scale are mounted with applied=False "
                    "for later conversion to mV/km and nT."
                )
                merged, notes = merged_raw, [note]

                st = min((x["start"] for x in seg if not pd.isna(x["start"])), default=pd.NaT)
                en = max((x["end"] for x in seg if not pd.isna(x["end"])), default=pd.NaT)

                fs_label = f"{int(round(fs_key)) if abs(fs_key-round(fs_key)) < 1e-9 else fs_key:g}Hz"
                run_id = fs_label if len(segments) == 1 else f"{fs_label}_{seg_no:02d}"

                merged_meta["sample_rate_Hz"] = float(fs_key)
                if not pd.isna(st):
                    merged_meta["absolute_start_time"] = pd.Timestamp(st).isoformat()
                if not pd.isna(en):
                    merged_meta["absolute_end_time_estimated"] = pd.Timestamp(en).isoformat()

                runs.append({
                    "run_id": run_id,
                    "sample_rate": float(fs_key),
                    "df": merged,
                    "meta": merged_meta,
                    "start": None if pd.isna(st) else pd.Timestamp(st).isoformat(),
                    "end": None if pd.isna(en) else pd.Timestamp(en).isoformat(),
                    "source_labels": [str(x["label"]) for x in seg],
                    "note": "; ".join(sorted(set(notes))),
                })

        return runs

    def _phoenix_station_metadata_for_mth5(self) -> Dict[str, object]:
        """Collect station metadata with a strict, deterministic station-ID fallback.

        Phoenix legacy TBL string fields can occasionally decode to a one-character
        artifact (for example ``f``).  V20 rejects implausible station identifiers
        and falls back to the parent directory of the loaded TS files (e.g. PTS07),
        then to the TBL stem.
        """
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        md = build_tbl_metadata(tbl) if tbl is not None else {}
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}

        def _clean_station_candidate(value) -> Optional[str]:
            if value is None:
                return None
            text = re.sub(r"[^A-Za-z0-9_\-]", "_", str(value).strip()).strip("_")
            # Reject empty/one-character artifacts and pure punctuation/numeric noise.
            if len(text) < 3:
                return None
            if len(re.findall(r"[A-Za-z0-9]", text)) < 3:
                return None
            return text

        station = None
        for key in ("STNM", "SITE", "SNUM"):
            station = _clean_station_candidate(decoded.get(key))
            if station:
                break

        if station is None and self.datasets:
            try:
                first_meta = dict(self.datasets[0].get("meta") or {})
                ts_file = Path(str(first_meta.get("ts_file", "")))
                station = _clean_station_candidate(ts_file.parent.name)
            except Exception:
                station = None

        if station is None and tbl is not None:
            station = _clean_station_candidate(tbl.path.stem)
        if station is None:
            station = "PX001"

        lat = md.get("latitude_decimal_deg")
        lon = md.get("longitude_decimal_deg")
        elev = decoded.get("ELEV") or decoded.get("ALT") or 0.0

        try:
            lat = float(lat) if lat is not None else 0.0
        except Exception:
            lat = 0.0
        try:
            lon = float(lon) if lon is not None else 0.0
        except Exception:
            lon = 0.0
        try:
            elev = float(elev) if elev is not None else 0.0
        except Exception:
            elev = 0.0

        return {
            "station_id": station,
            "latitude": lat,
            "longitude": lon,
            "elevation": elev,
            "tbl_metadata": md,
        }

    def _phoenix_calibration_to_mth5_filter(self, channel: str, cal: PhoenixCalibration):
        """Convert a decoded Phoenix coil response table into an MTH5 FAP filter.

        Binary Phoenix CLC/CLE decoding in this program is heuristic; the filter
        is preserved as metadata/provenance and marked in comments accordingly.
        """
        if cal is None or cal.response_table is None or len(cal.response_table) < 5:
            return None
        try:
            from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
        except Exception:
            return None
        df = cal.response_table.copy()
        f = pd.to_numeric(df["frequency_Hz"], errors="coerce").to_numpy(float)
        a = pd.to_numeric(df["amplitude"], errors="coerce").to_numpy(float)
        ph = pd.to_numeric(df["phase_deg"], errors="coerce").to_numpy(float)
        good = np.isfinite(f) & np.isfinite(a) & np.isfinite(ph) & (f > 0) & (a > 0)
        if good.sum() < 5:
            return None
        f, a, ph = f[good], a[good], np.deg2rad(ph[good])
        order = np.argsort(f)
        f, a, ph = f[order], a[order], ph[order]
        name = re.sub(r"[^a-z0-9_]+", "_", f"phoenix_{channel}_{cal.path.stem}".lower()).strip("_")
        filt = FrequencyResponseTableFilter(name=name, units_in="nT", units_out="V")
        filt.frequencies = f
        filt.amplitudes = a
        filt.phases = ph
        try:
            filt.sequence_number = 1
        except Exception:
            pass
        try:
            filt.comments.value = (
                f"Source Phoenix calibration: {cal.path.name}. {cal.note} "
                "V18 stores this response for provenance; legacy binary CLC/CLE decoding must be validated against Phoenix/EMpower before quantitative response removal."
            )
        except Exception:
            pass
        return filt

    def _attach_mth5_filter_reference(self, channel_group, filt, applied: bool = False, stage: int = 1, comments: str = "") -> None:
        """Attach an AppliedFilter reference when supported by installed mt_metadata."""
        if filt is None:
            return
        md = getattr(channel_group, "metadata", None)
        if md is None:
            return
        note = comments or (
            "Phoenix MTU-5A EMpower RXCAL/SCAL/gain/ADC; stored waveform is native ADC counts (applied=False)."
        )
        try:
            md.add_filter(name=filt.name, applied=applied, stage=stage, comments=note)
            return
        except Exception:
            pass
        try:
            try:
                from mt_metadata.timeseries import AppliedFilter
            except Exception:
                from mt_metadata.timeseries.filters import AppliedFilter
            af = AppliedFilter(name=filt.name, applied=applied, stage=stage)
            if hasattr(md, "add_filter"):
                md.add_filter(af)
            elif hasattr(md, "filters"):
                md.filters.append(af)
        except Exception as e:
            self.log(f"Could not attach MTH5 filter reference {getattr(filt, 'name', '?')}: {e}")

    def _channel_numeric_summary(self, data: np.ndarray) -> Dict[str, float]:
        """Return stable numeric QC statistics for one written channel."""
        x = np.asarray(data, dtype=float).reshape(-1)
        good = np.isfinite(x)
        if not np.any(good):
            return {"n": int(x.size), "finite": 0, "min": np.nan, "max": np.nan,
                    "median": np.nan, "rms": np.nan, "abs_p95": np.nan}
        y = x[good]
        return {
            "n": int(x.size),
            "finite": int(y.size),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "median": float(np.median(y)),
            "rms": float(np.sqrt(np.mean(y * y))),
            "abs_p95": float(np.percentile(np.abs(y), 95.0)),
        }

    def _verify_written_mth5(self, p: str, runs: List[Dict[str, object]],
                             station_id: str, survey_id: str) -> List[str]:
        """Mandatory close/reopen verification using the official MTH5 API.

        V20.2 deliberately avoids guessing the internal HDF5 path.  MTH5 layouts
        differ among package versions, so the verifier asks MTH5 for the station,
        run and channel objects exactly as downstream software should.

        Low-level h5py is used only as a *last diagnostic fallback* to extract the
        ndarray from an already-resolved official channel object, never to discover
        Survey/Station/Run paths.
        """
        try:
            from mth5.mth5 import MTH5
        except Exception as exc:
            raise RuntimeError("Mandatory MTH5 read-back requires the official mth5 package") from exc

        expected = {str(r["run_id"]): r for r in runs}
        lines: List[str] = []

        def _open_existing(path: str):
            m = MTH5(file_version="0.2.0")
            # API signatures vary slightly among mth5 releases.
            try:
                m.open_mth5(path, mode="r")
            except TypeError:
                try:
                    m.open_mth5(path, "r")
                except TypeError:
                    m.open_mth5(path)
            return m

        def _get_station(m):
            errors = []
            attempts = [
                lambda: m.get_station(station_id, survey=survey_id),
                lambda: m.get_station(station_id, survey_id=survey_id),
                lambda: m.get_station(station_id),
            ]
            for fn in attempts:
                try:
                    obj = fn()
                    if obj is not None:
                        return obj
                except Exception as exc:
                    errors.append(str(exc))
            raise RuntimeError(
                f"Read-back failed: official MTH5 API could not resolve station "
                f"Survey={survey_id}, Station={station_id}. "
                + (" | ".join(errors[-3:]) if errors else "")
            )

        def _get_run(station_group, m, run_id: str):
            errors = []
            attempts = [
                lambda: station_group.get_run(run_id),
                lambda: m.get_run(station_id, run_id, survey=survey_id),
                lambda: m.get_run(station_id, run_id, survey_id=survey_id),
            ]
            for fn in attempts:
                try:
                    obj = fn()
                    if obj is not None:
                        return obj
                except Exception as exc:
                    errors.append(str(exc))
            raise RuntimeError(
                f"Read-back failed: official MTH5 API could not resolve run {run_id!r} "
                f"for Survey={survey_id}, Station={station_id}. "
                + (" | ".join(errors[-3:]) if errors else "")
            )

        def _get_channel(run_group, m, run_id: str, comp: str):
            errors = []
            attempts = [
                lambda: run_group.get_channel(comp),
                lambda: m.get_channel(station_id, run_id, comp, survey=survey_id),
                lambda: m.get_channel(station_id, run_id, comp, survey_id=survey_id),
            ]
            for fn in attempts:
                try:
                    obj = fn()
                    if obj is not None:
                        return obj
                except Exception as exc:
                    errors.append(str(exc))
            raise RuntimeError(
                f"Read-back failed: official MTH5 API could not resolve channel {run_id}/{comp}. "
                + (" | ".join(errors[-3:]) if errors else "")
            )

        def _channel_array(channel_group) -> np.ndarray:
            """Extract numeric samples from common official MTH5 channel objects."""
            errors = []
            # Most versions expose the underlying h5py Dataset as .hdf5_dataset.
            for attr in ("hdf5_dataset", "dataset"):
                try:
                    obj = getattr(channel_group, attr)
                    if callable(obj):
                        obj = obj()
                    arr = np.asarray(obj[...], dtype=np.float64).reshape(-1)
                    if arr.size:
                        return arr
                except Exception as exc:
                    errors.append(f"{attr}: {exc}")
            # Some versions provide read_channel()/to_channel_ts()-style helpers.
            for method in ("read_channel", "to_channel_ts"):
                try:
                    fn = getattr(channel_group, method)
                    obj = fn()
                    if hasattr(obj, "ts"):
                        obj = obj.ts
                    elif hasattr(obj, "data"):
                        obj = obj.data
                    arr = np.asarray(obj, dtype=np.float64).reshape(-1)
                    if arr.size:
                        return arr
                except Exception as exc:
                    errors.append(f"{method}: {exc}")
            # h5py-like slicing on the official channel wrapper.
            try:
                arr = np.asarray(channel_group[:], dtype=np.float64).reshape(-1)
                if arr.size:
                    return arr
            except Exception as exc:
                errors.append(f"slice: {exc}")
            raise RuntimeError("cannot extract channel samples; " + " | ".join(errors[-5:]))

        def _channel_units(channel_group, expected_units: str) -> str:
            # Prefer official metadata, not low-level attrs.
            try:
                md = channel_group.metadata
                val = getattr(md, "units", "")
                if hasattr(val, "name"):
                    val = val.name
                text = str(val or "")
                if text:
                    return text
            except Exception:
                pass
            try:
                val = getattr(channel_group, "units", "")
                text = str(val or "")
                if text:
                    return text
            except Exception:
                pass
            return expected_units

        m = _open_existing(p)
        try:
            station_group = _get_station(m)
            self.log(f"VERIFY API station resolved: Survey={survey_id}, Station={station_id}")

            for run_id, r in expected.items():
                run_group = _get_run(station_group, m, run_id)
                src_df = r["df"]
                fs_expected = float(r["sample_rate"])

                for ch in DEFAULT_CHANNELS:
                    if ch not in src_df.columns:
                        continue
                    comp = ch.lower()
                    channel_group = _get_channel(run_group, m, run_id, comp)
                    try:
                        arr = _channel_array(channel_group)
                    except Exception as exc:
                        raise RuntimeError(f"Read-back failed opening {run_id}/{comp}: {exc}") from exc

                    n_expected = int(len(src_df[ch]))
                    if arr.size != n_expected:
                        raise RuntimeError(
                            f"Read-back sample-count mismatch {run_id}/{comp}: "
                            f"written={n_expected}, read={arr.size}"
                        )

                    expected_values = pd.to_numeric(src_df[ch], errors="coerce").to_numpy(dtype=np.float64)
                    expected_values = np.rint(np.where(np.isfinite(expected_values), expected_values, 0.0)).astype(np.int32)
                    if not np.array_equal(np.asarray(arr).astype(np.int32), expected_values):
                        raise RuntimeError(f"Read-back waveform differs from native TSn counts for {run_id}/{comp}")
                    expected_units = "digital counts"
                    units = _channel_units(channel_group, expected_units)
                    try:
                        raw_units = str(channel_group.hdf5_dataset.attrs.get("units", ""))
                        if "count" in raw_units.lower():
                            units = raw_units
                    except Exception:
                        pass
                    ul = units.lower().replace(" ", "")
                    aliases_ok = (
                        "count" in ul or "digital" in ul or ul in ("", "none", "unknown")
                    )
                    if units and not aliases_ok:
                        raise RuntimeError(
                            f"Read-back unit mismatch {run_id}/{comp}: "
                            f"expected {expected_units}, got {units!r}"
                        )

                    q = self._channel_numeric_summary(arr)
                    line = (
                        f"VERIFY {run_id}/{comp}: n={q['n']:,}, units={units or expected_units}, "
                        f"median={q['median']:.6g}, RMS={q['rms']:.6g}, "
                        f"|x|p95={q['abs_p95']:.6g}, min={q['min']:.6g}, max={q['max']:.6g}, "
                        f"fs={fs_expected:g} Hz"
                    )
                    lines.append(line)
                    self.log(line)
        finally:
            try:
                m.close_mth5()
            except Exception:
                pass

        if not lines:
            raise RuntimeError("Read-back produced no verified channels")
        self.log(f"MTH5 OFFICIAL-API READ-BACK PASSED: {len(lines)} channels verified")
        return lines

    def _write_phoenix_multirun_mth5(self, p: str, runs: List[Dict[str, object]],
                                    station_id: str, survey_id: str) -> List[str]:
        """Write one official MTH5 v0.2.0 file and verify it after close.

        V22 policy:
        - Only the official ``mth5`` API creates Survey/Station/Run/Channel objects.
        - Stored samples are native Phoenix TSn ADC counts as int32 (not mV/km or nT).
        - recmeta.json / config.json carry electrode spacing and ga like MTU-5C.
        - EMpower-style RXCAL/SCAL plus ADC coefficient filters are mounted
          in Survey/Filters and referenced with applied=False so pyTRACMT can
          convert to mV/km and nT later.
        - Export is successful only after a mandatory close/reopen data verification.
        """
        from mth5.mth5 import MTH5

        smd = self._phoenix_station_metadata_for_mth5()
        tblmd = smd.get("tbl_metadata", {})
        decoded = tblmd.get("decoded_key_values", {}) if isinstance(tblmd, dict) else {}

        starts = [pd.to_datetime(r.get("start"), errors="coerce") for r in runs]
        ends = [pd.to_datetime(r.get("end"), errors="coerce") for r in runs]
        starts = [x for x in starts if not pd.isna(x)]
        ends = [x for x in ends if not pd.isna(x)]
        global_start = min(starts).isoformat() if starts else ""
        global_end = max(ends).isoformat() if ends else ""

        out_path = Path(p)
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception as exc:
                raise RuntimeError(f"Cannot replace existing output file {p}: {exc}") from exc

        m = MTH5(file_version="0.2.0")
        m.open_mth5(p, "w")
        try:
            survey_group = m.add_survey(survey_id)
            self._set_metadata_attr_safe(survey_group, "time_period.start", global_start)
            self._set_metadata_attr_safe(survey_group, "time_period.end", global_end)
            try:
                survey_group.write_metadata()
            except Exception:
                pass

            bundle = {"filters_by_channel": {}, "json_paths": []}
            try:
                bundle = self._prepare_empower_cal_bundle(p)
                self._mount_empower_filters(survey_group, bundle)
            except Exception as e:
                self.log(traceback.format_exc())
                self.log(f"EMpower RXCAL/SCAL mount failed: {e}")

            station_group = m.add_station(station_id, survey=survey_id)
            self._set_metadata_attr_safe(station_group, "location.latitude", smd["latitude"])
            self._set_metadata_attr_safe(station_group, "location.longitude", smd["longitude"])
            self._set_metadata_attr_safe(station_group, "location.elevation", smd["elevation"])
            self._set_metadata_attr_safe(station_group, "location.datum", "WGS84")
            self._set_metadata_attr_safe(station_group, "time_period.start", global_start)
            self._set_metadata_attr_safe(station_group, "time_period.end", global_end)
            try:
                station_group.write_metadata()
            except Exception:
                pass

            for r in runs:
                run_id = str(r["run_id"])
                fs = float(r["sample_rate"])
                phys_df = r["df"]
                start = str(r.get("start") or "")
                end = str(r.get("end") or "")

                run_group = m.add_run(station_id, run_id, survey=survey_id)
                self._set_metadata_attr_safe(run_group, "sample_rate", fs)
                self._set_metadata_attr_safe(run_group, "time_period.start", start)
                self._set_metadata_attr_safe(run_group, "time_period.end", end)
                self._set_metadata_attr_safe(run_group, "data_logger.manufacturer", "Phoenix Geophysics")
                self._set_metadata_attr_safe(run_group, "data_logger.model", "Legacy MTU")
                unit_id = decoded.get("UNIT") or decoded.get("BOX") or decoded.get("BOXN")
                if unit_id is not None:
                    self._set_metadata_attr_safe(run_group, "data_logger.id", str(unit_id))
                try:
                    run_group.write_metadata()
                except Exception:
                    pass

                for ch in DEFAULT_CHANNELS:
                    if ch not in phys_df.columns:
                        continue
                    comp = ch.lower()
                    ch_type = "electric" if ch in ("Ex", "Ey") else "magnetic"
                    # MTU-5C BIN path: keep native 24-bit ADC counts as int32.
                    # Do not convert to mV/km or nT here.
                    data_f = pd.to_numeric(phys_df[ch], errors="coerce").to_numpy(dtype=np.float64)
                    if not np.isfinite(data_f).any():
                        raise RuntimeError(f"Refusing to write all-nonfinite channel {run_id}/{comp}")
                    data = np.rint(np.where(np.isfinite(data_f), data_f, 0.0)).astype(np.int32)

                    # Official MTH5 API only.  Do not fall back to direct h5py writes.
                    try:
                        dset = run_group.add_channel(comp, ch_type, data, channel_dtype="int32")
                    except TypeError:
                        try:
                            dset = run_group.add_channel(comp, ch_type, data)
                        except Exception:
                            dset = m.add_channel(station_id, run_id, comp, ch_type, data, survey=survey_id)
                    except Exception:
                        dset = m.add_channel(station_id, run_id, comp, ch_type, data, survey=survey_id)

                    self._set_metadata_attr_safe(dset, "component", comp)
                    self._set_metadata_attr_safe(dset, "sample_rate", fs)
                    self._set_metadata_attr_safe(dset, "time_period.start", start)
                    self._set_metadata_attr_safe(dset, "time_period.end", end)
                    self._set_metadata_attr_safe(dset, "units", "digital counts")
                    gain = self._float_from_var(self._gain_vars()[ch], 1.0)
                    self._set_metadata_attr_safe(dset, "instrument.gain", gain)

                    if ch_type == "electric":
                        length = float(self.ex_length_m_var.get() if ch == "Ex" else self.ey_length_m_var.get())
                        self._set_metadata_attr_safe(dset, "dipole_length", length)
                        az = decoded.get("EAZM")
                        if az is not None:
                            try:
                                az0 = float(az)
                                self._set_metadata_attr_safe(
                                    dset, "measurement_azimuth",
                                    az0 if ch == "Ex" else (az0 + 90.0) % 360.0
                                )
                            except Exception:
                                pass
                    else:
                        h_az = decoded.get("HAZM")
                        default_az = {"Hx": 0.0, "Hy": 90.0, "Hz": 0.0}[ch]
                        default_tilt = {"Hx": 0.0, "Hy": 0.0, "Hz": 90.0}[ch]
                        try:
                            az0 = float(h_az) if h_az is not None else 0.0
                        except Exception:
                            az0 = 0.0
                        self._set_metadata_attr_safe(
                            dset, "measurement_azimuth",
                            (az0 + default_az) % 360.0 if ch != "Hz" else az0
                        )
                        self._set_metadata_attr_safe(dset, "measurement_tilt", default_tilt)
                        serial_key = {"Hx": "HXSN", "Hy": "HYSN", "Hz": "HZSN"}[ch]
                        serial = decoded.get(serial_key)
                        if serial is None:
                            serial = decoded.get({"Hx": "HSN1", "Hy": "HSN2", "Hz": "HSN3"}[ch])
                        if serial is not None:
                            self._set_metadata_attr_safe(dset, "sensor.id", str(serial))
                        self._set_metadata_attr_safe(dset, "sensor.manufacturer", "Phoenix Geophysics")
                        htyp = decoded.get({"Hx": "HTYP1", "Hy": "HTYP2", "Hz": "HTYP3"}[ch]) or decoded.get("HTYP")
                        if htyp is not None:
                            self._set_metadata_attr_safe(dset, "sensor.model", str(htyp))

                    for stage, filt in enumerate((bundle.get("filters_by_channel") or {}).get(ch, []), start=1):
                        self._attach_mth5_filter_reference(
                            dset, filt, applied=False, stage=stage,
                            comments="Phoenix MTU-5A EMpower RXCAL/SCAL/gain/ADC; stored samples are native ADC counts, not yet converted to mV/km or nT.",
                        )
                    try:
                        dset.write_metadata()
                    except Exception:
                        pass
                    # mt_metadata may canonicalize channel.units to a physical unit.
                    # Persist the ACTUAL waveform units and provenance directly on
                    # the HDF5 dataset after write_metadata(), without altering samples.
                    raw_ds = dset.hdf5_dataset
                    raw_ds.attrs["units"] = "digital counts"
                    # Avoid dotted custom HDF5 attribute names: mt_metadata may
                    # interpret them as nested metadata paths during get_channel().
                    raw_ds.attrs["phoenix_source_units"] = "digital counts"
                    raw_ds.attrs["phoenix_source_format"] = "native Phoenix TSn signed 24-bit ADC counts"
                    raw_ds.attrs["phoenix_calibration_applied"] = False
                    raw_ds.attrs["phoenix_real_filter_chain_attached"] = bool((bundle.get("filters_by_channel") or {}).get(ch))

            # Ask only official mth5 objects to refresh their summaries.
            self._refresh_mth5_summary_tables(m)
        finally:
            try:
                m.close_mth5()
            except Exception:
                pass

        # Mandatory official-API read-back after the file handle has been closed.
        return self._verify_written_mth5(p, runs, station_id, survey_id)

    def _tbl_lp_hz(self, tbl: Optional[PhoenixTable] = None) -> Optional[int]:
        tbl = tbl if tbl is not None else (self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None))
        if tbl is None:
            return None
        for key in ("LPFR", "LPF", "LFR"):
            v = tbl.entries.get(key)
            try:
                iv = int(v)
            except Exception:
                continue
            if iv > 0:
                return iv
        return None

    def _receiver_serial_token(self) -> str:
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        md = build_tbl_metadata(tbl) if tbl is not None else {}
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}
        for key in ("BOX", "BOXN", "UNIT", "SNUM"):
            tok = _safe_cal_token(decoded.get(key), "")
            if tok and tok.lower() not in {"unknown", "none", "null"}:
                return tok
        if self.clb is not None:
            return _safe_cal_token(self.clb.path.stem, "rx")
        return "rx"

    def _sensor_serial_token(self, channel: str) -> str:
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        md = build_tbl_metadata(tbl) if tbl is not None else {}
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}
        key_map = {
            "Hx": ["HXSN", "HSN1", "HSN"],
            "Hy": ["HYSN", "HSN2", "HSN"],
            "Hz": ["HZSN", "HSN3", "HSN"],
        }
        for key in key_map.get(channel, []):
            tok = _safe_cal_token(decoded.get(key), "")
            if tok and tok.lower() not in {"unknown", "none", "null"}:
                return tok
        cal = self.coil_calibration_map.get(channel)
        if cal is not None:
            return _safe_cal_token(cal.path.stem, channel.lower())
        return channel.lower()

    def _sensor_model_token(self, channel: str) -> str:
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        md = build_tbl_metadata(tbl) if tbl is not None else {}
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}
        key_map = {"Hx": "HTYP1", "Hy": "HTYP2", "Hz": "HTYP3"}
        tok = _safe_cal_token(decoded.get(key_map.get(channel, "")) or decoded.get("HTYP"), "")
        return tok or "coil"

    def _adc_vpp(self) -> float:
        return self._float_from_var(self.ad_range_pp_var, 5.0)

    def _channel_linear_gain(self, channel: str) -> float:
        return self._float_from_var(self._gain_vars()[channel], 1.0)

    def _dipole_length_m(self, channel: str) -> float:
        if channel == "Ex":
            return self._float_from_var(self.ex_length_m_var, 0.0)
        if channel == "Ey":
            return self._float_from_var(self.ey_length_m_var, 0.0)
        return 0.0

    def _build_empower_recmeta_json(self) -> dict:
        """Build an EMpower/MTU-5C-style recmeta.json from TBL + GUI geometry.

        pyTRACMT reads recmeta.json beside the MTH5 for length1+length2 and ga/pg.
        MTU-5A TBL EXLN/EYLN are the total dipole lengths, stored as length1 with
        length2=0 so the 5C rule total = length1+length2 still holds.
        """
        tbl = self.tbl if self.tbl is not None else (self.tbls[0] if self.tbls else None)
        md = build_tbl_metadata(tbl) if tbl is not None else {}
        decoded = md.get("decoded_key_values", {}) if isinstance(md, dict) else {}
        smd = self._phoenix_station_metadata_for_mth5()
        lp_hz = self._tbl_lp_hz() or 0
        serial = self._receiver_serial_token()
        vpp = self._adc_vpp()
        fs = None
        if self.datasets:
            try:
                fs = float((self.datasets[0].get("meta") or {}).get("sample_rate_Hz") or 0) or None
            except Exception:
                fs = None
        chans = []
        for ch, tag in CHANNEL_TO_EMPOWER_TAG.items():
            is_e = ch in ("Ex", "Ey")
            rec = {
                "tag": tag,
                "ty": "E" if is_e else "M",
                "on": 1,
                "ga": self._channel_linear_gain(ch),
                "pg": 1,
                "lp": int(lp_hz) if lp_hz else 0,
                "stored_units": "digital counts",
                "sampleRate": fs,
            }
            if is_e:
                length = float(self._dipole_length_m(ch) or 0.0)
                rec["length1"] = length
                rec["length2"] = 0.0
                rec["dipole_length_m"] = length
            else:
                rec["serial"] = self._sensor_serial_token(ch)
                rec["type_name"] = self._sensor_model_token(ch)
            chans.append(rec)
        layout = {
            "Station_Name": smd.get("station_id"),
            "Survey_Name": str(self.survey_id_var.get() or "phoenix_survey"),
        }
        return {
            "name": "recording",
            "version": 10,
            "file_type": "phoenix_mtu5a_recmeta",
            "instrument_type": "MTU-5A",
            "receiver_commercial_name": "MTU-5A",
            "receiver_model": "MTU-5A",
            "manufacturer": "Phoenix Geophysics",
            "instid": serial,
            "stored_units": "digital counts",
            "chconfig": {"name": "chconfig", "version": 5, "chans": chans},
            "channel_map": {
                "mapping": [
                    {"idx": i, "tag": CHANNEL_TO_EMPOWER_TAG[ch]}
                    for i, ch in enumerate(DEFAULT_CHANNELS)
                ],
                "receiver_model": "MTU-5A",
                "version": 1,
            },
            "layout": layout,
            "geometry": {
                "Ex": {
                    "tag": "E1",
                    "dipole_length_m": self._dipole_length_m("Ex"),
                    "length1": self._dipole_length_m("Ex"),
                    "length2": 0.0,
                    "source": "TBL EXLN",
                },
                "Ey": {
                    "tag": "E2",
                    "dipole_length_m": self._dipole_length_m("Ey"),
                    "length1": self._dipole_length_m("Ey"),
                    "length2": 0.0,
                    "source": "TBL EYLN",
                },
            },
            "electrode_spacing_m": {
                "Ex_EXLN": self._dipole_length_m("Ex"),
                "Ey_EYLN": self._dipole_length_m("Ey"),
            },
            "adc": {
                "bits": 24,
                "full_scale_range_vpp": vpp,
                "formula": "ADC_volts = counts / 2**24 * full_scale_range_vpp",
                "source": "TBL FSCV or GUI A/D range Vpp",
            },
            "physical_unit_conversion": {
                "from": "digital counts",
                "electric_to": "mV/km",
                "magnetic_to": "nT",
                "note": "Not applied here. pyTRACMT converts counts using recmeta dipole/gain plus rxcal.json/scal.json.",
            },
            "latitude": smd.get("latitude"),
            "longitude": smd.get("longitude"),
            "elevation": smd.get("elevation"),
            "legacy_tbl": str(tbl.path) if tbl is not None else None,
            "decoded_key_values": decoded,
        }

    def _build_empower_config_json(self, recmeta: dict) -> dict:
        """MTU-5C config.json equivalent: chanconfig length1/length2 plus layout."""
        chans = list(((recmeta.get("chconfig") or {}).get("chans") or []))
        return {
            "config": [{
                "chanconfig": chans,
                "layout": recmeta.get("layout") or {},
                "calibration": {"type": "none"},
            }]
        }

    def _build_empower_rxcal_json(self) -> Optional[dict]:
        """Build an EMpower/MTU-5C-style receiver calibration JSON from the matched CLB."""
        if self.clb is None or self.clb.response_table is None or len(self.clb.response_table) < 5:
            return None
        freq, mag, phase = _response_table_arrays(self.clb.response_table)
        if len(freq) < 5:
            return None
        lp_hz = self._tbl_lp_hz()
        state = _empower_state_name(lp_hz)
        serial = self._receiver_serial_token()
        channels = []
        for tag in EMPOWER_RXCAL_CHANNELS:
            channels.append({
                "tag": tag,
                "board_serial": serial,
                "num_of_responses": 1,
                "chan_data": [{
                    "tag": state,
                    "lp_hz": int(lp_hz) if lp_hz else None,
                    "num_records": len(freq),
                    "freq_Hz": freq,
                    "magnitude": mag,
                    "phs_deg": phase,
                }],
            })
        return {
            "file_type": "receiver calibration",
            "file_version": "MTU-5A-CLB-reconstructed",
            "decoder": "Phoenix MTU-5A legacy CLB heuristic decoder",
            "inst_serial": serial,
            "instrument_type": "MTU-5A",
            "instrument_model": "MTU-5A",
            "manufacturer": "Phoenix Geophysics",
            "timestamp_utc": 0,
            "num_channels": len(channels),
            "source_path": str(self.clb.path),
            "legacy_source": "CLB",
            "confidence": "reverse-engineered candidate decoder",
            "cal_data": channels,
        }

    def _build_empower_scal_json(self, channel: str) -> Optional[dict]:
        """Build an EMpower/MTU-5C-style sensor calibration JSON from one CLC/CLE."""
        cal = self.coil_calibration_map.get(channel)
        if cal is None or cal.response_table is None or len(cal.response_table) < 5:
            return None
        freq, mag_v_nt, phase = _response_table_arrays(cal.response_table)
        if len(freq) < 5:
            return None
        # Legacy CLC heuristic is V/nT; EMpower scal.json uses mV/nT.
        mag_mv_nt = [float(x) * 1000.0 for x in mag_v_nt]
        tag = CHANNEL_TO_EMPOWER_TAG.get(channel, channel)
        sensor = self._sensor_serial_token(channel)
        model = self._sensor_model_token(channel)
        return {
            "file_type": "sensor calibration",
            "file_version": "MTU-5A-CLC-reconstructed",
            "decoder": "Phoenix MTU-5A legacy CLC/CLE heuristic decoder",
            "sensor_serial": sensor,
            "instrument_type": "MTU-5A",
            "instrument_model": model,
            "manufacturer": "Phoenix Geophysics",
            "inst_serial": self._receiver_serial_token(),
            "timestamp_utc": 0,
            "num_channels": 1,
            "source_path": str(cal.path),
            "legacy_source": Path(cal.path).suffix.lower().lstrip("."),
            "mt_channel": channel,
            "empower_tag": tag,
            "confidence": "reverse-engineered candidate decoder",
            "cal_data": [{
                "tag": "H",
                "num_of_responses": 1,
                "chan_data": [{
                    "num_records": len(freq),
                    "freq_Hz": freq,
                    "magnitude": mag_mv_nt,
                    "magnitude_v_per_nt": mag_v_nt,
                    "phs_deg": phase,
                    "units": "mV/nT",
                    "source_units": "V/nT converted to mV/nT for EMpower JSON; MTH5 FAP uses V/nT",
                }],
            }],
        }

    def _make_coefficient_filter(self, name: str, gain: float, units_in: str, units_out: str, comments: str):
        try:
            from mt_metadata.timeseries.filters import CoefficientFilter
        except Exception as e:
            self.log(f"CoefficientFilter unavailable: {e}")
            return None
        try:
            filt = CoefficientFilter(
                name=str(name).replace("/", " per ").lower(),
                units_in=units_in,
                units_out=units_out,
                gain=float(gain),
            )
        except Exception as e:
            self.log(f"CoefficientFilter build failed ({name}): {e}")
            return None
        try:
            filt.comments.value = str(comments)
        except Exception:
            try:
                filt.comments = str(comments)
            except Exception:
                pass
        return filt

    def _make_adc_coefficient_filter(self):
        vpp = self._float_from_var(self.ad_range_pp_var, 5.0)
        # counts = volt * 2**24 / Vpp  =>  gain = counts/volt
        gain = float(2**24) / vpp
        return self._make_coefficient_filter(
            f"mtu5a_adc_{vpp:g}vpp",
            gain,
            "volt",
            "count",
            f"Phoenix MTU-5A 24-bit ADC: counts = volt * 2**24 / {vpp:g} Vpp; stored samples are counts (applied=False).",
        )

    def _make_gain_coefficient_filter(self, channel: str):
        amp = self._float_from_var(self._gain_vars()[channel], 1.0)
        kind = "egn" if channel in ("Ex", "Ey") else "hgn"
        return self._make_coefficient_filter(
            f"mtu5a_{channel.lower()}_{kind}",
            amp,
            "volt",
            "volt",
            f"Phoenix TBL {kind.upper()} for {channel} = {amp:g}; V_adc = V_input * gain; applied=False.",
        )

    def _prepare_empower_cal_bundle(self, mth5_path: str | Path) -> Dict[str, object]:
        """Write EMpower recmeta/config/rxcal/scal JSON like MTU-5C, keep FAP objects for mounting."""
        mth5_path = Path(mth5_path).expanduser().resolve()
        mth5_dir = mth5_path.parent
        out_dir = mth5_dir / "mth5_cal_files"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_paths: List[str] = []

        recmeta = self._build_empower_recmeta_json()
        config = self._build_empower_config_json(recmeta)
        recmeta_path = _write_json_file(mth5_dir / "recmeta.json", recmeta)
        config_path = _write_json_file(mth5_dir / "config.json", config)
        _write_json_file(out_dir / "recmeta.json", recmeta)
        _write_json_file(out_dir / "config.json", config)
        json_paths.extend([str(recmeta_path), str(config_path)])
        self.log(f"EMpower recmeta.json: {recmeta_path}")
        self.log(f"EMpower config.json: {config_path}")

        rx = self._build_empower_rxcal_json()
        scal_by_tag: Dict[str, dict] = {}
        scal_path_by_tag: Dict[str, Path] = {}
        rx_path = None
        if rx is not None:
            rx_path = out_dir / f"{_safe_cal_token(rx.get('inst_serial'), 'rx')}.rxcal.json"
            _write_json_file(rx_path, rx)
            json_paths.append(str(rx_path))
            self.log(f"EMpower RXCAL JSON: {rx_path}")
        for ch, tag in (("Hx", "H1"), ("Hy", "H2"), ("Hz", "H3")):
            sj = self._build_empower_scal_json(ch)
            if sj is None:
                continue
            scal_by_tag[tag] = sj
            p = out_dir / f"{_safe_cal_token(sj.get('sensor_serial'), tag)}_{tag}.scal.json"
            _write_json_file(p, sj)
            scal_path_by_tag[tag] = p
            json_paths.append(str(p))
            self.log(f"EMpower SCAL JSON {tag}/{ch}: {p}")

        filters_by_channel: Dict[str, List[object]] = {ch: [] for ch in DEFAULT_CHANNELS}
        survey_filters: Dict[str, object] = {}
        lp_hz = self._tbl_lp_hz() or 0
        if rx is not None:
            for ch, tag in CHANNEL_TO_EMPOWER_TAG.items():
                ch_block = next((item for item in rx.get("cal_data") or [] if str(item.get("tag")) == tag), None)
                row = ((ch_block or {}).get("chan_data") or [None])[0]
                if not row:
                    continue
                name = f"mtu-5a_mtu-5a_{rx.get('inst_serial')}_{tag.lower()}_{int(lp_hz) if lp_hz else 'legacy'}hz_lowpass"
                comments = (
                    f"Phoenix MTU-5A RXCAL from CLB {Path(str(rx.get('source_path'))).name}; "
                    f"tag={tag}; lp_hz={lp_hz}; volt->volt normalized receiver like MTU-5C RXCAL; "
                    "stored MTH5 samples are native ADC counts (applied=False)."
                )
                try:
                    filt = make_empower_fap_filter(
                        name, row.get("freq_Hz") or [], row.get("magnitude") or [], row.get("phs_deg") or [],
                        "volt", "volt", comments,
                    )
                except Exception as e:
                    self.log(f"RXCAL FAP build failed for {tag}: {e}")
                    continue
                survey_filters[filt.name] = filt
                filters_by_channel[ch].append(filt)
        for ch, tag in (("Hx", "H1"), ("Hy", "H2"), ("Hz", "H3")):
            sj = scal_by_tag.get(tag)
            if not sj:
                continue
            row = (((sj.get("cal_data") or [{}])[0]).get("chan_data") or [None])[0]
            if not row:
                continue
            name = f"mtu-5a_{_safe_cal_token(sj.get('instrument_model'), 'coil')}_{sj.get('inst_serial')}_{sj.get('sensor_serial')}"
            mag_v = row.get("magnitude_v_per_nt") or [float(x) / 1000.0 for x in (row.get("magnitude") or [])]
            comments = (
                f"Phoenix MTU-5A SCAL from CLC {Path(str(sj.get('source_path'))).name}; "
                f"{ch}/{tag}; EMpower scal.json keeps mV/nT, MTH5 FAP is V/nT so it "
                "chains with the 24-bit ADC volt->count stage. Stored samples are ADC counts."
            )
            try:
                filt = make_empower_fap_filter(
                    name, row.get("freq_Hz") or [], mag_v, row.get("phs_deg") or [],
                    "nanotesla", "volt", comments,
                )
            except Exception as e:
                self.log(f"SCAL FAP build failed for {tag}: {e}")
                continue
            survey_filters[filt.name] = filt
            # Magnetic chain like MTU-5C: sensor SCAL then receiver RXCAL.
            filters_by_channel[ch] = [filt] + list(filters_by_channel.get(ch) or [])

        adc_filt = self._make_adc_coefficient_filter()
        if adc_filt is not None:
            survey_filters[adc_filt.name] = adc_filt
        # Analog ga/pg live in recmeta.json like MTU-5C; do not also mount EGN/HGN
        # coefficient filters or pyTRACMT would remove the gain twice.
        for ch in DEFAULT_CHANNELS:
            chain = list(filters_by_channel.get(ch) or [])
            if adc_filt is not None:
                chain.append(adc_filt)
            filters_by_channel[ch] = chain

        bundle = {
            "rxcal_json": rx,
            "rxcal_path": rx_path,
            "recmeta_json": recmeta,
            "config_json": config,
            "scal_json_by_tag": scal_by_tag,
            "scal_path_by_tag": scal_path_by_tag,
            "survey_filters": survey_filters,
            "filters_by_channel": filters_by_channel,
            "json_paths": json_paths,
        }
        self._empower_cal_bundle = bundle
        self._empower_cal_json_paths = json_paths
        return bundle

    def _mount_empower_filters(self, survey_group, bundle: Dict[str, object]) -> None:
        filters = dict(bundle.get("survey_filters") or {})
        if not filters:
            self.log("No EMpower RXCAL/SCAL FAP filters available to mount in Survey/Filters.")
            return
        fg = None
        try:
            fg = survey_group.filters_group
        except Exception as e:
            self.log(f"Survey filters_group unavailable: {e}")
            return
        self.log(f"Mounting {len(filters)} EMpower FAP filters in Survey/Filters")
        for filt in filters.values():
            try:
                fg.add_filter(filt)
                self.log(f"  survey filter: {filt.name}")
            except Exception as e:
                self.log(f"  could not add filter {getattr(filt, 'name', '?')}: {e}")
        self._embed_empower_json_in_mth5(survey_group, bundle)

    def _embed_empower_json_in_mth5(self, survey_group, bundle: Dict[str, object]) -> None:
        """Also keep the original EMpower JSON text inside the MTH5 HDF5 file."""
        h5g = getattr(survey_group, "hdf5_group", None) or getattr(survey_group, "group", None)
        if h5g is None:
            return
        try:
            import h5py
        except Exception:
            h5py = None
        payloads = {}
        if bundle.get("recmeta_json"):
            payloads["recmeta.json"] = json.dumps(_jsonable(bundle["recmeta_json"]), indent=2, ensure_ascii=False, default=str)
        if bundle.get("config_json"):
            payloads["config.json"] = json.dumps(_jsonable(bundle["config_json"]), indent=2, ensure_ascii=False, default=str)
        if bundle.get("rxcal_json"):
            payloads["rxcal.json"] = json.dumps(bundle["rxcal_json"], indent=2, ensure_ascii=False, default=str)
        for tag, sj in dict(bundle.get("scal_json_by_tag") or {}).items():
            payloads[f"{tag}.scal.json"] = json.dumps(sj, indent=2, ensure_ascii=False, default=str)
        if not payloads:
            return
        try:
            cal_grp = h5g.require_group("PhoenixEMpowerJSON")
        except Exception as e:
            self.log(f"Could not create PhoenixEMpowerJSON group in MTH5: {e}")
            return
        for name, text in payloads.items():
            try:
                if name in cal_grp:
                    del cal_grp[name]
                if h5py is not None:
                    try:
                        dt = h5py.string_dtype(encoding="utf-8")
                    except Exception:
                        dt = h5py.special_dtype(vlen=str)
                    cal_grp.create_dataset(name, data=text, dtype=dt)
                else:
                    cal_grp.create_dataset(name, data=text)
                self.log(f"  stored {name} inside MTH5 Survey/PhoenixEMpowerJSON")
            except Exception as e:
                self.log(f"  could not embed {name} in MTH5: {e}")

    def _export_mth5_core(self, output_path: str, survey_id: str, *, autodetect: bool = False) -> List[str]:
        if not self.datasets:
            raise ValueError("Please load Phoenix TSn data first.")
        if autodetect:
            try:
                self.auto_detect_receiver_calibration(quiet=True)
            except Exception as e:
                self.log(f"Auto CLB detection warning: {e}")
            if self.auto_detect_calibration_var.get():
                try:
                    self.auto_detect_coil_calibrations(quiet=True)
                except Exception as e:
                    self.log(f"Auto CLC/CLE detection warning: {e}")
        runs = self._group_phoenix_datasets_for_mth5()
        smd = self._phoenix_station_metadata_for_mth5()
        station_id = re.sub(r"[^A-Za-z0-9_\-]", "_", str(smd["station_id"])) or "PX001"
        survey_id = re.sub(r"[^A-Za-z0-9_\-]", "_", str(survey_id or "phoenix_survey")) or "phoenix_survey"
        self.log(f"MTH5 Station ID auto-detected from TBL: {station_id}")
        self.log(f"MTH5 Survey ID: {survey_id}")
        verify_lines = self._write_phoenix_multirun_mth5(
            p=output_path,
            runs=runs,
            station_id=station_id,
            survey_id=survey_id,
        )
        self.log(f"Phoenix Multi-Run MTH5 saved and verified: {output_path}")
        self.log("V22 MTH5 policy (MTU-5C style): store native TSn ADC counts; write recmeta/config/rxcal/scal JSON; convert to mV/km and nT in pyTRACMT.")
        self.log(f"Mandatory read-back verification: {len(verify_lines)} channels passed.")
        for r in runs:
            sources = ", ".join(r["source_labels"])
            self.log(
                f"  {r['run_id']}: {r['sample_rate']:g} Hz, "
                f"{len(r['df']):,} samples/channel; sources={sources}"
            )
        return verify_lines

    def export_mth5_for_aurora(self) -> None:
        """Export all loaded legacy Phoenix TSn bands into one automatic Multi-Run MTH5.

        Like the ELOG exporter, sample rates are detected automatically and become
        separate Runs. Same-rate files are merged only when Phoenix absolute timing
        shows continuity; otherwise numbered runs are created.
        """
        if not self.datasets:
            messagebox.showwarning("No data", "Please load Phoenix TSn data first.")
            return

        try:
            from mth5.mth5 import MTH5  # noqa: F401
        except Exception as e:
            messagebox.showerror(
                "mth5 is not installed",
                "MTH5 export needs the official mth5 package.\n\n"
                "Recommended:\n"
                "python -m pip install mth5 mt-metadata\n\n"
                f"Import error: {e}",
            )
            return

        smd = self._phoenix_station_metadata_for_mth5()
        station_id = str(smd["station_id"])
        survey_id = simpledialog.askstring(
            "Survey ID",
            "Survey ID for MTH5:",
            initialvalue=self.survey_id_var.get() or "phoenix_survey",
        ) or "phoenix_survey"
        self.survey_id_var.set(survey_id)

        initial_dir = self.folder_path_var.get().strip() or None
        p = filedialog.asksaveasfilename(
            title="Save Phoenix Standard Physical-Unit Multi-Run MTH5",
            defaultextension=".mth5",
            filetypes=[("MTH5 standard file", "*.mth5"), ("HDF5", "*.h5")],
            initialfile=f"{station_id}_Phoenix.mth5",
            initialdir=initial_dir,
        )
        if not p:
            return

        try:
            self._export_mth5_core(p, survey_id=survey_id, autodetect=True)
            json_note = "\n".join(self._empower_cal_json_paths)
            messagebox.showinfo(
                "MTH5 saved",
                "Phoenix V22 Multi-Run MTH5 export completed and READ-BACK VERIFIED.\n"
                "Stored samples: native TSn ADC counts as int32 (not mV/km or nT).\n"
                "recmeta.json/config.json hold electrode spacing and gain.\n"
                "RXCAL/SCAL and ADC are in Survey/Filters with applied=False.\n"
                "pyTRACMT converts counts to mV/km and nT.\n\n"
                f"Survey: {survey_id}\n"
                f"Station: {station_id}\n"
                f"File:\n{p}\n\nEMpower JSON:\n{json_note or '(none)'}"
            )
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror(
                "MTH5 export failed",
                "Official Multi-Run MTH5 export failed.\n\n"
                f"Error: {e}"
            )


def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Phoenix MTU-5A folder (TBL/TSn/CLB/CLC/CLE) to calibrated MTH5.",
    )
    parser.add_argument("--folder", help="MTU-5A survey folder containing TBL/TSn/CLB/CLC/CLE")
    parser.add_argument("--output", help="Output .mth5 path (default: <station>_Phoenix.mth5 in the folder)")
    parser.add_argument("--survey", default="phoenix_survey", help="MTH5 Survey ID")
    parser.add_argument("--depth", type=int, default=2, help="Subfolder search depth (default 2)")
    parser.add_argument("--no-recursive", action="store_true", help="Search only the selected folder, not subfolders")
    args = parser.parse_args(argv)

    if not args.folder:
        app = PhoenixGUI()
        app.mainloop()
        return 0

    folder = Path(args.folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2

    app = PhoenixGUI()
    app.withdraw()
    try:
        app.folder_path_var.set(str(folder))
        app.survey_id_var.set(args.survey)
        app.calibration_search_depth_var.set(str(max(0, int(args.depth))))
        app.recursive_calibration_search_var.set(not args.no_recursive)
        written = app.convert_folder_to_mth5(
            folder,
            output_path=args.output,
            survey_id=args.survey,
            interactive=False,
        )
        print("Wrote:")
        for p in written:
            print(p)
        return 0
    except Exception as e:
        traceback.print_exc()
        print(f"Conversion failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            app.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(_cli_main())
