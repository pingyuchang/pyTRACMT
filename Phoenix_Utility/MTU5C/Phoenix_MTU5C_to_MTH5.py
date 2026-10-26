# V71 native RXCAL/SCAL auto-decoding (V63 reconstructed phase, exact serial matching)
# V70 STANDARD MTH5 metadata-safe update (2026-09-19)
# - force standard channel units to survive close/reopen through MTH5 API
# - preserve/reverify standard RXCAL/SCAL AppliedFilter references
# - export is accepted only when units + response metadata reopen correctly
#
# V69 GPS channel coverage / continuity update (2026-09-19)
# - GPS / Time tab now reports Ex/Ey/Hx/Hy/Hz start/end UTC and continuity by sample rate
# - detects inter-file time gaps/overlaps and retains BIN frame-counter continuity checks
#
# V68 TD-Volt / acquisition-LP / M3 calibration metadata update (2026-09-17)
# - Phoenix .td_* float32 numerical samples are stored explicitly as volt
# - Phoenix TD is treated as already de-gained (GA removed before TD write); GA is metadata only
# - acquisition analog LP state/cutoff is recorded as provenance (e.g. LP=10000 Hz -> T0)
# - RXCAL/SCAL are attached as FORWARD instrument responses with applied=False for M3/Aurora deconvolution
# - magnetic SCAL remains nT -> V (source mV/nT converted to V/nT exactly once)
# - electric TD remains electrode potential difference in V; dipole length remains geometry metadata
#
# V65 calibration-chain update (2026-09-16)
# - official MakeMTH5.from_phoenix receives serial-keyed RXCAL/SCAL JSON dictionaries
# - decoded RXCAL/SCAL JSON is archived verbatim inside /Phoenix_Calibration_JSON
# - final calibration-chain build/verification runs for both official and direct-writer outputs
# - stored samples remain RAW; embedded JSON/filter metadata does not imply calibration_applied=True
#
# V64 safety update (2026-09-14)
# - magnetic SCAL: exact sensor-serial match only; no leftover/positional fallback
# - station ID: layout.Station_Name -> data-folder name -> receiver serial fallback
# - receiver serial remains data_logger.id, not station name
# - standard .td_150/.td_24k: official MakeMTH5.from_phoenix remains first choice
# - RXCAL binary anchor tables remain native 13/7/5/5 points; no synthetic densification
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phoenix MTU-5C -> MTH5 GUI v20 Professional
============================================
Purpose
-------
A Phoenix MTU-5C native binary + JSON converter/inspector with:
  * direct Ex/Ey/Hx/Hy/Hz synchronized waveform display
  * whole-recording frame continuity and missing-frame checks
  * GPS / UTC recording-time display
  * official MakeMTH5.from_phoenix export when compatible
  * direct MTH5 API fallback writer when the Phoenix collection client fails
  * automatic re-open/verification of Survey/Station/Run/Channel structure

V23 keeps numerical samples RAW/UNCALIBRATED and adds a conservative calibration
builder for Phoenix rxcal/scal JSON.  It writes real response objects only when
the JSON contains usable numerical response data and attaches a channel filter
reference only when sensor/receiver identity matching is sufficiently strong.

V36 adds a binary RXCAL formula + acquisition-state resolver.  For each validated 32-byte response
record it compares three interpretations of (f1,f2): Cartesian real/imaginary,
amplitude + phase in radians, and amplitude + phase in degrees.  The radian
interpretation is the primary hypothesis requested for testing, and V36 additionally tests GA/PG/LP/sample-rate compatibility. Binary RXCAL remains
diagnostic-only unless the confidence gate is satisfied and independently validated against official
EMpower JSON or another trusted reference.
"""
from __future__ import annotations

import json
import os
import re
import struct
import traceback
import shutil
import tempfile
import sys
import platform
import builtins
import io
import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

APP_TITLE = "Phoenix MTU-5C -> MTH5 Professional V72 LP-aware RXCAL/SCAL"
HEADER_BYTES = 128
FRAME_BYTES = 64
SAMPLES_PER_FRAME = 20
TAG_TO_MT_COMPONENT = {"H1": "hx", "H2": "hy", "H3": "hz", "E1": "ex", "E2": "ey"}
COMPONENT_ORDER = ["ex", "ey", "hx", "hy", "hz"]
COMPONENT_TO_TAG = {v: k for k, v in TAG_TO_MT_COMPONENT.items()}


@contextmanager
def _mt_metadata_utf8_open_patch():
    """Temporarily force UTF-8 when mt_metadata opens its packaged JSON files.

    Some mt_metadata releases call open(..., "r") without an explicit
    encoding.  On Traditional-Chinese Windows that falls back to CP950, while
    the packaged JSON (notably the license enumeration data) is UTF-8.
    Limit the workaround strictly to text-mode JSON reads under an
    ``mt_metadata`` path, then restore the original open functions.
    """
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

    builtins.open = patched_builtin_open
    io.open = patched_io_open
    try:
        yield
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open


def _import_mth5_utf8_safe():
    """Import MTH5/mt_metadata while shielding UTF-8 package JSON from CP950."""
    with _mt_metadata_utf8_open_patch():
        mth5_mod = importlib.import_module("mth5.mth5")
        ts_mod = importlib.import_module("mt_metadata.timeseries")
    return (
        mth5_mod.MTH5,
        getattr(ts_mod, "Electric", None),
        getattr(ts_mod, "Magnetic", None),
        getattr(ts_mod, "Run", None),
        getattr(ts_mod, "Station", None),
        getattr(ts_mod, "Survey", None),
    )


@dataclass
class MTU5CHeader:
    path: Path
    file_type: int
    file_version: int
    header_length: int
    instrument_type: str
    instrument_serial: str
    recording_id: int
    channel_id: int
    file_sequence: int
    fragment_period_s: int
    channel_board_model: str
    channel_board_serial: str
    channel_firmware: int
    config_fingerprint: Tuple[int, ...]
    sample_rate: float
    bytes_per_sample: int
    frame_size: int
    gps_longitude: float
    gps_latitude: float
    gps_elevation: float
    battery_voltage_v: float
    min_signal: float
    max_signal: float

    @property
    def recording_start_utc(self) -> str:
        try:
            return datetime.fromtimestamp(self.recording_id, tz=timezone.utc).isoformat()
        except Exception:
            return ""

    @property
    def n_frames(self) -> int:
        n = max(0, self.path.stat().st_size - self.header_length)
        return n // self.frame_size if self.frame_size > 0 else 0

    @property
    def n_samples(self) -> int:
        return self.n_frames * SAMPLES_PER_FRAME

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate > 0 else 0.0


def _ascii(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip("\x00 ")


def read_mtu5c_header(path: str | Path) -> MTU5CHeader:
    path = Path(path)
    with path.open("rb") as f:
        h = f.read(HEADER_BYTES)
    if len(h) < HEADER_BYTES:
        raise ValueError(f"File shorter than {HEADER_BYTES} bytes: {path}")
    u = lambda fmt, off: struct.unpack_from(fmt, h, off)
    instrument_type = _ascii(u("<8s", 4)[0])
    if "MTU-5C" not in instrument_type.upper():
        raise ValueError(f"Not MTU-5C native data: {instrument_type!r}")
    sr_base = u("<H", 59)[0]
    sr_exp = u("<b", 61)[0]
    sample_rate = float(sr_base * (10 ** sr_exp))
    frame_raw = u("<I", 63)[0]
    frame_size = int(frame_raw & 0x0FFFFFFF) or FRAME_BYTES
    return MTU5CHeader(
        path=path,
        file_type=u("<B", 0)[0], file_version=u("<B", 1)[0], header_length=u("<H", 2)[0],
        instrument_type=instrument_type, instrument_serial=_ascii(u("<8s", 12)[0]),
        recording_id=u("<I", 20)[0], channel_id=u("<B", 24)[0], file_sequence=u("<I", 25)[0],
        fragment_period_s=u("<H", 29)[0], channel_board_model=_ascii(u("<8s", 31)[0]),
        channel_board_serial=_ascii(u("<8s", 39)[0]), channel_firmware=u("<I", 47)[0],
        config_fingerprint=tuple(u("<8B", 51)), sample_rate=sample_rate,
        bytes_per_sample=u("<B", 62)[0], frame_size=frame_size,
        gps_longitude=float(u("<f", 71)[0]), gps_latitude=float(u("<f", 75)[0]),
        gps_elevation=float(u("<f", 79)[0]), battery_voltage_v=float(u("<H", 105)[0]) / 1000.0,
        min_signal=float(u("<f", 107)[0]), max_signal=float(u("<f", 111)[0]),
    )


def decode_24bit_be_samples(payload60: bytes) -> np.ndarray:
    if len(payload60) < 60:
        return np.empty(0, dtype=np.int32)
    b = np.frombuffer(payload60[:60], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    v = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    v = np.where(v & 0x800000, v - 0x1000000, v)
    return v.astype(np.int32)



def _is_td_file(path: str | Path) -> bool:
    """Phoenix time-domain float32 files such as .td_150 and .td_24k."""
    name = Path(path).name.lower()
    return ".td_" in name


def read_td_float32(path: str | Path, max_samples: Optional[int] = None) -> np.ndarray:
    """
    Read Phoenix .td_* time-domain waveform.

    Observed MTU-5C layouts:
      * .td_150 : 128-byte MTU-5C header + contiguous little-endian float32 samples.
      * .td_24k : 128-byte MTU-5C header + repeated 1-second records:
                  8 float32 metadata words + Fs float32 waveform samples.

    The block-metadata layout is detected generically from Fs rather than by filename.
    """
    h = read_mtu5c_header(path)
    with Path(path).open("rb") as f:
        f.seek(h.header_length)
        payload = f.read()

    n4 = len(payload) // 4
    if n4 <= 0:
        return np.empty(0, dtype=np.float32)
    a = np.frombuffer(payload[:n4 * 4], dtype="<f4")

    fs_i = int(round(h.sample_rate))
    block_words = fs_i + 8 if fs_i > 0 else 0

    # High-rate Phoenix TD files use 8 float32 metadata words per 1-s block.
    # Example: 6 * (24000 + 8) words.
    if block_words > 8 and a.size >= block_words and a.size % block_words == 0:
        a = a.reshape(-1, block_words)[:, 8:].reshape(-1)

    # Remove non-finite sentinel values defensively, without changing sample count
    # unless they occur outside the waveform after format detection.
    a = np.asarray(a, dtype=np.float32)

    if max_samples is not None:
        a = a[:max_samples]
    return a


def native_sample_count(path: str | Path) -> int:
    """Return actual waveform sample count for BIN or TD files."""
    h = read_mtu5c_header(path)
    if _is_td_file(path):
        payload_bytes = max(0, Path(path).stat().st_size - h.header_length)
        nwords = payload_bytes // 4
        fs_i = int(round(h.sample_rate))
        block_words = fs_i + 8 if fs_i > 0 else 0
        if block_words > 8 and nwords >= block_words and nwords % block_words == 0:
            return (nwords // block_words) * fs_i
        return nwords
    return h.n_frames * SAMPLES_PER_FRAME

def read_frame_counters(path: str | Path) -> np.ndarray:
    h = read_mtu5c_header(path)
    out = np.empty(h.n_frames, dtype=np.int64)
    with Path(path).open("rb") as f:
        f.seek(h.header_length)
        n = 0
        for i in range(h.n_frames):
            frame = f.read(h.frame_size)
            if len(frame) < h.frame_size:
                break
            if len(frame) >= 64:
                footer = struct.unpack_from("<I", frame, 60)[0]
                out[n] = int(footer & 0x0FFFFFFF)
                n += 1
    return out[:n]


def read_counts(path: str | Path, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    h = read_mtu5c_header(path)

    # Phoenix .td_* is already time-domain float32 waveform, not packed 24-bit BIN.
    if _is_td_file(path):
        y = read_td_float32(path, max_samples=max_samples)
        return y, np.empty(0, dtype=np.int64)

    max_frames = h.n_frames
    if max_samples is not None:
        max_frames = min(max_frames, int(np.ceil(max_samples / SAMPLES_PER_FRAME)))
    samples: List[np.ndarray] = []
    counters = np.empty(max_frames, dtype=np.int64)
    nc = 0
    with Path(path).open("rb") as f:
        f.seek(h.header_length)
        for _ in range(max_frames):
            frame = f.read(h.frame_size)
            if len(frame) < h.frame_size:
                break
            y = decode_24bit_be_samples(frame[:60])
            if y.size:
                samples.append(y)
            if len(frame) >= 64:
                footer = struct.unpack_from("<I", frame, 60)[0]
                counters[nc] = int(footer & 0x0FFFFFFF)
                nc += 1
    y = np.concatenate(samples) if samples else np.empty(0, dtype=np.int32)
    if max_samples is not None:
        y = y[:max_samples]
    return y, counters[:nc]


def load_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def find_json_role(path: str | Path) -> str:
    try:
        d = load_json(path)
    except Exception:
        return "unknown"
    if "receiver_commercial_name" in d or ("channel_map" in d and "timing" in d):
        return "recmeta"
    if "receiver" in d and "config" in d:
        return "config"
    return "unknown"


def extract_channel_map(recmeta: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    try:
        for item in recmeta.get("channel_map", {}).get("mapping", []):
            out[int(item["idx"])] = str(item["tag"]).upper()
    except Exception:
        pass
    return out


def extract_recmeta_channel_info(recmeta: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    try:
        for ch in recmeta.get("chconfig", {}).get("chans", []):
            tag = str(ch.get("tag", "")).upper()
            if tag:
                out[tag] = dict(ch)
    except Exception:
        pass
    return out


def extract_layout(config: dict, recmeta: dict) -> dict:
    layout = {}
    try:
        layout.update(config.get("config", [{}])[0].get("layout", {}))
    except Exception:
        pass
    try:
        layout.update({k: v for k, v in recmeta.get("layout", {}).items() if v not in (None, "")})
    except Exception:
        pass
    return layout


def describe_dipoles(config: dict) -> Dict[str, dict]:
    out = {}
    try:
        for c in config.get("config", [{}])[0].get("chanconfig", []):
            tag = str(c.get("tag", "")).upper()
            if tag.startswith("E"):
                out[tag] = {"dipole": c.get("dipole", {}), "length1": c.get("length1"), "length2": c.get("length2")}
    except Exception:
        pass
    return out


def safe_id(value: object, fallback: str) -> str:
    s = str(value or "").strip()
    if not s:
        return fallback
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or fallback


def unix_iso(value) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def dipole_azimuth_and_length(config: dict, tag: str) -> Tuple[Optional[float], Optional[float]]:
    """Return azimuth clockwise from geographic/receiver North and total length."""
    try:
        for c in config.get("config", [{}])[0].get("chanconfig", []):
            if str(c.get("tag", "")).upper() != tag.upper():
                continue
            d = c.get("dipole", {}) or {}
            ne, nn = float(d.get("NegEast", 0)), float(d.get("NegNorth", 0))
            pe, pn = float(d.get("PosEast", 0)), float(d.get("PosNorth", 0))
            de, dn = pe-ne, pn-nn
            length = float(np.hypot(de, dn))
            az = float((np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0)
            return az, length
    except Exception:
        pass
    return None, None


"""Phoenix MTU-5C V63 reconstructed RXCAL/SCAL decoder.
RXCAL phase is reverse engineered, not manufacturer certified.
"""
from pathlib import Path
import json, struct, math, re

STATES=[('T0_10kHz',10000,84,13),('T1_1kHz',1000,512,7),('T2_100Hz',100,812,5),('T3_10Hz',10,1144,5)]
PHI={10000:.130751,1000:-.173619,100:.301190,10:.059604}
TAU=.001466

def records(data,start,n,rx=False,lp=10000):
    out=[]
    for j in range(n):
        pos=start+32*j
        if pos+32>len(data): raise ValueError(f'truncated record at {pos}')
        w=struct.unpack_from('<8f',data,pos)
        f,real,imag=w[:3]
        if not (math.isfinite(f) and f>0 and math.isfinite(real) and math.isfinite(imag)):
            raise ValueError(f'invalid record at {pos}: {w[:3]}')
        mag=math.hypot(real,imag)
        phase=math.degrees(math.atan2(imag,real))
        if rx: phase+=PHI[lp]-360*f*TAU
        phase=(phase+180)%360-180
        out.append((f,mag,phase))
    out.sort()
    if any(out[k][0]>=out[k+1][0] for k in range(len(out)-1)):
        raise ValueError('frequency anchors not strictly increasing')
    return {'freq_Hz':[x[0] for x in out], 'magnitude':[x[1] for x in out], 'phs_deg':[x[2] for x in out]}

def decode_rxcal(path):
    path=Path(path); b=path.read_bytes()
    if len(b)<256+5*1464: raise ValueError('RXCAL too short')
    serial=path.name.split('_')[0]
    if not serial.isdigit(): raise ValueError('RXCAL filename must begin with receiver serial')
    blocks=[]
    for i,tag in enumerate(('E1','E2','H1','H2','H3')):
        base=256+1464*i
        rows=[]
        for state,lp,off,n in STATES:
            row=records(b,base+off,n,True,lp)
            row.update(tag=state,lp_hz=lp,num_records=n)
            rows.append(row)
        blocks.append({'tag':tag,'num_of_responses':4,'chan_data':rows})
    return {'file_type':'receiver calibration','file_version':'V63-reconstructed','decoder':'Phoenix V63 RXCAL', 'inst_serial':serial,'instrument_type':'MTU-5C','manufacturer':'Phoenix Geophysics','phase_model':{'formula':'phase = atan2(w2,w1) + phi0_state - 360*f*tau','tau_seconds':TAU,'phi0_deg_by_state':{s:PHI[lp] for s,lp,_,_ in STATES},'status':'reverse-engineered; not manufacturer certified'},'num_channels':5,'source_path':str(path),'cal_data':blocks}

def decode_scal(path,receiver_serial=None):
    path=Path(path); b=path.read_bytes()
    serial=path.name.split('_')[0]
    if not serial.isdigit(): raise ValueError('SCAL filename must begin with sensor serial')
    if len(b)<340+32*68: raise ValueError('SCAL too short for 68 records')
    row=records(b,340,68)
    row.update(num_records=68,units='mV/nT')
    obj={'file_type':'sensor calibration','file_version':'V63-reconstructed','decoder':'Phoenix V63 SCAL','sensor_serial':serial,'instrument_type':'MTU-5C','instrument_model':'MTC-150','manufacturer':'Phoenix Geophysics','num_channels':1,'source_path':str(path),'cal_data':[{'tag':'H','num_of_responses':1,'chan_data':[row]}]}
    if receiver_serial: obj['inst_serial']=str(receiver_serial)
    return obj

def convert_folder(folder, receiver_serial, sensor_serials, output_dir=None):
    folder=Path(folder); dest=Path(output_dir) if output_dir else folder
    dest.mkdir(parents=True,exist_ok=True)
    rxfiles=sorted(p for p in folder.rglob('*.rxcal') if p.name.split('_')[0]==str(receiver_serial))
    if len(rxfiles)!=1: raise ValueError(f'Expected exactly one RXCAL for {receiver_serial}; found {len(rxfiles)}')
    rx=decode_rxcal(rxfiles[0]); output=[]
    target=dest/(f'{receiver_serial}_V63_reconstructed.rxcal.json')
    target.write_text(json.dumps(rx,indent=2,ensure_ascii=False),encoding='utf-8'); output.append(target)
    for tag,serial in sensor_serials.items():
        matches=sorted(p for p in folder.rglob('*.scal') if p.name.split('_')[0]==str(serial))
        if len(matches)!=1: raise ValueError(f'Expected exactly one SCAL for {tag}={serial}; found {len(matches)}')
        obj=decode_scal(matches[0],receiver_serial)
        target=dest/(f'{serial}_{tag}_V63_reconstructed.scal.json')
        target.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); output.append(target)
    return output



class PhoenixMTU5CApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1500x900")
        self.minsize(1180, 720)

        self.station_dir = tk.StringVar()
        self.recmeta_path = tk.StringVar()
        self.config_path = tk.StringVar()
        self.calibration_dir = tk.StringVar()
        self.output_path = tk.StringVar()
        self._last_auto_output_path = ""
        self.sample_rates = tk.StringVar(value="150,24000")
        self.preview_seconds = tk.DoubleVar(value=2.0)
        self.waveform_mode = tk.StringVar(value="Full recording")
        self.max_display_points = 50000
        # V22: never relabel uncalibrated samples as physical units.
        # When enabled and a calibration folder is supplied, an export is only
        # considered calibrated if real non-empty channel filter references are
        # found after writing.
        self.require_real_calibration = tk.BooleanVar(value=True)
        self.v23_last_calibration_report = ""

        self.bin_headers: List[MTU5CHeader] = []
        self.recmeta: dict = {}
        self.config: dict = {}
        self.channel_map: Dict[int, str] = {}
        self.channel_info: Dict[str, dict] = {}
        self.board_serial_to_tag: Dict[str, str] = {}
        self.continuity_report: Dict[str, dict] = {}

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        top.columnconfigure(1, weight=1)
        self._path_row(top, 0, "MTU-5C station/data folder", self.station_dir, self.choose_station_dir)
        self._path_row(top, 1, "recmeta.json", self.recmeta_path, self.choose_recmeta)
        self._path_row(top, 2, "config.json", self.config_path, self.choose_config)
        self._path_row(top, 3, "Calibration folder (EMpower *.rxcal.json / *.scal.json preferred; binary diagnostic only)", self.calibration_dir, self.choose_calibration_dir)
        self._path_row(top, 4, "Output MTH5", self.output_path, self.choose_output)

        opt = ttk.Frame(top)
        opt.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Label(opt, text="Sample rates").pack(side=tk.LEFT)
        ttk.Entry(opt, textvariable=self.sample_rates, width=15).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(opt, text="Waveform display").pack(side=tk.LEFT)
        ttk.Combobox(opt, textvariable=self.waveform_mode, values=("Full recording","Preview seconds"), state="readonly", width=16).pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(opt, text="Preview seconds").pack(side=tk.LEFT)
        ttk.Spinbox(opt, textvariable=self.preview_seconds, from_=0.05, to=3600, increment=0.25, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            opt,
            text="Require real channel calibration chain",
            variable=self.require_real_calibration
        ).pack(side=tk.LEFT, padx=(14,4))

        btn = ttk.Frame(top)
        btn.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Button(btn, text="Scan / Read MTU-5C", command=self.scan_station).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Show multi-rate waveforms", command=self.preview_all_channels).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Check full recording continuity", command=self.check_full_continuity).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Export MTH5 + verify", command=self.export_mth5).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Verify existing MTH5", command=self.verify_selected_mth5).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Check Python packages", command=self.check_packages).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="Run V47.1 RXCAL JSON-supervised decoder", command=self.run_v47_1_rxcal_resolver).pack(side=tk.LEFT, padx=3)

        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        left, right = ttk.Frame(pane), ttk.Frame(pane)
        pane.add(left, weight=2); pane.add(right, weight=4)

        cols = ("tag", "component", "channel", "seq", "fs", "duration", "frames", "serial")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=20)
        heads = {"tag":"Phoenix tag","component":"MT comp.","channel":"ch_id","seq":"sequence","fs":"sample rate","duration":"duration (s)","frames":"frames","serial":"board serial"}
        widths = {"tag":80,"component":70,"channel":50,"seq":55,"fs":80,"duration":85,"frames":75,"serial":90}
        for c in cols:
            self.tree.heading(c, text=heads[c]); self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", lambda e: self.preview_selected())

        nb = ttk.Notebook(right); nb.pack(fill=tk.BOTH, expand=True)
        tab_meta, tab_wave, tab_cont, tab_gps, tab_verify, tab_log = [ttk.Frame(nb) for _ in range(6)]
        for tab, name in zip([tab_meta,tab_wave,tab_cont,tab_gps,tab_verify,tab_log], ["Metadata","5-Ch Waveforms","Continuity","GPS / Time","MTH5 Verify","Log"]):
            nb.add(tab, text=name)
        self.nb = nb
        self.tab_wave = tab_wave; self.tab_cont = tab_cont; self.tab_gps = tab_gps; self.tab_verify = tab_verify

        self.meta_text = tk.Text(tab_meta, wrap="none", font=("Consolas", 9)); self.meta_text.pack(fill=tk.BOTH, expand=True)
        self.cont_text = tk.Text(tab_cont, wrap="none", font=("Consolas", 9)); self.cont_text.pack(fill=tk.BOTH, expand=True)
        self.gps_text = tk.Text(tab_gps, wrap="none", font=("Consolas", 10)); self.gps_text.pack(fill=tk.BOTH, expand=True)
        self.verify_text = tk.Text(tab_verify, wrap="none", font=("Consolas", 9)); self.verify_text.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(tab_log, wrap="word", font=("Consolas", 9)); self.log_text.pack(fill=tk.BOTH, expand=True)

        self.fig = self.canvas = None
        self.rate_notebook = None
        self.rate_plot_widgets = {}
        if HAVE_MPL:
            self.rate_notebook = ttk.Notebook(tab_wave)
            self.rate_notebook.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(tab_wave, text="matplotlib is not available.").pack(padx=20, pady=20)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def _path_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0,6), pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, padx=(6,0), pady=2)

    def log(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{stamp}] {msg}\n"); self.log_text.see(tk.END); self.update_idletasks()

    def _metadata_output_filename(self) -> str:
        """Build an MTH5 filename from Phoenix metadata.

        Preferred pattern:
        Survey_Station_InstrumentID_YYYYMMDD.mth5
        Missing fields are simply omitted.
        """
        layout = extract_layout(self.config if isinstance(self.config, dict) else {},
                                self.recmeta if isinstance(self.recmeta, dict) else {})
        survey = safe_id(layout.get("Survey_Name"), "")
        station = safe_id(layout.get("Station_Name"), "")
        instid = safe_id((self.recmeta or {}).get("instid"), "") if isinstance(self.recmeta, dict) else ""

        date_text = ""
        if isinstance(self.recmeta, dict):
            start_value = self.recmeta.get("start")
            try:
                if start_value not in (None, ""):
                    date_text = datetime.fromtimestamp(float(start_value), tz=timezone.utc).strftime("%Y%m%d")
            except Exception:
                date_text = ""

        parts = [p for p in (survey, station, instid, date_text) if p]
        if not parts:
            # Last-resort metadata-independent fallback.
            base_name = Path(self.station_dir.get().strip()).name if self.station_dir.get().strip() else "Phoenix_MTU5C"
            parts = [safe_id(base_name, "Phoenix_MTU5C")]
        return "_".join(parts) + ".mth5"

    def _update_output_from_metadata(self, force: bool = False):
        """Auto-fill Output MTH5 while preserving a user-edited filename."""
        base_s = self.station_dir.get().strip()
        if not base_s:
            return
        current = self.output_path.get().strip()
        # Update automatically only when empty, when it is our previous auto-name,
        # or when force=True (e.g. first station selection).
        if not force and current and current != self._last_auto_output_path:
            return
        target = Path(base_s) / self._metadata_output_filename()
        self.output_path.set(str(target))
        self._last_auto_output_path = str(target)
        self.log(f"Output MTH5 auto-named from metadata: {target.name}")

    def choose_station_dir(self):
        p = filedialog.askdirectory(title="Select Phoenix MTU-5C station/data directory")
        if not p: return
        self.station_dir.set(p)
        base = Path(p)
        self.recmeta_path.set(""); self.config_path.set("")
        self._auto_find_metadata_recursive(base)
        # V36.1 FAST START: do not scan an entire Dropbox/station tree here.
        # The user starts the scan explicitly with the Scan / Read MTU-5C button.
        self.status.set("Station folder selected. Press 'Scan / Read MTU-5C' to load files.")
        self.update_idletasks()

    def choose_recmeta(self):
        p = filedialog.askopenfilename(title="Select recmeta JSON", filetypes=[("JSON","*.json"),("All","*.*")])
        if p:
            self.recmeta_path.set(p)
            self.recmeta = load_json(p)
            self._update_output_from_metadata(force=(not bool(self.output_path.get().strip())))

    def choose_config(self):
        p = filedialog.askopenfilename(title="Select config JSON", filetypes=[("JSON","*.json"),("All","*.*")])
        if p:
            self.config_path.set(p)
            self.config = load_json(p)
            self._update_output_from_metadata(force=(not bool(self.output_path.get().strip())))

    def choose_calibration_dir(self):
        p = filedialog.askdirectory(title="Select folder containing Phoenix rxcal/scal JSON")
        if p: self.calibration_dir.set(p)

    def choose_output(self):
        p = filedialog.asksaveasfilename(title="Save MTH5", defaultextension=".mth5", filetypes=[("MTH5","*.mth5"),("HDF5","*.h5"),("All","*.*")])
        if p:
            self.output_path.set(p)
            # Treat a Save-As choice as a manual override.
            self._last_auto_output_path = ""

    def _auto_find_metadata_recursive(self, base: Path):
        """V36.1 bounded metadata search; avoids full recursive Dropbox scan."""
        roots = [base]
        try:
            common = {"metadata", "meta", "config", "configuration", "phoenix", "data"}
            roots += [p for p in base.iterdir() if p.is_dir() and p.name.lower() in common]
        except Exception:
            pass
        seen = set()
        for root in roots:
            try:
                candidates = list(root.glob("*.json"))
            except Exception:
                candidates = []
            for p in candidates:
                key = str(p).lower()
                if key in seen:
                    continue
                seen.add(key)
                role = find_json_role(p)
                if role == "recmeta" and not self.recmeta_path.get():
                    self.recmeta_path.set(str(p))
                elif role == "config" and not self.config_path.get():
                    self.config_path.set(str(p))
            if self.recmeta_path.get() and self.config_path.get():
                break

    def _discover_bins(self, base: Path) -> List[Path]:
        """V36.1 faster waveform discovery; avoids rglob('*')."""
        import os
        files = []
        skip = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "calibration", "calibrations", "01_校正檔",
            "plots", "figures", "output", "outputs", "diagnostics"
        }
        skip_lower = {x.lower() for x in skip}
        for root, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d.lower() not in skip_lower]
            for name in names:
                low = name.lower()
                if low.endswith(".bin") or ".td_" in low:
                    files.append(Path(root) / name)
        return sorted(set(files), key=lambda p: str(p).lower())

    def scan_station(self):
        base_s=self.station_dir.get().strip()
        if not base_s or not Path(base_s).exists():
            messagebox.showerror("Input","Please select the Phoenix MTU-5C station/data directory first."); return
        base=Path(base_s)
        try:
            self.status.set("Scanning MTU-5C files...")
            self._auto_find_metadata_recursive(base)
            self.recmeta=load_json(self.recmeta_path.get()); self.config=load_json(self.config_path.get())
            self._update_output_from_metadata(force=(not bool(self.output_path.get().strip())))
            self.channel_map=extract_channel_map(self.recmeta); self.channel_info=extract_recmeta_channel_info(self.recmeta)
            self.board_serial_to_tag = {}
            for _tag, _info in self.channel_info.items():
                _ser = str(_info.get("ser", "") or "").strip().upper()
                if _ser:
                    self.board_serial_to_tag[_ser] = _tag.upper()
            headers=[]; errors=[]
            bins=self._discover_bins(base)
            for p in bins:
                try: headers.append(read_mtu5c_header(p))
                except Exception as e: errors.append(f"{p.name}: {e}")
            headers.sort(key=lambda h:(h.recording_id,h.channel_id,h.file_sequence,str(h.path)))
            self.bin_headers=headers
            if self.board_serial_to_tag:
                self.log("TD board-serial mapping: " + ", ".join(f"{k}->{v}" for k,v in sorted(self.board_serial_to_tag.items())))
            for iid in self.tree.get_children(): self.tree.delete(iid)
            for i,h in enumerate(headers):
                cid=self._effective_channel_id(h); tag=self.channel_map.get(cid,f"ch{cid}"); comp=TAG_TO_MT_COMPONENT.get(tag.upper(),"?")
                ns = native_sample_count(h.path)
                dur = (ns / h.sample_rate) if h.sample_rate > 0 else 0.0
                frame_label = h.n_frames if not _is_td_file(h.path) else f"TD/{ns}"
                self.tree.insert("",tk.END,iid=str(i),values=(tag,comp,cid,h.file_sequence,f"{h.sample_rate:g}",f"{dur:.3f}",frame_label,h.channel_board_serial))
            self._render_metadata(); self._render_gps_time()
            self.log(f"Scanned {len(bins)} Phoenix native files (.bin/.td_*); recognized {len(headers)} MTU-5C files.")
            if headers: self.log(f"Receiver: {headers[0].instrument_type} {headers[0].instrument_serial}; record id={headers[0].recording_id}")
            if self.channel_map:
                self.log(f"Channel map from recmeta: {self.channel_map}")
                tdmap={}
                for h in headers:
                    if _is_td_file(h.path):
                        cid=self._effective_channel_id(h); tdmap[cid]=tdmap.get(cid,0)+1
                if tdmap: self.log(f"TD effective channel IDs from folder/header: {tdmap}")
            if errors: self.log(f"Skipped {len(errors)} files. First: {errors[0]}")
            self.status.set(f"Recognized {len(headers)} MTU-5C native files")
            self.preview_all_channels(silent=True)
            self.check_full_continuity(silent=True)
        except Exception as e:
            self.status.set("Scan failed"); self.log(traceback.format_exc()); messagebox.showerror("Scan failed",str(e))

    def _render_metadata(self):
        timing=self.recmeta.get("timing",{}) if isinstance(self.recmeta,dict) else {}
        data={
            "receiver":self.recmeta.get("receiver_commercial_name") or self.recmeta.get("receiver_model"),
            "instrument_id":self.recmeta.get("instid"),
            "recording_start_unix":self.recmeta.get("start"), "recording_stop_unix":self.recmeta.get("stop"),
            "recording_start_utc":unix_iso(self.recmeta.get("start")), "recording_stop_utc":unix_iso(self.recmeta.get("stop")),
            "gps":{"latitude":timing.get("gps_lat"),"longitude":timing.get("gps_lon"),"elevation":timing.get("gps_alt"),"satellites":timing.get("tm_satcount"),"leap_seconds":timing.get("tm_leapSecAtRecStart")},
            "layout":extract_layout(self.config,self.recmeta), "channel_map":self.channel_map,
            "channel_configuration":self.channel_info, "electric_dipoles":describe_dipoles(self.config), "bin_file_count":len(self.bin_headers),
        }
        self.meta_text.delete("1.0",tk.END); self.meta_text.insert("1.0",json.dumps(data,indent=2,ensure_ascii=False,default=str))

    def _render_gps_time(self):
        timing=self.recmeta.get("timing",{}) if isinstance(self.recmeta,dict) else {}
        start=self.recmeta.get("start"); stop=self.recmeta.get("stop")
        lines=[
            "PHOENIX MTU-5C GPS / RECORDING TIME", "="*118,
            f"Receiver commercial name : {self.recmeta.get('receiver_commercial_name','')}",
            f"Receiver / instrument ID : {self.recmeta.get('instid','')}",
            f"Start Unix               : {start}", f"Start UTC                : {unix_iso(start)}",
            f"Stop Unix                : {stop}", f"Stop UTC                 : {unix_iso(stop)}",
            f"Duration from recmeta    : {(float(stop)-float(start)) if start is not None and stop is not None else 'N/A'} s",
            "", f"GPS latitude             : {timing.get('gps_lat')}", f"GPS longitude            : {timing.get('gps_lon')}",
            f"GPS elevation (m)        : {timing.get('gps_alt')}", f"GPS satellites           : {timing.get('tm_satcount')}",
            f"Leap seconds at start    : {timing.get('tm_leapSecAtRecStart')}", f"Timing drift             : {timing.get('tm_drift')} {timing.get('tm_drift_units','')}",
            f"UTC set / time set       : {timing.get('tm_utcset')} / {timing.get('tm_timeset')}",
        ]

        # V69: show real Ex/Ey/Hx/Hy/Hz time coverage and continuity in the GPS/Time tab.
        # Channels are separated by sample rate so a 150-Hz run is never mixed with 24-kHz data.
        grouped=self._headers_by_component_rate() if self.bin_headers else {}
        lines += ["", "CHANNEL GPS/UTC COVERAGE + CONTINUITY", "="*118,
                  "Comp  Fs(Hz) Files Samples        Start UTC                         End UTC                           Duration(s)   Continuity",
                  "-"*118]
        for fs, comp_groups in grouped.items():
            for comp in COMPONENT_ORDER:
                hs=list(comp_groups.get(comp,[]))
                if not hs:
                    continue
                hs.sort(key=lambda h:(h.recording_id,h.file_sequence,str(h.path)))
                counts=[native_sample_count(h.path) for h in hs]
                total=int(sum(counts))

                # Phoenix fragments encountered in the field use either an absolute
                # recording_id per fragment or one common recording_id plus sequence.
                # Use absolute IDs when they vary; otherwise advance by fragment_period.
                recids=[int(h.recording_id) for h in hs]
                varying_recid=len(set(recids))>1
                intervals=[]
                for h,n in zip(hs,counts):
                    t0=float(h.recording_id)
                    if (not varying_recid) and h.fragment_period_s>0:
                        seq0=min(int(x.file_sequence) for x in hs)
                        t0 += (int(h.file_sequence)-seq0)*float(h.fragment_period_s)
                    t1=t0+(float(n)/float(h.sample_rate) if h.sample_rate>0 else 0.0)
                    intervals.append((t0,t1,h))
                intervals.sort(key=lambda x:x[0])
                first=intervals[0][0]; last=max(x[1] for x in intervals)

                time_gaps=0; overlaps=0; gap_seconds=0.0
                prev_end=intervals[0][1]
                tol=max(1.0/float(fs),1e-6)
                for t0,t1,_h in intervals[1:]:
                    delta=t0-prev_end
                    if delta>tol:
                        time_gaps+=1; gap_seconds+=delta
                    elif delta < -tol:
                        overlaps+=1
                    prev_end=max(prev_end,t1)

                # BIN files additionally carry frame counters; TD products do not.
                cr=self._continuity_for_headers(hs)
                frame_bad=int(cr.get("discontinuities",0) or 0)
                continuous=(time_gaps==0 and overlaps==0 and frame_bad==0)
                if continuous:
                    status="OK / CONTINUOUS"
                else:
                    parts=[]
                    if time_gaps: parts.append(f"{time_gaps} time gap(s), {gap_seconds:.3f}s")
                    if overlaps: parts.append(f"{overlaps} overlap(s)")
                    if frame_bad: parts.append(f"{frame_bad} frame gap(s)")
                    status="CHECK: "+", ".join(parts)
                lines.append(f"{comp.upper():<5} {fs:>7g} {len(hs):>5} {total:>13,}  "
                             f"{datetime.fromtimestamp(first,tz=timezone.utc).isoformat():<33} "
                             f"{datetime.fromtimestamp(last,tz=timezone.utc).isoformat():<33} "
                             f"{(last-first):>11.3f}   {status}")

        # Explicitly show missing/disabled channels as well.
        present_components={comp for _fs,g in grouped.items() for comp in COMPONENT_ORDER if g.get(comp)}
        for comp in COMPONENT_ORDER:
            if comp not in present_components:
                tag=COMPONENT_TO_TAG[comp]; info=self.channel_info.get(tag,{})
                state="DISABLED" if info and int(info.get("on",1))==0 else "NO DATA"
                lines.append(f"{comp.upper():<5} {'-':>7} {0:>5} {'-':>13}  {'-':<33} {'-':<33} {'-':>11}   {state}")

        if self.bin_headers:
            h=self.bin_headers[0]; lines += ["", "First native header", "-"*118, f"Header recording UTC     : {h.recording_start_utc}", f"Header GPS lat/lon/elev  : {h.gps_latitude}, {h.gps_longitude}, {h.gps_elevation}", f"Battery voltage          : {h.battery_voltage_v:.3f} V"]
        self.gps_text.delete("1.0",tk.END); self.gps_text.insert("1.0","\n".join(lines))

    def _effective_channel_id(self, h: MTU5CHeader) -> int:
        """Return Phoenix logical channel index when it can be inferred safely.

        TD headers commonly contain channel_id=0.  V20.4 therefore tries:
          1) channel-board serial -> recmeta chconfig tag -> channel_map index
          2) nearest numeric parent folder
          3) native header channel_id
        """
        if _is_td_file(h.path):
            # Most robust discriminator in EMpower TD exports: BCM board serial.
            bser = str(h.channel_board_serial or "").strip().upper()
            tag = self.board_serial_to_tag.get(bser)
            if tag:
                for cid, mapped_tag in self.channel_map.items():
                    if str(mapped_tag).upper() == tag:
                        return int(cid)

            # Secondary fallback: numeric channel directory.
            try:
                base = Path(self.station_dir.get()).resolve()
            except Exception:
                base = None
            p = h.path.parent
            while True:
                name = p.name.strip()
                if name.isdigit():
                    n = int(name)
                    if n in self.channel_map:
                        return n
                    if (n - 1) in self.channel_map and n not in self.channel_map:
                        return n - 1
                if base is not None:
                    try:
                        if p.resolve() == base:
                            break
                    except Exception:
                        pass
                if p.parent == p:
                    break
                p = p.parent
        return int(h.channel_id)

    def _tag_for_header(self, h: MTU5CHeader) -> str:
        # For TD, board serial can identify the tag directly even when the
        # header channel_id is always zero.
        if _is_td_file(h.path):
            bser = str(h.channel_board_serial or "").strip().upper()
            tag = self.board_serial_to_tag.get(bser)
            if tag:
                return tag
        cid = self._effective_channel_id(h)
        return self.channel_map.get(cid, f"ch{cid}")

    def _headers_by_component(self) -> Dict[str,List[MTU5CHeader]]:
        groups={c:[] for c in COMPONENT_ORDER}
        for h in self.bin_headers:
            tag=self._tag_for_header(h).upper(); comp=TAG_TO_MT_COMPONENT.get(tag)
            if comp in groups: groups[comp].append(h)
        for comp in groups: groups[comp].sort(key=lambda x:(x.sample_rate,x.file_sequence,str(x.path)))
        return groups

    def _headers_by_component_rate(self) -> Dict[float, Dict[str,List[MTU5CHeader]]]:
        """Group native files first by sample rate, then MT component."""
        rates: Dict[float, Dict[str,List[MTU5CHeader]]] = {}
        for h in self.bin_headers:
            tag=self._tag_for_header(h).upper()
            comp=TAG_TO_MT_COMPONENT.get(tag)
            if comp not in COMPONENT_ORDER:
                continue
            fs=float(h.sample_rate)
            rates.setdefault(fs,{c:[] for c in COMPONENT_ORDER})[comp].append(h)
        for fs in rates:
            for comp in COMPONENT_ORDER:
                rates[fs][comp].sort(key=lambda x:(x.file_sequence,str(x.path)))
        return dict(sorted(rates.items(), key=lambda kv: kv[0]))

    def _read_headers_prefix(self, hs:List[MTU5CHeader], seconds:float) -> Tuple[np.ndarray,float]:
        if not hs: return np.empty(0,dtype=np.float32),0.0
        fs=float(hs[0].sample_rate); target=max(1,int(round(seconds*fs))); pieces=[]; remaining=target
        for h in hs:
            if remaining<=0: break
            y,_=read_counts(h.path,max_samples=remaining); pieces.append(y); remaining-=len(y)
        return (np.concatenate(pieces)[:target] if pieces else np.empty(0,dtype=np.float32)),fs

    def _read_headers_full_for_display(self, hs:List[MTU5CHeader]) -> Tuple[np.ndarray,np.ndarray,float,int]:
        """Read a full channel for plotting only, decimating to max_display_points."""
        if not hs: return np.empty(0),np.empty(0),0.0,0
        fs=float(hs[0].sample_rate)
        total=int(sum(native_sample_count(h.path) for h in hs))
        if total<=0 or fs<=0: return np.empty(0),np.empty(0),fs,total
        stride=max(1,int(np.ceil(total/float(self.max_display_points))))
        ys=[]; ts=[]; sample_offset=0
        for h in hs:
            y,_=read_counts(h.path)
            if y.size:
                idx=np.arange(0,y.size,stride,dtype=np.int64)
                ys.append(np.asarray(y[idx]))
                ts.append((sample_offset+idx.astype(float))/fs)
            sample_offset += len(y)
        if not ys: return np.empty(0),np.empty(0),fs,total
        return np.concatenate(ts),np.concatenate(ys),fs,total

    def _clear_rate_tabs(self):
        if self.rate_notebook is None: return
        for tab_id in self.rate_notebook.tabs():
            self.rate_notebook.forget(tab_id)
        self.rate_plot_widgets.clear()

    def preview_selected(self):
        sel=self.tree.selection()
        if sel:
            i=int(sel[0]); h=self.bin_headers[i]
            tag=self._tag_for_header(h)
            self.log(f"Selected {h.path.name}: {tag}, fs={h.sample_rate:g} Hz")
        self.preview_all_channels()

    def preview_all_channels(self, silent:bool=False):
        if not self.bin_headers:
            if not silent: messagebox.showinfo("Waveform","Scan the station first.")
            return
        try:
            if self.rate_notebook is None:
                if not silent: messagebox.showerror("Waveform","matplotlib is not available.")
                return
            mode=self.waveform_mode.get().strip() or "Full recording"
            sec=float(self.preview_seconds.get())
            grouped=self._headers_by_component_rate()
            self._clear_rate_tabs()
            if not grouped:
                if not silent: messagebox.showinfo("Waveform","No mapped Ex/Ey/Hx/Hy/Hz data found.")
                return

            summary=[]
            for fs,comp_groups in grouped.items():
                frame=ttk.Frame(self.rate_notebook)
                label=f"{fs:g} Hz"
                self.rate_notebook.add(frame,text=label)
                fig=Figure(figsize=(9,7),dpi=100)
                axes=[fig.add_subplot(5,1,i+1) for i in range(5)]
                canvas=FigureCanvasTkAgg(fig,master=frame)
                canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
                toolbar=NavigationToolbar2Tk(canvas,frame); toolbar.update()
                present=[]
                durations=[]
                for ax,comp in zip(axes,COMPONENT_ORDER):
                    hs=comp_groups.get(comp,[])
                    if hs:
                        present.append(comp)
                        if mode=="Full recording":
                            t,y,ch_fs,total=self._read_headers_full_for_display(hs)
                            duration=(total/ch_fs) if ch_fs>0 else 0.0
                            durations.append(duration)
                            if y.size:
                                ax.plot(t,y,linewidth=0.55)
                                ax.set_xlim(0,max(duration,float(t[-1]) if t.size else duration))
                            info=f"{ch_fs:g} Hz | total n={total:,} | plotted n={len(y):,} | {duration:.3f} s"
                        else:
                            y,ch_fs=self._read_headers_prefix(hs,sec)
                            t=np.arange(y.size,dtype=float)/ch_fs if ch_fs>0 else np.empty(0)
                            if y.size:
                                ax.plot(t,y,linewidth=0.55)
                                ax.set_xlim(0,max(sec,float(t[-1]) if t.size else sec))
                            info=f"{ch_fs:g} Hz | n={len(y):,} | preview {sec:g} s"
                        ax.text(0.995,0.88,info,transform=ax.transAxes,ha="right",va="top",fontsize=8)
                    else:
                        tag=COMPONENT_TO_TAG[comp]; info_cfg=self.channel_info.get(tag,{})
                        state="disabled in recmeta" if info_cfg and int(info_cfg.get("on",1))==0 else f"NO DATA AT {fs:g} Hz"
                        ax.text(0.5,0.5,f"{comp.upper()} ({tag}): {state}",transform=ax.transAxes,ha="center",va="center")
                    ax.set_ylabel(comp.upper()); ax.grid(True,alpha=0.2)
                axes[-1].set_xlabel("Time from recording start (s)")
                title_mode="full recording" if mode=="Full recording" else f"first {sec:g} s"
                fig.suptitle(f"Phoenix MTU-5C {fs:g} Hz waveforms | {title_mode}",fontsize=11)
                fig.tight_layout(rect=[0,0,1,0.97]); canvas.draw_idle()
                self.rate_plot_widgets[fs]=(frame,fig,axes,canvas,toolbar)
                summary.append(f"{fs:g} Hz: {present if present else 'no mapped channels'}")

            self.log("Multi-rate waveform displayed | "+"; ".join(summary)+f" | mode={mode}")
            self.nb.select(self.tab_wave)
        except Exception as e:
            self.log(traceback.format_exc())
            if not silent: messagebox.showerror("Multi-rate waveform failed",str(e))

    def _continuity_for_headers(self, hs:List[MTU5CHeader]) -> dict:
        if not hs: return {"present":False,"files":0,"frames":0,"missing_frames":0,"discontinuities":0}

        # .td_* files are float32 waveform products and do not carry the BIN footer
        # frame counters used by read_frame_counters(). Report sample continuity by
        # file sequence and decoded sample length instead of inventing counters.
        if all(_is_td_file(h.path) for h in hs):
            hs2 = sorted(hs, key=lambda x:x.file_sequence)
            fs = hs2[0].sample_rate
            counts = [native_sample_count(h.path) for h in hs2]
            total = int(sum(counts))
            details = [{
                "file": h.path.name,
                "sequence": h.file_sequence,
                "frames": 0,
                "first_counter": None,
                "last_counter": None,
                "internal_discontinuities": 0,
                "internal_missing_frames": 0
            } for h in hs2]
            return {
                "present": True, "files": len(hs2), "sample_rate": fs,
                "frames": 0, "samples": total,
                "duration_s": total/fs if fs else None,
                "first_counter": None, "last_counter": None,
                "missing_frames": 0, "discontinuities": 0,
                "counter_resets_or_backsteps": 0, "continuous": True,
                "files_detail": details
            }

        total_frames=0; missing=0; discontinuities=0; resets=0; first_counter=None; last_counter=None; details=[]
        prev_last=None
        for h in sorted(hs,key=lambda x:x.file_sequence):
            c=read_frame_counters(h.path); total_frames+=len(c)
            local_bad=0; local_missing=0
            if c.size:
                if first_counter is None: first_counter=int(c[0])
                if prev_last is not None:
                    d0=int(c[0])-int(prev_last)
                    if d0!=1:
                        discontinuities+=1
                        if d0>1: missing+=d0-1
                        else: resets+=1
                if len(c)>1:
                    d=np.diff(c); bad=d[d!=1]; local_bad=len(bad); discontinuities+=local_bad
                    local_missing=int(np.sum(bad[bad>1]-1)) if np.any(bad>1) else 0; missing+=local_missing; resets+=int(np.sum(bad<=0))
                last_counter=int(c[-1]); prev_last=last_counter
            details.append({"file":h.path.name,"sequence":h.file_sequence,"frames":len(c),"first_counter":int(c[0]) if c.size else None,"last_counter":int(c[-1]) if c.size else None,"internal_discontinuities":local_bad,"internal_missing_frames":local_missing})
        fs=hs[0].sample_rate; n_samples=total_frames*SAMPLES_PER_FRAME
        return {"present":True,"files":len(hs),"sample_rate":fs,"frames":total_frames,"samples":n_samples,"duration_s":n_samples/fs if fs else None,"first_counter":first_counter,"last_counter":last_counter,"missing_frames":missing,"discontinuities":discontinuities,"counter_resets_or_backsteps":resets,"continuous":discontinuities==0,"files_detail":details}

    def check_full_continuity(self, silent:bool=False):
        if not self.bin_headers:
            if not silent: messagebox.showinfo("Continuity","Scan the station first.")
            return
        try:
            self.status.set("Checking every frame counter..."); self.update_idletasks()
            groups=self._headers_by_component(); report={comp:self._continuity_for_headers(groups[comp]) for comp in COMPONENT_ORDER}
            self.continuity_report=report
            lines=["FULL RECORDING FRAME CONTINUITY CHECK","="*100,"Component Tag Present Files  Fs(Hz)   Frames      Samples       Missing  Discont  Status","-"*100]
            for comp in COMPONENT_ORDER:
                r=report[comp]; tag=COMPONENT_TO_TAG[comp]
                if not r.get("present"):
                    info=self.channel_info.get(tag,{})
                    status="DISABLED" if info and int(info.get("on",1))==0 else "NO DATA"
                    lines.append(f"{comp.upper():<9} {tag:<3} {'NO':<7} {0:>5} {'-':>8} {'-':>10} {'-':>13} {'-':>9} {'-':>8}  {status}")
                else:
                    status="OK" if r['continuous'] else "CHECK"
                    lines.append(f"{comp.upper():<9} {tag:<3} {'YES':<7} {r['files']:>5} {r['sample_rate']:>8g} {r['frames']:>10,} {r['samples']:>13,} {r['missing_frames']:>9,} {r['discontinuities']:>8,}  {status}")
            lines += ["", "Per-file details", "-"*100]
            for comp in COMPONENT_ORDER:
                r=report[comp]
                if not r.get('present'): continue
                lines.append(f"[{comp.upper()} / {COMPONENT_TO_TAG[comp]}]")
                for d in r['files_detail']:
                    lines.append(f"  seq={d['sequence']:>3} frames={d['frames']:>7} counter={d['first_counter']}..{d['last_counter']} internal gaps={d['internal_discontinuities']} missing={d['internal_missing_frames']}  {d['file']}")
            self.cont_text.delete("1.0",tk.END); self.cont_text.insert("1.0","\n".join(lines))
            self.log("Full recording continuity checked: "+", ".join(f"{c}={'OK' if report[c].get('continuous') else ('NO DATA' if not report[c].get('present') else 'CHECK')}" for c in COMPONENT_ORDER))
            self.status.set("Continuity check complete")
            if not silent: self.nb.select(self.tab_cont)
        except Exception as e:
            self.status.set("Continuity check failed"); self.log(traceback.format_exc())
            if not silent: messagebox.showerror("Continuity check failed",str(e))

    def _parse_sample_rates(self) -> List[int]:
        vals=[]
        for s in self.sample_rates.get().replace(";",",").split(","):
            s=s.strip()
            if s: vals.append(int(float(s)))
        return sorted(set(vals)) or [150,24000]

    def _validate_station_folder_for_mth5(self, base:Path) -> Tuple[bool,str]:
        bins=list(base.rglob("*.bin"))
        tds=[p for p in base.rglob("*") if p.is_file() and ".td_" in p.name.lower()]
        native=bins+tds
        if not native: return False,"No Phoenix .bin or .td_* files found."
        numeric=sum(1 for p in native if p.parent.name.isdigit())
        return True,f"Found {len(native)} native files ({len(bins)} BIN, {len(tds)} TD); {numeric} are under numeric Phoenix channel folders."

    def _read_complete_headers(self, hs: List[MTU5CHeader]) -> Tuple[np.ndarray,float]:
        if not hs:
            return np.empty(0,dtype=np.float32),0.0
        fs=float(hs[0].sample_rate); pieces=[]
        for h in hs:
            y,_=read_counts(h.path); pieces.append(y)
        if not pieces:
            return np.empty(0,dtype=np.float32),fs
        return np.concatenate(pieces),fs


    # ------------------------------------------------------------------
    # V29 OFFICIAL EMPOWER JSON -> MTH5 CALIBRATION
    # ------------------------------------------------------------------
    @staticmethod
    def _v29_json_files(cal_dir: Path):
        """Find EMpower-exported calibration JSON files conservatively."""
        cal_dir = Path(cal_dir)
        pats = ("*.rxcal.json", "*_rxcal.json", "*.rx_cal.json", "*_rx_cal.json",
                "*.scal.json", "*_scal.json")
        files = []
        seen = set()
        for pat in pats:
            for p in cal_dir.rglob(pat):
                rp = str(p.resolve()).lower()
                if rp not in seen:
                    seen.add(rp)
                    files.append(p)
        return sorted(files)

    @staticmethod
    def _v29_load_json(path: Path):
        """Load JSON robustly on Windows/CP950 systems."""
        raw = Path(path).read_bytes()
        last = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return json.loads(raw.decode(enc)), enc
            except Exception as exc:
                last = exc
        raise last

    @staticmethod
    def _v29_flatten_json(obj, prefix=""):
        """Flatten scalar JSON values for identity search."""
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.update(PhoenixMTU5CApp._v29_flatten_json(v, key))
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:200]):
                key = f"{prefix}[{i}]"
                out.update(PhoenixMTU5CApp._v29_flatten_json(v, key))
        else:
            out[prefix] = obj
        return out

    @classmethod
    def _v29_json_identity(cls, path: Path):
        """Extract likely receiver/sensor identity from EMpower export JSON."""
        data, enc = cls._v29_load_json(path)
        flat = cls._v29_flatten_json(data)
        text = "\n".join(f"{k}={v}" for k, v in flat.items())

        # Gather plausible serial/model values without assuming exact schema.
        serials, models, kinds = [], [], []
        for k, v in flat.items():
            lk = k.lower()
            sv = str(v).strip()
            if not sv:
                continue
            if any(x in lk for x in ("serial", "serial_number", "sensor_id", "receiver_id", "instid", "instrument_id")):
                serials.append(sv)
            if any(x in lk for x in ("model", "type_name", "sensor_type", "receiver_model", "commercial_name")):
                models.append(sv)
            if any(x in lk for x in ("calibration.type", "kind", "calibration_kind")):
                kinds.append(sv)

        # Filename is often the strongest Phoenix exported identity.
        name = path.name.lower()
        inferred_kind = "receiver" if ("rxcal" in name or "rx_cal" in name) else ("sensor" if "scal" in name else "unknown")

        return {
            "path": str(path),
            "encoding": enc,
            "kind": inferred_kind,
            "serial_candidates": sorted(set(serials)),
            "model_candidates": sorted(set(models)),
            "kind_candidates": sorted(set(kinds)),
            "flat": flat,
            "raw": data,
            "search_text": text,
        }

    def _v29_recording_identity(self):
        """Return authoritative actual-recording identities for matching."""
        receiver_ids = []
        receiver_models = []
        try:
            # Common places used in earlier versions / Phoenix metadata.
            for attr in ("receiver_id", "receiver_serial", "instid"):
                v = getattr(self, attr, None)
                if v not in (None, "", 0):
                    receiver_ids.append(str(v))
        except Exception:
            pass

        # Pull likely receiver metadata from station/recording dictionaries if present.
        for container_name in ("recording_meta", "station_meta", "recmeta", "metadata"):
            d = getattr(self, container_name, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    lk = str(k).lower()
                    if any(x in lk for x in ("instid", "receiver_id", "receiver_serial")) and v not in (None, ""):
                        receiver_ids.append(str(v))
                    if "receiver_model" in lk and v not in (None, ""):
                        receiver_models.append(str(v))

        # Known channel identities from actual recording metadata.
        channels = {}
        try:
            for tag in ("E1","E2","H1","H2","H3"):
                info = dict(self.channel_info.get(tag, {}) or {})
                channels[tag] = {
                    "serial": str(info.get("serial") or info.get("ser") or ""),
                    "model": str(info.get("type_name") or info.get("model") or ""),
                    "ga": info.get("ga"),
                    "pg": info.get("pg"),
                    "lp": info.get("lp"),
                    "sampleRate": info.get("sampleRate"),
                }
        except Exception:
            pass

        return {
            "receiver_ids": sorted(set(x for x in receiver_ids if x)),
            "receiver_models": sorted(set(x for x in receiver_models if x)),
            "channels": channels,
        }

    @staticmethod
    def _v29_match_score(identity, serial="", model="", kind=None):
        """Conservative EMpower JSON matcher."""
        txt = identity.get("search_text", "").lower()
        name = Path(identity["path"]).name.lower()
        score = 0
        reasons = []

        if kind and identity.get("kind") == kind:
            score += 30
            reasons.append(f"kind={kind}")

        serial = str(serial or "").strip()
        if serial:
            if serial.lower() in name:
                score += 120
                reasons.append("serial-in-filename")
            if serial.lower() in txt:
                score += 100
                reasons.append("serial-in-json")

        model = str(model or "").strip()
        if model:
            if model.lower() in txt:
                score += 30
                reasons.append("model-in-json")
            if model.lower().replace("-", "") in name.replace("-", ""):
                score += 20
                reasons.append("model-in-filename")

        return score, reasons

    def _v29_match_calibration_jsons(self, cal_dir: Path):
        """Match official EMpower exported JSON calibrations to actual recording."""
        files = self._v29_json_files(cal_dir)
        identities = []
        for p in files:
            try:
                identities.append(self._v29_json_identity(p))
            except Exception as exc:
                identities.append({
                    "path": str(p), "kind": "unknown", "error": repr(exc),
                    "serial_candidates": [], "model_candidates": [], "search_text": ""
                })

        rec = self._v29_recording_identity()

        # Receiver: if actual receiver ID could not be recovered programmatically,
        # still allow unique receiver JSON as a safe fallback.
        receiver_candidates = []
        for ident in identities:
            if ident.get("kind") != "receiver":
                continue
            best_score, best_reasons = 0, []
            for rid in rec["receiver_ids"] or [""]:
                score, reasons = self._v29_match_score(ident, serial=rid, kind="receiver")
                if score > best_score:
                    best_score, best_reasons = score, reasons
            receiver_candidates.append((best_score, ident, best_reasons))

        receiver_candidates.sort(key=lambda x: x[0], reverse=True)
        receiver_match = None
        receiver_status = "NO_MATCH"
        if receiver_candidates:
            if receiver_candidates[0][0] >= 100:
                receiver_match = receiver_candidates[0][1]
                receiver_status = "STRONG_MATCH"
            elif len([x for x in receiver_candidates if x[0] > 0]) == 1:
                receiver_match = receiver_candidates[0][1]
                receiver_status = "WEAK_UNIQUE_MATCH"
            elif len(receiver_candidates) == 1 and not rec["receiver_ids"]:
                receiver_match = receiver_candidates[0][1]
                receiver_status = "UNIQUE_RECEIVER_JSON"

        # Sensors: match only magnetic channels with real actual-recording serials.
        sensor_matches = {}
        sensor_diagnostics = {}
        for tag in ("H1","H2","H3"):
            ch = rec["channels"].get(tag, {})
            serial = ch.get("serial", "")
            model = ch.get("model", "")
            cand = []
            for ident in identities:
                if ident.get("kind") != "sensor":
                    continue
                score, reasons = self._v29_match_score(ident, serial=serial, model=model, kind="sensor")
                cand.append((score, ident, reasons))
            cand.sort(key=lambda x: x[0], reverse=True)

            match = None
            status = "NO_MATCH"
            if cand and cand[0][0] >= 100:
                match = cand[0][1]
                status = "STRONG_MATCH"
            elif cand and len([x for x in cand if x[0] > 0]) == 1 and cand[0][0] >= 50:
                match = cand[0][1]
                status = "WEAK_UNIQUE_MATCH"

            sensor_matches[tag] = match
            sensor_diagnostics[tag] = {
                "actual_serial": serial,
                "actual_model": model,
                "status": status,
                "top_candidates": [
                    {
                        "score": score,
                        "file": Path(ident["path"]).name,
                        "reasons": reasons,
                    }
                    for score, ident, reasons in cand[:5]
                ],
            }

        return {
            "files": identities,
            "recording_identity": rec,
            "receiver_match": receiver_match,
            "receiver_status": receiver_status,
            "receiver_candidates": [
                {"score": score, "file": Path(ident["path"]).name, "reasons": reasons}
                for score, ident, reasons in receiver_candidates[:10]
            ],
            "sensor_matches": sensor_matches,
            "sensor_diagnostics": sensor_diagnostics,
        }

    @staticmethod
    def _v29_build_calibration_dicts(match_result):
        """Build dicts accepted by current MTH5 Phoenix workflows.

        Values are file paths, keyed by receiver/sensor identity.  If the
        installed MTH5 release expects another concrete type, the official
        importer attempt below reports that incompatibility rather than
        silently fabricating a filter.
        """
        receiver_dict = {}
        sensor_dict = {}

        r = match_result.get("receiver_match")
        if r:
            # Use every serial candidate; fallback to file stem.
            keys = r.get("serial_candidates") or [Path(r["path"]).stem]
            for k in keys:
                receiver_dict[str(k)] = r["path"]

        for tag, ident in match_result.get("sensor_matches", {}).items():
            if not ident:
                continue
            keys = ident.get("serial_candidates") or [Path(ident["path"]).stem]
            for k in keys:
                sensor_dict[str(k)] = ident["path"]

        return receiver_dict, sensor_dict

    def _v29_try_official_phoenix_import(self, data_path: Path, out_mth5: Path, receiver_dict, sensor_dict):
        """Try official MTH5 Phoenix APIs with calibration dictionaries.

        This is deliberately introspective because MTH5 Phoenix APIs evolved
        between releases. It never applies a guessed response.
        """
        import inspect, traceback, shutil, tempfile, os

        attempts = []

        def note(name, ok, detail):
            attempts.append({"attempt": name, "ok": bool(ok), "detail": str(detail)})

        # UTF-8 safe import patch retained from earlier versions.
        try:
            self._import_mth5_utf8_safe()
        except Exception:
            pass

        # 1) PhoenixClient
        try:
            from mth5.clients.phoenix import PhoenixClient
            sig = inspect.signature(PhoenixClient)
            kwargs = {}
            if "data_path" in sig.parameters:
                kwargs["data_path"] = Path(data_path)
            elif "path" in sig.parameters:
                kwargs["path"] = Path(data_path)

            if "receiver_calibration_dict" in sig.parameters:
                kwargs["receiver_calibration_dict"] = receiver_dict
            if "sensor_calibration_dict" in sig.parameters:
                kwargs["sensor_calibration_dict"] = sensor_dict
            if "sample_rates" in sig.parameters:
                kwargs["sample_rates"] = [150, 24000]

            client = PhoenixClient(**kwargs)

            # Probe common method names.
            methods = (
                "make_mth5_from_phoenix",
                "make_mth5",
                "run",
            )
            for mname in methods:
                if not hasattr(client, mname):
                    continue
                meth = getattr(client, mname)
                msig = inspect.signature(meth)
                mkwargs = {}
                for key in ("save_path", "output_path", "mth5_path", "filename", "file_name"):
                    if key in msig.parameters:
                        if key in ("filename", "file_name"):
                            mkwargs[key] = str(out_mth5.name)
                        else:
                            mkwargs[key] = Path(out_mth5.parent)

                result = meth(**mkwargs)
                note(f"PhoenixClient.{mname}", True, f"result={result!r}")

                # Locate output if method returned a path.
                if isinstance(result, (str, Path)) and Path(result).exists():
                    rp = Path(result)
                    if rp.resolve() != out_mth5.resolve():
                        shutil.copy2(rp, out_mth5)
                if out_mth5.exists():
                    return True, attempts
        except Exception:
            note("PhoenixClient", False, traceback.format_exc())

        # 2) MakeMTH5 fallback
        try:
            from mth5.mth5 import MTH5
            from mth5.utils.helpers import initialize_mth5  # may not exist
            note("MakeMTH5 fallback", False, "No compatible PhoenixClient output method produced the target file.")
        except Exception:
            note("MakeMTH5 fallback", False, traceback.format_exc())

        return out_mth5.exists(), attempts

    @staticmethod
    def _v29_verify_mth5_filters(path: Path):
        """Strictly verify real filter objects and channel references."""
        import h5py, json, math
        result = {
            "file": str(path),
            "real_filter_objects": [],
            "channels": {},
            "all_channels_with_refs": False,
        }
        if not Path(path).exists():
            result["error"] = "MTH5 file not found"
            return result

        with h5py.File(path, "r") as h5:
            # Real filter objects are child objects below /Filters categories.
            for root_name in ("/Filters", "/Experiment/Filters"):
                if root_name not in h5:
                    continue
                root = h5[root_name]
                def visitor(name, obj):
                    # require at least category/name depth
                    full = root_name.rstrip("/") + "/" + name
                    rel_parts = [p for p in name.split("/") if p]
                    if len(rel_parts) >= 2:
                        result["real_filter_objects"].append(full)
                try:
                    root.visititems(visitor)
                except Exception:
                    pass

            # Search channel-like groups/datasets by component attrs/name.
            def walk(name, obj):
                low = name.lower()
                component = None
                for c in ("ex","ey","hx","hy","hz"):
                    if low.endswith("/"+c) or low == c:
                        component = c
                        break
                try:
                    attrs = {str(k): obj.attrs[k] for k in obj.attrs.keys()}
                except Exception:
                    attrs = {}

                if component:
                    refs = []
                    for key in ("filters","filter","filter_name","filter_names","response","calibration"):
                        if key in attrs:
                            v = attrs[key]
                            try:
                                if hasattr(v, "tolist"):
                                    v = v.tolist()
                            except Exception:
                                pass
                            sv = str(v)
                            if sv not in ("", "[]", "None", "none", "null"):
                                refs.append({"attr": key, "value": sv})
                    result["channels"].setdefault(component, []).append({
                        "path": "/" + name.strip("/"),
                        "refs": refs,
                        "units": str(attrs.get("units","")),
                    })
            try:
                h5.visititems(walk)
            except Exception:
                pass

        comps = ("ex","ey","hx","hy","hz")
        result["all_channels_with_refs"] = all(
            any(item["refs"] for item in result["channels"].get(c, []))
            for c in comps
        )
        return result

    def _v29_official_calibration_workflow(self, out_mth5: Path):
        """Main V29 workflow: official EMpower JSON first, binary only diagnostics."""
        import json, traceback

        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            self.log("V29: calibration folder unavailable; official JSON calibration not attempted.")
            return False, "NO_CALIBRATION_FOLDER"

        match = self._v29_match_calibration_jsons(Path(cal))

        # Always write transparent matching report next to final MTH5.
        outdir = Path(out_mth5).with_suffix("")
        outdir = outdir.parent / (outdir.name + "_V29_official_calibration")
        outdir.mkdir(parents=True, exist_ok=True)

        serializable = {
            "recording_identity": match["recording_identity"],
            "receiver_status": match["receiver_status"],
            "receiver_match": match["receiver_match"]["path"] if match["receiver_match"] else None,
            "receiver_candidates": match["receiver_candidates"],
            "sensor_matches": {
                k: (v["path"] if v else None)
                for k, v in match["sensor_matches"].items()
            },
            "sensor_diagnostics": match["sensor_diagnostics"],
            "all_json_files": [
                {
                    "path": i.get("path"),
                    "kind": i.get("kind"),
                    "serial_candidates": i.get("serial_candidates", []),
                    "model_candidates": i.get("model_candidates", []),
                    "error": i.get("error"),
                }
                for i in match["files"]
            ],
        }

        mp = outdir / "V29_calibration_matching.json"
        mp.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

        receiver_dict, sensor_dict = self._v29_build_calibration_dicts(match)

        # Require one receiver and all three magnetic sensor calibrations.
        complete = bool(match["receiver_match"]) and all(match["sensor_matches"].get(k) for k in ("H1","H2","H3"))

        report = [
            "V29 OFFICIAL EMPOWER JSON -> MTH5 CALIBRATION",
            "="*120,
            f"MTH5 target        : {out_mth5}",
            f"Calibration folder : {cal}",
            f"Output diagnostics : {outdir}",
            "",
            f"Receiver status    : {match['receiver_status']}",
            f"Receiver match     : {Path(match['receiver_match']['path']).name if match['receiver_match'] else 'NONE'}",
        ]
        for tag in ("H1","H2","H3"):
            m = match["sensor_matches"].get(tag)
            d = match["sensor_diagnostics"].get(tag, {})
            report.append(
                f"{tag}: serial={d.get('actual_serial','')} model={d.get('actual_model','')} "
                f"status={d.get('status','')} match={Path(m['path']).name if m else 'NONE'}"
            )

        report += [
            "",
            f"Official JSON set complete: {complete}",
            "Binary .rxcal/.scal remain diagnostic only in V29.",
        ]

        if not complete:
            report += [
                "",
                "V29 RESULT: OFFICIAL CALIBRATION NOT APPLIED.",
                "Reason: one or more required EMpower-exported calibration JSON files were not confidently matched.",
                "Expected receiver file pattern: *.rxcal.json / *_rxcal.json",
                "Expected sensor file pattern  : *.scal.json / *_scal.json",
                "No guessed filter is written.",
            ]
            rp = outdir / "V29_official_calibration_report.txt"
            rp.write_text("\n".join(report), encoding="utf-8")
            self.log("\n" + "\n".join(report))
            return False, "\n".join(report)

        # IMPORTANT: official MTH5 import should rebuild a calibrated MTH5.
        # We do not modify an already written raw file in-place.
        calibrated_out = out_mth5.with_name(out_mth5.stem + "_official_calibrated.mth5")

        # Determine raw Phoenix data folder from GUI's data path if present.
        data_path = None
        for attr in ("input_dir", "data_dir", "phoenix_dir", "source_dir"):
            obj = getattr(self, attr, None)
            try:
                if hasattr(obj, "get"):
                    val = obj.get().strip()
                else:
                    val = str(obj).strip() if obj else ""
                if val and Path(val).exists():
                    data_path = Path(val)
                    break
            except Exception:
                pass

        if data_path is None:
            report += [
                "",
                "V29 RESULT: OFFICIAL CALIBRATION JSON MATCHED, BUT RAW PHOENIX DATA PATH COULD NOT BE RESOLVED.",
                "No calibrated MTH5 was generated; existing raw MTH5 was left unchanged.",
            ]
            rp = outdir / "V29_official_calibration_report.txt"
            rp.write_text("\n".join(report), encoding="utf-8")
            self.log("\n" + "\n".join(report))
            return False, "\n".join(report)

        ok, attempts = self._v29_try_official_phoenix_import(
            data_path, calibrated_out, receiver_dict, sensor_dict
        )

        ap = outdir / "V29_official_api_attempts.json"
        ap.write_text(json.dumps(attempts, indent=2, ensure_ascii=False), encoding="utf-8")

        verify = self._v29_verify_mth5_filters(calibrated_out) if calibrated_out.exists() else {
            "file": str(calibrated_out), "error": "calibrated output not generated"
        }
        vp = outdir / "V29_mth5_filter_verification.json"
        vp.write_text(json.dumps(verify, indent=2, ensure_ascii=False), encoding="utf-8")

        report += [
            "",
            f"Official API import success : {ok}",
            f"Calibrated output            : {calibrated_out if calibrated_out.exists() else 'NOT CREATED'}",
            f"Real filter objects          : {len(verify.get('real_filter_objects', []))}",
            f"All Ex/Ey/Hx/Hy/Hz have refs : {verify.get('all_channels_with_refs', False)}",
        ]

        scientifically_valid = bool(
            ok
            and calibrated_out.exists()
            and verify.get("real_filter_objects")
            and verify.get("all_channels_with_refs")
        )

        report += [
            "",
            f"V29 SCIENTIFIC VALIDATION: {'PASS' if scientifically_valid else 'FAIL'}",
        ]
        if not scientifically_valid:
            report += [
                "The raw MTH5 remains the authoritative output.",
                "The official-calibrated candidate is NOT promoted because filter objects/channel references were not fully verified.",
            ]
        else:
            report += [
                "Official EMpower JSON calibration is present in MTH5 and channel filter references were verified.",
                "Use the *_official_calibrated.mth5 file for Aurora/response-removal testing.",
            ]

        rp = outdir / "V29_official_calibration_report.txt"
        rp.write_text("\n".join(report), encoding="utf-8")
        pointer = out_mth5.with_suffix(".V29_official_calibration_location.txt")
        pointer.write_text(str(outdir), encoding="utf-8")

        self.log("\n" + "\n".join(report))
        return scientifically_valid, "\n".join(report)

    # ------------------------------------------------------------------
    # V30 RXCAL AMP-PHASE (RADIAN) FORMULA RESOLVER
    # ------------------------------------------------------------------
    @staticmethod
    def _v28_unwrap_deg(phases):
        """Simple phase unwrap in degrees."""
        import math
        out = []
        if not phases:
            return out
        prev = float(phases[0])
        out.append(prev)
        for p in phases[1:]:
            p = float(p)
            while p - prev > 180.0:
                p -= 360.0
            while p - prev < -180.0:
                p += 360.0
            out.append(p)
            prev = p
        return out

    @staticmethod
    def _v28_transform_complex(z, mode):
        if mode == "z":
            return z
        if mode == "inv_z":
            return 1.0/z if z != 0 else complex(float("nan"), float("nan"))
        if mode == "neg_z":
            return -z
        if mode == "conj_z":
            return z.conjugate()
        if mode == "inv_conj_z":
            zc = z.conjugate()
            return 1.0/zc if zc != 0 else complex(float("nan"), float("nan"))
        raise ValueError(mode)

    @staticmethod
    def _v30_metrics_from_complex(zz):
        """Return smoothness diagnostics for one candidate RXCAL response curve."""
        import math, cmath
        mags = [abs(z) for z in zz]
        phases_rad = [cmath.phase(z) for z in zz]
        phases_unwrapped_rad = list(np.unwrap(np.asarray(phases_rad, dtype=float))) if phases_rad else []
        phases_deg = [math.degrees(v) for v in phases_rad]
        phases_unwrapped_deg = [math.degrees(v) for v in phases_unwrapped_rad]

        dphi = [phases_unwrapped_deg[i+1] - phases_unwrapped_deg[i]
                for i in range(len(phases_unwrapped_deg)-1)]
        dmag = []
        for i in range(len(mags)-1):
            a, b = mags[i], mags[i+1]
            if a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b):
                dmag.append(math.log10(b) - math.log10(a))

        ratio_mag, ratio_phase_deg = [], []
        for i in range(len(zz)-1):
            a, b = zz[i], zz[i+1]
            if a == 0 or not (math.isfinite(a.real) and math.isfinite(a.imag)):
                ratio_mag.append(float("nan")); ratio_phase_deg.append(float("nan"))
            else:
                rr = b/a
                ratio_mag.append(abs(rr)); ratio_phase_deg.append(math.degrees(cmath.phase(rr)))

        return {
            "magnitude": mags,
            "phase_rad": phases_rad,
            "phase_unwrapped_rad": phases_unwrapped_rad,
            "phase_deg": phases_deg,
            "phase_unwrapped_deg": phases_unwrapped_deg,
            "phase_total_variation_deg": sum(abs(v) for v in dphi),
            "phase_max_step_deg": max([abs(v) for v in dphi], default=0.0),
            "logmag_total_variation": sum(abs(v) for v in dmag),
            "logmag_max_step": max([abs(v) for v in dmag], default=0.0),
            "ratio_magnitude": ratio_mag,
            "ratio_phase_deg": ratio_phase_deg,
        }

    @classmethod
    def _v28_analyze_table_rows(cls, rows):
        """V30 RXCAL resolver: compare Cartesian, amp/phase(rad), amp/phase(deg).

        The V27 fixed binary structure is retained.  Only the physical meaning of
        f1/f2 is tested here.  No candidate is silently promoted to an MTH5 filter.
        """
        import math
        if not rows:
            return {}

        freqs = [float(r["f0"]) for r in rows]
        f1 = [float(r["f1"]) for r in rows]
        f2 = [float(r["f2"]) for r in rows]

        # Three mutually exclusive physical interpretations.
        z_cart = [complex(a, p) for a, p in zip(f1, f2)]
        z_rad  = [complex(a*math.cos(p), a*math.sin(p)) for a, p in zip(f1, f2)]
        z_deg  = [complex(a*math.cos(math.radians(p)), a*math.sin(math.radians(p)))
                  for a, p in zip(f1, f2)]

        formulas = {
            "cartesian_re_im": cls._v30_metrics_from_complex(z_cart),
            "amp_phase_rad": cls._v30_metrics_from_complex(z_rad),
            "amp_phase_deg": cls._v30_metrics_from_complex(z_deg),
        }

        # Extra diagnostics that are directly informative for the radian hypothesis.
        finite_f2 = [v for v in f2 if math.isfinite(v)]
        rad_in_pi_fraction = (sum(abs(v) <= math.pi*1.05 for v in finite_f2)/len(finite_f2)) if finite_f2 else 0.0
        deg_in_180_fraction = (sum(abs(v) <= 180.0 for v in finite_f2)/len(finite_f2)) if finite_f2 else 0.0
        amp_positive_fraction = (sum(v > 0 for v in f1 if math.isfinite(v)) /
                                 max(1, sum(math.isfinite(v) for v in f1)))

        # Composite score: phase continuity is strongest, then magnitude smoothness.
        # A small evidence bonus is given to radian only when raw f2 actually lives
        # mostly inside +-pi; this does NOT force radian to win.
        ranking=[]
        for mode, data in formulas.items():
            score = (
                data["phase_total_variation_deg"]
                + 0.5*data["phase_max_step_deg"]
                + 20.0*data["logmag_total_variation"]
                + 10.0*data["logmag_max_step"]
            )
            if mode == "amp_phase_rad" and rad_in_pi_fraction >= 0.8:
                score *= 0.85
            ranking.append({
                "mode": mode,
                "score": score,
                "phase_total_variation_deg": data["phase_total_variation_deg"],
                "phase_max_step_deg": data["phase_max_step_deg"],
                "logmag_total_variation": data["logmag_total_variation"],
                "logmag_max_step": data["logmag_max_step"],
            })
        ranking.sort(key=lambda x: x["score"] if math.isfinite(x["score"]) else float("inf"))

        state = {}
        for key in ("f3","f4","f5","f6","f7","u32_3","u32_4","u32_5","u32_6","u32_7"):
            vals = [r.get(key) for r in rows]
            uniq=[]
            for v in vals:
                if v not in uniq:
                    uniq.append(v)
            state[key] = {"unique_count":len(uniq), "unique_values":uniq[:10], "constant":len(uniq)==1}

        return {
            "resolver_version": "V30",
            "n": len(rows),
            "frequency_hz": freqs,
            "f1_raw": f1,
            "f2_raw": f2,
            "f2_abs_max": max([abs(v) for v in finite_f2], default=float("nan")),
            "f2_within_1p05pi_fraction": rad_in_pi_fraction,
            "f2_within_180deg_fraction": deg_in_180_fraction,
            "f1_positive_fraction": amp_positive_fraction,
            "primary_hypothesis": "amp_phase_rad",
            "formulas": formulas,
            "ranking": ranking,
            "best_mode": ranking[0]["mode"] if ranking else None,
            "state_fields": state,
            "warning": "Binary RXCAL remains diagnostic-only until independently validated.",
        }

    @staticmethod
    def _v28_cross_channel_factorization(tables):
        """Compare same table index across channels/files.

        The goal is to determine whether response differences look like
        channel-specific multiplicative complex gains rather than different
        filter-state tables.
        """
        import math, cmath, statistics
        result = {}
        by_table = {}
        for item in tables:
            by_table.setdefault(item["table_index"], []).append(item)

        for tidx, group in by_table.items():
            if len(group) < 2:
                continue

            # Group by frequency, use first channel as reference.
            ref = group[0]
            ref_map = {float(r["f0"]): complex(float(r["f1"]), float(r["f2"])) for r in ref["rows"]}
            comparisons = []
            for item in group[1:]:
                ratios = []
                for r in item["rows"]:
                    f = float(r["f0"])
                    if f not in ref_map:
                        continue
                    zref = ref_map[f]
                    z = complex(float(r["f1"]), float(r["f2"]))
                    if zref == 0:
                        continue
                    ratios.append(z/zref)

                if ratios:
                    mags = [abs(x) for x in ratios]
                    phases = [math.degrees(cmath.phase(x)) for x in ratios]
                    uph = PhoenixMTU5CApp._v28_unwrap_deg(phases)
                    comparisons.append({
                        "reference": ref["label"],
                        "target": item["label"],
                        "n_common": len(ratios),
                        "ratio_mag_median": statistics.median(mags),
                        "ratio_mag_min": min(mags),
                        "ratio_mag_max": max(mags),
                        "ratio_phase_median_deg": statistics.median(uph),
                        "ratio_phase_min_deg": min(uph),
                        "ratio_phase_max_deg": max(uph),
                        "ratio_phase_span_deg": max(uph)-min(uph),
                    })
            result[str(tidx)] = comparisons
        return result

    @staticmethod
    def _v31_num(v):
        """Best-effort numeric extraction from Phoenix acquisition-state metadata."""
        import re, math
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
            return x if math.isfinite(x) else None
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(v))
        if not m:
            return None
        try:
            x = float(m.group(0))
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def _v31_recording_state(self, tag):
        info = dict(self.channel_info.get(str(tag).upper(), {}) or {})
        return {
            "tag": str(tag).upper(),
            "serial": str(info.get("serial") or info.get("ser") or ""),
            "type_name": str(info.get("type_name") or info.get("model") or ""),
            "ga_raw": info.get("ga"),
            "pg_raw": info.get("pg"),
            "lp_raw": info.get("lp"),
            "sample_rate_raw": info.get("sampleRate"),
            "ga": self._v31_num(info.get("ga")),
            "pg": self._v31_num(info.get("pg")),
            "lp": self._v31_num(info.get("lp")),
            "sample_rate_hz": self._v31_num(info.get("sampleRate")),
        }

    @classmethod
    def _v31_table_features(cls, rows, ana):
        import math, statistics
        freq = [float(r.get("f0", float("nan"))) for r in rows]
        freq = [f for f in freq if math.isfinite(f) and f > 0]
        sf = sorted(freq)
        fmin = min(sf) if sf else None
        fmax = max(sf) if sf else None
        ratios=[]
        for a,b in zip(sf[:-1],sf[1:]):
            if a>0 and b>0 and b!=a:
                ratios.append(b/a)
        best = (ana.get("ranking") or [{}])[0]
        rad = ana.get("formulas",{}).get("amp_phase_rad",{})
        return {
            "n_records": len(rows),
            "fmin_hz": fmin,
            "fmax_hz": fmax,
            "frequency_span_decades": (math.log10(fmax/fmin) if fmin and fmax and fmax>fmin else 0.0),
            "median_adjacent_frequency_ratio": statistics.median(ratios) if ratios else None,
            "f2_within_1p05pi_fraction": ana.get("f2_within_1p05pi_fraction"),
            "f1_positive_fraction": ana.get("f1_positive_fraction"),
            "best_formula": ana.get("best_mode"),
            "best_formula_score": best.get("score"),
            "rad_phase_total_variation_deg": rad.get("phase_total_variation_deg"),
            "rad_phase_max_step_deg": rad.get("phase_max_step_deg"),
            "rad_logmag_total_variation": rad.get("logmag_total_variation"),
            "state_fields": ana.get("state_fields",{}),
        }

    @classmethod
    def _v31_state_compatibility(cls, state, feat):
        """Score table compatibility without pretending undocumented GA/PG semantics are known.

        Strong evidence: frequency grid must be physically compatible with sample-rate Nyquist.
        Moderate evidence: radian-phase evidence + curve quality.
        LP/GA/PG are carried into the fingerprint, but are NOT assigned numerical response
        effects unless their Phoenix binary encoding has been independently established.
        """
        import math
        score = 0.0
        evidence=[]
        penalties=[]
        fs = state.get("sample_rate_hz")
        fmax = feat.get("fmax_hz")
        if fs and fs > 0 and fmax and fmax > 0:
            nyq = fs/2.0
            ratio = fmax/nyq
            if ratio <= 1.02:
                score += 35.0
                evidence.append(f"fmax={fmax:g} Hz is <= Nyquist={nyq:g} Hz")
                # Frequency grids ending reasonably near the usable band get a mild bonus.
                if ratio >= 0.2:
                    score += 10.0
                    evidence.append(f"fmax/Nyquist={ratio:.3g} gives useful band coverage")
            else:
                score -= 70.0
                penalties.append(f"fmax={fmax:g} Hz exceeds Nyquist={nyq:g} Hz")
        else:
            evidence.append("sample-rate/Nyquist test unavailable")

        radfrac = feat.get("f2_within_1p05pi_fraction")
        if isinstance(radfrac,(int,float)) and math.isfinite(radfrac):
            if radfrac >= 0.9:
                score += 20.0; evidence.append(f"{radfrac:.0%} of raw f2 lies inside ±1.05π")
            elif radfrac >= 0.7:
                score += 10.0; evidence.append(f"{radfrac:.0%} of raw f2 lies inside ±1.05π")
            else:
                score -= 10.0; penalties.append(f"only {radfrac:.0%} of raw f2 lies inside ±1.05π")

        if feat.get("best_formula") == "amp_phase_rad":
            score += 15.0
            evidence.append("amp_phase_rad is the smoothness winner")

        pos = feat.get("f1_positive_fraction")
        if isinstance(pos,(int,float)) and pos >= 0.95:
            score += 5.0; evidence.append("f1 is almost entirely positive (amplitude-like)")

        # Record state is deliberately reported but not over-interpreted.
        unresolved=[]
        for k in ("ga_raw","pg_raw","lp_raw"):
            if state.get(k) not in (None, ""):
                unresolved.append(f"{k[:-4].upper()}={state.get(k)}")
        if unresolved:
            evidence.append("state fingerprint present: " + ", ".join(unresolved))
            evidence.append("GA/PG/LP numerical semantics are not assumed without a validated Phoenix mapping")

        return {"score":score,"evidence":evidence,"penalties":penalties}

    @staticmethod
    def _v31_confidence_gate(candidates):
        """Return selected table only when one candidate is materially better."""
        if not candidates:
            return {"selected_table":None,"confidence":"NONE","reason":"no candidate tables"}
        c = sorted(candidates, key=lambda x:x["state_score"], reverse=True)
        top=c[0]; second=c[1] if len(c)>1 else None
        margin = top["state_score"] - (second["state_score"] if second else 0.0)
        if top["state_score"] >= 70 and (second is None or margin >= 20):
            conf="HIGH"
        elif top["state_score"] >= 45 and (second is None or margin >= 10):
            conf="MEDIUM"
        else:
            conf="LOW"
        return {
            "selected_table": top["table_index"] if conf in ("HIGH","MEDIUM") else None,
            "best_candidate_table": top["table_index"],
            "confidence": conf,
            "top_score": top["state_score"],
            "margin_to_second": margin,
            "reason": ("candidate may be used for experimental diagnostics only" if conf=="MEDIUM" else
                       "state-to-table mapping is sufficiently separated" if conf=="HIGH" else
                       "insufficient evidence to choose a binary RXCAL table safely"),
        }

    def _v31_write_state_resolver(self, out_mth5: Path):
        import csv, json, traceback

        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            self.log("V36: calibration folder is empty or unavailable.")
            return [], "NO_CALIBRATION_FOLDER"

        outdir = Path(out_mth5).with_suffix("")
        outdir = outdir.parent / (outdir.name + "_V36_fixed_rxcal_frequency_section_resolver")
        outdir.mkdir(parents=True, exist_ok=True)
        products=[]
        report=[
            "V36 PHOENIX BINARY RXCAL STATE RESOLVER",
            "="*120,
            f"MTH5              : {out_mth5}",
            f"Calibration folder: {cal}",
            f"Resolver output   : {outdir}",
            "",
            "FORMULA HYPOTHESES",
            "  cartesian_re_im : H = f1 + i f2",
            "  amp_phase_rad   : H = f1 exp(i f2)       [PRIMARY HYPOTHESIS]",
            "  amp_phase_deg   : H = f1 exp(i deg2rad(f2))",
            "",
            "STATE RESOLUTION POLICY",
            "  * sampleRate is used only through the physically mandatory Nyquist constraint.",
            "  * GA / PG / LP are recorded as acquisition-state fingerprints.",
            "  * Their undocumented numerical semantics are NOT invented or hard-coded.",
            "  * A table is selected only when the confidence gate is sufficiently separated.",
            "  * LOW confidence => no table selection and no binary calibration attachment.",
            "",
        ]

        all_tables=[]
        channel_results={}
        for p in sorted(Path(cal).rglob("*.rxcal")):
            try:
                blocks=self._v27_assign_tags_from_recording(self._v27_decode_rxcal_fixed(p))
            except Exception:
                report += [f"[RXCAL ERROR] {p.name}", traceback.format_exc()]
                continue
            report.append(f"[RXCAL] {p.name}")
            for b in blocks:
                tag=b["tag"]
                state=self._v31_recording_state(tag)
                candidates=[]
                report.append(
                    f"  {tag}: serial={state['serial'] or '?'} GA={state['ga_raw']} PG={state['pg_raw']} "
                    f"LP={state['lp_raw']} fs={state['sample_rate_raw']}"
                )
                for t in b["tables"]:
                    tid=t["table_index"]
                    label=f"{p.stem}:{tag}:T{tid}"
                    ana=self._v28_analyze_table_rows(t["rows"])
                    ana["resolver_version"]="V36"
                    feat=self._v31_table_features(t["rows"], ana)
                    comp=self._v31_state_compatibility(state,feat)
                    candidate={
                        "file":p.name,"tag":tag,"table_index":tid,
                        "recording_state":state,"features":feat,
                        "state_score":comp["score"],"evidence":comp["evidence"],"penalties":comp["penalties"],
                        "formula_analysis":ana,
                    }
                    candidates.append(candidate)
                    all_tables.append({"label":label,"file":p.name,"tag":tag,"table_index":tid,"rows":t["rows"],"analysis":ana})

                    jp=outdir/f"{p.stem}_{tag}_T{tid}_V36_analysis.json"
                    jp.write_text(json.dumps(candidate,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
                    products.append(jp)

                    cp=outdir/f"{p.stem}_{tag}_T{tid}_V36_formula_analysis.csv"
                    fields=["record_index","frequency_hz","f1","f2"]
                    for mode in ("cartesian_re_im","amp_phase_rad","amp_phase_deg"):
                        fields += [f"{mode}_mag",f"{mode}_phase_rad",f"{mode}_phase_deg",f"{mode}_phase_unwrapped_deg"]
                    with cp.open("w",newline="",encoding="utf-8-sig") as fh:
                        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
                        for i,r in enumerate(t["rows"]):
                            row={"record_index":i,"frequency_hz":r["f0"],"f1":r["f1"],"f2":r["f2"]}
                            for mode in ("cartesian_re_im","amp_phase_rad","amp_phase_deg"):
                                fm=ana["formulas"][mode]
                                row[f"{mode}_mag"]=fm["magnitude"][i]
                                row[f"{mode}_phase_rad"]=fm["phase_rad"][i]
                                row[f"{mode}_phase_deg"]=fm["phase_deg"][i]
                                row[f"{mode}_phase_unwrapped_deg"]=fm["phase_unwrapped_deg"][i]
                            w.writerow(row)
                    products.append(cp)

                    report.append(
                        f"    T{tid}: score={comp['score']:.1f}, f={feat['fmin_hz']}..{feat['fmax_hz']} Hz, "
                        f"best_formula={feat['best_formula']}, radian_fraction={feat['f2_within_1p05pi_fraction']}"
                    )

                gate=self._v31_confidence_gate(candidates)
                channel_results[f"{p.name}:{tag}"]={"recording_state":state,"gate":gate,"candidates":candidates}
                report.append(
                    f"    => best candidate T{gate.get('best_candidate_table')} | selected={gate.get('selected_table')} | "
                    f"confidence={gate['confidence']} | margin={gate.get('margin_to_second')}"
                )
                report.append(f"       {gate['reason']}")

        summary={
            "resolver_version":"V36",
            "primary_formula_hypothesis":"amp_phase_rad",
            "policy":"Binary RXCAL remains experimental; LOW confidence never selects a table or attaches calibration.",
            "channels":channel_results,
        }
        sp=outdir/"V36_fixed_rxcal_frequency_section_summary.json"
        sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
        products.append(sp)

        # Keep cross-channel diagnostics, but evaluate the raw table identity only; it is evidence, not specification.
        factor=self._v28_cross_channel_factorization(all_tables)
        fp=outdir/"V36_cross_channel_factorization.json"
        fp.write_text(json.dumps(factor,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
        products.append(fp)

        report += [
            "", "-"*120, "V36 SAFETY / STATUS",
            "  * V30 stale CSV mode-name bug is fixed.",
            "  * Binary .rxcal works even when rxcal.json is absent.",
            "  * amp_phase_rad remains a hypothesis and is tested against the other two formula interpretations.",
            "  * sampleRate/Nyquist is used as hard physical state evidence.",
            "  * GA/PG/LP values are preserved and reported, but their binary semantics are not fabricated.",
            "  * MEDIUM confidence is experimental only; HIGH still requires external validation before scientific production use.",
            "  * LOW confidence attaches NOTHING to MTH5.",
            "  * TD/BIN numerical samples remain RAW/UNCALIBRATED in the binary-RXCAL path.",
        ]
        text="\n".join(report)
        rp=outdir/"V36_fixed_rxcal_frequency_section_resolver_report.txt"
        rp.write_text(text,encoding="utf-8"); products.append(rp)
        pointer=Path(out_mth5).with_suffix(".V36_state_resolver_location.txt")
        pointer.write_text(str(outdir),encoding="utf-8"); products.append(pointer)
        self.log("\n"+text)
        self.log(f"V36 binary RXCAL state resolver folder: {outdir}")
        return products,text

    # ------------------------------------------------------------------
    # V27 FIXED-STRUCTURE RXCAL DECODER + OUTPUT-FOLDER FIX
    # ------------------------------------------------------------------
    @staticmethod
    def _v27_safe_output_dir(out_mth5: Path):
        """Create decoder folder beside the FINAL MTH5, never inside temp ASCII staging."""
        out_mth5 = Path(out_mth5)
        parent = out_mth5.parent
        stem = out_mth5.stem
        outdir = parent / f"{stem}_V27_calibration_decode"
        outdir.mkdir(parents=True, exist_ok=True)
        # Verify directory really exists and is writable.
        probe = outdir / ".v27_write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
        return outdir

    @staticmethod
    def _v27_fixed_tables_for_block(block_start: int):
        """Validated RXCAL structure from many Phoenix receiver files.

        Per 1464-byte channel block:
          table 0 = +84,   13 x 32-byte records
          table 1 = +512,   7 x 32-byte records
          table 2 = +812,   5 x 32-byte records
          table 3 = +1144,  5 x 32-byte records
        """
        return [
            {"table_index": 0, "rel_offset": 84,   "n_records": 13},
            {"table_index": 1, "rel_offset": 512,  "n_records": 7},
            {"table_index": 2, "rel_offset": 812,  "n_records": 5},
            {"table_index": 3, "rel_offset": 1144, "n_records": 5},
        ]

    @classmethod
    def _v27_decode_rxcal_fixed(cls, path: Path):
        """Decode RXCAL using fixed validated structure, not heuristic alignment search."""
        import re, math, struct
        blob = path.read_bytes()
        anchors = [m.start() for m in re.finditer(b"BCM05-B", blob)]
        blocks = []

        for i, a in enumerate(anchors):
            block_start = max(0, a - 10)
            block_end = (anchors[i+1] - 10) if i + 1 < len(anchors) else len(blob)
            if block_end - block_start < 1200:
                continue

            # First 180 bytes provide board serial / tag-like strings.
            chunk = blob[block_start:block_end]
            strings = []
            for m in re.finditer(rb"[\x20-\x7e]{3,}", chunk[:180]):
                strings.append((block_start + m.start(), m.group().decode("ascii", errors="replace")))

            serial_hex = ""
            for _, txt in strings:
                mm = re.search(r"\b([0-9A-Fa-f]{6})\b", txt)
                if mm:
                    serial_hex = mm.group(1).upper()
                    break

            tables = []
            for td in cls._v27_fixed_tables_for_block(block_start):
                start = block_start + td["rel_offset"]
                rows = []
                for j in range(td["n_records"]):
                    off = start + 32*j
                    if off + 32 > block_end:
                        break

                    f8 = tuple(float(v) for v in struct.unpack_from("<8f", blob, off))
                    u8 = tuple(int(v) for v in struct.unpack_from("<8I", blob, off))

                    # Candidate interpretation:
                    # f0,f1,f2 are preserved exactly; derived complex diagnostics
                    # are computed only as diagnostics, never asserted as Phoenix spec.
                    freq = f8[0]
                    re_c = f8[1]
                    im_c = f8[2]
                    mag = math.hypot(re_c, im_c) if math.isfinite(re_c) and math.isfinite(im_c) else float("nan")
                    ph = math.degrees(math.atan2(im_c, re_c)) if math.isfinite(re_c) and math.isfinite(im_c) else float("nan")

                    rows.append({
                        "record_index": j,
                        "offset": off,
                        "f0": f8[0], "f1": f8[1], "f2": f8[2], "f3": f8[3],
                        "f4": f8[4], "f5": f8[5], "f6": f8[6], "f7": f8[7],
                        "u32_0": u8[0], "u32_1": u8[1], "u32_2": u8[2], "u32_3": u8[3],
                        "u32_4": u8[4], "u32_5": u8[5], "u32_6": u8[6], "u32_7": u8[7],
                        "frequency_candidate_hz": freq,
                        "re_candidate": re_c,
                        "im_candidate": im_c,
                        "magnitude_candidate": mag,
                        "phase_candidate_deg": ph,
                    })

                tables.append({
                    **td,
                    "start_offset": start,
                    "rows": rows,
                })

            blocks.append({
                "block_index": i,
                "start_offset": block_start,
                "end_offset": block_end,
                "size": block_end-block_start,
                "serial_hex": serial_hex,
                "header_strings": strings,
                "tables": tables,
            })
        return blocks

    def _v27_assign_tags_from_recording(self, blocks):
        """Assign block tags using actual recording-board serial strings, then fallback by block order."""
        serial_to_tag = {}
        try:
            for tag, info in self.channel_info.items():
                ser = str((info or {}).get("ser") or "").strip().upper()
                if ser:
                    serial_to_tag[ser] = str(tag).upper()
        except Exception:
            pass

        fallback = ["E1","E2","H1","H2","H3"]
        for i, b in enumerate(blocks):
            tag = serial_to_tag.get(str(b.get("serial_hex","")).upper(), "")
            if not tag and i < len(fallback):
                tag = fallback[i]
                b["tag_source"] = "block_order_fallback"
            else:
                b["tag_source"] = "recording_serial_match" if tag else "unknown"
            b["tag"] = tag or f"B{i}"
        return blocks

    @staticmethod
    def _v27_table_stats(rows):
        import math
        if not rows:
            return {}
        vals = {}
        for key in ["f0","f1","f2","f3","f4","f5","f6","f7",
                    "frequency_candidate_hz","magnitude_candidate","phase_candidate_deg"]:
            x = [r[key] for r in rows if isinstance(r.get(key), (int,float)) and math.isfinite(r[key])]
            if not x:
                continue
            vals[key] = {
                "min": min(x),
                "max": max(x),
                "mean": sum(x)/len(x),
                "n": len(x),
            }
        # Check whether f0 looks physically like a frequency sequence.
        f = [r["f0"] for r in rows if isinstance(r.get("f0"), (int,float)) and math.isfinite(r["f0"])]
        if len(f) >= 2:
            vals["f0_frequency_quality"] = {
                "positive_fraction": sum(v > 0 for v in f)/len(f),
                "nonincreasing_fraction": sum(f[k+1] <= f[k]*1.05 for k in range(len(f)-1))/max(1,len(f)-1),
                "range_hz": [min(f), max(f)],
            }
        return vals

    def _v27_write_fixed_decode(self, out_mth5: Path):
        """Write fixed-structure RXCAL/SCAL diagnostics beside the final MTH5."""
        import csv, json, traceback

        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            self.log("V27: calibration folder is empty or unavailable; decoder folder not created.")
            return [], "NO_CALIBRATION_FOLDER"

        outdir = self._v27_safe_output_dir(Path(out_mth5))
        products = []

        report = [
            "V27 FIXED-STRUCTURE PHOENIX RXCAL DECODER",
            "="*120,
            f"MTH5              : {out_mth5}",
            f"Calibration folder: {cal}",
            f"Decoder output    : {outdir}",
            "",
            "FIXED RXCAL STRUCTURE",
            "  Channel block size = 1464 bytes",
            "  Table 0 = block + 84 bytes,   13 records",
            "  Table 1 = block + 512 bytes,   7 records",
            "  Table 2 = block + 812 bytes,   5 records",
            "  Table 3 = block + 1144 bytes,  5 records",
            "  Record size = 32 bytes = 8 x float32 = 8 x uint32 bit patterns",
            "",
        ]

        # Recording state / geometry.
        report.append("ACTUAL RECORDING STATE")
        for tag in ("E1","E2","H1","H2","H3"):
            info = self.channel_info.get(tag, {}) or {}
            if tag.startswith("E"):
                az, dlen, src = self._v26_electric_geometry(tag)
                geom = f", dipole={dlen} m, az={az} deg, geom_source={src}"
            else:
                geom = ""
            report.append(
                f"  {tag}: ser={info.get('ser','')}, serial={info.get('serial','')}, "
                f"type={info.get('type_name','')}, ga={info.get('ga','')}, "
                f"pg={info.get('pg','')}, lp={info.get('lp','')}, "
                f"sampleRate={info.get('sampleRate','')}{geom}"
            )

        report.append("")

        rxfiles = sorted(Path(cal).rglob("*.rxcal"))
        for p in rxfiles:
            try:
                blocks = self._v27_assign_tags_from_recording(self._v27_decode_rxcal_fixed(p))
            except Exception:
                report.append(f"[RXCAL ERROR] {p.name}")
                report.append(traceback.format_exc())
                continue

            report.append(f"[RXCAL] {p.name} | blocks={len(blocks)}")
            for b in blocks:
                tag = b["tag"]
                report.append(
                    f"  block {b['block_index']}: tag={tag}, source={b['tag_source']}, "
                    f"serial_hex={b.get('serial_hex') or '?'}, "
                    f"offset={b['start_offset']}..{b['end_offset']-1}, size={b['size']}"
                )

                for t in b["tables"]:
                    csvp = outdir / f"{p.stem}_RXCAL_{tag}_T{t['table_index']}_FIXED.csv"
                    fields = [
                        "record_index","offset",
                        "f0","f1","f2","f3","f4","f5","f6","f7",
                        "u32_0","u32_1","u32_2","u32_3","u32_4","u32_5","u32_6","u32_7",
                        "frequency_candidate_hz","re_candidate","im_candidate",
                        "magnitude_candidate","phase_candidate_deg",
                    ]
                    with csvp.open("w", newline="", encoding="utf-8-sig") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        w.writerows(t["rows"])
                    products.append(csvp)

                    stats = self._v27_table_stats(t["rows"])
                    statp = outdir / f"{p.stem}_RXCAL_{tag}_T{t['table_index']}_stats.json"
                    statp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
                    products.append(statp)

                    fq = stats.get("f0_frequency_quality", {})
                    report.append(
                        f"    T{t['table_index']}: +{t['rel_offset']} bytes, rows={len(t['rows'])}, "
                        f"f0 range={fq.get('range_hz','?')}, "
                        f"positive={fq.get('positive_fraction','?')}, "
                        f"nonincreasing={fq.get('nonincreasing_fraction','?')}"
                    )

        # SCAL decode remains V25 parser, but write into V27 folder too.
        scal_files = sorted(Path(cal).rglob("*.scal"))
        for p in scal_files:
            try:
                tables = self._v25_decode_scal(p)
            except Exception:
                continue
            for t in tables:
                csvp = outdir / f"{p.stem}_SCAL_T{t['table_index']}_{t['kind']}.csv"
                with csvp.open("w", newline="", encoding="utf-8-sig") as f:
                    fields = ["offset","frequency_hz","value1","value2",
                              "amplitude_candidate","phase_deg_candidate"]
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(t["rows"])
                products.append(csvp)
            if tables:
                report.append(f"[SCAL] {p.name}: {len(tables)} table(s) written")

        report += [
            "",
            "-"*120,
            "V27 SAFETY / STATUS",
            "  * Heuristic alignment search is DISABLED for the scientific decoder.",
            "  * RXCAL uses the fixed +84/+512/+812/+1144 table structure.",
            "  * f0/f1/f2 are still candidate semantic labels; all 8 raw float32 values are preserved.",
            "  * TD/BIN numerical samples remain RAW/UNCALIBRATED.",
            "  * No uncertain binary response is attached to MTH5 filters yet.",
            f"  * Output folder successfully created: {outdir}",
        ]

        text = "\n".join(report)
        reportp = outdir / "V27_fixed_decoder_report.txt"
        reportp.write_text(text, encoding="utf-8")
        products.append(reportp)

        # Easy-to-find pointer beside MTH5.
        pointer = Path(out_mth5).with_suffix(".V27_decoder_location.txt")
        pointer.write_text(str(outdir), encoding="utf-8")
        products.append(pointer)

        self.log("\n" + text)
        self.log(f"V27 decoder folder: {outdir}")
        return products, text

    # ------------------------------------------------------------------
    # V26 FULL RXCAL BLOCK DECODER + ELECTRIC GEOMETRY FIX
    # ------------------------------------------------------------------
    def _v26_electric_geometry(self, tag: str):
        """Return (azimuth_deg, total_length_m, source).

        Priority:
          1) recording metadata (actual acquisition state)
          2) schedule/config metadata
          3) explicit dipole coordinates if present

        Phoenix length1/length2 are the two electrode-arm distances from the
        recorder reference.  Total dipole length is therefore length1+length2.
        """
        tag = str(tag).upper()
        # 1. Actual recording metadata is authoritative.
        info = self.channel_info.get(tag, {}) or {}
        try:
            l1 = float(info.get("length1"))
            l2 = float(info.get("length2"))
            if l1 >= 0 and l2 >= 0 and (l1 + l2) > 0:
                az = 0.0 if tag == "E1" else 90.0 if tag == "E2" else None
                return az, l1 + l2, "recmeta.length1+length2"
        except Exception:
            pass

        # 2. Schedule/config length1 + length2.
        try:
            for c in self.config.get("config", [{}])[0].get("chanconfig", []):
                if str(c.get("tag", "")).upper() != tag:
                    continue
                l1 = float(c.get("length1"))
                l2 = float(c.get("length2"))
                if l1 >= 0 and l2 >= 0 and (l1 + l2) > 0:
                    az = 0.0 if tag == "E1" else 90.0 if tag == "E2" else None
                    return az, l1 + l2, "config.length1+length2"
        except Exception:
            pass

        # 3. Existing coordinate-based helper.
        try:
            az, dlen = dipole_azimuth_and_length(self.config, tag)
            if dlen is not None and float(dlen) > 0:
                return az, float(dlen), "config.dipole_coordinates"
        except Exception:
            pass
        return None, None, "not_found"

    @staticmethod
    def _v26_unpack_8f(blob: bytes, off: int):
        import struct
        return tuple(float(v) for v in struct.unpack_from("<8f", blob, off))

    @staticmethod
    def _v26_unpack_8u32(blob: bytes, off: int):
        import struct
        return tuple(int(v) for v in struct.unpack_from("<8I", blob, off))

    @classmethod
    def _v26_decode_rxcal_full_blocks(cls, path: Path):
        """Decode every 32-byte record inside each RXCAL channel block.

        The five repeated BCM05-B anchors define channel-block boundaries.
        Each 32-byte record is exported as both 8 little-endian float32 values
        and 8 uint32 bit patterns.  This is intentionally lossless/diagnostic:
        fields that are integer enums or bitfields are not mis-labelled as floats.
        """
        import re, math
        blob = path.read_bytes()
        anchors = [m.start() for m in re.finditer(b"BCM05-B", blob)]
        blocks = []
        for i, a in enumerate(anchors):
            start = max(0, a - 10)
            stop = (anchors[i+1] - 10) if i + 1 < len(anchors) else len(blob)
            chunk = blob[start:stop]

            # Header strings from first 180 bytes.
            strings = []
            for m in re.finditer(rb"[\x20-\x7e]{3,}", chunk[:180]):
                strings.append((start + m.start(), m.group().decode("ascii", errors="replace")))

            # Known serial-string seen in Phoenix channel board metadata.
            serial_hex = ""
            for _, txt in strings:
                mm = re.search(r"\b([0-9A-Fa-f]{6})\b", txt)
                if mm:
                    serial_hex = mm.group(1).upper()
                    break

            # Map observed board serial string to tag using current recording metadata.
            tag = ""
            for t, info in getattr(cls, "_dummy", {}).items():
                pass
            # Fallback: infer by exact serial string outside this class later.

            # The first response record in supplied RXCAL starts 84 bytes after block start.
            # Do not blindly assume all bytes are response records: scan all possible 32-byte
            # alignments and choose the alignment with the strongest frequency-like column.
            best = None
            for rel0 in range(0, min(160, max(0, stop-start-32)), 4):
                rows = []
                off = start + rel0
                while off + 32 <= stop:
                    f8 = cls._v26_unpack_8f(blob, off)
                    u8 = cls._v26_unpack_8u32(blob, off)
                    rows.append((off, f8, u8))
                    off += 32
                if len(rows) < 5:
                    continue

                # Score each of 8 columns as possible frequency column.
                col_scores = []
                for col in range(8):
                    vals = [r[1][col] for r in rows]
                    finite = [v for v in vals if math.isfinite(v)]
                    posfreq = [v for v in finite if 0.01 <= v <= 200000.0]
                    if not posfreq:
                        col_scores.append((0.0, col))
                        continue
                    # Prefer many plausible frequencies and recognizable Phoenix rates.
                    score = len(posfreq)
                    score += 4 * sum(
                        any(abs(v-f0) <= max(1e-3, abs(f0)*1e-6)
                            for f0 in (10240,9600,9216,8192,7680,6400,6144,5120,
                                       4096,3200,2048,1440,1024,512,256,150,120,
                                       90,72,60,45,30,24,15,10,7.8,7,4.2,2,1.4))
                        for v in posfreq
                    )
                    col_scores.append((score, col))
                score, freq_col = max(col_scores)
                candidate = (score, rel0, freq_col, rows)
                if best is None or candidate[0] > best[0]:
                    best = candidate

            if best is None:
                best_score, rel0, freq_col, rows = 0, 0, 0, []
            else:
                best_score, rel0, freq_col, rows = best

            blocks.append({
                "block_index": i,
                "start_offset": start,
                "end_offset": stop,
                "size": stop-start,
                "serial_hex": serial_hex,
                "header_strings": strings,
                "alignment_rel": rel0,
                "frequency_column_candidate": freq_col,
                "alignment_score": best_score,
                "rows": rows,
            })
        return blocks

    def _v26_assign_rxcal_tags(self, blocks):
        """Assign RXCAL blocks to E1/E2/H1/H2/H3 using real recording metadata."""
        serial_to_tag = {}
        for tag, info in self.channel_info.items():
            ser = str(info.get("ser") or "").strip().upper()
            if ser:
                serial_to_tag[ser] = tag.upper()

        # Expected serial strings from actual supplied recording:
        # E1 03276F, E2 0326F2, H1 03277A, H2 032760, H3 0326F0.
        for b in blocks:
            b["tag"] = serial_to_tag.get(str(b.get("serial_hex","")).upper(), "")
        return blocks

    def _v26_write_full_rxcal_decode(self, out_mth5: Path):
        import csv, json, math
        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            return [], "V26: no calibration folder selected."

        outdir = Path(out_mth5).with_suffix("")
        outdir = outdir.parent / (outdir.name + "_V26_full_rxcal_decode")
        outdir.mkdir(parents=True, exist_ok=True)

        products = []
        lines = [
            "V26 FULL RXCAL BLOCK DECODER + GEOMETRY AUDIT",
            "="*118,
            f"Calibration folder: {cal}",
            f"Output folder     : {outdir}",
            "",
            "ACTUAL RECORDING CHANNEL STATE",
        ]
        for tag in ("E1","E2","H1","H2","H3"):
            info = self.channel_info.get(tag, {}) or {}
            az, dlen, src = self._v26_electric_geometry(tag) if tag.startswith("E") else (None,None,"")
            lines.append(
                f"  {tag}: ser={info.get('ser','')} serial={info.get('serial','')} "
                f"type={info.get('type_name','')} ga={info.get('ga','')} pg={info.get('pg','')} "
                f"lp={info.get('lp','')} fs={info.get('sampleRate','')} "
                + (f"dipole={dlen:g} m az={az:g} deg source={src}" if dlen else "")
            )

        for p in sorted(Path(cal).rglob("*.rxcal")):
            blocks = self._v26_assign_rxcal_tags(self._v26_decode_rxcal_full_blocks(p))
            lines += ["", f"[RXCAL] {p.name}: {len(blocks)} repeated channel blocks"]
            for b in blocks:
                tag = b.get("tag") or f"B{b['block_index']}"
                lines.append(
                    f"  block {b['block_index']}: tag={tag}, serial_hex={b.get('serial_hex') or '?'}, "
                    f"offset={b['start_offset']}..{b['end_offset']-1}, size={b['size']}, "
                    f"best 32-byte alignment=+{b['alignment_rel']} bytes, "
                    f"frequency-column candidate=f{b['frequency_column_candidate']}, "
                    f"score={b['alignment_score']}"
                )

                csvp = outdir / f"{p.stem}_RXCAL_{tag}_FULL_8FLOAT.csv"
                fields = ["record_index","offset"]
                fields += [f"f{i}" for i in range(8)]
                fields += [f"u32_{i}" for i in range(8)]
                fields += ["frequency_candidate","complex_mag_f1_f2","complex_phase_f1_f2_deg"]
                with csvp.open("w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for j,(off,f8,u8) in enumerate(b["rows"]):
                        row = {"record_index":j,"offset":off}
                        row.update({f"f{i}":f8[i] for i in range(8)})
                        row.update({f"u32_{i}":u8[i] for i in range(8)})
                        fc = f8[b["frequency_column_candidate"]]
                        row["frequency_candidate"] = fc if math.isfinite(fc) else ""
                        row["complex_mag_f1_f2"] = math.hypot(f8[1],f8[2]) if math.isfinite(f8[1]) and math.isfinite(f8[2]) else ""
                        row["complex_phase_f1_f2_deg"] = math.degrees(math.atan2(f8[2],f8[1])) if math.isfinite(f8[1]) and math.isfinite(f8[2]) else ""
                        w.writerow(row)
                products.append(csvp)

                metap = outdir / f"{p.stem}_RXCAL_{tag}_blockmeta.json"
                metap.write_text(json.dumps({
                    k:v for k,v in b.items() if k != "rows"
                }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                products.append(metap)

        lines += [
            "",
            "-"*118,
            "V26 STATUS",
            "  * RXCAL channel-block matching now uses ACTUAL recmeta board serial strings.",
            "  * Every 32-byte candidate record is exported losslessly as 8 float32 + 8 uint32 bit patterns.",
            "  * The best candidate frequency column is reported per block rather than hard-coded.",
            "  * E1/E2 dipole_length is fixed from actual recording length1+length2 when available.",
            "  * Numerical TD/BIN samples remain RAW/UNCALIBRATED.",
            "  * No uncertain RXCAL/SCAL transfer function is yet attached as an MTH5 filter.",
        ]
        text = "\n".join(lines)
        rp = outdir / "V26_full_decoder_report.txt"
        rp.write_text(text, encoding="utf-8")
        products.append(rp)
        self.log("\n" + text)
        return products, text

    # ------------------------------------------------------------------
    # V25 PHOENIX BINARY CALIBRATION DECODER
    # ------------------------------------------------------------------
    @staticmethod
    def _v25_f32(blob, off):
        import struct
        return float(struct.unpack_from("<f", blob, off)[0])

    @classmethod
    def _v25_find_32byte_response_runs(cls, blob: bytes, start=0, stop=None):
        """Detect repeated 32-byte response records.

        Phoenix SCAL/RXCAL examples contain records in which the first three
        little-endian float32 values behave as:
            frequency, response_value_1, response_value_2
        Records are accepted only when:
          * frequency is finite, positive and <= 200 kHz
          * frequency generally descends
          * at least 5 consecutive 32-byte records are present
        This avoids converting arbitrary binary bytes into calibration tables.
        """
        import math
        if stop is None:
            stop = len(blob)
        runs = []
        off = start - (start % 4)
        candidates = []
        while off + 12 <= stop:
            try:
                f = cls._v25_f32(blob, off)
                a = cls._v25_f32(blob, off + 4)
                p = cls._v25_f32(blob, off + 8)
            except Exception:
                off += 4
                continue
            ok = (
                math.isfinite(f) and math.isfinite(a) and math.isfinite(p)
                and 0.001 <= f <= 200000.0
                and abs(a) <= 1e8 and abs(p) <= 1e8
            )
            if ok:
                candidates.append((off, f, a, p))
            off += 4

        # Start only at alignments that continue every 32 bytes.
        by_off = {x[0]: x for x in candidates}
        visited = set()
        for off0, *_ in candidates:
            if off0 in visited:
                continue
            run = []
            off = off0
            prev_f = None
            while off in by_off:
                row = by_off[off]
                f = row[1]
                # Frequencies may repeat at table boundaries; inside a run
                # require non-increasing values with no wild upward jump.
                if prev_f is not None and f > prev_f * 1.05:
                    break
                run.append(row)
                visited.add(off)
                prev_f = f
                off += 32
            if len(run) >= 5:
                runs.append(run)

        # Keep maximal non-overlapping runs.
        runs.sort(key=lambda r: (r[0][0], -len(r)))
        maximal = []
        last_end = -1
        for r in runs:
            rs, re_ = r[0][0], r[-1][0] + 32
            if rs >= last_end:
                maximal.append(r)
                last_end = re_
        return maximal

    @classmethod
    def _v25_decode_scal(cls, path: Path):
        """Decode validated SCAL FAP-style runs.

        In the supplied SCAL sample a 32-byte record sequence begins with
        frequencies 10240, 9600, 9216, ... and the next two float32 values
        form a smooth amplitude/phase response.  V25 labels these columns
        conservatively as amplitude_candidate and phase_deg_candidate.
        """
        blob = path.read_bytes()
        runs = cls._v25_find_32byte_response_runs(blob)
        out = []
        for i, run in enumerate(runs):
            freqs = [r[1] for r in run]
            vals1 = [r[2] for r in run]
            vals2 = [r[3] for r in run]
            # Strong SCAL/FAP signature: positive amplitude-like column and
            # phase-like column mostly within +/- 180 deg.
            positive_fraction = sum(v > 0 for v in vals1) / len(vals1)
            phase_fraction = sum(abs(v) <= 180 for v in vals2) / len(vals2)
            kind = "FAP_CANDIDATE" if positive_fraction > 0.8 and phase_fraction > 0.9 else "RESPONSE_CANDIDATE"
            out.append({
                "table_index": i,
                "kind": kind,
                "start_offset": run[0][0],
                "end_offset": run[-1][0] + 32,
                "rows": [
                    {
                        "offset": off,
                        "frequency_hz": f,
                        "value1": a,
                        "value2": p,
                        "amplitude_candidate": a if kind == "FAP_CANDIDATE" else None,
                        "phase_deg_candidate": p if kind == "FAP_CANDIDATE" else None,
                    }
                    for off, f, a, p in run
                ],
            })
        return out

    @classmethod
    def _v25_rxcal_channel_blocks(cls, path: Path):
        """Detect RXCAL channel blocks from repeated BCM05-B anchors.

        The supplied RXCAL contains five repeated channel blocks separated by
        1464 bytes.  V25 does not hard-code 1464: it derives each block from the
        repeated 'BCM05-B' header anchors and nearby serial/tag strings.
        """
        blob = path.read_bytes()
        anchors = [m.start() for m in re.finditer(b"BCM05-B", blob)]
        blocks = []
        # First BCM05-B at ~266 belongs to channel block 1; subsequent anchors
        # delimit later channel blocks.
        for i, a in enumerate(anchors):
            # Header starts 10 bytes before BCM05-B in supplied file.
            start = max(0, a - 10)
            stop = (anchors[i+1] - 10) if i + 1 < len(anchors) else len(blob)
            chunk = blob[start:stop]

            # Extract printable strings for identity.
            strings = []
            for m in re.finditer(rb"[\x20-\x7e]{3,}", chunk[:160]):
                strings.append((start + m.start(), m.group().decode("ascii", errors="replace")))

            # Look for E1/E2/H1/H2/H3 in first 160 bytes.
            tag = ""
            for _, txt in strings:
                mm = re.search(r"\b([EH][123])\b", txt)
                if mm:
                    tag = mm.group(1)
                    break
            # Phoenix files may embed tag in a longer printable string.
            if not tag:
                txtall = " ".join(x[1] for x in strings)
                mm = re.search(r"([EH][123])", txtall)
                if mm:
                    tag = mm.group(1)

            runs = cls._v25_find_32byte_response_runs(blob, start, stop)
            tables = []
            for j, run in enumerate(runs):
                # RXCAL values often look like two Cartesian components.  Provide
                # derived magnitude/phase as diagnostics, but do not claim this is
                # the final documented Phoenix response until independently verified.
                rows = []
                import math
                for off, f, x, y in run:
                    mag = math.hypot(x, y)
                    phase = math.degrees(math.atan2(y, x))
                    rows.append({
                        "offset": off,
                        "frequency_hz": f,
                        "x_candidate": x,
                        "y_candidate": y,
                        "magnitude_candidate": mag,
                        "phase_deg_candidate": phase,
                    })
                tables.append({
                    "table_index": j,
                    "kind": "COMPLEX_RESPONSE_CANDIDATE",
                    "start_offset": run[0][0],
                    "end_offset": run[-1][0] + 32,
                    "rows": rows,
                })

            blocks.append({
                "block_index": i,
                "start_offset": start,
                "end_offset": stop,
                "size": stop-start,
                "tag": tag,
                "header_strings": strings,
                "tables": tables,
            })
        return blocks

    def _v25_write_decoded_csvs(self, out_mth5: Path):
        """Decode all selected binary calibration files and write CSV diagnostics."""
        import csv
        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            return [], "V25: no calibration folder selected."

        outdir = Path(out_mth5).with_suffix("")
        outdir = outdir.parent / (outdir.name + "_V25_calibration_decode")
        outdir.mkdir(parents=True, exist_ok=True)

        lines = [
            "V25 PHOENIX BINARY CALIBRATION DECODER",
            "="*110,
            f"Calibration folder: {cal}",
            f"Decode output     : {outdir}",
            "",
        ]
        products = []

        for p in sorted(Path(cal).rglob("*")):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf == ".scal":
                tables = self._v25_decode_scal(p)
                lines.append(f"[SCAL] {p.name}: {len(tables)} response table(s)")
                for t in tables:
                    csvp = outdir / f"{p.stem}_SCAL_table{t['table_index']:02d}_{t['kind']}.csv"
                    with csvp.open("w", newline="", encoding="utf-8-sig") as f:
                        w = csv.DictWriter(f, fieldnames=[
                            "offset","frequency_hz","value1","value2",
                            "amplitude_candidate","phase_deg_candidate"
                        ])
                        w.writeheader()
                        w.writerows(t["rows"])
                    products.append(csvp)
                    lines.append(
                        f"  table {t['table_index']}: {t['kind']} "
                        f"offset={t['start_offset']} rows={len(t['rows'])} -> {csvp.name}"
                    )

            elif suf == ".rxcal":
                blocks = self._v25_rxcal_channel_blocks(p)
                lines.append(f"[RXCAL] {p.name}: {len(blocks)} channel block(s)")
                for b in blocks:
                    header_preview = "; ".join(txt for _, txt in b["header_strings"][:8])
                    lines.append(
                        f"  block {b['block_index']}: tag={b['tag'] or '?'} "
                        f"offset={b['start_offset']} size={b['size']} | {header_preview}"
                    )
                    for t in b["tables"]:
                        tag = b["tag"] or f"B{b['block_index']}"
                        csvp = outdir / (
                            f"{p.stem}_RXCAL_{tag}_table{t['table_index']:02d}_complex_candidate.csv"
                        )
                        with csvp.open("w", newline="", encoding="utf-8-sig") as f:
                            w = csv.DictWriter(f, fieldnames=[
                                "offset","frequency_hz","x_candidate","y_candidate",
                                "magnitude_candidate","phase_deg_candidate"
                            ])
                            w.writeheader()
                            w.writerows(t["rows"])
                        products.append(csvp)
                        lines.append(
                            f"    table {t['table_index']}: "
                            f"offset={t['start_offset']} rows={len(t['rows'])} -> {csvp.name}"
                        )

        lines += [
            "",
            "-"*110,
            "INTERPRETATION RULES",
            "1. SCAL smooth positive amplitude + phase within +/-180 deg is reported as FAP_CANDIDATE.",
            "2. RXCAL columns 2/3 are treated as possible Cartesian response components; V25 additionally",
            "   reports magnitude=sqrt(x^2+y^2) and phase=atan2(y,x), but does NOT yet claim that this",
            "   transform is the official Phoenix calibration formula.",
            "3. V25 still DOES NOT apply numerical calibration to TD/BIN samples.",
            "4. V25 still DOES NOT attach an uncertain binary response to MTH5 channel.filters.",
        ]
        text = "\n".join(lines)
        reportp = outdir / "V25_decoder_report.txt"
        reportp.write_text(text, encoding="utf-8")
        products.append(reportp)
        self.log("\n" + text)
        return products, text

    # ------------------------------------------------------------------
    # V24 PHOENIX BINARY RXCAL / SCAL PROBE
    # ------------------------------------------------------------------
    @staticmethod
    def _v24_ascii_strings(blob: bytes, min_len: int = 4):
        """Return printable ASCII strings with byte offsets."""
        out = []
        start = None
        for i, c in enumerate(blob + b"\x00"):
            if 32 <= c <= 126:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_len:
                    out.append((start, blob[start:i].decode("ascii", errors="replace")))
                start = None
        return out

    @staticmethod
    def _v24_plausible_float32(blob: bytes, endian="<"):
        """Collect finite float32 values without pretending their field meaning is known."""
        import struct, math
        rows = []
        fmt = endian + "f"
        for off in range(0, len(blob)-3, 4):
            try:
                v = struct.unpack_from(fmt, blob, off)[0]
            except Exception:
                continue
            if math.isfinite(v) and v != 0.0 and 1e-30 <= abs(v) <= 1e30:
                rows.append((off, float(v)))
        return rows

    def _v24_probe_binary_calibration_file(self, path: Path):
        """Inspect Phoenix binary calibration structure conservatively.

        V24 intentionally does NOT turn unknown offsets into an MTH5 response.
        It extracts reproducible evidence (headers, strings, float candidates,
        repeated-record clues) so the binary layout can be validated first.
        """
        import hashlib, struct
        blob = path.read_bytes()
        strings = self._v24_ascii_strings(blob)
        le = self._v24_plausible_float32(blob, "<")
        be = self._v24_plausible_float32(blob, ">")

        # Recognizable Phoenix identifiers/models are useful for matching.
        keywords = []
        for off, text in strings:
            up = text.upper()
            if any(k in up for k in (
                "MTC", "AMTC", "MTU", "RMT", "BCM", "BTM",
                "HWV", "FWV", "FP_", "FX"
            )):
                keywords.append((off, text))

        # Look for repeated ASCII anchors; repeated records often reveal block layout.
        repeated = {}
        for off, text in strings:
            if len(text) >= 5:
                repeated.setdefault(text, []).append(off)
        repeated = {k:v for k,v in repeated.items() if len(v) >= 2}

        return {
            "path": str(path),
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "first16_hex": blob[:16].hex(" "),
            "first4_u8": list(blob[:4]),
            "ascii_strings": strings,
            "keywords": keywords,
            "float32_le": le,
            "float32_be": be,
            "repeated_strings": repeated,
        }

    def _v24_probe_calibration_folder(self):
        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            return [], "V24: no valid calibration folder selected."
        files = sorted(
            [p for p in Path(cal).rglob("*")
             if p.is_file() and p.suffix.lower() in (".rxcal", ".scal")]
        )
        reports = []
        lines = [
            "V24 PHOENIX BINARY CALIBRATION PROBE",
            "=" * 110,
            f"Folder: {cal}",
            f"Binary calibration files: {len(files)}",
            "",
        ]
        for p in files:
            r = self._v24_probe_binary_calibration_file(p)
            reports.append(r)
            lines += [
                f"[{p.name}]",
                f"  type/size : {r['suffix']} / {r['size']:,} bytes",
                f"  SHA256    : {r['sha256']}",
                f"  first16   : {r['first16_hex']}",
                f"  first4 u8 : {r['first4_u8']}",
                "  Phoenix/header strings:",
            ]
            for off, text in r["keywords"][:40]:
                lines.append(f"    0x{off:04X} ({off:5d}) : {text}")
            if not r["keywords"]:
                lines.append("    (none recognized)")
            lines.append(
                f"  plausible float32 candidates: LE={len(r['float32_le'])}, "
                f"BE={len(r['float32_be'])}"
            )
            lines.append("  first LE float32 candidates:")
            for off, val in r["float32_le"][:24]:
                lines.append(f"    0x{off:04X} : {val:.9g}")
            if r["repeated_strings"]:
                lines.append("  repeated ASCII anchors:")
                for text, offs in list(r["repeated_strings"].items())[:20]:
                    lines.append(f"    {text!r}: {offs}")
            lines.append("")

        lines += [
            "-" * 110,
            "SAFETY STATUS:",
            "  Binary .rxcal/.scal were inspected, but V24 does NOT yet attach them as MTH5 filters.",
            "  A response is attached only after frequency/amplitude/phase or ZPK record boundaries",
            "  are independently validated. This prevents an incorrect binary offset from becoming",
            "  a scientifically wrong calibration.",
        ]
        return reports, "\n".join(lines)

    def _v24_write_probe_report(self, out_mth5: Path):
        """Write a sidecar text report next to the MTH5 and log it."""
        reports, text = self._v24_probe_calibration_folder()
        try:
            report_path = Path(out_mth5).with_suffix(".V24_calibration_probe.txt")
            report_path.write_text(text, encoding="utf-8")
            self.log(f"V24 calibration probe report: {report_path}")
        except Exception as e:
            self.log(f"V24 probe report write warning: {e}")
        self.log("\n" + text)
        return reports, text

    # ------------------------------------------------------------------
    # V23 CALIBRATION BUILDER
    # ------------------------------------------------------------------
    @staticmethod
    def _v23_scalar(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return v

    @staticmethod
    def _v23_norm_key(k: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(k).lower())

    @staticmethod
    def _v23_float_array(v):
        """Convert a JSON value to a finite 1-D float array when possible."""
        if v is None:
            return None
        if isinstance(v, dict):
            # Common wrappers used by calibration JSONs.
            for kk in ("values", "data", "value", "array"):
                if kk in v:
                    return PhoenixMTU5CApp._v23_float_array(v[kk])
            return None
        if not isinstance(v, (list, tuple, np.ndarray)):
            return None
        out = []
        for x in v:
            try:
                fx = float(x)
                if np.isfinite(fx):
                    out.append(fx)
            except Exception:
                return None
        return np.asarray(out, dtype=float) if out else None

    @classmethod
    def _v23_find_first(cls, d, aliases):
        """Recursively find the first scalar/array value whose normalized key matches."""
        aliases = {cls._v23_norm_key(x) for x in aliases}
        stack = [d]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if cls._v23_norm_key(k) in aliases:
                        return v
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend(x for x in cur if isinstance(x, (dict, list)))
        return None

    @classmethod
    def _v23_collect_candidate_records(cls, obj, source_file=""):
        """Collect JSON dictionaries that look like calibration/response records.

        This is deliberately conservative.  A record must contain at least one
        frequency/response representation OR explicit poles/zeros/coefficients.
        """
        records = []
        def walk(node, path=""):
            if isinstance(node, dict):
                keys = {cls._v23_norm_key(k) for k in node.keys()}
                has_freq = bool(keys & {
                    "frequency","frequencies","freq","freqs","hz"
                })
                has_amp = bool(keys & {
                    "amplitude","amplitudes","magnitude","magnitudes","gain","gains","response"
                })
                has_phase = bool(keys & {"phase","phases","phasedeg","phaseangle"})
                has_pz = bool(keys & {"poles","zeros","normalizationfactor","a0"})
                has_coef = bool(keys & {"coefficients","coefficient","numerator","denominator","fir"})
                if (has_freq and (has_amp or has_phase)) or has_pz or has_coef:
                    rec = dict(node)
                    rec["_v23_source_file"] = source_file
                    rec["_v23_json_path"] = path or "/"
                    records.append(rec)
                for k, v in node.items():
                    if isinstance(v, (dict, list)):
                        walk(v, f"{path}/{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    if isinstance(v, (dict, list)):
                        walk(v, f"{path}[{i}]")
        walk(obj)
        return records

    @classmethod
    def _v23_record_identity(cls, rec):
        """Extract likely sensor/receiver/channel identity from a calibration record."""
        aliases = {
            "serial": ("serial","sensorserial","serialnumber","sensorid","id","ser"),
            "model": ("model","sensormodel","modelname","typename","type_name"),
            "tag": ("tag","channel","component","channelname","input","axis"),
            "kind": ("kind","type","filtertype","responsetype"),
            "sample_rate": ("samplerate","sample_rate","samplingrate","fs","rate"),
        }
        out = {}
        for name, aa in aliases.items():
            v = cls._v23_find_first(rec, aa)
            if isinstance(v, (str, int, float, np.integer, np.floating)):
                out[name] = str(cls._v23_scalar(v)).strip()
        return out

    @classmethod
    def _v23_response_from_record(cls, rec):
        """Return normalized response dict or None.

        Supported conservative forms:
          * FAP: frequency + amplitude/magnitude + phase
          * ZPK: poles + zeros (+ gain/a0)
          * FIR/coefficients
        """
        freq = cls._v23_float_array(cls._v23_find_first(
            rec, ("frequency","frequencies","freq","freqs","hz")
        ))
        amp = cls._v23_float_array(cls._v23_find_first(
            rec, ("amplitude","amplitudes","magnitude","magnitudes","gain","gains")
        ))
        phase = cls._v23_float_array(cls._v23_find_first(
            rec, ("phase","phases","phasedeg","phase_deg","phaseangle")
        ))
        if freq is not None and amp is not None:
            n = min(len(freq), len(amp), len(phase) if phase is not None else 10**9)
            if n >= 2:
                if phase is None:
                    phase = np.zeros(n, dtype=float)
                return {
                    "type": "fap",
                    "frequency": np.asarray(freq[:n], dtype=float),
                    "amplitude": np.asarray(amp[:n], dtype=float),
                    "phase_deg": np.asarray(phase[:n], dtype=float),
                }

        poles = cls._v23_find_first(rec, ("poles",))
        zeros = cls._v23_find_first(rec, ("zeros",))
        if poles is not None or zeros is not None:
            def complex_array(v):
                if v is None:
                    return np.asarray([], dtype=np.complex128)
                out = []
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    try:
                        if isinstance(x, dict):
                            re_v = float(x.get("real", x.get("re", 0.0)))
                            im_v = float(x.get("imag", x.get("im", 0.0)))
                            out.append(complex(re_v, im_v))
                        elif isinstance(x, (list, tuple)) and len(x) >= 2:
                            out.append(complex(float(x[0]), float(x[1])))
                        else:
                            out.append(complex(x))
                    except Exception:
                        pass
                return np.asarray(out, dtype=np.complex128)
            p = complex_array(poles)
            z = complex_array(zeros)
            gain_v = cls._v23_find_first(rec, ("gain","normalizationfactor","a0","constant"))
            try:
                gain = float(gain_v) if gain_v is not None else 1.0
            except Exception:
                gain = 1.0
            if p.size or z.size:
                return {"type": "zpk", "poles": p, "zeros": z, "gain": gain}

        coef = cls._v23_float_array(cls._v23_find_first(
            rec, ("coefficients","coefficient","fir","numerator")
        ))
        if coef is not None and len(coef) >= 1:
            return {"type": "fir", "coefficients": coef}
        return None

    def _v23_scan_calibration_folder(self):
        """Parse JSON calibration files only; V24 separately probes binary .rxcal/.scal."""
        cal = self.calibration_dir.get().strip()
        out = []
        errors = []
        if not cal or not Path(cal).exists():
            return out, ["No valid calibration folder selected."]
        files = sorted(Path(cal).rglob("*.json"))
        for p in files:
            try:
                obj = load_json(p)
                for rec in self._v23_collect_candidate_records(obj, str(p)):
                    response = self._v23_response_from_record(rec)
                    if response is None:
                        continue
                    ident = self._v23_record_identity(rec)
                    out.append({
                        "source_file": str(p),
                        "json_path": rec.get("_v23_json_path","/"),
                        "identity": ident,
                        "response": response,
                    })
            except Exception as e:
                errors.append(f"{p.name}: {type(e).__name__}: {e}")
        return out, errors

    @staticmethod
    def _v23_safe_filter_name(text):
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
        return text or "phoenix_filter"

    def _v23_target_identity(self, comp, hs):
        tag = COMPONENT_TO_TAG.get(comp, comp.upper())
        info = self.channel_info.get(tag, {}) or {}
        ident = {
            "component": comp.lower(),
            "tag": tag.upper(),
            "serial": "",
            "model": "",
            "sample_rate": str(float(hs[0].sample_rate)) if hs else "",
        }
        if comp.startswith("h"):
            ident["serial"] = str(
                info.get("serial") or info.get("ser") or
                (hs[0].channel_board_serial if hs else "") or ""
            ).strip()
            ident["model"] = str(
                info.get("type_name") or info.get("model") or ""
            ).strip()
        else:
            # Electric channels are receiver/electrode paths, so receiver ID/model
            # are more relevant than a coil serial number.
            ident["serial"] = str(
                (self.recmeta or {}).get("instid") or
                (hs[0].instrument_serial if hs else "") or ""
            ).strip()
            ident["model"] = str(
                (self.recmeta or {}).get("receiver_commercial_name") or "MTU-5C"
            ).strip()
        return ident

    def _v23_score_record(self, record, target):
        """Score a calibration record against a channel; 0 means do not attach."""
        ident = record.get("identity", {})
        score = 0
        why = []
        serial = ident.get("serial","").strip().lower()
        model = ident.get("model","").strip().lower()
        tag = ident.get("tag","").strip().lower()
        rate = ident.get("sample_rate","").strip().lower()

        tserial = target.get("serial","").strip().lower()
        tmodel = target.get("model","").strip().lower()
        ttag = target.get("tag","").strip().lower()
        tcomp = target.get("component","").strip().lower()
        trate = target.get("sample_rate","").strip().lower()

        if serial and tserial and serial == tserial:
            score += 100; why.append("serial")
        if tag and tag in (ttag, tcomp):
            score += 40; why.append("tag/component")
        if model and tmodel and (model == tmodel or model in tmodel or tmodel in model):
            score += 20; why.append("model")
        if rate and trate:
            try:
                if abs(float(rate)-float(trate)) < 1e-6:
                    score += 10; why.append("sample_rate")
            except Exception:
                pass

        # Prevent magnetic calibration from attaching to electric data solely by
        # a weak generic model match, and vice versa.
        src_text = (
            " ".join(str(v) for v in ident.values()) + " " +
            Path(record.get("source_file","")).name
        ).lower()
        if tcomp.startswith("h"):
            if any(x in src_text for x in ("electric","electrode","ex","ey")) and "mtc" not in src_text:
                score -= 50
        else:
            if "mtc-" in src_text or "induction coil" in src_text:
                score -= 50
        return max(score, 0), ",".join(why)

    def _v23_choose_records_for_channel(self, records, comp, hs):
        """Choose calibration records conservatively.

        V64 safety rule for magnetic channels:
        H1/H2/H3 sensor calibration is attached ONLY when the calibration
        sensor serial exactly matches the sensor serial recorded for that channel.
        There is no positional/leftover fallback.  A missing SCAL is safer than
        silently attaching a different coil from the calibration archive.
        """
        target = self._v23_target_identity(comp, hs)
        ranked = []
        for r in records:
            sc, why = self._v23_score_record(r, target)
            if sc > 0:
                ranked.append((sc, why, r))
        ranked.sort(key=lambda x: x[0], reverse=True)

        tserial = str(target.get("serial", "") or "").strip().lower()

        # Magnetic sensor responses must match the expected sensor serial exactly.
        if str(comp).lower().startswith("h"):
            if not tserial:
                return target, []
            chosen = []
            for sc, why, r in ranked:
                rserial = str((r.get("identity", {}) or {}).get("serial", "") or "").strip().lower()
                if rserial and rserial == tserial:
                    chosen.append((sc, (why + ",exact_sensor_serial").strip(","), r))
            return target, chosen

        # Electric/receiver records may still use the existing strong identity
        # rules because they are receiver-path calibrations rather than coil files.
        chosen = []
        for sc, why, r in ranked:
            if sc >= 100 or sc >= 60:
                chosen.append((sc, why, r))
        return target, chosen

    def _v23_write_filter_object(self, h5, survey_path, filt_name, response, source_file, identity):
        """Write a conservative filter object beneath the standard MTH5 Filters groups.

        The object stores the full numerical response plus provenance.  Channel
        ``filters`` contains the symbolic filter name.  This is intentionally
        simple and inspectable, rather than inventing undocumented Phoenix math.
        """
        import h5py
        ftype = response["type"]
        base = h5.require_group(f"{survey_path}/Filters/{ftype}")
        name = self._v23_safe_filter_name(filt_name)
        if name in base:
            del base[name]
        g = base.create_group(name)
        g.attrs["name"] = name
        g.attrs["type"] = ftype
        g.attrs["phoenix_v23.source_file"] = str(source_file)
        g.attrs["phoenix_v23.identity_json"] = json.dumps(identity, ensure_ascii=False)

        if ftype == "fap":
            g.create_dataset("frequency", data=np.asarray(response["frequency"], dtype=float))
            g.create_dataset("amplitude", data=np.asarray(response["amplitude"], dtype=float))
            # Phoenix calibration phases are treated as degrees unless the JSON
            # explicitly proves otherwise.  The raw degree values are retained.
            g.create_dataset("phase_deg", data=np.asarray(response["phase_deg"], dtype=float))
            g.attrs["phase_units"] = "degrees"
        elif ftype == "zpk":
            p = np.asarray(response["poles"], dtype=np.complex128)
            z = np.asarray(response["zeros"], dtype=np.complex128)
            g.create_dataset("poles_real", data=p.real)
            g.create_dataset("poles_imag", data=p.imag)
            g.create_dataset("zeros_real", data=z.real)
            g.create_dataset("zeros_imag", data=z.imag)
            g.attrs["gain"] = float(response.get("gain",1.0))
        elif ftype == "fir":
            g.create_dataset("coefficients", data=np.asarray(response["coefficients"], dtype=float))
        return name

    def _v23_build_real_calibration_chain(self, path: Path):
        """Build and attach real response objects only when JSON records match.

        This does NOT calibrate the numerical time series.  It only puts actual
        response metadata into MTH5 and attaches symbolic filter names to channels.
        """
        import h5py

        records, parse_errors = self._v23_scan_calibration_folder()
        grouped = self._headers_by_component_rate()
        lines = [
            "V23 CALIBRATION BUILDER",
            "="*110,
            f"Usable calibration response records found: {len(records)}",
        ]
        if parse_errors:
            lines.append(f"Calibration JSON parse warnings: {len(parse_errors)}")
            lines.extend("  " + x for x in parse_errors[:10])

        attached_count = 0
        channel_count = 0

        with h5py.File(path, "r+") as h5:
            # Find the survey base(s) from channel dataset paths.
            channel_items = []
            def collect(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                comp = name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER:
                    return
                parts = name.split("/")
                try:
                    i = parts.index("Surveys")
                    survey_path = "/" + "/".join(parts[:i+2])
                except Exception:
                    return
                channel_items.append((name, obj, comp, survey_path))
            h5.visititems(collect)

            for name, obj, comp, survey_path in channel_items:
                channel_count += 1
                try:
                    fs = float(obj.attrs.get("sample_rate",0.0) or 0.0)
                except Exception:
                    fs = 0.0
                hs = grouped.get(fs,{}).get(comp,[])
                target, chosen = self._v23_choose_records_for_channel(records, comp, hs)

                if not chosen:
                    obj.attrs["phoenix_v23.calibration_chain_status"] = "NO_CONFIDENT_MATCH"
                    if comp.startswith("h"):
                        lines.append(
                            f"{comp.upper():<3} {fs:g} Hz | WARNING: NO EXACT SENSOR-SERIAL MATCH | "
                            f"expected sensor serial={target.get('serial','')} "
                            f"model={target.get('model','')} | calibration left unattached"
                        )
                    else:
                        lines.append(
                            f"{comp.upper():<3} {fs:g} Hz | NO MATCH | "
                            f"target serial={target.get('serial','')} model={target.get('model','')}"
                        )
                    continue

                filter_names = []
                # Multiple confidently matched records can represent receiver +
                # sensor responses. De-duplicate by source/json-path/type.
                seen = set()
                for sc, why, rec in chosen:
                    key = (
                        rec["source_file"], rec["json_path"],
                        rec["response"]["type"]
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    ident = rec["identity"]
                    srcstem = Path(rec["source_file"]).stem
                    serial = ident.get("serial") or target.get("serial") or comp
                    fname = self._v23_safe_filter_name(
                        f"phoenix_{srcstem}_{serial}_{rec['response']['type']}"
                    )
                    fname = self._v23_write_filter_object(
                        h5, survey_path, fname, rec["response"],
                        rec["source_file"], ident
                    )
                    filter_names.append(fname)
                    lines.append(
                        f"{comp.upper():<3} {fs:g} Hz | ATTACH {fname} | "
                        f"score={sc} ({why})"
                    )

                if filter_names:
                    dt = h5py.string_dtype("utf-8")
                    obj.attrs["filters"] = np.asarray(filter_names, dtype=dt)
                    obj.attrs["phoenix_v22.real_filter_chain_attached"] = True
                    obj.attrs["phoenix_v23.calibration_chain_status"] = "ATTACHED"
                    obj.attrs["phoenix_v23.filter_names_json"] = json.dumps(filter_names)
                    # Data are still uncalibrated numeric samples.
                    obj.attrs["phoenix_v22.calibration_applied"] = False
                    attached_count += 1

        lines += [
            "-"*110,
            f"Channel datasets found      : {channel_count}",
            f"Channels with attached chain: {attached_count}",
            (
                "RESULT: ALL CHANNEL DATASETS HAVE REAL MATCHED RESPONSE METADATA."
                if channel_count and attached_count == channel_count
                else "RESULT: CALIBRATION CHAIN IS INCOMPLETE; SEE UNMATCHED CHANNELS ABOVE."
            ),
            "NOTE: V23 attaches calibration metadata only. It does not numerically calibrate the stored samples.",
        ]
        text = "\n".join(lines)
        self.v23_last_calibration_report = text
        return {
            "records": records,
            "channel_count": channel_count,
            "attached_count": attached_count,
            "complete": bool(channel_count and attached_count == channel_count),
            "text": text,
        }

    @staticmethod
    def _v22_nonempty_filter_value(value) -> bool:
        """True only for an actually non-empty filter reference."""
        if value is None:
            return False
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            return value.size > 0 and any(
                PhoenixMTU5CApp._v22_nonempty_filter_value(x)
                for x in value.ravel().tolist()
            )
        if isinstance(value, (list, tuple)):
            return len(value) > 0 and any(
                PhoenixMTU5CApp._v22_nonempty_filter_value(x) for x in value
            )
        t = str(value).strip().lower()
        return t not in ("", "[]", "{}", "()", "none", "null", "false", "nan")

    def _v22_mark_direct_raw_channels(self, path: Path):
        """Mark direct-writer channels truthfully as uncalibrated/raw.

        No response is fabricated. BIN remains digital counts. Phoenix .td_*
        float32 samples are explicitly voltage-domain data (V).  They are NOT yet
        RXCAL/SCAL calibrated: filter.applied/calibration_applied remain False.
        Phoenix acquisition GA has already been removed before TD write.
        """
        import h5py
        grouped = self._headers_by_component_rate()
        with h5py.File(path, "r+") as h5:
            def visitor(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                comp = name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER:
                    return
                try:
                    fs = float(obj.attrs.get("sample_rate", 0.0) or 0.0)
                except Exception:
                    fs = 0.0
                hs = grouped.get(fs, {}).get(comp, [])
                is_td = bool(hs) and all(_is_td_file(h.path) for h in hs)

                if is_td:
                    # V68 confirmed interpretation: Phoenix TD numerical samples are volts.
                    # Do NOT relabel as mV/km or nT: those are post-calibration physical fields.
                    obj.attrs["units"] = "volt"
                    obj.attrs["phoenix_v22.source_units"] = "volt"
                    obj.attrs["phoenix_v68.td_domain"] = "voltage"
                    obj.attrs["phoenix_v68.ga_removed_before_td_write"] = True

                    # Record the acquisition analog low-pass state separately from
                    # the later RXCAL deconvolution.  LP was physically active during
                    # acquisition; RXCAL itself is attached later with applied=False.
                    tag = COMPONENT_TO_TAG.get(comp, comp.upper())
                    lp, lp_source = self._v66_lp_hz_for_channel(tag, fs)
                    lp_state = {10000:"T0", 1000:"T1", 100:"T2", 10:"T3"}.get(int(lp), "UNKNOWN")
                    obj.attrs["phoenix_v68.acquisition_lp_hz"] = int(lp)
                    obj.attrs["phoenix_v68.acquisition_lp_state"] = lp_state
                    obj.attrs["phoenix_v68.acquisition_lp_source"] = str(lp_source)
                    obj.attrs["phoenix_v68.analog_lp_physically_applied"] = True
                    obj.attrs["phoenix_v68.post_td_calibration"] = (
                        "M3/Aurora deconvolution: inverse RXCAL; magnetic channels also inverse SCAL"
                    )
                    if comp.startswith("e"):
                        obj.attrs["phoenix_v68.electric_td_quantity"] = "electrode potential difference"
                        obj.attrs["phoenix_v68.dipole_distance_applied_to_td"] = False
                else:
                    obj.attrs["units"] = "digital counts"
                    obj.attrs["phoenix_v22.source_units"] = "24-bit ADC digital counts"

                obj.attrs["phoenix_v22.calibration_applied"] = False
                obj.attrs["phoenix_v22.real_filter_chain_attached"] = False
                obj.attrs["phoenix_v22.source_format"] = "TD" if is_td else "BIN"

                # Never invent standard MTH5 filter references.
                if "filters" in obj.attrs and self._v22_nonempty_filter_value(obj.attrs["filters"]):
                    # Keep a real pre-existing reference if one somehow exists.
                    pass
                else:
                    try:
                        obj.attrs["filters"] = np.asarray([], dtype=h5py.string_dtype("utf-8"))
                    except Exception:
                        obj.attrs["filters"] = "[]"
            h5.visititems(visitor)

    def _v22_calibration_report(self, path: Path) -> dict:
        """Inspect whether real filter objects and channel references exist."""
        import h5py
        report = {
            "channels": [],
            "filter_objects": [],
            "all_channels_have_refs": False,
            "has_real_filter_objects": False,
        }
        with h5py.File(path, "r") as h5:
            # Look below every Filters container, but do not count the standard
            # empty category groups (coefficient/fap/fir/time_delay/zpk) themselves.
            def collect_filters(name, obj):
                parts = [p for p in name.split("/") if p]
                if "Filters" not in parts:
                    return
                idx = parts.index("Filters")
                rel = parts[idx + 1:]
                if len(rel) < 2:
                    return
                # A path such as Filters/fap/<actual-filter-name> is a real object.
                report["filter_objects"].append("/" + name)

            h5.visititems(collect_filters)
            report["filter_objects"] = sorted(set(report["filter_objects"]))
            report["has_real_filter_objects"] = bool(report["filter_objects"])

            def collect_channels(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                comp = name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER:
                    return
                fval = obj.attrs.get("filters", None)
                has_ref = self._v22_nonempty_filter_value(fval)
                units = obj.attrs.get("units", "")
                if isinstance(units, bytes):
                    units = units.decode("utf-8", errors="replace")
                report["channels"].append({
                    "component": comp,
                    "path": "/" + name,
                    "filters": str(fval),
                    "has_ref": bool(has_ref),
                    "units": str(units),
                })

            h5.visititems(collect_channels)

        report["all_channels_have_refs"] = bool(report["channels"]) and all(
            x["has_ref"] for x in report["channels"]
        )
        return report

    def _v22_render_calibration_report(self, path: Path, title="V22 CALIBRATION-CHAIN CHECK") -> str:
        r = self._v22_calibration_report(path)
        lines = [title, "=" * 110]
        lines.append(
            f"Real filter objects below /Filters/type/name : "
            f"{'YES' if r['has_real_filter_objects'] else 'NO'}"
        )
        if r["filter_objects"]:
            for p in r["filter_objects"][:30]:
                lines.append(f"  FILTER {p}")
            if len(r["filter_objects"]) > 30:
                lines.append(f"  ... {len(r['filter_objects'])-30} more")
        lines += ["", "CHANNEL REFERENCES", "-" * 110]
        for c in r["channels"]:
            lines.append(
                f"{c['component'].upper():<3} ref={'YES' if c['has_ref'] else 'NO ':<3} "
                f"units={c['units']!r:<24} filters={c['filters']} | {c['path']}"
            )
        lines += ["-" * 110]
        lines.append(
            "All channel datasets have non-empty filter references: "
            + ("YES" if r["all_channels_have_refs"] else "NO")
        )
        if r["has_real_filter_objects"] and r["all_channels_have_refs"]:
            lines.append("RESULT: REAL MTH5 CALIBRATION CHAIN PRESENT.")
        else:
            lines.append("RESULT: FILE IS NOT PROVEN TO HAVE A COMPLETE CALIBRATION CHAIN.")
        return "\n".join(lines)

    def _direct_mth5_export(self,out:Path) -> Path:
        """Write one MTH5 run per sample rate, preserving TD float32 values."""
        self.log("[MTH5 IMPORT] applying UTF-8 safeguard for mt_metadata packaged JSON ...")
        MTH5, Electric, Magnetic, Run, Station, Survey = _import_mth5_utf8_safe()
        self.log("[MTH5 IMPORT] OK")

        layout=extract_layout(self.config,self.recmeta)
        survey_id=safe_id(layout.get("Survey_Name"),"Phoenix_MTU5C")
        # V64: station identity and receiver identity are separate concepts.
        # Prefer recmeta/config layout.Station_Name (e.g. YP307).  If absent,
        # use the selected data-folder name, and use receiver serial only as
        # the final fallback.
        receiver_id = str(
            (self.recmeta or {}).get("instid")
            or (self.bin_headers[0].instrument_serial if self.bin_headers else "")
            or ""
        ).strip()
        station_from_layout = str(layout.get("Station_Name") or "").strip()
        try:
            station_from_folder = Path(self.station_dir.get().strip()).name.strip()
        except Exception:
            station_from_folder = ""
        station_id=safe_id(
            station_from_layout or station_from_folder or receiver_id,
            "MTU5C"
        )
        self.log(
            f"V64 station mapping: Station_Name={station_from_layout!r}, "
            f"folder={station_from_folder!r}, receiver_id={receiver_id!r} -> "
            f"station_id={station_id!r}"
        )
        timing=self.recmeta.get("timing",{}) if isinstance(self.recmeta,dict) else {}
        start_unix=self.recmeta.get("start") or (self.bin_headers[0].recording_id if self.bin_headers else None)
        start_iso=unix_iso(start_unix) or "1980-01-01T00:00:00+00:00"
        grouped=self._headers_by_component_rate()
        if not grouped:
            raise RuntimeError("No recognized MT components are available for direct MTH5 writing.")

        if out.exists(): out.unlink()
        m=MTH5(file_version="0.2.0")
        try:
            self.log("[MTH5 STEP] open_mth5(w) ...")
            m.open_mth5(out,"w")
            self.log("[MTH5 STEP] open_mth5(w) OK")
            survey_meta=Survey(id=survey_id) if Survey else None
            if survey_meta is not None:
                try: survey_meta.geographic_name=str(layout.get("Survey_Name") or survey_id)
                except Exception: pass
            self.log(f"[MTH5 STEP] add_survey({survey_id}) ...")
            survey=m.add_survey(survey_id,survey_metadata=survey_meta)
            self.log("[MTH5 STEP] add_survey OK")

            station_meta=Station(id=station_id) if Station else None
            if station_meta is not None:
                try:
                    station_meta.location.latitude=float(timing.get("gps_lat",0) or 0)
                    station_meta.location.longitude=float(timing.get("gps_lon",0) or 0)
                    station_meta.location.elevation=float(timing.get("gps_alt",0) or 0)
                    station_meta.time_period.start=start_iso
                except Exception: pass
            self.log(f"[MTH5 STEP] add_station({station_id}) ...")
            station=m.add_station(station_id,station_metadata=station_meta,survey=survey_id)
            self.log("[MTH5 STEP] add_station OK")

            station_end_iso=start_iso
            for fs, comp_groups in grouped.items():
                present=[c for c in COMPONENT_ORDER if comp_groups.get(c)]
                if not present:
                    continue
                lengths={c:sum(native_sample_count(h.path) for h in comp_groups[c]) for c in present}
                n_common=min(lengths.values())
                if n_common<=0:
                    continue
                if len(set(lengths.values()))>1:
                    self.log(f"WARNING {fs:g} Hz: channel sample counts differ {lengths}; trimming each channel to n={n_common:,}")
                end_iso=(datetime.fromtimestamp(float(start_unix),tz=timezone.utc)+timedelta(seconds=(n_common-1)/fs)).isoformat() if start_unix is not None else start_iso
                station_end_iso=max(station_end_iso,end_iso)
                rate_label=(str(int(fs)) if float(fs).is_integer() else str(fs).replace('.','p'))
                run_id=f"sr{rate_label}Hz"

                run_meta=Run(id=run_id) if Run else None
                if run_meta is not None:
                    try:
                        run_meta.sample_rate=float(fs); run_meta.time_period.start=start_iso; run_meta.time_period.end=end_iso
                        run_meta.data_logger.manufacturer="Phoenix Geophysics"; run_meta.data_logger.model="MTU-5C"; run_meta.data_logger.id=(receiver_id or station_id)
                    except Exception: pass
                self.log(f"[MTH5 STEP] add_run({run_id}) ...")
                run=m.add_run(station_id,run_id,run_metadata=run_meta,survey=survey_id)
                self.log(f"[MTH5 STEP] add_run({run_id}) OK")
                self.log(f"Direct writer: creating run {run_id}, fs={fs:g} Hz, channels={present}, n={n_common:,}")

                for comp in COMPONENT_ORDER:
                    hs=comp_groups.get(comp,[])
                    if not hs: continue
                    self.status.set(f"Writing {run_id}: {comp.upper()}..."); self.update_idletasks()
                    data,ch_fs=self._read_complete_headers(hs); data=data[:n_common]
                    is_td=all(_is_td_file(h.path) for h in hs)
                    if is_td:
                        data=np.asarray(data,dtype=np.float32)
                        channel_dtype="float32"
                    else:
                        data=np.asarray(data,dtype=np.int32)
                        channel_dtype="int32"
                    tag=COMPONENT_TO_TAG[comp]; info=self.channel_info.get(tag,{})
                    ctype="electric" if comp.startswith("e") else "magnetic"
                    meta=(Electric(component=comp) if ctype=="electric" else Magnetic(component=comp)) if Electric else None
                    if meta is not None:
                        meta.sample_rate=float(ch_fs); meta.time_period.start=start_iso; meta.time_period.end=end_iso
                        try:
                            # V67.5: Empower TD float32 is provisionally treated as voltage-domain data.
                            # Legacy/non-TD integer data remain digital counts.
                            meta.units = "volt" if is_td else "digital counts"
                        except Exception:
                            pass
                        try: meta.channel_number=int(self._effective_channel_id(hs[0]))
                        except Exception: pass
                        if ctype=="electric":
                            az,dlen,_geom_src=self._v26_electric_geometry(tag)
                            if az is not None: meta.measurement_azimuth=az
                            if dlen is not None: meta.dipole_length=dlen
                            try:
                                self.log(f"V27 electric geometry {tag}: dipole_length={dlen} m, azimuth={az}, source={_geom_src}")
                            except Exception:
                                pass
                        else:
                            try:
                                meta.sensor.manufacturer="Phoenix Geophysics"; meta.sensor.model=str(info.get("type_name") or "MTC-150")
                                meta.sensor.id=str(info.get("serial") or info.get("ser") or hs[0].channel_board_serial); meta.sensor.type="induction coil"
                            except Exception: pass
                            meta.measurement_azimuth={"hx":0.0,"hy":90.0,"hz":0.0}.get(comp,0.0)
                            try: meta.measurement_tilt=90.0 if comp=="hz" else 0.0
                            except Exception: pass
                    self.log(f"  adding {comp}: n={len(data):,}, fs={ch_fs:g} Hz, dtype={channel_dtype}, source={'TD float32' if is_td else 'BIN counts'}")
                    self.log(f"[MTH5 STEP] add_channel({run_id}/{comp}) ...")
                    ch=m.add_channel(station_id,run_id,comp,ctype,data,channel_dtype=channel_dtype,channel_metadata=meta,survey=survey_id)
                    self.log(f"[MTH5 STEP] add_channel({run_id}/{comp}) OK")
                    try: ch.write_metadata()
                    except Exception: pass
                try:
                    run.metadata.sample_rate=float(fs); run.metadata.time_period.start=start_iso; run.metadata.time_period.end=end_iso; run.write_metadata()
                except Exception as e:
                    self.log(f"Run metadata warning ({run_id}): {e}")

            try:
                station.metadata.time_period.start=start_iso; station.metadata.time_period.end=station_end_iso; station.write_metadata(); survey.write_metadata()
            except Exception as e:
                self.log(f"Metadata finalization warning: {e}")
            try:
                valid=m.validate_file(); self.log(f"MTH5 validate_file() before close: {valid}")
            except Exception as e:
                self.log(f"validate_file() warning: {e}")
        finally:
            try: m.close_mth5()
            except Exception: pass
        self._v22_mark_direct_raw_channels(out)
        self.log(
            "V24 direct fallback: numerical samples were preserved RAW/UNCALIBRATED. "
            "BIN=24-bit digital counts; TD=float32 provisionally stored with units=volt (V67.6 assumption)."
        )
        # V38: no diagnostics are written here because `out` may be an
        # ASCII-only temporary staging file.  The single V37 folder is created
        # only after the FINAL MTH5 path is known.
        return out

    def _v65_prepare_official_calibration(self, cal_dir: Path):
        """Resolve the selected RXCAL/SCAL JSON files to the actual recording.

        Returns the exact serial-keyed dictionaries to pass to the official
        Phoenix MTH5 importer.  Magnetic SCAL keeps the V64 exact-serial policy.
        """
        match = self._v29_match_calibration_jsons(Path(cal_dir))
        receiver_dict, sensor_dict = self._v29_build_calibration_dicts(match)
        self.log(
            "V65 calibration matching: receiver="
            f"{match.get('receiver_status')} ({len(receiver_dict)} keys), "
            f"sensors={len(sensor_dict)} keys"
        )
        for tag in ("H1", "H2", "H3"):
            d = match.get("sensor_diagnostics", {}).get(tag, {})
            self.log(
                f"V65 {tag} SCAL: serial={d.get('actual_serial','')} "
                f"model={d.get('actual_model','')} status={d.get('status','NO_MATCH')}"
            )
        return match, receiver_dict, sensor_dict

    def _v65_embed_calibration_json(self, mth5_path: Path, match=None):
        """Archive the decoded RXCAL/SCAL JSON inside the MTH5 without applying it.

        The custom /Phoenix_Calibration_JSON group is provenance/archive data.
        Numerical response objects used by processing remain in Survey/Filters;
        channel filter references are built separately.  JSON is stored verbatim
        as UTF-8 text plus matching metadata so the original decoded calibration
        can always be recovered from the MTH5 file.
        """
        import h5py
        cal = self.calibration_dir.get().strip()
        if not cal or not Path(cal).exists():
            return {"stored": 0, "files": []}
        if match is None:
            match = self._v29_match_calibration_jsons(Path(cal))

        selected = []
        r = match.get("receiver_match")
        if r:
            selected.append(("RXCAL", r))
        for tag in ("H1", "H2", "H3"):
            ident = match.get("sensor_matches", {}).get(tag)
            if ident:
                selected.append((f"SCAL_{tag}", ident))

        # Also preserve any decoded rxcal/scal JSON not selected, under ALL_JSON.
        all_idents = [x for x in match.get("files", []) if x.get("path")]
        selected_paths = {str(Path(x[1]["path"]).resolve()).lower() for x in selected}
        for ident in all_idents:
            rp = str(Path(ident["path"]).resolve()).lower()
            if rp not in selected_paths:
                selected.append(("ALL_JSON", ident))

        stored=[]
        with h5py.File(mth5_path, "r+") as h5:
            root = h5.require_group("/Phoenix_Calibration_JSON")
            root.attrs["description"] = (
                "Original decoded Phoenix RXCAL/SCAL JSON archived by converter V66; "
                "not evidence that numerical calibration was applied."
            )
            root.attrs["calibration_applied_to_samples"] = False
            root.attrs["receiver_match_status"] = str(match.get("receiver_status", ""))
            root.attrs["matching_summary_json"] = json.dumps({
                "receiver_status": match.get("receiver_status"),
                "sensor_diagnostics": match.get("sensor_diagnostics", {}),
            }, ensure_ascii=False, default=str)
            dt=h5py.string_dtype(encoding="utf-8")
            for role, ident in selected:
                src=Path(ident["path"])
                try:
                    raw=src.read_text(encoding="utf-8-sig", errors="replace")
                except Exception:
                    raw=src.read_bytes().decode("utf-8", errors="replace")
                role_group=root.require_group(role)
                name=self._v23_safe_filter_name(src.name)
                if name in role_group:
                    del role_group[name]
                g=role_group.create_group(name)
                g.create_dataset("json_text", data=raw, dtype=dt)
                g.attrs["source_filename"] = src.name
                g.attrs["source_path"] = str(src)
                g.attrs["kind"] = str(ident.get("kind", ""))
                g.attrs["serial_candidates_json"] = json.dumps(ident.get("serial_candidates", []), ensure_ascii=False)
                g.attrs["model_candidates_json"] = json.dumps(ident.get("model_candidates", []), ensure_ascii=False)
                g.attrs["selected_role"] = role
                stored.append(f"{role}/{src.name}")
        self.log(f"V66 embedded {len(stored)} RXCAL/SCAL JSON file(s) inside MTH5 /Phoenix_Calibration_JSON")
        return {"stored": len(stored), "files": stored}

    # ------------------------------------------------------------------
    # V66 DECODED EMPOWER-STYLE JSON -> MTH5 FAP CHAIN
    # ------------------------------------------------------------------
    @staticmethod
    def _v66_json_kind(obj, path):
        ft = str((obj or {}).get("file_type", "") or "").lower()
        name = Path(path).name.lower()
        if "receiver calibration" in ft or "rxcal" in name:
            return "rxcal"
        if "sensor calibration" in ft or "scal" in name:
            return "scal"
        return "unknown"

    @staticmethod
    def _v66_valid_fap_row(row):
        if not isinstance(row, dict):
            return False
        f = row.get("freq_Hz", row.get("freq", row.get("frequency")))
        a = row.get("magnitude", row.get("amplitude"))
        p = row.get("phs_deg", row.get("phase_deg", row.get("phase")))
        try:
            return len(f or []) >= 2 and len(a or []) >= 2 and len(p or []) >= 2
        except Exception:
            return False

    def _v66_load_calibration_jsons(self):
        """Load decoded RXCAL/SCAL JSON using the schema produced by V47/V63.

        RXCAL: file_type=receiver calibration, inst_serial, cal_data[tag=E1..H3],
               chan_data[{lp_hz, freq_Hz/freq, magnitude, phs_deg}]
        SCAL : file_type=sensor calibration, sensor_serial, cal_data[].chan_data[].

        UNVERIFIED V47 files are archived but never attached as production filters.
        """
        cal = self.calibration_dir.get().strip()
        out = {"rxcal": [], "scal": [], "ignored": [], "errors": []}
        if not cal or not Path(cal).exists():
            return out
        for p in sorted(Path(cal).rglob("*.json")):
            try:
                obj = load_json(p)
                kind = self._v66_json_kind(obj, p)
                if "unverified" in p.name.lower():
                    out["ignored"].append((str(p), "UNVERIFIED filename"))
                    continue
                usable = False
                for cb in (obj.get("cal_data") or []):
                    for row in (cb.get("chan_data") or []):
                        if self._v66_valid_fap_row(row):
                            usable = True; break
                    if usable: break
                if kind in ("rxcal", "scal") and usable:
                    out[kind].append((p, obj))
                elif kind != "unknown":
                    out["ignored"].append((str(p), "no usable FAP arrays"))
            except Exception as e:
                out["errors"].append(f"{p.name}: {type(e).__name__}: {e}")
        return out

    @staticmethod
    def _v66_norm_serial(v):
        s = str(v or "").strip().upper()
        # Phoenix JSON may serialize a numeric serial with harmless leading zeros.
        return s.lstrip("0") or s

    def _v66_receiver_serial(self):
        return self._v66_norm_serial((self.recmeta or {}).get("instid") or
                                     (self.bin_headers[0].instrument_serial if self.bin_headers else ""))

    def _v66_sensor_serial(self, tag):
        info = self.channel_info.get(str(tag).upper(), {}) or {}
        return self._v66_norm_serial(info.get("serial") or info.get("ser") or "")

    def _v66_choose_rx_json(self, candidates):
        target = self._v66_receiver_serial()
        ranked=[]
        for p,obj in candidates:
            serial = self._v66_norm_serial(obj.get("inst_serial") or obj.get("receiver_serial") or obj.get("serial"))
            if target and serial == target:
                # Prefer an explicitly emitted candidate/official/reconstructed JSON over generic copies.
                name=p.name.lower()
                pref = 3 if "candidate" in name else (2 if "reconstructed" in name else 1)
                ranked.append((pref, p, obj))
        ranked.sort(key=lambda x:(x[0], str(x[1])), reverse=True)
        return (ranked[0][1], ranked[0][2]) if ranked else (None, None)

    def _v66_choose_scal_json(self, candidates, tag):
        target = self._v66_sensor_serial(tag)
        if not target:
            return None, None
        ranked=[]
        for p,obj in candidates:
            serial=self._v66_norm_serial(obj.get("sensor_serial") or obj.get("inst_serial") or obj.get("serial"))
            if serial == target:
                ranked.append((p,obj))
        ranked.sort(key=lambda x:str(x[0]))
        return ranked[0] if ranked else (None,None)

    def _v66_lp_hz_for_channel(self, tag, fs):
        """Use Phoenix acquisition LP state first; sample rate is fallback only."""
        info = self.channel_info.get(str(tag).upper(), {}) or {}
        for key in ("lp", "lp_hz", "low_pass", "lowpass"):
            try:
                v=float(info.get(key))
                if v > 0: return int(round(v)), f"recmeta.{key}"
            except Exception:
                pass
        # Fail-safe fallback for the two MTU-5C TD products in this converter.
        # This is used only when acquisition metadata has no LP state.
        fs=float(fs or 0)
        if fs >= 10000: return 10000, "sample-rate fallback"
        if fs >= 100: return 100, "sample-rate fallback"
        return 10, "sample-rate fallback"

    @staticmethod
    def _v66_find_channel_block(rx, tag):
        for cb in (rx.get("cal_data") or []):
            if str(cb.get("tag") or "").upper() == str(tag).upper():
                return cb
        return None

    def _v66_select_rx_row(self, cb, lp_hz):
        rows=list((cb or {}).get("chan_data") or [])
        rows=[r for r in rows if self._v66_valid_fap_row(r)]
        if not rows: return None
        for r in rows:
            try:
                if int(round(float(r.get("lp_hz")))) == int(lp_hz): return r
            except Exception: pass
        state={10000:"T0",1000:"T1",100:"T2",10:"T3"}.get(int(lp_hz))
        if state:
            for r in rows:
                if str(r.get("tag") or "").upper().startswith(state): return r
        return None

    @staticmethod
    def _v66_row_response(row):
        f=row.get("freq_Hz", row.get("freq", row.get("frequency"))) or []
        a=row.get("magnitude", row.get("amplitude")) or []
        p=row.get("phs_deg", row.get("phase_deg", row.get("phase"))) or []
        n=min(len(f),len(a),len(p))
        if n < 2: return None
        f=np.asarray(f[:n],float); a=np.asarray(a[:n],float); p=np.asarray(p[:n],float)
        ok=np.isfinite(f)&np.isfinite(a)&np.isfinite(p)&(f>0)
        f,a,p=f[ok],a[ok],p[ok]
        if len(f)<2: return None
        order=np.argsort(f)
        return {"type":"fap","frequency":f[order],"amplitude":a[order],"phase_deg":p[order]}

    @staticmethod
    def _v72_limit_fap(resp, lp_hz, receiver=False):
        """Restrict response to selected LP. RXCAL alone has unity low-frequency extension.

        SCAL is never flattened. Low-frequency RXCAL is explicitly (1, 0 deg).
        High-frequency extrapolation is forbidden. Return None on inadequate coverage.
        """
        if resp is None:
            return None
        f=np.asarray(resp["frequency"],float)
        a=np.asarray(resp["amplitude"],float)
        phase=np.asarray(resp["phase_deg"],float)
        lp=float(lp_hz)
        keep=(f>0)&(f<=lp)&np.isfinite(a)&np.isfinite(phase)
        if not np.any(keep):
            return None
        ff=f[keep]; aa=a[keep]; pp=phase[keep]
        if receiver:
            # Add a unity anchor at the SCAL low end, without inventing RXCAL
            # high-frequency measurements. A second anchor just below f_min
            # preserves the exact original first RXCAL sample.
            fmin=float(ff[0])
            low=min(8e-5, fmin/100.0)
            near=np.nextafter(fmin, 0.0)
            if low < near:
                ff=np.r_[low,near,ff]
                aa=np.r_[1.0,1.0,aa]
                pp=np.r_[0.0,0.0,pp]
        return {"type":"fap","frequency":ff,"amplitude":aa,"phase_deg":pp}

    def _v66_build_decoded_json_fap_chain(self, path: Path):
        """Attach decoded RXCAL + exact-serial SCAL FAP responses to raw MTH5 samples.

        Electric chain: RXCAL only (normalized receiver response).
        Magnetic chain: SCAL (nT -> mV) followed by RXCAL (mV -> mV).
        The numerical TD samples remain untouched and calibration_applied=False.
        """
        import h5py
        bundle=self._v66_load_calibration_jsons()
        rx_path,rx=self._v66_choose_rx_json(bundle["rxcal"])
        scal={tag:self._v66_choose_scal_json(bundle["scal"],tag) for tag in ("H1","H2","H3")}
        lines=["V66 DECODED JSON -> MTH5 FAP BUILDER","="*110,
               f"Receiver target serial: {self._v66_receiver_serial()}",
               f"RXCAL selected: {rx_path if rx_path else 'NONE'}"]
        for tag in ("H1","H2","H3"):
            lines.append(f"SCAL {tag} target sensor serial={self._v66_sensor_serial(tag) or '?'} selected={scal[tag][0] if scal[tag][0] else 'NONE'}")
        if bundle["ignored"]: lines.append(f"Ignored JSON files: {len(bundle['ignored'])}")
        if bundle["errors"]: lines.extend("JSON error: "+x for x in bundle["errors"][:10])

        grouped=self._headers_by_component_rate()
        attached=0; total=0
        with h5py.File(path,"r+") as h5:
            items=[]
            def collect(name,obj):
                if not isinstance(obj,h5py.Dataset): return
                comp=name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER: return
                parts=name.split("/")
                try:
                    i=parts.index("Surveys"); survey_path="/"+"/".join(parts[:i+2])
                except Exception: return
                items.append((name,obj,comp,survey_path))
            h5.visititems(collect)
            for name,obj,comp,survey_path in items:
                total+=1
                try: fs=float(obj.attrs.get("sample_rate",0) or 0)
                except Exception: fs=0.0
                tag=COMPONENT_TO_TAG.get(comp,comp.upper())
                filters=[]

                # Magnetic sensor response: exact sensor serial only.
                if comp.startswith("h"):
                    sp,sj=scal.get(tag,(None,None))
                    if sj is not None:
                        rows=[]
                        for cb in (sj.get("cal_data") or []): rows.extend(cb.get("chan_data") or [])
                        sr=next((r for r in rows if self._v66_valid_fap_row(r)),None)
                        lp,_=self._v66_lp_hz_for_channel(tag,fs)
                        resp=self._v72_limit_fap(self._v66_row_response(sr) if sr else None,lp,receiver=False)
                        if resp:
                            serial=self._v66_sensor_serial(tag)
                            fname=self._v23_safe_filter_name(f"phoenix_scal_{tag}_{serial}_fap")
                            ident={"kind":"sensor","tag":tag,"serial":serial,
                                   "model":sj.get("instrument_model",""),"units":sr.get("units","mV/nT")}
                            fname=self._v23_write_filter_object(h5,survey_path,fname,resp,str(sp),ident)
                            fg=h5[f"{survey_path}/Filters/fap/{fname}"]
                            fg.attrs["input_units"]="nanotesla"; fg.attrs["output_units"]="millivolt"
                            fg.attrs["phoenix_v66.role"]="SCAL"
                            filters.append(fname)

                # Receiver response for every E/H channel.
                lp,lp_source=self._v66_lp_hz_for_channel(tag,fs)
                cb=self._v66_find_channel_block(rx or {},tag) if rx else None
                rr=self._v66_select_rx_row(cb,lp) if cb else None
                resp=self._v72_limit_fap(self._v66_row_response(rr) if rr else None,lp,receiver=True)
                if resp:
                    fname=self._v23_safe_filter_name(f"phoenix_rxcal_{self._v66_receiver_serial()}_{tag}_{lp}Hz_fap")
                    ident={"kind":"receiver","tag":tag,"serial":self._v66_receiver_serial(),
                           "model":str((rx or {}).get("instrument_type","MTU-5C")),
                           "sample_rate":str(fs),"lp_hz":str(lp),"lp_source":lp_source}
                    fname=self._v23_write_filter_object(h5,survey_path,fname,resp,str(rx_path),ident)
                    fg=h5[f"{survey_path}/Filters/fap/{fname}"]
                    if comp.startswith("h"):
                        fg.attrs["input_units"]="millivolt"; fg.attrs["output_units"]="millivolt"
                    else:
                        fg.attrs["input_units"]="volt"; fg.attrs["output_units"]="volt"
                    fg.attrs["phoenix_v66.role"]="RXCAL"; fg.attrs["phoenix_v66.lp_hz"]=int(lp)
                    filters.append(fname)

                required=2 if comp.startswith("h") else 1
                if len(filters)==required:
                    dt=h5py.string_dtype("utf-8")
                    obj.attrs["filters"]=np.asarray(filters,dtype=dt)
                    obj.attrs["filter.applied"]=np.asarray([False]*len(filters),dtype=np.bool_)
                    obj.attrs["phoenix_v22.calibration_applied"]=False
                    obj.attrs["phoenix_v22.real_filter_chain_attached"]=True
                    obj.attrs["phoenix_v66.calibration_chain_status"]="ATTACHED"
                    obj.attrs["phoenix_v66.filter_names_json"]=json.dumps(filters)
                    attached+=1
                    lines.append(f"{comp.upper():<3} {fs:g} Hz | ATTACHED {filters} | LP={lp} ({lp_source})")
                else:
                    obj.attrs["phoenix_v66.calibration_chain_status"]="INCOMPLETE"
                    lines.append(f"{comp.upper():<3} {fs:g} Hz | INCOMPLETE {len(filters)}/{required} stages | LP={lp} ({lp_source})")
        lines += ["-"*110,f"Channel datasets found: {total}",f"Complete V66 chains: {attached}",
                  "RESULT: COMPLETE" if total and attached==total else "RESULT: INCOMPLETE"]
        return {"complete": bool(total and attached==total),"attached":attached,"total":total,"text":"\n".join(lines)}


    # ------------------------------------------------------------------
    # V68 TD-VOLT + ACQUISITION-LP + M3 CALIBRATION STANDARD MTH5 RESPONSE WRITER / REOPEN VERIFIER
    # ------------------------------------------------------------------
    def _v67_standardize_response_chain(self, path: Path):
        """Write V66 decoded responses through the public MTH5/mt_metadata API.

        V66 wrote inspectable HDF5 groups itself.  V67.4 deliberately does not
        regard those groups as sufficient.  It rebuilds the response objects
        with FrequencyResponseTableFilter, adds them through FiltersGroup,
        attaches channel metadata ``filter.name``/``filter.applied``, closes
        the file, reopens it, and requires ``channel_response`` to reconstruct.

        Samples are NOT numerically calibrated here; filter.applied stays False.
        """
        import numpy as _np
        from mth5.mth5 import MTH5
        from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

        bundle = self._v66_load_calibration_jsons()
        rx_path, rx = self._v66_choose_rx_json(bundle["rxcal"])
        scal = {tag: self._v66_choose_scal_json(bundle["scal"], tag)
                for tag in ("H1", "H2", "H3")}
        if rx is None:
            raise RuntimeError("V67: no usable decoded RXCAL JSON was selected.")

        # Discover exact MTH5 survey/station/run/channel identities first.
        import h5py
        channels = []
        with h5py.File(path, "r") as h5:
            def _collect(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                comp = name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER:
                    return
                p = [x for x in name.split("/") if x]
                try:
                    i = p.index("Surveys")
                    survey, station, run = p[i+1], p[i+3], p[i+4]
                except Exception:
                    return
                try: fs = float(obj.attrs.get("sample_rate", 0) or 0)
                except Exception: fs = 0.0
                channels.append((survey, station, run, comp, fs))
            h5.visititems(_collect)

        if not channels:
            raise RuntimeError("V67: no standard MTH5 channel datasets were found.")

        # V67.4: mt_metadata is strict.  Custom Phoenix attributes must NOT live
        # on channel datasets because Electric/Magnetic.from_dict() treats every
        # dataset attribute as standard channel metadata.  Archive them under a
        # separate provenance group, then remove them before any MTH5 get_channel().
        with h5py.File(path, "r+") as h5:
            pg = h5.require_group("Phoenix_Metadata").require_group("channel_provenance")
            for survey, station, run, comp, fs in channels:
                dpath = f"/Experiment/Surveys/{survey}/Stations/{station}/{run}/{comp}"
                if dpath not in h5:
                    continue
                ds = h5[dpath]
                custom = {}
                for key in list(ds.attrs.keys()):
                    if str(key).startswith("phoenix_"):
                        val = ds.attrs[key]
                        try:
                            if isinstance(val, bytes): val = val.decode("utf-8", errors="replace")
                            elif hasattr(val, "tolist"): val = val.tolist()
                            json.dumps(val)
                        except Exception:
                            val = str(val)
                        custom[str(key)] = val
                        del ds.attrs[key]
                if custom:
                    g = pg.require_group(survey).require_group(station).require_group(run).require_group(comp)
                    g.attrs["phoenix_attributes_json"] = json.dumps(custom, ensure_ascii=False)
                    g.attrs["source_dataset"] = dpath
                    g.attrs["sample_rate"] = float(fs)

        lines = ["V68 TD-VOLT + ACQUISITION-LP + M3 CALIBRATION STANDARD MTH5 RESPONSE WRITER", "="*110,
                 "V67.5 sanitized non-standard phoenix_* channel attributes -> /Phoenix_Metadata/channel_provenance"]
        m = MTH5()
        m.open_mth5(path, mode="a")
        try:
            for survey, station, run, comp, fs in channels:
                tag = COMPONENT_TO_TAG.get(comp, comp.upper())
                sg = m.get_survey(survey)
                fg = sg.filters_group
                ch = m.get_channel(station, run, comp, survey=survey)
                names = []
                lp, lp_source = self._v66_lp_hz_for_channel(tag, fs)

                # Magnetic sensor FORWARD FAP: physical nT -> TD voltage domain.
                # M3/Aurora calibrates stored TD[V] by applying the inverse response.
                if comp.startswith("h"):
                    sp, sj = scal.get(tag, (None, None))
                    if sj is None:
                        raise RuntimeError(f"V67: {survey}/{station}/{run}/{comp}: exact-serial SCAL JSON missing")
                    rows = []
                    for cb in (sj.get("cal_data") or []):
                        rows.extend(cb.get("chan_data") or [])
                    sr = next((r for r in rows if self._v66_valid_fap_row(r)), None)
                    resp = self._v72_limit_fap(self._v66_row_response(sr) if sr else None, lp, receiver=False)
                    if not resp:
                        raise RuntimeError(f"V72: {comp}: SCAL has no usable points at LP={lp}")
                    serial = self._v66_sensor_serial(tag)
                    name = self._v23_safe_filter_name(f"phoenix_scal_{tag}_{serial}_fap").lower()
                    fap = FrequencyResponseTableFilter(
                        name=name, units_in="nanotesla", units_out="volt")
                    fap.frequencies = _np.asarray(resp["frequency"], dtype=float)
                    # SCAL source magnitude is mV/nT; confirmed TD unit is V, so
                    # convert response amplitude to V/nT exactly once.
                    fap.amplitudes = _np.asarray(resp["amplitude"], dtype=float) / 1000.0
                    # mt_metadata FAP phase is radians.
                    fap.phases = _np.deg2rad(_np.asarray(resp["phase_deg"], dtype=float))
                    fg.add_filter(fap)
                    names.append(name)

                # Receiver FAP for all E/H channels.
                lp, lp_source = self._v66_lp_hz_for_channel(tag, fs)
                cb = self._v66_find_channel_block(rx, tag)
                rr = self._v66_select_rx_row(cb, lp) if cb else None
                resp = self._v72_limit_fap(self._v66_row_response(rr) if rr else None, lp, receiver=True)
                if not resp:
                    raise RuntimeError(f"V72: {comp}: RXCAL response missing for {tag}, LP={lp}")
                name = self._v23_safe_filter_name(
                    f"phoenix_rxcal_{self._v66_receiver_serial()}_{tag}_{lp}Hz_fap").lower()
                # RXCAL is the FORWARD normalized receiver response. TD float32 is V,
                # so keep this stage volt -> volt. M3/Aurora uses its inverse.
                fap = FrequencyResponseTableFilter(
                    name=name, units_in="volt", units_out="volt")
                fap.frequencies = _np.asarray(resp["frequency"], dtype=float)
                fap.amplitudes = _np.asarray(resp["amplitude"], dtype=float)
                fap.phases = _np.deg2rad(_np.asarray(resp["phase_deg"], dtype=float))
                fg.add_filter(fap)
                names.append(name)

                # V67.6: use the CURRENT documented mt_metadata channel API.
                # Channel.filters is a list[AppliedFilter]; add_filter() assigns the
                # name/applied/stage values used by MTH5 ChannelResponse reconstruction.
                # Do NOT delete unrelated channel attributes (V67.5 did that and could
                # remove otherwise valid Electric/Magnetic metadata).
                attrs = ch.hdf5_dataset.attrs
                _probe_before = {}
                for _key in ("filters", "filter.name", "filter.applied", "filtered.name", "filtered.applied"):
                    if _key in attrs:
                        try:
                            _v = attrs[_key]
                            _probe_before[_key] = {"shape": getattr(_v, "shape", None), "dtype": str(getattr(_v, "dtype", type(_v))), "value": str(_v)}
                        except Exception:
                            _probe_before[_key] = "unreadable"

                # Clear any in-memory legacy/current filter state, then add the
                # requested chain with explicit stage order.  This is the public
                # mt_metadata API documented for current Channel metadata.
                try:
                    ch.metadata.filters = []
                    for _stage, _name in enumerate(names, start=1):
                        ch.metadata.add_filter(name=str(_name), applied=False, stage=int(_stage))
                    _schema = "filters/AppliedFilter"
                except Exception as _new_exc:
                    # Compatibility fallback for older mt_metadata releases.
                    ch.metadata.set_attr_from_name("filter.name", list(names))
                    ch.metadata.set_attr_from_name("filter.applied", [False] * len(names))
                    _schema = f"legacy filter.name/filter.applied ({type(_new_exc).__name__})"

                # Runtime schema probe: record exactly what this installed
                # mt_metadata serializes before touching HDF5.
                try:
                    _md_probe = ch.metadata.to_dict(single=True, required=False)
                except TypeError:
                    _md_probe = ch.metadata.to_dict(single=True)
                _filter_probe = {k: str(v) for k, v in _md_probe.items()
                                 if ("filter" in str(k).lower())}
                try:
                    _filter_names_probe = list(getattr(ch.metadata, "filter_names", []) or [])
                except Exception:
                    _filter_names_probe = []

                # MTH5 BaseGroup.write_metadata uses attrs.modify(), which cannot
                # resize an existing attribute.  Delete ONLY filter-related storage
                # keys that the installed schema may recreate; preserve all other
                # channel metadata.
                _deleted = []
                for _key in ("filters", "filter.name", "filter.applied", "filtered.name", "filtered.applied"):
                    if _key in attrs:
                        del attrs[_key]
                        _deleted.append(_key)

                ch.write_metadata()
                try:
                    ch.hdf5_dataset.file.flush()
                except Exception:
                    pass

                _probe_after = {}
                for _key in attrs.keys():
                    if "filter" in str(_key).lower():
                        try:
                            _v = attrs[_key]
                            _probe_after[str(_key)] = {"shape": getattr(_v, "shape", None), "dtype": str(getattr(_v, "dtype", type(_v))), "value": str(_v)}
                        except Exception:
                            _probe_after[str(_key)] = "unreadable"
                lines.append(
                    f"{survey}/{station}/{run}/{comp.upper():<2} {fs:g} Hz | "
                    f"V67.6 CHANNELRESPONSE-PROBE schema={_schema} requested={list(names)} "
                    f"filter_names={_filter_names_probe} deleted_filter_attrs={_deleted} | "
                    f"metadata_filter_fields={_filter_probe} | hdf5_before={_probe_before} | "
                    f"hdf5_after={_probe_after} | LP={lp} ({lp_source})"
                )
            m.flush()
        finally:
            m.close_mth5()

        # Critical V67 rule: writer does not verify itself with h5py.  Close and
        # reopen through MTH5, reconstruct each ChannelResponse, and verify each
        # named filter is visible through the standard FiltersGroup API.
        verified = 0
        failures = []
        m2 = MTH5()
        m2.open_mth5(path, mode="r")
        try:
            for survey, station, run, comp, fs in channels:
                try:
                    sg = m2.get_survey(survey)
                    ch = m2.get_channel(station, run, comp, survey=survey)
                    # V67.6 readback: current schema first (AppliedFilter), then
                    # legacy flattened names for compatibility with older releases.
                    names, applied, stages = [], [], []
                    try:
                        _afs = list(getattr(ch.metadata, "filters", []) or [])
                        for _af in _afs:
                            _n = getattr(_af, "name", None)
                            if _n:
                                names.append(str(_n))
                                applied.append(bool(getattr(_af, "applied", False)))
                                stages.append(getattr(_af, "stage", None))
                    except Exception:
                        pass
                    if not names:
                        try:
                            names = list(ch.metadata.get_attr_from_name("filter.name") or [])
                            applied = list(ch.metadata.get_attr_from_name("filter.applied") or [])
                        except Exception:
                            try:
                                md = ch.metadata.to_dict(single=True, required=False)
                            except TypeError:
                                md = ch.metadata.to_dict(single=True)
                            names = list(md.get("filter.name", []) or [])
                            applied = list(md.get("filter.applied", []) or [])
                    if not names:
                        raise RuntimeError("channel metadata has no filters after reopen")
                    if len(applied) != len(names) or any(bool(x) for x in applied):
                        raise RuntimeError(f"filter applied-state mismatch: {applied}")
                    # Public API must be able to reconstruct every stored filter.
                    objects = [sg.filters_group.to_filter_object(n) for n in names]
                    response = ch.channel_response
                    rlist = list(getattr(response, "filters_list", []) or [])
                    if len(rlist) != len(names):
                        raise RuntimeError(f"channel_response has {len(rlist)} filters, expected {len(names)}")
                    for obj in objects:
                        ff = _np.asarray(getattr(obj, "frequencies", []), dtype=float)
                        aa = _np.asarray(getattr(obj, "amplitudes", []), dtype=float)
                        pp = _np.asarray(getattr(obj, "phases", []), dtype=float)
                        if ff.size < 2 or aa.size != ff.size or pp.size != ff.size:
                            raise RuntimeError(f"invalid reconstructed FAP {getattr(obj, 'name', '?')}")
                        if not (_np.all(_np.isfinite(ff)) and _np.all(_np.isfinite(aa)) and _np.all(_np.isfinite(pp))):
                            raise RuntimeError(f"non-finite reconstructed FAP {getattr(obj, 'name', '?')}")
                    verified += 1
                    lines.append(f"REOPEN PASS {survey}/{station}/{run}/{comp}: names={names} applied={applied} stages={stages} channel_response={len(rlist)}")
                except Exception as exc:
                    failures.append(f"{survey}/{station}/{run}/{comp}: {exc}")
                    lines.append(f"REOPEN FAIL {survey}/{station}/{run}/{comp}: {exc}")
            valid = bool(m2.validate_file())
            lines.append(f"MTH5 validate_file(): {valid}")
            if not valid:
                failures.append("MTH5 validate_file() returned False")
        finally:
            m2.close_mth5()

        lines += ["-"*110, f"Standard channels verified: {verified}/{len(channels)}",
                  "V67 STANDARD RESPONSE RESULT: PASS" if not failures and verified == len(channels)
                  else "V67 STANDARD RESPONSE RESULT: FAIL"]
        if failures:
            lines.extend("  - " + x for x in failures)
        return {"complete": not failures and verified == len(channels),
                "verified": verified, "total": len(channels), "text": "\n".join(lines)}

    def export_mth5(self):
        base_s=self.station_dir.get().strip()
        if not base_s or not Path(base_s).exists(): messagebox.showerror("Export","Select a Phoenix station/data folder first."); return
        if not self.bin_headers: self.scan_station()
        base=Path(base_s); out_s=self.output_path.get().strip()
        if not out_s: self.choose_output(); out_s=self.output_path.get().strip()
        if not out_s: return
        out=Path(out_s); out.parent.mkdir(parents=True,exist_ok=True)
        ok,note=self._validate_station_folder_for_mth5(base); self.log(note)
        if not ok: messagebox.showerror("Export",note); return

        official_error=None; final=None
        has_td=any(_is_td_file(h.path) for h in self.bin_headers)
        cal=self.calibration_dir.get().strip()
        cal_ok=bool(cal and Path(cal).exists())

        # V71: decode native MTU-5C calibration files into standard V63 JSON.
        # Never accept mismatched receiver/sensor serials; never silently use a
        # calibration file for another station. Original binary is preserved.
        if cal_ok:
            try:
                _serial=self._v66_receiver_serial()
                _sensors={tag:self._v66_sensor_serial(tag) for tag in ('H1','H2','H3')}
                _binary_rx=list(Path(cal).rglob('*.rxcal'))
                _binary_sc=list(Path(cal).rglob('*.scal'))
                if _binary_rx and all(_sensors.values()):
                    _generated=convert_folder(Path(cal),_serial,_sensors)
                    for _f in _generated:
                        self.log(f'V71 binary calibration decoded: {_f.name}')
                elif _binary_rx or _binary_sc:
                    self.log('V71: native calibration files present but missing receiver or exact sensor serial metadata; strict JSON matching remains active.')
            except Exception as _decode_error:
                self.log(f'V71 native RXCAL/SCAL decode FAILED: {_decode_error}')
                if bool(self.require_real_calibration.get()):
                    messagebox.showerror('V71 native calibration decode failed',str(_decode_error))
                    return

        # V64 strategy:
        # 1) Standard Phoenix .td_150 / .td_24k are NOT a reason to bypass the
        #    official client.  Always try MakeMTH5.from_phoenix first.
        # 2) The official path is preferred for normal TD data because it knows
        #    Phoenix continuous/segmented run handling.
        # 3) Use the direct writer only when the official client fails or the
        #    data are a non-standard/native case.  Direct fallback never claims
        #    numerical calibration that has not actually been applied.
        try:
            self.status.set("Trying official Phoenix MTH5 client + calibration chain...")
            self.update_idletasks()
            with _mt_metadata_utf8_open_patch():
                from mth5.clients import MakeMTH5
            rates=self._parse_sample_rates()
            kwargs={
                "data_path":base,
                "mth5_filename":out.name,
                "save_path":out.parent,
                "sample_rates":rates,
            }
            v65_match = None
            if cal_ok:
                # V65: do not pass the whole folder as if it were a calibration
                # dictionary. Resolve the actual receiver + exact-serial sensors
                # first, then pass serial-keyed dictionaries to the official client.
                v65_match, receiver_dict, sensor_dict = self._v65_prepare_official_calibration(Path(cal))
                if receiver_dict:
                    kwargs["receiver_calibration_dict"] = receiver_dict
                if sensor_dict:
                    kwargs["sensor_calibration_dict"] = sensor_dict
                self.log(
                    f"V65 official calibration dictionaries: receiver_keys={list(receiver_dict.keys())}; "
                    f"sensor_keys={list(sensor_dict.keys())}"
                )
                if bool(self.require_real_calibration.get()) and not receiver_dict:
                    raise RuntimeError("V65: no confidently matched RXCAL JSON for the actual receiver.")
            else:
                self.log("V65: no valid calibration folder supplied; official client will run without external calibration dictionaries.")

            self.log(
                f"V22 MakeMTH5.from_phoenix(data_path={base}, sample_rates={rates}, "
                f"TD_present={has_td}, calibration={cal_ok})"
            )
            with _mt_metadata_utf8_open_patch():
                result=MakeMTH5.from_phoenix(**kwargs)
            rp=Path(result) if result else out
            final=rp if rp.exists() else (out if out.exists() else None)

            if final is not None and final.exists():
                txt=self._v22_render_calibration_report(final, "V22 OFFICIAL-CLIENT CALIBRATION CHECK")
                self.log("\n"+txt)
                rr=self._v22_calibration_report(final)
                if cal_ok and not (rr["has_real_filter_objects"] and rr["all_channels_have_refs"]):
                    self.log(
                        "V23: official client did not produce a complete channel filter chain; "
                        "trying the V23 JSON calibration builder."
                    )
                    br = self._v66_build_decoded_json_fap_chain(final)
                    self.log("\n" + br["text"])
                    rr=self._v22_calibration_report(final)

                if cal_ok and bool(self.require_real_calibration.get()):
                    if not (rr["has_real_filter_objects"] and rr["all_channels_have_refs"]):
                        raise RuntimeError(
                            "Official Phoenix export + V23 calibration builder did not "
                            "produce a complete real filter chain on every channel. "
                            "The export remains RAW/UNCALIBRATED."
                        )
        except Exception:
            official_error=traceback.format_exc()
            self.log(
                "V22 official Phoenix client/calibration-chain path failed or did not "
                "pass strict verification. Falling back to RAW direct MTH5 writing.\n"
                + official_error
            )
            final=None

        if final is None or not final.exists():
            stage_dir = None
            try:
                self.status.set("Writing MTH5 through ASCII staging path..."); self.update_idletasks()
                # Some Windows/PyTables/MTH5 combinations fall back to the ANSI
                # code page when a path contains non-ASCII characters.  Build the
                # HDF5 file in the system temp directory using an ASCII-only name,
                # close it, then copy it to the user-selected Unicode path.
                stage_dir = Path(tempfile.mkdtemp(prefix="phoenix_mth5_"))
                staged = stage_dir / "phoenix_output.mth5"
                self.log(f"V38 ASCII staging path: {staged}")
                staged_final = self._direct_mth5_export(staged)
                if not staged_final.exists():
                    raise RuntimeError("Staged MTH5 writer returned without creating a file.")
                if out.exists():
                    out.unlink()
                shutil.copy2(staged_final, out)
                final = out
                self.log(f"ASCII-stage MTH5 completed and copied to: {final}")
                try:
                    self.log("\n" + self._v22_render_calibration_report(
                        final, "V22 RAW-FALLBACK CALIBRATION CHECK"
                    ))
                except Exception as _e:
                    self.log(f"V22 calibration report warning: {_e}")
            except Exception as e:
                self.status.set("MTH5 export failed")
                tb=traceback.format_exc()
                env = (
                    f"Python: {sys.version}\n"
                    f"Platform: {platform.platform()}\n"
                    f"Default encoding: {sys.getdefaultencoding()}\n"
                    f"Filesystem encoding: {sys.getfilesystemencoding()}\n"
                    f"Stdout encoding: {getattr(sys.stdout, 'encoding', None)}\n"
                )
                diagnostic=(official_error or "")+"\n\nDIRECT ASCII-STAGE FALLBACK:\n"+env+"\n"+tb
                self.log("\n========== V38 MTH5 FULL TRACEBACK ==========\n"+diagnostic)
                try:
                    print("\n========== V38 MTH5 FULL TRACEBACK ==========", file=sys.stderr)
                    print(diagnostic, file=sys.stderr)
                except Exception:
                    pass
                saved=[]
                # V20.8: always store diagnostic log beside the selected MTU-5C station/data folder.
                data_dir = Path(self.station_dir.get()).expanduser()
                if data_dir.is_file():
                    data_dir = data_dir.parent
                log_targets = [data_dir / "Phoenix_MTH5_error.log"]
                for log_path in log_targets:
                    try:
                        log_path.write_text(diagnostic,encoding="utf-8",errors="replace")
                        saved.append(str(log_path))
                    except Exception as log_e:
                        self.log(f"Could not write diagnostic log to {log_path}: {log_e}")
                try:
                    self.nb.select(self.tab_log)
                except Exception:
                    pass
                where = saved[0] if saved else "(log file could not be written; use the Log tab text)"
                messagebox.showerror("MTH5 export failed", f"{e}\n\nFull traceback is now in the Log tab.\nDiagnostic log: {where}")
                return
            finally:
                if stage_dir is not None:
                    try: shutil.rmtree(stage_dir, ignore_errors=True)
                    except Exception: pass

        # V67 FINAL STANDARD CALIBRATION PASS -- applies to BOTH official and direct-writer outputs.
        # Stored samples remain raw; this pass only attaches proven response metadata
        # and archives the decoded source JSON inside the MTH5.
        if cal_ok and final is not None and final.exists():
            try:
                v65_final_match = self._v29_match_calibration_jsons(Path(cal))
                # V67 always rewrites the response through the public MTH5 API.
                # Do not accept the older V66 h5py-only self-verification as proof.
                self.log("V67.6 final pass: writing STANDARD MTH5 RXCAL/SCAL response chain using official Channel metadata filters/AppliedFilter API and ChannelResponse reopen probe.")
                br = self._v67_standardize_response_chain(final)
                self.log("\n" + br["text"])
                emb = self._v65_embed_calibration_json(final, v65_final_match)
                self.log(f"V67 embedded calibration archive: {emb['stored']} JSON file(s).")
                if bool(self.require_real_calibration.get()) and not br["complete"]:
                    raise RuntimeError(
                        "V67 strict STANDARD MTH5 response verification failed after close/reopen. "
                        "At least one channel response could not be reconstructed through the public MTH5 API."
                    )
            except Exception as _v65e:
                self.log("V67.5 final standard calibration/archive failed:\n" + traceback.format_exc())
                self.status.set("MTH5 calibration verification failed")
                messagebox.showerror("V67 standard calibration verification failed", str(_v65e))
                return

        # V70 FINAL STANDARD-METADATA SAFETY PASS.  No numerical samples are changed.
        if final is not None and final.exists():
            try:
                v70 = self._v70_enforce_standard_channel_metadata(final)
                self.log("\n" + v70["text"])
                if not v70["complete"]:
                    raise RuntimeError("V70 standard MTH5 metadata reopen verification failed: " + "; ".join(v70["failures"]))
            except Exception as _v70e:
                self.log("V70 final standard metadata verification failed:\n" + traceback.format_exc())
                self.status.set("MTH5 standard metadata verification failed")
                messagebox.showerror("V70 standard metadata verification failed", str(_v70e))
                return

        # V38 SINGLE-DIAGNOSTICS-FOLDER POLICY
        # At this point `final` is the real user-selected MTH5 path for BOTH
        # official-client and RAW-fallback routes. Remove stale diagnostic
        # folders made by earlier versions for this exact MTH5 stem, then create
        # exactly one V37 folder.
        try:
            self._v45_cleanup_old_diagnostic_outputs(final)
            self._v45_write_single_rxcal_folder(final)
        except Exception:
            self.log("V38 final RXCAL diagnostics failed:\n" + traceback.format_exc())

        self.status.set("MTH5 created; verifying..."); self.update_idletasks()
        report=self.verify_mth5(final)
        self.status.set("MTH5 export + verification completed")
        msg=f"MTH5 created and reopened for verification.\n\nOutput:\n{final}\n\n{report.splitlines()[0] if report else ''}"
        if official_error: msg += "\n\nNote: official Phoenix collection client failed in your environment, so V23 used the UTF-8-safe ASCII-stage RAW direct MTH5 API fallback + calibration builder."
        messagebox.showinfo("MTH5 saved and verified",msg)

    def _v70_enforce_standard_channel_metadata(self, path: Path) -> dict:
        """V70: make units/filter references standard MTH5 metadata, then hard-reopen verify.

        This does not alter waveform samples.  TD stays volt; BIN stays digital counts.
        Existing V67 RXCAL/SCAL AppliedFilter references are preserved and must reconstruct
        through ChannelResponse after close/reopen.
        """
        MTH5, _, _, _, _, _ = _import_mth5_utf8_safe()
        import h5py

        # Discover channels and determine source-domain units from the actual native inputs.
        grouped = self._headers_by_component_rate()
        identities = []
        with h5py.File(path, "r") as h5:
            def collect(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                comp = name.split("/")[-1].lower()
                if comp not in COMPONENT_ORDER:
                    return
                p = [x for x in name.split("/") if x]
                try:
                    i = p.index("Surveys")
                    survey, station, run = p[i+1], p[i+3], p[i+4]
                except Exception:
                    return
                try: fs = float(obj.attrs.get("sample_rate", 0) or 0)
                except Exception: fs = 0.0
                hs = grouped.get(fs, {}).get(comp, [])
                is_td = bool(hs) and all(_is_td_file(h.path) for h in hs)
                expected_units = "volt" if is_td else "digital counts"
                identities.append((survey, station, run, comp, fs, expected_units))
            h5.visititems(collect)

        if not identities:
            raise RuntimeError("V70: no standard MTH5 channel datasets found")

        # Write using the public MTH5/mt_metadata API.  Do not touch numerical data.
        m = MTH5()
        m.open_mth5(path, mode="a")
        try:
            for survey, station, run, comp, fs, expected_units in identities:
                ch = m.get_channel(station, run, comp, survey=survey)
                # Capture current filter state before metadata rewrite.
                old_filters = []
                try:
                    for af in list(getattr(ch.metadata, "filters", []) or []):
                        n = getattr(af, "name", None)
                        if n:
                            old_filters.append((str(n), bool(getattr(af, "applied", False)), getattr(af, "stage", None)))
                except Exception:
                    pass
                if not old_filters:
                    try:
                        ns = list(ch.metadata.get_attr_from_name("filter.name") or [])
                        aps = list(ch.metadata.get_attr_from_name("filter.applied") or [])
                        old_filters = [(str(n), bool(aps[i]) if i < len(aps) else False, i+1) for i,n in enumerate(ns)]
                    except Exception:
                        pass

                ch.metadata.units = expected_units
                # Rebuild current AppliedFilter objects explicitly if a response chain exists.
                if old_filters:
                    try:
                        ch.metadata.filters = []
                        for i, (n, applied, stage) in enumerate(old_filters, start=1):
                            ch.metadata.add_filter(name=n, applied=bool(applied), stage=int(stage or i))
                    except Exception:
                        ch.metadata.set_attr_from_name("filter.name", [x[0] for x in old_filters])
                        ch.metadata.set_attr_from_name("filter.applied", [bool(x[1]) for x in old_filters])
                ch.write_metadata()
            m.flush()
        finally:
            m.close_mth5()

        # Hard reopen verification through the same public API pyTRACMT/Aurora depend on.
        rows=[]; failures=[]
        m2=MTH5(); m2.open_mth5(path, mode="r")
        try:
            for survey, station, run, comp, fs, expected_units in identities:
                try:
                    ch=m2.get_channel(station, run, comp, survey=survey)
                    units=str(getattr(ch.metadata,"units","") or "").strip()
                    names=[]; applied=[]
                    try:
                        for af in list(getattr(ch.metadata,"filters",[]) or []):
                            n=getattr(af,"name",None)
                            if n:
                                names.append(str(n)); applied.append(bool(getattr(af,"applied",False)))
                    except Exception:
                        pass
                    if not names:
                        try:
                            names=list(ch.metadata.get_attr_from_name("filter.name") or [])
                            applied=list(ch.metadata.get_attr_from_name("filter.applied") or [])
                        except Exception:
                            pass
                    if units.lower() != expected_units.lower():
                        raise RuntimeError(f"units reopened as {units!r}, expected {expected_units!r}")
                    # When calibration is requested/available, filters must be standard and reconstructable.
                    if bool(self.require_real_calibration.get()):
                        if not names:
                            raise RuntimeError("standard filter references are empty after reopen")
                        response=ch.channel_response
                        rlist=list(getattr(response,"filters_list",[]) or [])
                        if len(rlist) != len(names):
                            raise RuntimeError(f"ChannelResponse has {len(rlist)} filters, metadata has {len(names)}")
                    rows.append((survey,station,run,comp,units,names,applied,"PASS"))
                except Exception as e:
                    failures.append(f"{survey}/{station}/{run}/{comp}: {e}")
                    rows.append((survey,station,run,comp,"",[],[],"FAIL"))
        finally:
            m2.close_mth5()
        text=["V70 STANDARD CHANNEL METADATA VERIFY", "="*100,
              "Component   Units            Filter names                                      Applied        Result",
              "-"*100]
        for survey,station,run,comp,units,names,applied,result in rows:
            text.append(f"{comp.upper():<11} {units:<16} {str(names):<49} {str(applied):<14} {result}")
        if failures:
            text += ["", "FAILURES:"] + ["  "+x for x in failures]
        return {"complete": not failures, "text":"\n".join(text), "failures":failures}

    def verify_selected_mth5(self):
        p=self.output_path.get().strip()
        if not p or not Path(p).exists():
            p=filedialog.askopenfilename(title="Select MTH5/H5 to verify",filetypes=[("MTH5/HDF5","*.mth5 *.h5"),("All","*.*")])
        if p: self.verify_mth5(Path(p))

    def verify_mth5(self,path:Path) -> str:
        lines=[f"MTH5 RE-OPEN VERIFICATION: {path}","="*100]
        try:
            MTH5, _, _, _, _, _ = _import_mth5_utf8_safe()
            m=MTH5(file_version="0.2.0")
            try:
                m.open_mth5(path,"r")
                try: lines.append(f"validate_file(): {m.validate_file()}")
                except Exception as e: lines.append(f"validate_file(): warning/error: {e}")
                try:
                    cs=m.channel_summary.to_dataframe() if hasattr(m.channel_summary,"to_dataframe") else m.channel_summary
                    lines.append("\nChannel summary:")
                    lines.append(str(cs))
                except Exception as e: lines.append(f"channel_summary unavailable: {e}")
            finally:
                try: m.close_mth5()
                except Exception: pass
        except Exception as e:
            lines.append(f"MTH5 API reopen warning: {e}")

        # HDF5 tree is an independent structural verification.
        try:
            import h5py
            lines.append("\nHDF5 hierarchy:")
            with h5py.File(path,"r") as h5:
                found=[]
                def visitor(name,obj):
                    if isinstance(obj,h5py.Dataset):
                        shape=obj.shape; found.append(f"DATASET /{name} shape={shape} dtype={obj.dtype}")
                    else: found.append(f"GROUP   /{name}")
                h5.visititems(visitor)
                lines.extend(found[:300])
                # concise MT component check
                comps=[]
                def find_comp(name,obj):
                    if isinstance(obj,h5py.Dataset) and name.split("/")[-1].lower() in COMPONENT_ORDER:
                        comps.append((name.split("/")[-1].lower(),name,obj.shape))
                h5.visititems(find_comp)
                lines.append("\nDetected MT channels:")
                for comp,name,shape in comps: lines.append(f"  {comp.upper():<3}  /{name}  shape={shape}")
                expected=[]
                for comp in COMPONENT_ORDER:
                    tag=COMPONENT_TO_TAG[comp]; info=self.channel_info.get(tag,{})
                    if any(TAG_TO_MT_COMPONENT.get(self.channel_map.get(h.channel_id,""))==comp for h in self.bin_headers): expected.append(comp)
                    elif info and int(info.get("on",1))==0: lines.append(f"  {comp.upper():<3}  expected absent: {tag} disabled in recmeta")
                missing=[c for c in expected if c not in [x[0] for x in comps]]
                lines.append(f"\nExpected channels from BIN: {expected}")
                lines.append(f"Missing expected channels: {missing if missing else 'NONE'}")
                lines.insert(0,"VERIFICATION RESULT: "+("PASS" if not missing else "CHECK"))
        except Exception as e:
            lines.append(f"HDF5 hierarchy verification failed: {e}")
            lines.insert(0,"VERIFICATION RESULT: CHECK")
        try:
            lines += ["", self._v22_render_calibration_report(path)]
        except Exception as e:
            lines += ["", f"V22 calibration-chain verification failed: {e}"]
        try:
            if self.v23_last_calibration_report:
                lines += ["", self.v23_last_calibration_report]
        except Exception:
            pass
        try:
            v70 = self._v70_enforce_standard_channel_metadata(path)
            lines += ["", v70["text"]]
            if not v70["complete"]:
                lines[0] = "VERIFICATION RESULT: CHECK"
        except Exception as e:
            lines += ["", f"V70 standard metadata verification failed: {e}"]
            lines[0] = "VERIFICATION RESULT: CHECK"
        text="\n".join(lines); self.verify_text.delete("1.0",tk.END); self.verify_text.insert("1.0",text); self.nb.select(self.tab_verify); self.log(text.splitlines()[0]+f" | {path}")
        return text

    def _v45_cleanup_old_diagnostic_outputs(self, out_mth5: Path):
        """Remove only known stale diagnostic artifacts for this exact MTH5 stem."""
        import shutil
        p = Path(out_mth5)
        stem = p.with_suffix("").name
        parent = p.parent

        # Exact legacy folders generated by V27/V31-V36 family.
        exact_folders = [
            parent / f"{stem}_V27_calibration_decode",
            parent / f"{stem}_V36_fixed_rxcal_frequency_section_resolver",
            parent / f"{stem}_V36_RXCAL",
            parent / f"{stem}_V36_sections",
            parent / f"{stem}_V37_RXCAL",
            parent / f"{stem}_V38_RXCAL",
        ]
        for d in exact_folders:
            if d.exists() and d.is_dir():
                try:
                    shutil.rmtree(d)
                    self.log(f"V38 removed stale diagnostics folder: {d}")
                except Exception as e:
                    self.log(f"V38 could not remove stale folder {d}: {e}")

        # Exact legacy pointer files only.
        exact_files = [
            p.with_suffix(".V27_decoder_location.txt"),
            p.with_suffix(".V36_state_resolver_location.txt"),
        ]
        for f in exact_files:
            if f.exists() and f.is_file():
                try:
                    f.unlink()
                    self.log(f"V38 removed stale diagnostics pointer: {f}")
                except Exception as e:
                    self.log(f"V38 could not remove stale pointer {f}: {e}")

    def _v45_write_single_rxcal_folder(self, out_mth5: Path):
        """Generate one V45 multi-response-state diagnostics folder beside the selected MTH5."""
        cal_s = self.calibration_dir.get().strip()
        if not cal_s or not Path(cal_s).exists():
            self.log("V45 RXCAL skipped: no valid calibration folder.")
            return None
        rxfiles = sorted(Path(cal_s).glob("*.rxcal"))
        if not rxfiles:
            self.log("V45 RXCAL skipped: no *.rxcal directly in calibration folder.")
            return None
        rid = str((self.recmeta or {}).get("instid", "") or "").strip()
        rx = next((p for p in rxfiles if rid and rid in p.name), rxfiles[0])

        states = {}
        for tag, info in (self.channel_info or {}).items():
            states[tag.upper()] = {
                "serial": info.get("ser", info.get("serial", "")),
                "ga": info.get("ga"),
                "pg": info.get("pg"),
                "lp": info.get("lp"),
                "sampleRate": info.get("sampleRate", info.get("sample_rate")),
                "model": info.get("model", info.get("type_name", "")),
            }

        outdir = Path(out_mth5).with_suffix("")
        outdir = outdir.parent / (outdir.name + "_V45_RXCAL")

        if outdir.exists():
            import shutil
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        result = v46_decode_rxcal_binary_record_types(rx, states, outdir)
        self.log(f"V45 diagnostics folder: {outdir}")
        return result


    def run_v47_1_rxcal_resolver(self):
        """V47.1 independent official-JSON-supervised RXCAL decoder."""
        cal_s = self.calibration_dir.get().strip()
        if not cal_s or not Path(cal_s).exists():
            messagebox.showerror("V47.1 RXCAL", "Select a valid calibration folder first.")
            return

        rxfiles = sorted(Path(cal_s).glob("*.rxcal"))
        if not rxfiles:
            messagebox.showerror("V47.1 RXCAL", "No *.rxcal found in the selected calibration folder.")
            return

        rid = str((self.recmeta or {}).get("instid", "") or "").strip()
        rx = next((p for p in rxfiles if rid and rid in p.name), rxfiles[0])

        states = {}
        for tag, info in (self.channel_info or {}).items():
            states[tag.upper()] = {
                "serial": info.get("ser", info.get("serial", "")),
                "ga": info.get("ga"),
                "pg": info.get("pg"),
                "lp": info.get("lp"),
                "sampleRate": info.get("sampleRate", info.get("sample_rate")),
                "model": info.get("model", info.get("type_name", "")),
            }

        try:
            banner = (
                "\n=== V47.1 OFFICIAL JSON SUPERVISED RXCAL DECODER ===\n"
                "T0 = LP 10 kHz\n"
                "T1 = LP 1 kHz\n"
                "T2 = LP 100 Hz\n"
                "T3 = LP 10 Hz\n"
                "Crossover continuity constraint = OFF\n"
                "Legacy V45/V46 resolver = NOT CALLED\n"
            )
            self.log(banner)
            self.status.set("Running V47.1 independent RXCAL decoder...")
            self.update_idletasks()

            # Deliberately independent from V45/V46 output routing.
            if self.output_path.get().strip():
                base = Path(self.output_path.get()).with_suffix("")
                outdir = base.parent / (base.name + "_V47_1_RXCAL")
            else:
                outdir = Path(cal_s) / (rx.stem + "_V47_1_RXCAL")

            if outdir.exists():
                import shutil
                shutil.rmtree(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            # Sentinel proves which code path actually executed.
            sentinel = outdir / "THIS_IS_V47_1.txt"
            sentinel.write_text(
                "V47.1 OFFICIAL JSON SUPERVISED RXCAL DECODER\n"
                "T0=10kHz; T1=1kHz; T2=100Hz; T3=10Hz\n"
                "No V45/V46 resolver was called by this action.\n",
                encoding="utf-8"
            )

            result = v47_decode_rxcal_official_json_supervised(rx, states, outdir)

            # Rename V47 products to V47.1 so there can be no ambiguity.
            rename_pairs = {
                "V47_RXCAL_decoded_response_candidates.csv":
                    "V47_1_RXCAL_decoded_response_candidates.csv",
                "V47_RXCAL_official_json_supervised_summary.json":
                    "V47_1_RXCAL_official_json_supervised_summary.json",
                "V47_RXCAL_official_json_supervised_report.txt":
                    "V47_1_RXCAL_official_json_supervised_report.txt",
            }
            for old_name, new_name in rename_pairs.items():
                oldp = outdir / old_name
                if oldp.exists():
                    oldp.rename(outdir / new_name)

            # Also rename generated JSONs.
            for p in list(outdir.glob("*_V47_UNVERIFIED_rxcal.json")):
                p.rename(outdir / p.name.replace("_V47_UNVERIFIED_", "_V47_1_UNVERIFIED_"))
            for p in list(outdir.glob("*_V47_candidate_rxcal.json")):
                p.rename(outdir / p.name.replace("_V47_candidate_", "_V47_1_candidate_"))

            report = outdir / "V47_1_RXCAL_official_json_supervised_report.txt"
            csvp = outdir / "V47_1_RXCAL_decoded_response_candidates.csv"
            summary = outdir / "V47_1_RXCAL_official_json_supervised_summary.json"

            self.log(f"V47.1 output folder: {outdir}")
            self.log(f"V47.1 report: {report}")
            self.log(f"V47.1 decoded CSV: {csvp}")
            self.log(f"V47.1 summary: {summary}")
            self.status.set("V47.1 RXCAL decoder complete.")

            messagebox.showinfo(
                "V47.1 RXCAL",
                "V47.1 completed.\n\n"
                f"Output folder:\n{outdir}\n\n"
                "Verify that THIS_IS_V47_1.txt exists.\n"
                "No V46 word-view files should be created in this folder."
            )
        except Exception as e:
            self.status.set("V47.1 RXCAL failed.")
            self.log("V47.1 RXCAL failed:\n" + traceback.format_exc())
            messagebox.showerror("V47.1 RXCAL", str(e))

    def check_packages(self):
        lines=[]
        for name in ["mth5","mt_metadata","mt_io","numpy","pandas","matplotlib","h5py"]:
            try:
                mod=__import__(name); lines.append(f"{name}: {getattr(mod,'__version__','installed')}")
            except Exception as e: lines.append(f"{name}: NOT AVAILABLE ({e})")
        text="\n".join(lines); self.log(text); messagebox.showinfo("Python packages",text)








# =============================================================================
# V45 RXCAL RESPONSE + CONTROL-FIELD FACTORIZATION RESOLVER
# =============================================================================
#
# Key change from V41
# -------------------
# V41 could decode .scal directly, but failed to locate H-channel SCAL files
# because channel state serials may be receiver-board serials rather than
# induction-coil sensor serials.
#
# V45 explicitly resolves magnetic sensor serials from channel metadata/state,
# prioritizing known sensor-specific keys and plausible Phoenix sensor serials.
# It then searches the calibration directory for <sensor_serial>.scal.
#
# Confirmed SCAL binary layout remains:
#   offset=340, record size=32 bytes, f0=freq, f1=Re, f2=Im
#
# RXCAL remains diagnostic-only.

V45_TABLES = (
    (0, 84,   13),
    (1, 512,   7),
    (2, 812,   5),
    (3, 1144,  5),
)
V45_RECORD_BYTES = 32
V45_BLOCK_BYTES = 1464

V45_SCAL_DATA_OFFSET = 340
V45_SCAL_RECORD_BYTES = 32
V45_SCAL_MAX_RECORDS = 256

def _v45_unpack_fixed(blob, block_start):
    sections = []
    for ti, rel, n in V45_TABLES:
        rows = []
        start = block_start + rel
        for i in range(n):
            p = start + i * V45_RECORD_BYTES
            q = p + V45_RECORD_BYTES
            if q > len(blob):
                break
            rows.append(tuple(float(x) for x in struct.unpack("<8f", blob[p:q])))
        sections.append({
            "table_index": int(ti),
            "relative_offset": int(rel),
            "absolute_start": int(start),
            "rows": np.asarray(rows, dtype=float),
        })
    return sections

def _v45_valid_frequency_rows(arr):
    if arr.size == 0:
        return arr
    f = np.asarray(arr[:, 0], dtype=float)
    return arr[np.isfinite(f) & (f > 0.0) & (f <= 1e7)]

def _v45_read_scal_binary(scal_path):
    p = Path(scal_path)
    blob = p.read_bytes()
    rows = []

    max_possible = max(0, (len(blob) - V45_SCAL_DATA_OFFSET) // V45_SCAL_RECORD_BYTES)
    nmax = min(max_possible, V45_SCAL_MAX_RECORDS)

    for i in range(nmax):
        off = V45_SCAL_DATA_OFFSET + i * V45_SCAL_RECORD_BYTES
        rec = blob[off:off + V45_SCAL_RECORD_BYTES]
        if len(rec) != V45_SCAL_RECORD_BYTES:
            break
        vals = struct.unpack("<8f", rec)
        f = float(vals[0])
        if not np.isfinite(f) or f <= 0.0 or f > 1e7:
            continue

        re_v = float(vals[1])
        im_v = float(vals[2])
        if not (np.isfinite(re_v) and np.isfinite(im_v)):
            continue

        z = complex(re_v, im_v)
        rows.append({
            "index": i,
            "offset": off,
            "frequency_hz": f,
            "real": re_v,
            "imag": im_v,
            "magnitude": float(abs(z)),
            "phase_deg": float(np.degrees(np.angle(z))),
            "f3": float(vals[3]),
            "f4": float(vals[4]),
            "f5": float(vals[5]),
            "f6": float(vals[6]),
            "f7": float(vals[7]),
        })

    # keep unique frequency values
    dedup = {}
    for r in rows:
        key = round(r["frequency_hz"], 12)
        if key not in dedup:
            dedup[key] = r
    rows = list(dedup.values())
    rows.sort(key=lambda r: r["frequency_hz"])

    return {
        "path": str(p),
        "size_bytes": len(blob),
        "data_offset": V45_SCAL_DATA_OFFSET,
        "record_bytes": V45_SCAL_RECORD_BYTES,
        "n_records": len(rows),
        "rows": rows,
    }

def _v45_normalize_serial(v):
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # keep digits only for sensor serial matching
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits

def _v45_plausible_sensor_serial(s):
    """
    Phoenix induction-coil sensor serials in this dataset are numeric, typically
    4-5 digits (e.g., 1452, 53372, 53401, 53406).
    Receiver board serials such as 03277A contain hex letters and should not win.
    """
    s = _v45_normalize_serial(s)
    return bool(s) and s.isdigit() and 3 <= len(s) <= 6

def _v45_resolve_sensor_serial(ch, state):
    """
    Resolve magnetic sensor serial using explicit sensor-specific keys first.
    Returns (serial, source_key, diagnostics).
    """
    diagnostics = []
    if not isinstance(state, dict):
        return "", "", diagnostics

    priority_keys = [
        "sensor_serial",
        "sensorSerial",
        "coil_serial",
        "coilSerial",
        "mag_sensor_serial",
        "magSensorSerial",
        "induction_coil_serial",
        "inductionCoilSerial",
        "sensor",
        "sensor_id",
        "sensorId",
    ]

    for k in priority_keys:
        if k in state:
            raw = state.get(k)
            s = _v45_normalize_serial(raw)
            diagnostics.append({"key": k, "raw": raw, "normalized": s})
            if _v45_plausible_sensor_serial(s):
                return s, k, diagnostics

    # Secondary scan over any key mentioning sensor/coil.
    for k, raw in state.items():
        kl = str(k).lower()
        if "sensor" in kl or "coil" in kl:
            s = _v45_normalize_serial(raw)
            diagnostics.append({"key": k, "raw": raw, "normalized": s})
            if _v45_plausible_sensor_serial(s):
                return s, k, diagnostics

    # Last-resort: plain serial only if it is fully numeric and plausible.
    # Hex-like receiver board serials (e.g. 03277A) will fail this.
    if "serial" in state:
        raw = state.get("serial")
        raw_s = str(raw).strip()
        norm = _v45_normalize_serial(raw)
        diagnostics.append({"key": "serial", "raw": raw, "normalized": norm})
        if raw_s.isdigit() and _v45_plausible_sensor_serial(norm):
            return norm, "serial", diagnostics

    return "", "", diagnostics

def _v45_find_scal_by_sensor_serial(rxcal_path, sensor_serial):
    """
    Search for <sensor serial>.scal in likely calibration folders.
    Uses exact serial stem match first, then bounded wildcard search.
    """
    sensor_serial = _v45_normalize_serial(sensor_serial)
    if not sensor_serial:
        return None

    rp = Path(rxcal_path)
    roots = []
    for r in [rp.parent, rp.parent.parent, rp.parent.parent.parent]:
        if r.exists() and r not in roots:
            roots.append(r)

    common = ("01_校正檔", "calibration", "Calibration", "SCAL", "scal")
    dirs = []
    for root in roots:
        dirs.append(root)
        for nm in common:
            d = root / nm
            if d.is_dir():
                dirs.append(d)

    # Exact filename first.
    exact_names = [
        f"{sensor_serial}.scal",
        f"{sensor_serial}.SCAL",
    ]
    for d in dirs:
        for name in exact_names:
            p = d / name
            if p.is_file():
                return p.resolve()

    # Bounded wildcard within selected dirs only.
    seen = set()
    for d in dirs:
        for pat in (f"*{sensor_serial}*.scal", f"*{sensor_serial}*.SCAL"):
            for p in d.glob(pat):
                try:
                    q = p.resolve()
                except Exception:
                    continue
                if q in seen or not q.is_file():
                    continue
                seen.add(q)
                # Prefer filename stem containing exact sensor digits.
                if sensor_serial in q.stem:
                    return q
    return None

def _v45_scal_frequency_supervision(sections, scal_decoded):
    if not scal_decoded or not scal_decoded.get("rows"):
        return {"available": False, "reason": "no decoded SCAL records"}
    sf = np.asarray([r["frequency_hz"] for r in scal_decoded["rows"]], dtype=float)
    rf = []
    for s in sections:
        a = _v45_valid_frequency_rows(s["rows"])
        if len(a):
            rf.extend(a[:, 0].tolist())
    rf = np.asarray(sorted(set(float(x) for x in rf)), dtype=float)

    matched = 0
    unmatched = []
    for f in rf:
        if np.any(np.isclose(sf, f, rtol=1e-5, atol=max(1e-8, 1e-6*abs(f)))):
            matched += 1
        else:
            unmatched.append(float(f))
    return {
        "available": True,
        "rxcal_unique_frequencies": int(len(rf)),
        "scal_frequency_count": int(len(sf)),
        "matched_rxcal_frequencies": int(matched),
        "match_fraction": float(matched/max(len(rf),1)),
        "unmatched_rxcal_frequencies": unmatched,
    }

def _v45_raw_field_stats(v):
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    if len(v)==0:
        return {"n":0,"unique_count":0,"range":None,"std":None,
                "subnormal_fraction":None,"near_zero_fraction":None,"constant_fraction":None}
    av=np.abs(v); scale=max(float(np.max(av)),1e-30)
    vr=float(np.max(v)-np.min(v)); tol=max(1e-12,1e-7*max(scale,1.0))
    return {
        "n":int(len(v)),
        "unique_count":int(len(np.unique(np.round(v,10)))),
        "range":vr,
        "std":float(np.std(v)),
        "subnormal_fraction":float(np.mean((av>0)&(av<1e-35))),
        "near_zero_fraction":float(np.mean(av<max(1e-12,1e-7*scale))),
        "constant_fraction":float(1.0 if vr<=tol else 0.0),
    }

def _v45_curve_stats(f,z):
    f=np.asarray(f,dtype=float); z=np.asarray(z,dtype=complex)
    good=np.isfinite(f)&np.isfinite(z.real)&np.isfinite(z.imag)&(np.abs(z)>1e-30)
    f=f[good]; z=z[good]
    if len(f)==0:
        return {"n":0,"fmin":None,"fmax":None,"mag_min":None,"mag_max":None,
                "mag_median":None,"logmag_range":None,"phase_range_deg":None,
                "smoothness":1e6,"unity_log_rms_lt3khz":None,
                "unity_median_abs_error_lt3khz":None,"n_unity_points":0}
    o=np.argsort(f); f=f[o]; z=z[o]
    mag=np.abs(z); ph=np.unwrap(np.angle(z)); lm=np.log(np.maximum(mag,1e-30))
    if len(z)>1:
        x=np.log10(f); dx=np.diff(x); dx[np.abs(dx)<1e-12]=1.0
        sm=float(np.median(np.abs(np.diff(ph)/dx))+0.35*np.median(np.abs(np.diff(lm)/dx)))
    else: sm=1e6
    sel=f<3000.0
    if np.any(sel):
        mm=mag[sel]
        ulr=float(np.sqrt(np.mean(np.log(np.maximum(mm,1e-30))**2)))
        umae=float(np.median(np.abs(mm-1.0))); nu=int(np.sum(sel))
    else:
        ulr=umae=None; nu=0
    return {
        "n":int(len(f)),"fmin":float(np.min(f)),"fmax":float(np.max(f)),
        "mag_min":float(np.min(mag)),"mag_max":float(np.max(mag)),
        "mag_median":float(np.median(mag)),
        "logmag_range":float(np.max(lm)-np.min(lm)),
        "phase_range_deg":float(np.degrees(np.max(ph)-np.min(ph))),
        "smoothness":sm,
        "unity_log_rms_lt3khz":ulr,
        "unity_median_abs_error_lt3khz":umae,
        "n_unity_points":nu,
    }

def _v45_candidate_penalty(rs,ims,gst):
    p=0.0; reasons=[]
    for name,st in (("Re",rs),("Im",ims)):
        if st["n"]==0:
            p+=1e5; reasons.append(f"{name}: empty field"); continue
        if st["unique_count"]<=3 or st["constant_fraction"]>=0.75:
            p+=300.0; reasons.append(f"{name}: constant/low-uniqueness +300")
        sf=st["subnormal_fraction"] or 0.0
        if sf>0.05:
            q=600*sf; p+=q; reasons.append(f"{name}: subnormal +{q:.3g}")
    ulr=gst.get("unity_log_rms_lt3khz")
    if ulr is not None:
        q=70*ulr; p+=q; reasons.append(f"<3kHz unity log-RMS +{q:.3g}")
    umae=gst.get("unity_median_abs_error_lt3khz")
    if umae is not None and umae>0.25:
        q=30*min(umae,10); p+=q; reasons.append(f"<3kHz unity median-error +{q:.3g}")
    sm=gst.get("smoothness")
    if sm is not None and np.isfinite(sm):
        q=min(20,0.5*sm); p+=q
        if q>2: reasons.append(f"smoothness +{q:.3g}")
    return float(p),reasons

def _v45_rank_cartesian_fields(sections):
    all_rows=[]
    for s in sections:
        a=_v45_valid_frequency_rows(s["rows"])
        if len(a): all_rows.append(a)
    if not all_rows: return []
    joined=np.vstack(all_rows); cands=[]
    for rc in range(1,8):
        for ic in range(1,8):
            if rc==ic: continue
            rs=_v45_raw_field_stats(joined[:,rc]); ims=_v45_raw_field_stats(joined[:,ic])
            for direction in ("direct","inverse"):
                secs=[]; fa=[]; za=[]
                for s in sections:
                    arr=_v45_valid_frequency_rows(s["rows"])
                    if not len(arr): continue
                    z0=arr[:,rc].astype(float)+1j*arr[:,ic].astype(float)
                    z=np.where(np.abs(z0)>1e-30,1.0/z0,np.nan+1j*np.nan) if direction=="inverse" else z0
                    st=_v45_curve_stats(arr[:,0],z)
                    secs.append({"table_index":s["table_index"],"rows":arr,"z":z,"stats":st})
                    fa.extend(arr[:,0].tolist()); za.extend(z.tolist())
                gst=_v45_curve_stats(np.asarray(fa),np.asarray(za))
                score,reasons=_v45_candidate_penalty(rs,ims,gst)
                cands.append({
                    "score":score,"re_col":rc,"im_col":ic,"direction":direction,
                    "semantic":"real_imag" if direction=="direct" else "inverse_real_imag",
                    "raw_re_stats":rs,"raw_im_stats":ims,"global_curve_stats":gst,
                    "penalty_reasons":reasons,"_sections":secs
                })
    cands.sort(key=lambda x:x["score"])
    return cands

def _v45_overlap_diagnostics(secs):
    out=[]
    for i in range(len(secs)):
        a=secs[i]; fa=a["rows"][:,0]; za=a["z"]
        for j in range(i+1,len(secs)):
            b=secs[j]; fb=b["rows"][:,0]; zb=b["z"]
            for ia,f in enumerate(fa):
                js=np.where(np.isclose(fb,f,rtol=1e-7,atol=1e-7))[0]
                for ib in js:
                    z1,z2=complex(za[ia]),complex(zb[ib])
                    scale=max(0.5*(abs(z1)+abs(z2)),1e-20)
                    mm=float(abs(z1-z2)/scale)
                    out.append({"frequency_hz":float(f),"table_a":int(a["table_index"]),
                                "table_b":int(b["table_index"]),"normalized_mismatch":mm,
                                "same_curve_compatible":bool(mm<0.10)})
    return out

def _v45_write_scal_csv(outdir,ch,scal_decoded):
    if not scal_decoded or not scal_decoded.get("rows"): return None
    import csv
    p=outdir/f"{ch}_V45_direct_SCAL_response.csv"
    with p.open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh)
        w.writerow(["index","offset","frequency_hz","real","imag","magnitude","phase_deg","f3","f4","f5","f6","f7"])
        for r in scal_decoded["rows"]:
            w.writerow([r[k] for k in ("index","offset","frequency_hz","real","imag","magnitude","phase_deg","f3","f4","f5","f6","f7")])
    return str(p)

def _v45_write_cartesian_outputs(outdir,ch,best):
    import csv
    p=outdir/f"{ch}_V45_cartesian_candidate.csv"
    with p.open("w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh)
        w.writerow(["table","frequency_hz","real","imag","magnitude","phase_deg","raw_re","raw_im","re_col","im_col","direction"])
        for sec in best["_sections"]:
            for row,zz in zip(sec["rows"],sec["z"]):
                w.writerow([sec["table_index"],float(row[0]),float(np.real(zz)),float(np.imag(zz)),
                            float(np.abs(zz)),float(np.degrees(np.angle(zz))),
                            float(row[best["re_col"]]),float(row[best["im_col"]]),
                            best["re_col"],best["im_col"],best["direction"]])
    return str(p)

def _v45_clean_candidate(r):
    return {k:v for k,v in r.items() if not k.startswith("_")}

def v45_decode_rxcal_sensorserial_scal(rxcal_path, channel_states=None, output_dir=None):
    rxcal_path=Path(rxcal_path)
    blob=rxcal_path.read_bytes()

    anchors=[m.start() for m in re.finditer(b"BCM05-B",blob)]
    block_starts=[max(0,a-10) for a in anchors]
    if len(block_starts)<5:
        fallback=[i*V45_BLOCK_BYTES for i in range(5) if i*V45_BLOCK_BYTES<len(blob)]
        if len(fallback)>len(block_starts): block_starts=fallback

    output_dir=Path(output_dir or (rxcal_path.parent/(rxcal_path.stem+"_V45_RXCAL")))
    output_dir.mkdir(parents=True,exist_ok=True)

    labels=["E1","E2","H1","H2","H3"]
    summary={
        "resolver_version":"V45",
        "rxcal_file":str(rxcal_path),
        "policy":"Resolve magnetic sensor serial explicitly, then load matching binary .scal; diagnostic-only RXCAL.",
        "channels":{}
    }

    report=[
        "V45 RXCAL RESPONSE + CONTROL-FIELD FACTORIZATION RESOLVER",
        "="*104,
        f"RXCAL: {rxcal_path}",
        f"Output folder: {output_dir}",
        "",
        "Magnetic SCAL association now uses SENSOR SERIAL, not receiver-board serial.",
        "SCAL layout: offset=340, record=32 bytes, f0=Hz, f1=Re, f2=Im.",
        "Safety: diagnostic only; no binary RXCAL filter is attached to MTH5.",
        ""
    ]

    for bi,b0 in enumerate(block_starts[:5]):
        ch=labels[bi]
        state=(channel_states or {}).get(ch,{}) if isinstance(channel_states,dict) else {}
        sections=_v45_unpack_fixed(blob,b0)
        ranked=_v45_rank_cartesian_fields(sections)

        sensor_serial=""
        sensor_serial_source=""
        serial_diag=[]
        scal_path=None
        scal_decoded=None
        supervision={"available":False,"reason":"not magnetic channel or no matching SCAL"}

        if ch.startswith("H"):
            sensor_serial,sensor_serial_source,serial_diag=_v45_resolve_sensor_serial(ch,state)
            scal_path=_v45_find_scal_by_sensor_serial(rxcal_path,sensor_serial) if sensor_serial else None
            if scal_path:
                try:
                    scal_decoded=_v45_read_scal_binary(scal_path)
                    supervision=_v45_scal_frequency_supervision(sections,scal_decoded)
                except Exception as exc:
                    supervision={"available":False,"reason":f"SCAL decode failed: {exc}"}

        if not ranked:
            summary["channels"][ch]={
                "block_index":bi,"block_start":b0,"recording_state":state,
                "sensor_serial":sensor_serial,"sensor_serial_source":sensor_serial_source,
                "serial_diagnostics":serial_diag,
                "scal_file":str(scal_path) if scal_path else None,
                "status":"NO_CARTESIAN_CANDIDATE"
            }
            report += [f"[{ch}] NO CARTESIAN CANDIDATE",""]
            continue

        best=ranked[0]
        margin=float(ranked[1]["score"]-best["score"]) if len(ranked)>1 else float("inf")
        gst=best["global_curve_stats"]
        overlap=_v45_overlap_diagnostics(best["_sections"])
        cand_csv=_v45_write_cartesian_outputs(output_dir,ch,best)
        scal_csv=_v45_write_scal_csv(output_dir,ch,scal_decoded)

        conf="LOW"
        if margin>10 and best["score"]<25: conf="MEDIUM"
        elif margin>2 and best["score"]<80: conf="LOW-MEDIUM"

        summary["channels"][ch]={
            "block_index":bi,
            "block_start":b0,
            "recording_state":state,
            "sensor_serial":sensor_serial,
            "sensor_serial_source":sensor_serial_source,
            "serial_diagnostics":serial_diag,
            "cartesian_candidate":{
                "re_col":best["re_col"],"im_col":best["im_col"],
                "direction":best["direction"],"semantic":best["semantic"],
                "score":float(best["score"]),"margin_to_second":margin,
                "confidence":conf,"global_curve_stats":gst,
                "penalty_reasons":best["penalty_reasons"]
            },
            "top20_cartesian_candidates":[_v45_clean_candidate(r) for r in ranked[:20]],
            "overlap_diagnostics":overlap,
            "scal_file":str(scal_path) if scal_path else None,
            "scal_decoded":{
                "n_records":scal_decoded.get("n_records"),
                "data_offset":scal_decoded.get("data_offset"),
                "record_bytes":scal_decoded.get("record_bytes"),
                "size_bytes":scal_decoded.get("size_bytes")
            } if scal_decoded else None,
            "scal_frequency_supervision":supervision,
            "candidate_csv":cand_csv,
            "scal_response_csv":scal_csv
        }

        report += [
            f"[{ch}] block={bi} start={b0}",
            f"  CARTESIAN CANDIDATE: Re=f{best['re_col']} Im=f{best['im_col']} direction={best['direction']}",
            f"  score={best['score']:.6g} margin={margin:.6g} confidence={conf}",
            f"  |H| median={gst.get('mag_median')} range={gst.get('mag_min')} .. {gst.get('mag_max')}",
        ]

        if ch.startswith("H"):
            report += [
                f"  SENSOR SERIAL={sensor_serial if sensor_serial else 'UNRESOLVED'}",
                f"  SENSOR SERIAL SOURCE={sensor_serial_source if sensor_serial_source else 'NONE'}",
                f"  SCAL binary={str(scal_path) if scal_path else 'NONE'}",
            ]
            if scal_decoded:
                report += [
                    f"  SCAL decoded records={scal_decoded.get('n_records')}",
                    f"  SCAL format offset={scal_decoded.get('data_offset')} record_bytes={scal_decoded.get('record_bytes')}",
                    f"  SCAL/RXCAL frequency match={supervision.get('matched_rxcal_frequencies')}/"
                    f"{supervision.get('rxcal_unique_frequencies')} "
                    f"({supervision.get('match_fraction',0):.3f})",
                ]
            else:
                report.append(f"  SERIAL DIAGNOSTICS={serial_diag}")

        report.append("  overlap diagnostics:")
        for c in overlap:
            report.append(
                f"    {c['frequency_hz']:g} Hz: T{c['table_a']}-T{c['table_b']} "
                f"mismatch={c['normalized_mismatch']:.6g} "
                f"same_curve_compatible={c['same_curve_compatible']}"
            )
        report.append("")

    sp=output_dir/"V45_RXCAL_sensorserial_scal_summary.json"
    rp=output_dir/"V45_RXCAL_sensorserial_scal_report.txt"
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    rp.write_text("\n".join(report),encoding="utf-8")
    summary["summary_json"]=str(sp)
    summary["report_txt"]=str(rp)
    return summary



# =============================================================================
# V45 RESPONSE + CONTROL-FIELD FACTORIZATION CORE
# =============================================================================
# Scientific policy:
#   * f0 is the Phoenix calibration frequency.
#   * f1/f2 are FIXED as the primary Cartesian receiver-response hypothesis.
#   * f6/f7 (and all f3..f7) are forbidden as stand-alone response candidates.
#   * f3..f7 are evaluated only as possible control / normalization predictors.
#   * T0/T1/T2 are overlapping frequency sections; T3 is not forced to attach.
#   * diagnostic only: nothing here is attached to MTH5.


def _v45_complex_fit_scale(z_source, z_target):
    zs=np.asarray(z_source,dtype=complex); zt=np.asarray(z_target,dtype=complex)
    good=(np.isfinite(zs.real)&np.isfinite(zs.imag)&np.isfinite(zt.real)&np.isfinite(zt.imag)&
          (np.abs(zs)>1e-30)&(np.abs(zt)>1e-30))
    zs=zs[good]; zt=zt[good]
    if len(zs)==0: return np.nan+1j*np.nan
    den=np.vdot(zs,zs)
    if abs(den)<1e-30: return np.nan+1j*np.nan
    return np.vdot(zs,zt)/den


def _v45_overlap_indices(sec_a, sec_b):
    fa=np.asarray(sec_a['rows'][:,0],float); fb=np.asarray(sec_b['rows'][:,0],float)
    out=[]
    for ia,f in enumerate(fa):
        js=np.where(np.isclose(fb,f,rtol=1e-7,atol=max(1e-8,1e-7*abs(f))))[0]
        for ib in js: out.append((float(f),int(ia),int(ib)))
    return out


def _v45_pair_residual(za, zb):
    za=np.asarray(za,complex); zb=np.asarray(zb,complex)
    den=np.maximum(0.5*(np.abs(za)+np.abs(zb)),1e-20)
    mm=np.abs(za-zb)/den
    pe=np.degrees(np.angle(zb*np.conj(za)))
    ar=np.abs(zb)/np.maximum(np.abs(za),1e-30)
    return {
        'rms':float(np.sqrt(np.mean(mm**2))) if len(mm) else float('nan'),
        'mean':float(np.mean(mm)) if len(mm) else float('nan'),
        'max':float(np.max(mm)) if len(mm) else float('nan'),
        'mean_abs_phase_error_deg':float(np.mean(np.abs(pe))) if len(pe) else float('nan'),
        'mean_abs_log_amp_ratio':float(np.mean(np.abs(np.log(np.maximum(ar,1e-30))))) if len(ar) else float('nan')
    }


def _v45_make_fixed_response_sections(sections, swapped=False, inverse=False):
    out=[]
    for s in sections:
        a=_v45_valid_frequency_rows(s['rows'])
        if not len(a): continue
        if swapped: z0=a[:,2].astype(float)+1j*a[:,1].astype(float)
        else: z0=a[:,1].astype(float)+1j*a[:,2].astype(float)
        if inverse:
            z=np.where(np.abs(z0)>1e-30,1.0/z0,np.nan+1j*np.nan)
        else: z=z0
        out.append({'table_index':int(s['table_index']),'rows':a,'z':z})
    return out


def _v45_section_normalization(secs):
    by={int(s['table_index']):s for s in secs}; scales={0:1+0j}; fits=[]
    for a,b in ((0,1),(1,2),(2,3)):
        if a not in by or b not in by or a not in scales: continue
        pairs=_v45_overlap_indices(by[a],by[b])
        if not pairs:
            fits.append({'table_a':a,'table_b':b,'status':'NO_OVERLAP','n_overlap':0}); continue
        za=np.asarray([by[a]['z'][ia] for _,ia,_ in pairs],complex)*scales[a]
        zb=np.asarray([by[b]['z'][ib] for _,_,ib in pairs],complex)
        cb=_v45_complex_fit_scale(zb,za)
        if not np.isfinite(cb.real) or not np.isfinite(cb.imag):
            fits.append({'table_a':a,'table_b':b,'status':'FIT_FAILED','n_overlap':len(pairs)}); continue
        scales[b]=cb
        resid=_v45_pair_residual(za,zb*cb)
        fits.append({'table_a':a,'table_b':b,'status':'FIT','n_overlap':len(pairs),
                     'frequencies_hz':[p[0] for p in pairs],
                     'scale_real':float(cb.real),'scale_imag':float(cb.imag),
                     'scale_magnitude':float(abs(cb)),'scale_phase_deg':float(np.degrees(np.angle(cb))),**resid})
    return scales,fits


def _v45_control_transform_value(row, col, mode):
    q=float(row[col])
    if not np.isfinite(q): return np.nan+0j
    if mode=='multiply': return complex(q,0.0)
    if mode=='divide':
        return complex(1.0/q,0.0) if abs(q)>1e-30 else np.nan+0j
    if mode=='abs_multiply': return complex(abs(q),0.0)
    if mode=='abs_divide':
        return complex(1.0/abs(q),0.0) if abs(q)>1e-30 else np.nan+0j
    return 1+0j


def _v45_evaluate_control_field(secs, col, mode):
    """Apply one control field point-by-point to fixed f1+i f2, then fit only section constants."""
    transformed=[]
    for s in secs:
        arr=s['rows']; z=np.asarray(s['z'],complex).copy(); zt=[]
        for row,zz in zip(arr,z): zt.append(zz*_v45_control_transform_value(row,col,mode))
        transformed.append({'table_index':s['table_index'],'rows':arr,'z':np.asarray(zt,complex)})
    scales,fits=_v45_section_normalization(transformed)
    usable=[x for x in fits if x.get('status')=='FIT']
    score=float(np.mean([x['rms'] for x in usable])) if usable else float('inf')
    return {'control_col':col,'mode':mode,'score':score,'scales':scales,'fits':fits,'sections':transformed}


def _v45_control_ratio_diagnostics(secs):
    """At each crossover compare the required complex response ratio with raw f3..f7 field ratios."""
    by={int(s['table_index']):s for s in secs}; rows=[]
    for a,b in ((0,1),(1,2),(2,3)):
        if a not in by or b not in by: continue
        for f,ia,ib in _v45_overlap_indices(by[a],by[b]):
            za=complex(by[a]['z'][ia]); zb=complex(by[b]['z'][ib])
            req=za/zb if abs(zb)>1e-30 else np.nan+1j*np.nan
            rec={'table_a':a,'table_b':b,'frequency_hz':f,
                 'required_ratio_real':float(req.real),'required_ratio_imag':float(req.imag),
                 'required_ratio_mag':float(abs(req)),'required_ratio_phase_deg':float(np.degrees(np.angle(req)))}
            ra=by[a]['rows'][ia]; rb=by[b]['rows'][ib]
            for c in range(3,8):
                qa=float(ra[c]); qb=float(rb[c])
                rec[f'f{c}_a']=qa; rec[f'f{c}_b']=qb
                rec[f'f{c}_a_over_b']=float(qa/qb) if np.isfinite(qa) and np.isfinite(qb) and abs(qb)>1e-30 else float('nan')
                rec[f'f{c}_b_over_a']=float(qb/qa) if np.isfinite(qa) and np.isfinite(qb) and abs(qa)>1e-30 else float('nan')
            rows.append(rec)
    return rows


def _v45_cross_channel_similarity(channel_models):
    """Quantify whether fixed f1/f2 responses are implausibly identical across receiver boards."""
    labels=list(channel_models)
    out=[]
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            a,b=labels[i],labels[j]; sa=channel_models[a]; sb=channel_models[b]
            # compare matching T section/frequency points without normalization
            vals=[]
            bya={x['table_index']:x for x in sa}; byb={x['table_index']:x for x in sb}
            for ti in sorted(set(bya)&set(byb)):
                for _,ia,ib in _v45_overlap_indices(bya[ti],byb[ti]):
                    za=bya[ti]['z'][ia]; zb=byb[ti]['z'][ib]
                    den=max(0.5*(abs(za)+abs(zb)),1e-20); vals.append(abs(za-zb)/den)
            out.append({'channel_a':a,'channel_b':b,'n':len(vals),
                        'mean_normalized_difference':float(np.mean(vals)) if vals else float('nan'),
                        'max_normalized_difference':float(np.max(vals)) if vals else float('nan')})
    return out


def _v45_write_ratio_csv(outdir,ch,rows):
    import csv
    p=outdir/f'{ch}_V45_control_field_crossover_ratios.csv'
    if not rows:
        p.write_text('',encoding='utf-8'); return str(p)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return str(p)


def _v45_write_fixed_response_csv(outdir,ch,secs,scales):
    import csv
    p=outdir/f'{ch}_V45_fixed_f1_f2_response.csv'
    with p.open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh); w.writerow(['table','frequency_hz','f1_real','f2_imag','raw_mag','raw_phase_deg',
                                      'section_scale_real','section_scale_imag','normalized_real','normalized_imag','normalized_mag','normalized_phase_deg','anchored'])
        for s in secs:
            c=scales.get(int(s['table_index']))
            for row,z in zip(s['rows'],s['z']):
                zn=z*c if c is not None else np.nan+1j*np.nan
                w.writerow([s['table_index'],float(row[0]),float(z.real),float(z.imag),float(abs(z)),float(np.degrees(np.angle(z))),
                            float(c.real) if c is not None else np.nan,float(c.imag) if c is not None else np.nan,
                            float(zn.real) if c is not None else np.nan,float(zn.imag) if c is not None else np.nan,
                            float(abs(zn)) if c is not None else np.nan,float(np.degrees(np.angle(zn))) if c is not None else np.nan,c is not None])
    return str(p)


def v45_decode_rxcal_response_control_factorization(rxcal_path, channel_states=None, output_dir=None):
    rxcal_path=Path(rxcal_path); blob=rxcal_path.read_bytes()
    anchors=[m.start() for m in re.finditer(b'BCM05-B',blob)]
    block_starts=[max(0,a-10) for a in anchors]
    if len(block_starts)<5:
        fallback=[256+i*V45_BLOCK_BYTES for i in range(5) if 256+i*V45_BLOCK_BYTES<len(blob)]
        if len(fallback)>len(block_starts): block_starts=fallback
    output_dir=Path(output_dir or (rxcal_path.parent/(rxcal_path.stem+'_V45_RXCAL'))); output_dir.mkdir(parents=True,exist_ok=True)
    labels=['E1','E2','H1','H2','H3']
    summary={'resolver_version':'V45','rxcal_file':str(rxcal_path),
             'policy':'FIXED f1+i*f2 Cartesian response; f3..f7 control-field factorization only; diagnostic-only; no MTH5 attachment.',
             'channels':{}}
    report=['V45 RXCAL RESPONSE + CONTROL-FIELD FACTORIZATION RESOLVER','='*112,f'RXCAL: {rxcal_path}',f'Output folder: {output_dir}','',
            'FIXED RESPONSE HYPOTHESIS: Hraw = f1 + i*f2. f3..f7 are NOT allowed to compete as response fields.',
            'CONTROL TEST: f3..f7 are tested only as point-wise multiply/divide predictors before section complex normalization.',
            'CROSSOVERS: T0/T1 = 1024,1440,2048 Hz; T1/T2 = 144,225 Hz. T3 remains unanchored without overlap.',
            'Safety: diagnostic only; no binary RXCAL filter is attached to MTH5.','']
    channel_fixed={}
    for bi,b0 in enumerate(block_starts[:5]):
        ch=labels[bi]; state=(channel_states or {}).get(ch,{}) if isinstance(channel_states,dict) else {}
        sections=_v45_unpack_fixed(blob,b0)
        fixed=_v45_make_fixed_response_sections(sections,swapped=False,inverse=False)
        channel_fixed[ch]=fixed
        scales,fits=_v45_section_normalization(fixed)
        fitrows=[x for x in fits if x.get('status')=='FIT']
        base_rms=float(np.mean([x['rms'] for x in fitrows])) if fitrows else float('inf')
        base_max=float(max([x['max'] for x in fitrows])) if fitrows else float('inf')
        control=[]
        for c in range(3,8):
            for mode in ('multiply','divide','abs_multiply','abs_divide'):
                control.append(_v45_evaluate_control_field(fixed,c,mode))
        control.sort(key=lambda x:x['score'])
        bestc=control[0] if control else None
        improvement=(base_rms-bestc['score'])/base_rms if bestc and np.isfinite(base_rms) and base_rms>1e-30 else 0.0
        ratio_rows=_v45_control_ratio_diagnostics(fixed)
        ratio_csv=_v45_write_ratio_csv(output_dir,ch,ratio_rows)
        resp_csv=_v45_write_fixed_response_csv(output_dir,ch,fixed,scales)
        # Verdict deliberately conservative.
        if base_rms<0.05: verdict='F1_F2_SECTION_NORMALIZATION_SUPPORTED'
        elif bestc and bestc['score']<0.05 and improvement>0.5: verdict='CONTROL_FIELD_MAY_EXPLAIN_SECTION_NORMALIZATION'
        else: verdict='F1_F2_REQUIRES_ADDITIONAL_DECODING_OR_STATE_MODEL'
        summary['channels'][ch]={
            'block_index':bi,'block_start':b0,'recording_state':state,
            'fixed_response':'f1+i*f2','baseline_section_scales':{str(k):{'real':float(v.real),'imag':float(v.imag),'magnitude':float(abs(v)),'phase_deg':float(np.degrees(np.angle(v)))} for k,v in scales.items()},
            'baseline_fits':fits,'baseline_mean_rms':base_rms,'baseline_max':base_max,
            'best_control':None if bestc is None else {'control_col':bestc['control_col'],'mode':bestc['mode'],'score':bestc['score'],'improvement_fraction':improvement,'fits':bestc['fits']},
            'top_control_candidates':[{'control_col':r['control_col'],'mode':r['mode'],'score':r['score']} for r in control[:12]],
            'verdict':verdict,'fixed_response_csv':resp_csv,'control_ratio_csv':ratio_csv}
        report += [f'[{ch}] block={bi} start={b0}',
                   f'  FIXED Hraw=f1+i*f2: baseline crossover mean RMS={base_rms:.6g}, max={base_max:.6g}',
                   f'  VERDICT={verdict}']
        for ti in range(4):
            c=scales.get(ti)
            if c is None: report.append(f'  C{ti}: UNANCHORED')
            else: report.append(f'  C{ti}: |C|={abs(c):.9g}, phase={np.degrees(np.angle(c)):.6g} deg, Re={c.real:.9g}, Im={c.imag:.9g}')
        for x in fits:
            if x.get('status')=='FIT':
                report.append(f"    T{x['table_a']}-T{x['table_b']}: f={x['frequencies_hz']} RMS={x['rms']:.6g} max={x['max']:.6g} phase_err={x['mean_abs_phase_error_deg']:.6g} deg")
            else: report.append(f"    T{x['table_a']}-T{x['table_b']}: {x.get('status')}")
        if bestc:
            report.append(f"  BEST CONTROL: f{bestc['control_col']} mode={bestc['mode']} -> RMS={bestc['score']:.6g}; improvement={100*improvement:.3f}%")
            report.append('  TOP CONTROL CANDIDATES: '+', '.join([f"f{r['control_col']}:{r['mode']}={r['score']:.4g}" for r in control[:8]]))
        report += [f'  response CSV={resp_csv}',f'  crossover control-ratio CSV={ratio_csv}','']
    xs=_v45_cross_channel_similarity(channel_fixed)
    summary['cross_channel_fixed_f1_f2_similarity']=xs
    report += ['CROSS-CHANNEL FIXED f1+i*f2 DIFFERENCE','-'*112]
    for x in xs:
        report.append(f"  {x['channel_a']}-{x['channel_b']}: n={x['n']} mean={x['mean_normalized_difference']:.6g} max={x['max_normalized_difference']:.6g}")
    report += ['', 'INTERPRETATION GUIDE:',
               '  * Small baseline RMS => f1/f2 likely response with section-specific complex normalization.',
               '  * Large baseline RMS but a large, repeatable control-field improvement across channels => that field may encode normalization/state.',
               '  * A control field that is identical across all five channels is NOT accepted as receiver response; it is only a structural predictor.',
               '  * T3 remains unresolved until a state/header relation or another overlapping anchor is identified.']
    sp=output_dir/'V45_RXCAL_response_control_factorization_summary.json'; rp=output_dir/'V45_RXCAL_response_control_factorization_report.txt'
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8'); rp.write_text('\n'.join(report),encoding='utf-8')
    summary['summary_json']=str(sp); summary['report_txt']=str(rp); return summary

# GUI/export compatibility entry point.
def v45_decode_rxcal_sensorserial_scal(rxcal_path, channel_states=None, output_dir=None):
    return v45_decode_rxcal_response_control_factorization(rxcal_path, channel_states, output_dir)


# =============================================================================
# V45 RXCAL MULTI-RESPONSE STATE DECODER (authoritative V45 core)
# =============================================================================
# Scientific policy:
#   * T0..T3 are treated as independent candidate response states, NOT stitched bands.
#   * Shared frequencies are diagnostic only; equality across T states is NOT required.
#   * f0 is fixed as frequency.
#   * f1/f2 representation is resolved independently for each T state from physically
#     motivated candidate encodings: Cartesian, amplitude/phase(rad), amplitude/phase(deg),
#     plus inverse variants. f3..f7 are retained as state/control descriptors only.
#   * No binary RXCAL calibration is attached to MTH5 in V45.


def _v45_repr_candidates(rows):
    a=np.asarray(rows,float)
    x=a[:,1].astype(float); y=a[:,2].astype(float)
    eps=1e-30
    cands={}
    z=x+1j*y
    cands['cartesian_f1_re_f2_im_direct']=z
    cands['cartesian_f1_re_f2_im_inverse']=np.where(np.abs(z)>eps,1.0/z,np.nan+1j*np.nan)
    zs=y+1j*x
    cands['cartesian_swapped_direct']=zs
    cands['cartesian_swapped_inverse']=np.where(np.abs(zs)>eps,1.0/zs,np.nan+1j*np.nan)
    # Amplitude/phase interpretations. Negative amplitudes are allowed mathematically,
    # but receive a plausibility penalty below.
    zr=x*np.exp(1j*y)
    zd=x*np.exp(1j*np.deg2rad(y))
    cands['amp_f1_phase_f2_rad_direct']=zr
    cands['amp_f1_phase_f2_rad_inverse']=np.where(np.abs(zr)>eps,1.0/zr,np.nan+1j*np.nan)
    cands['amp_f1_phase_f2_deg_direct']=zd
    cands['amp_f1_phase_f2_deg_inverse']=np.where(np.abs(zd)>eps,1.0/zd,np.nan+1j*np.nan)
    return cands


def _v45_representation_metrics(freq,z,raw_amp=None,raw_phase=None,kind=''):
    f=np.asarray(freq,float); z=np.asarray(z,complex)
    good=np.isfinite(f)&(f>0)&np.isfinite(z.real)&np.isfinite(z.imag)&(np.abs(z)>1e-30)
    f=f[good]; z=z[good]
    if len(f)<2:
        return {'n':int(len(f)),'score':float('inf')}
    order=np.argsort(f); f=f[order]; z=z[order]
    mag=np.abs(z); ph=np.unwrap(np.angle(z)); lnf=np.log(f)
    dlnf=np.diff(lnf); dlnf=np.where(np.abs(dlnf)>1e-12,dlnf,np.nan)
    dlogm=np.diff(np.log(np.maximum(mag,1e-30)))/dlnf
    dph=np.diff(ph)/dlnf
    # Robust smoothness measures. Receiver electronics should vary smoothly with frequency.
    sm_mag=float(np.nanmedian(np.abs(dlogm))) if len(dlogm) else float('inf')
    sm_ph=float(np.nanmedian(np.abs(dph))) if len(dph) else float('inf')
    jump_mag=float(np.nanmax(np.abs(np.diff(np.log10(np.maximum(mag,1e-30)))))) if len(mag)>1 else 0.0
    jump_ph=float(np.degrees(np.nanmax(np.abs(np.diff(ph))))) if len(ph)>1 else 0.0
    # Receiver prior is deliberately weak: most MTU receiver responses should not be orders
    # of magnitude away from unity over their valid state band.
    unity=float(np.nanmedian(np.abs(np.log(np.maximum(mag,1e-30)))))
    dyn=float(np.nanpercentile(mag,95)/max(np.nanpercentile(mag,5),1e-30)) if len(mag)>=3 else 1.0
    neg_amp_pen=0.0
    if kind.startswith('amp_') and raw_amp is not None:
        aa=np.asarray(raw_amp,float)[good][order]
        neg_amp_pen=5.0*float(np.mean(aa<0))
    # Penalize gross phase jumps more than gentle analog roll-off.
    score=(1.2*sm_mag + 0.35*sm_ph + 0.35*unity + 0.20*max(0.0,jump_mag-0.5)
           + 0.004*max(0.0,jump_ph-45.0) + 0.05*max(0.0,np.log10(max(dyn,1.0))-2.0)
           + neg_amp_pen)
    return {
        'n':int(len(f)),'score':float(score),'median_magnitude':float(np.nanmedian(mag)),
        'min_magnitude':float(np.nanmin(mag)),'max_magnitude':float(np.nanmax(mag)),
        'dynamic_range_p95_p05':float(dyn),'median_abs_log_unity':unity,
        'median_abs_dlogmag_dlogf':sm_mag,'median_abs_dphase_rad_dlogf':sm_ph,
        'max_log10_mag_step':jump_mag,'max_unwrapped_phase_step_deg':jump_ph,
        'phase_start_deg':float(np.degrees(ph[0])),'phase_end_deg':float(np.degrees(ph[-1]))
    }


def _v45_state_descriptor(rows):
    a=np.asarray(rows,float)
    out={}
    for c in range(3,8):
        q=a[:,c]
        finite=q[np.isfinite(q)]
        if len(finite)==0:
            out[f'f{c}']={'n_finite':0}; continue
        uniq=np.unique(finite)
        out[f'f{c}']={
            'n_finite':int(len(finite)),'n_unique':int(len(uniq)),
            'min':float(np.min(finite)),'max':float(np.max(finite)),
            'median':float(np.median(finite)),
            'first_values':[float(v) for v in finite[:min(6,len(finite))]]
        }
    return out


def _v45_shared_frequency_diagnostics(sections, best_by_t):
    by={int(s['table_index']):s for s in sections}
    rows=[]
    for ta in sorted(by):
        for tb in sorted(by):
            if tb<=ta: continue
            A=by[ta]['rows']; B=by[tb]['rows']
            fa=A[:,0]; fb=B[:,0]
            for ia,f in enumerate(fa):
                jj=np.where(np.isclose(fb,f,rtol=1e-7,atol=max(1e-8,1e-7*abs(float(f)))))[0]
                for ib in jj:
                    rec={'table_a':ta,'table_b':tb,'frequency_hz':float(f),'note':'same frequency, different candidate response state; equality NOT required'}
                    for c in range(1,8):
                        rec[f'f{c}_a']=float(A[ia,c]); rec[f'f{c}_b']=float(B[ib,c])
                    # Compare best-decoded state responses only as a descriptive state ratio.
                    za=best_by_t.get(ta,{}).get('z'); zb=best_by_t.get(tb,{}).get('z')
                    if za is not None and zb is not None:
                        va=za[ia]; vb=zb[ib]
                        if np.isfinite(va.real) and np.isfinite(vb.real) and abs(va)>1e-30:
                            rr=vb/va
                            rec['state_b_over_a_mag']=float(abs(rr)); rec['state_b_minus_a_phase_deg']=float(np.degrees(np.angle(rr)))
                    rows.append(rec)
    return rows


def _v45_write_csv(path, rows):
    import csv
    path=Path(path)
    if not rows:
        path.write_text('',encoding='utf-8'); return str(path)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return str(path)


def _v45_write_state_response_csv(outdir,ch,sections,best_by_t):
    rows=[]
    for s in sections:
        ti=int(s['table_index']); a=np.asarray(s['rows'],float); info=best_by_t.get(ti,{})
        z=info.get('z')
        for i,row in enumerate(a):
            rec={'channel':ch,'table':ti,'frequency_hz':float(row[0]),'representation':info.get('representation','')}
            for c in range(1,8): rec[f'f{c}']=float(row[c])
            if z is not None:
                zz=z[i]; rec.update({'decoded_real':float(zz.real),'decoded_imag':float(zz.imag),'decoded_magnitude':float(abs(zz)),'decoded_phase_deg':float(np.degrees(np.angle(zz)))})
            rows.append(rec)
    return _v45_write_csv(Path(outdir)/f'{ch}_V45_multi_response_states.csv',rows)


def v45_decode_rxcal_multi_response_states(rxcal_path, channel_states=None, output_dir=None):
    rxcal_path=Path(rxcal_path); blob=rxcal_path.read_bytes()
    anchors=[m.start() for m in re.finditer(b'BCM05-B',blob)]
    block_starts=[max(0,a-10) for a in anchors]
    if len(block_starts)<5:
        fallback=[256+i*V45_BLOCK_BYTES for i in range(5) if 256+i*V45_BLOCK_BYTES<len(blob)]
        if len(fallback)>len(block_starts): block_starts=fallback
    output_dir=Path(output_dir or (rxcal_path.parent/(rxcal_path.stem+'_V45_RXCAL'))); output_dir.mkdir(parents=True,exist_ok=True)
    labels=['E1','E2','H1','H2','H3']
    summary={'resolver_version':'V45','rxcal_file':str(rxcal_path),
             'policy':'T0..T3 treated as independent candidate response states; no crossover equality; f0 frequency; f1/f2 representation resolved per state; f3..f7 descriptors only; diagnostic-only.',
             'channels':{}}
    report=['V45 RXCAL MULTI-RESPONSE STATE DECODER','='*116,f'RXCAL: {rxcal_path}',f'Output folder: {output_dir}','',
            'MODEL: T0..T3 are independent candidate receiver-response states, NOT stitched frequency sections.',
            'NO CROSSOVER-EQUALITY CONSTRAINT is used. Shared frequencies are reported only as same-frequency/different-state diagnostics.',
            'REPRESENTATIONS TESTED PER STATE: Cartesian f1+i*f2, swapped Cartesian, amplitude+phase(rad), amplitude+phase(deg), and inverse variants.',
            'f3..f7 are retained only as state/control descriptors. Safety: diagnostic only; no RXCAL filter is attached to MTH5.','']
    for bi,b0 in enumerate(block_starts[:5]):
        ch=labels[bi]; state=(channel_states or {}).get(ch,{}) if isinstance(channel_states,dict) else {}
        sections=_v45_unpack_fixed(blob,b0)
        best_by_t={}; state_records=[]
        report += [f'[{ch}] block={bi} start={b0}',f'  recording_state={state}']
        chsum={'block_index':bi,'block_start':b0,'recording_state':state,'response_states':{}}
        for sec in sections:
            ti=int(sec['table_index']); a=_v45_valid_frequency_rows(sec['rows'])
            if not len(a): continue
            cands=_v45_repr_candidates(a)
            ranked=[]
            for name,z in cands.items():
                m=_v45_representation_metrics(a[:,0],z,raw_amp=a[:,1],raw_phase=a[:,2],kind=name)
                ranked.append((m.get('score',float('inf')),name,z,m))
            ranked.sort(key=lambda q:q[0])
            score,name,z,metrics=ranked[0]
            margin=(ranked[1][0]-score) if len(ranked)>1 and np.isfinite(ranked[1][0]) and np.isfinite(score) else float('nan')
            best_by_t[ti]={'representation':name,'z':z,'metrics':metrics,'score_margin':margin}
            desc=_v45_state_descriptor(a)
            fmin=float(np.min(a[:,0])); fmax=float(np.max(a[:,0]))
            chsum['response_states'][str(ti)]={
                'n_records':int(len(a)),'frequency_min_hz':fmin,'frequency_max_hz':fmax,
                'best_representation':name,'best_metrics':metrics,'score_margin':margin,
                'top_candidates':[{'representation':nm,'score':float(sc),**met} for sc,nm,_,met in ranked[:8]],
                'control_descriptors':desc}
            report.append(f'  T{ti}: n={len(a)} band={fmin:g}..{fmax:g} Hz')
            report.append(f'    BEST={name} score={score:.6g} margin={margin:.6g}')
            report.append(f"    |H| median={metrics.get('median_magnitude',float('nan')):.6g} range={metrics.get('min_magnitude',float('nan')):.6g}..{metrics.get('max_magnitude',float('nan')):.6g}; phase {metrics.get('phase_start_deg',float('nan')):.3f}->{metrics.get('phase_end_deg',float('nan')):.3f} deg")
            report.append(f"    smooth: dlog|H|/dlogf={metrics.get('median_abs_dlogmag_dlogf',float('nan')):.6g}, dphase/dlogf={metrics.get('median_abs_dphase_rad_dlogf',float('nan')):.6g}, max phase step={metrics.get('max_unwrapped_phase_step_deg',float('nan')):.3f} deg")
            report.append('    TOP: '+', '.join([f'{nm}={sc:.4g}' for sc,nm,_,_ in ranked[:4]]))
            report.append('    CONTROL: '+', '.join([f"f{c}:unique={desc[f'f{c}'].get('n_unique',0)} med={desc[f'f{c}'].get('median',float('nan')):.6g}" for c in range(3,8)]))
        shared=_v45_shared_frequency_diagnostics(sections,best_by_t)
        shared_csv=_v45_write_csv(output_dir/f'{ch}_V45_shared_frequency_state_diagnostics.csv',shared)
        resp_csv=_v45_write_state_response_csv(output_dir,ch,sections,best_by_t)
        chsum['shared_frequency_diagnostics_csv']=shared_csv; chsum['multi_response_states_csv']=resp_csv
        summary['channels'][ch]=chsum
        report.append(f'  shared-frequency diagnostics={shared_csv}')
        report.append(f'  decoded state responses={resp_csv}')
        report.append('')
    report += ['INTERPRETATION GUIDE:','-'*116,
               '  * Do NOT compare T0/T1/T2/T3 at a shared frequency as if they were one curve; they are candidate acquisition/calibration states.',
               '  * A good representation should be smooth within each state band and should not require arbitrary section normalization.',
               '  * State/control descriptors f3..f7 are reported for later mapping to GA/PG/LP/sample-rate/AC-DC metadata.',
               '  * The best representation is a hypothesis ranking, not yet authorization to attach binary RXCAL to MTH5.',
               '  * Next decisive step: map T-state descriptors to recording states and/or compare one official EMpower-exported rxcal.json.']
    sp=output_dir/'V45_RXCAL_multi_response_state_summary.json'; rp=output_dir/'V45_RXCAL_multi_response_state_report.txt'
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8'); rp.write_text('\n'.join(report),encoding='utf-8')
    summary['summary_json']=str(sp); summary['report_txt']=str(rp); return summary

# Override the compatibility entry point used by the GUI/export workflow.
def v45_decode_rxcal_sensorserial_scal(rxcal_path, channel_states=None, output_dir=None):
    return v45_decode_rxcal_multi_response_states(rxcal_path, channel_states, output_dir)


# =============================================================================
# V46 RXCAL BINARY RECORD TYPE RESOLVER (authoritative V46 diagnostic core)
# =============================================================================
# Purpose:
#   Resolve the 32-byte RXCAL record layout BEFORE assigning transfer-function
#   semantics.  Each 4-byte word is interpreted simultaneously as float32,
#   uint32 and int32, and each half-word as uint16/int16.  This is specifically
#   designed to detect mixed-type structs where integer enum/state words were
#   previously misread as IEEE-754 subnormal floats.
#
# Safety:
#   Diagnostic only. No binary RXCAL filter is attached to MTH5.

V46_TABLES = V45_TABLES
V46_RECORD_BYTES = 32
V46_BLOCK_BYTES = V45_BLOCK_BYTES


def _v46_word_views(word4: bytes):
    """Return lossless simultaneous interpretations of one 4-byte little-endian word."""
    if len(word4) != 4:
        raise ValueError("V46 word must contain exactly 4 bytes")
    f32 = struct.unpack('<f', word4)[0]
    u32 = struct.unpack('<I', word4)[0]
    i32 = struct.unpack('<i', word4)[0]
    u16a, u16b = struct.unpack('<HH', word4)
    i16a, i16b = struct.unpack('<hh', word4)
    b0, b1, b2, b3 = word4
    return {
        'hex_le': word4.hex().upper(),
        'hex_u32': f'0x{u32:08X}',
        'f32': float(f32),
        'u32': int(u32),
        'i32': int(i32),
        'u16_lo': int(u16a), 'u16_hi': int(u16b),
        'i16_lo': int(i16a), 'i16_hi': int(i16b),
        'bytes_dec': f'{b0},{b1},{b2},{b3}',
    }


def _v46_float_class(x: float):
    if np.isnan(x): return 'nan'
    if np.isposinf(x): return '+inf'
    if np.isneginf(x): return '-inf'
    if x == 0.0: return 'zero'
    ax = abs(float(x))
    tiny = np.finfo(np.float32).tiny
    if ax < tiny: return 'subnormal'
    return 'normal'


def _v46_word_role_hint(v):
    """Conservative forensic hints only; never used as a decoder decision."""
    f=float(v['f32']); u=int(v['u32']); i=int(v['i32'])
    fc=_v46_float_class(f)
    hints=[]
    if fc == 'subnormal' and u <= 65535:
        hints.append(f'LIKELY_INTEGER_OR_ENUM(u32={u})')
    if fc == 'zero' and u == 0:
        hints.append('ZERO_BITS')
    if 0 < u <= 65535 and (fc in ('subnormal','nan','+inf','-inf') or abs(f) < 1e-30):
        hints.append('SMALL_U32')
    if 0 < abs(i) <= 65535 and u > 0x7FFFFFFF and abs(f) < 1e-30:
        hints.append('SMALL_I32')
    if np.isfinite(f) and 1e-6 <= abs(f) <= 1e7:
        hints.append('PLAUSIBLE_FLOAT_MAGNITUDE')
    return ';'.join(dict.fromkeys(hints)) or 'UNRESOLVED'


def _v46_unpack_records_raw(blob: bytes, block_start: int):
    sections=[]
    for ti, rel, n in V46_TABLES:
        records=[]
        start=block_start+rel
        for ri in range(n):
            off=start+ri*V46_RECORD_BYTES
            raw=blob[off:off+V46_RECORD_BYTES]
            if len(raw) != V46_RECORD_BYTES:
                break
            words=[]
            for wi in range(8):
                wraw=raw[wi*4:(wi+1)*4]
                v=_v46_word_views(wraw)
                v['word_index']=wi
                v['role_hint']=_v46_word_role_hint(v)
                words.append(v)
            records.append({'record_index':ri,'absolute_offset':off,'raw_hex':raw.hex().upper(),'words':words})
        sections.append({'table_index':int(ti),'relative_offset':int(rel),'absolute_start':int(start),'records':records})
    return sections


def _v46_find_block_starts(blob: bytes):
    anchors=[m.start() for m in re.finditer(b'BCM05-B',blob)]
    starts=[max(0,a-10) for a in anchors]
    fallback=[256+i*V46_BLOCK_BYTES for i in range(5) if 256+i*V46_BLOCK_BYTES < len(blob)]
    if len(starts)<5 and len(fallback)>len(starts):
        starts=fallback
    return starts[:5]


def _v46_write_csv(path, rows):
    path=Path(path)
    if not rows:
        path.write_text('',encoding='utf-8'); return str(path)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return str(path)


def _v46_channel_rows(ch, bi, b0, sections):
    rows=[]
    for sec in sections:
        ti=sec['table_index']
        for rec in sec['records']:
            base={'channel':ch,'block_index':bi,'block_start':b0,'table':ti,
                  'table_relative_offset':sec['relative_offset'],
                  'record_index':rec['record_index'],'record_absolute_offset':rec['absolute_offset'],
                  'record_hex':rec['raw_hex']}
            # one CSV row per 4-byte word, preserving exact raw location
            for w in rec['words']:
                r=dict(base)
                r.update({'word':w['word_index'],'word_byte_offset':rec['absolute_offset']+4*w['word_index'],
                          'hex_le':w['hex_le'],'hex_u32':w['hex_u32'],'f32':w['f32'],
                          'float_class':_v46_float_class(w['f32']),'u32':w['u32'],'i32':w['i32'],
                          'u16_lo':w['u16_lo'],'u16_hi':w['u16_hi'],'i16_lo':w['i16_lo'],'i16_hi':w['i16_hi'],
                          'bytes_dec':w['bytes_dec'],'role_hint':w['role_hint']})
                rows.append(r)
    return rows


def _v46_fingerprint_sections(sections):
    out={}
    for sec in sections:
        ti=sec['table_index']; recs=sec['records']; sm={}
        for wi in range(8):
            vals=[r['words'][wi] for r in recs]
            u32=[v['u32'] for v in vals]; i32=[v['i32'] for v in vals]; f32=[v['f32'] for v in vals]
            classes=[_v46_float_class(x) for x in f32]
            sm[f'w{wi}']={
                'u32_unique':sorted(set(u32)),
                'i32_unique':sorted(set(i32)),
                'f32_unique_count':len(set(float(x) for x in f32 if np.isfinite(x))),
                'float_classes':{c:classes.count(c) for c in sorted(set(classes))},
                'constant_bits':len(set(u32))==1,
                'hex_unique':[f'0x{x:08X}' for x in sorted(set(u32))[:32]],
                'small_integer_candidate': bool(u32 and len(set(u32))<=8 and max(u32)<=65535),
            }
        out[f'T{ti}']=sm
    return out


def _v46_cross_channel_fingerprint(channel_results):
    """Find words whose bit patterns are identical across E1/E2/H1/H2/H3 for a given table."""
    result={}
    labels=list(channel_results)
    for ti,_,_ in V46_TABLES:
        tkey=f'T{ti}'; result[tkey]={}
        for wi in range(8):
            sigs=[]
            for ch in labels:
                sec=next((s for s in channel_results[ch]['sections'] if s['table_index']==ti),None)
                if not sec: continue
                sig=tuple(r['words'][wi]['u32'] for r in sec['records'])
                sigs.append((ch,sig))
            identical=bool(sigs) and all(sig==sigs[0][1] for _,sig in sigs[1:])
            result[tkey][f'w{wi}']={'identical_across_channels':identical,
                                    'signatures':{ch:[f'0x{x:08X}' for x in sig] for ch,sig in sigs}}
    return result


def v46_decode_rxcal_binary_record_types(rxcal_path, channel_states=None, output_dir=None):
    rxcal_path=Path(rxcal_path); blob=rxcal_path.read_bytes()
    starts=_v46_find_block_starts(blob)
    if not starts:
        raise ValueError('No RXCAL channel blocks found.')
    output_dir=Path(output_dir or (rxcal_path.parent/(rxcal_path.stem+'_V46_RXCAL')))
    output_dir.mkdir(parents=True,exist_ok=True)
    labels=['E1','E2','H1','H2','H3']
    summary={'resolver_version':'V46','rxcal_file':str(rxcal_path),
             'policy':'Binary forensic only: 32-byte records decoded simultaneously as f32/u32/i32/u16/i16/hex; no transfer-function representation selected; no MTH5 attachment.',
             'channels':{}}
    report=['V46 RXCAL BINARY RECORD TYPE RESOLVER','='*118,
            f'RXCAL: {rxcal_path}',f'File bytes: {len(blob)}',f'Output folder: {output_dir}','',
            'PURPOSE: determine the true 32-byte record struct before assigning RXCAL transfer-function semantics.',
            'Each 4-byte word is interpreted simultaneously as float32 / uint32 / int32; halfwords as uint16 / int16.',
            'IMPORTANT: IEEE-754 subnormal float + small uint32 is flagged as a likely integer/enum word.',
            'NO response ranking, NO crossover constraint, NO RXCAL filter attachment.','']
    channel_results={}
    for bi,b0 in enumerate(starts[:5]):
        ch=labels[bi] if bi<len(labels) else f'CH{bi}'
        state=(channel_states or {}).get(ch,{}) if isinstance(channel_states,dict) else {}
        sections=_v46_unpack_records_raw(blob,b0)
        channel_results[ch]={'sections':sections}
        csv_path=_v46_write_csv(output_dir/f'{ch}_V46_word_views.csv',_v46_channel_rows(ch,bi,b0,sections))
        fp=_v46_fingerprint_sections(sections)
        summary['channels'][ch]={'block_index':bi,'block_start':b0,'recording_state':state,
                                 'word_views_csv':csv_path,'table_fingerprints':fp}
        report += [f'[{ch}] block={bi} start={b0}',f'  recording_state={state}',f'  word views={csv_path}']
        for tkey,tdata in fp.items():
            report.append(f'  {tkey}:')
            for wi in range(8):
                d=tdata[f'w{wi}']
                uu=d['u32_unique']
                sample=', '.join(str(x) for x in uu[:8])
                flags=[]
                if d['constant_bits']: flags.append('CONSTANT_BITS')
                if d['small_integer_candidate']: flags.append('SMALL_INTEGER_CANDIDATE')
                cls=', '.join(f'{k}:{v}' for k,v in d['float_classes'].items())
                report.append(f"    w{wi}: u32_unique=[{sample}] f32_unique={d['f32_unique_count']} classes={{{cls}}} {' '.join(flags)}")
        report.append('')
    cross=_v46_cross_channel_fingerprint(channel_results)
    summary['cross_channel_fingerprints']=cross
    report += ['CROSS-CHANNEL IDENTICAL BIT-PATTERN TEST','-'*118]
    for tkey,wd in cross.items():
        same=[w for w,d in wd.items() if d['identical_across_channels']]
        report.append(f"  {tkey}: identical across E1/E2/H1/H2/H3 -> {', '.join(same) if same else '(none)'}")
    report += ['', 'INTERPRETATION RULES','-'*118,
               '  * If a word is a subnormal float but reinterprets to a small uint32 (e.g., 1, 119, 124), treat integer/enum as the leading hypothesis.',
               '  * Words identical across all five channels are more likely shared state/control metadata than channel-specific analog response samples.',
               '  * Frequency-like values should vary systematically with record index and remain plausible as float32.',
               '  * Do not infer Re/Im or magnitude/phase until mixed-type word roles are established.',
               '  * Magnetic model is no longer defaulted to MTC-185 when metadata is absent; blank means unresolved.']
    sp=output_dir/'V46_RXCAL_binary_record_type_summary.json'
    rp=output_dir/'V46_RXCAL_binary_record_type_report.txt'
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    rp.write_text('\n'.join(report),encoding='utf-8')
    summary['summary_json']=str(sp); summary['report_txt']=str(rp)
    return summary


# =============================================================================
# V47 OFFICIAL-JSON-SUPERVISED RXCAL DECODER
# =============================================================================
# Official Phoenix schema constraints used here:
#   T0 = 10 kHz low-pass response
#   T1 = 1 kHz low-pass response
#   T2 = 100 Hz low-pass response
#   T3 = 10 Hz low-pass response
# Each table is an independent receiver response. Crossover equality is NOT used.
#
V47_TABLES = (
    (0, 84,   13, 10000.0, "LP_10kHz"),
    (1, 512,   7,  1000.0, "LP_1kHz"),
    (2, 812,   5,   100.0, "LP_100Hz"),
    (3, 1144,  5,    10.0, "LP_10Hz"),
)
V47_RECORD_BYTES = 32
V47_BLOCK_BYTES = 1464

def _v47_find_block_starts(blob):
    anchors = [m.start() for m in re.finditer(b"BCM05-B", blob)]
    starts = [max(0, a - 10) for a in anchors]
    if len(starts) < 5:
        fallback = [
            256 + i * V47_BLOCK_BYTES for i in range(5)
            if 256 + i * V47_BLOCK_BYTES < len(blob)
        ]
        if len(fallback) > len(starts):
            starts = fallback
    return starts[:5]

def _v47_unpack_table(blob, block_start, rel, n):
    rows = []
    for i in range(n):
        p = block_start + rel + i * V47_RECORD_BYTES
        q = p + V47_RECORD_BYTES
        if q > len(blob):
            break
        rows.append(struct.unpack("<8f", blob[p:q]))
    a = np.asarray(rows, dtype=float)
    if not len(a):
        return a
    good = np.isfinite(a[:, 0]) & (a[:, 0] > 0.0) & (a[:, 0] < 1e7)
    return a[good]

def _v47_candidates(a):
    f1 = np.asarray(a[:, 1], dtype=float)
    f2 = np.asarray(a[:, 2], dtype=float)
    return {
        "cartesian_f1_plus_i_f2": f1 + 1j * f2,
        "cartesian_f2_plus_i_f1": f2 + 1j * f1,
        "amp_phase_rad_f1_f2": f1 * np.exp(1j * f2),
        "amp_phase_deg_f1_f2": f1 * np.exp(1j * np.deg2rad(f2)),
        "absamp_phase_rad_f1_f2": np.abs(f1) * np.exp(1j * f2),
        "absamp_phase_deg_f1_f2": np.abs(f1) * np.exp(1j * np.deg2rad(f2)),
    }

def _v47_score(freq, z, lp_hz):
    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    mag = np.abs(z)
    phase = np.unwrap(np.angle(z))
    good = np.isfinite(freq) & np.isfinite(mag) & np.isfinite(phase) & (freq > 0) & (mag > 0)
    if int(np.sum(good)) < max(3, len(freq) // 2):
        return 1e9, {"reason": "insufficient finite positive samples"}

    f = freq[good]
    m = mag[good]
    p = phase[good]
    order = np.argsort(f)
    f, m, p = f[order], m[order], p[order]
    lm = np.log(np.clip(m, 1e-20, None))
    lf = np.log(np.clip(f, 1e-20, None))

    med = float(np.median(m))
    unity_pen = abs(math.log(max(med, 1e-20))) * 6.0
    range_pen = float(np.mean(np.maximum(0.0, np.abs(lm) - math.log(3.0)))) * 4.0

    if len(f) > 1:
        dph = np.diff(p)
        dmag = np.diff(lm)
        smooth_phase = float(np.median(np.abs(dph)))
        smooth_mag = float(np.median(np.abs(dmag)))
        dlf = np.diff(lf)
        slope = dmag / np.where(np.abs(dlf) > 1e-12, dlf, 1e-12)
        high = f[1:] >= 0.2 * lp_hz
        grow_pen = float(np.median(np.maximum(0.0, slope[high]))) if np.any(high) else float(np.median(np.maximum(0.0, slope)))
    else:
        smooth_phase = smooth_mag = grow_pen = 0.0

    score = unity_pen + 1.5 * smooth_phase + smooth_mag + 3.0 * grow_pen + range_pen
    return float(score), {
        "median_magnitude": med,
        "min_magnitude": float(np.min(m)),
        "max_magnitude": float(np.max(m)),
        "phase_low_deg": float(np.degrees(p[0])),
        "phase_high_deg": float(np.degrees(p[-1])),
        "median_abs_phase_step_deg": float(np.degrees(smooth_phase)),
        "median_abs_logmag_step": smooth_mag,
        "high_frequency_growth_penalty": grow_pen,
        "lp_hz": float(lp_hz),
    }

def _v47_confidence(ranked):
    if not ranked:
        return "NONE", 0.0
    if len(ranked) == 1:
        return "LOW", 0.0
    s0, s1 = ranked[0][0], ranked[1][0]
    margin = float(s1 - s0)
    rel = margin / max(abs(float(s0)), 1e-9)
    if margin > 2.0 and rel > 0.25:
        return "HIGH", margin
    if margin > 0.5 and rel > 0.08:
        return "MEDIUM", margin
    return "LOW", margin

def _v47_json_header(rxcal_path):
    stem = Path(rxcal_path).stem
    serial = ""
    m = re.match(r"(\d+)", stem)
    if m:
        serial = m.group(1)
    return {
        "manufacturer": "Phoenix Geophysics",
        "file_type": "receiver calibration",
        "file_version": "1.0",
        "instrument_type": "MTU-5C",
        "inst_serial": serial,
        "cal_data": [],
    }

def v47_decode_rxcal_official_json_supervised(rxcal_path, channel_states=None, output_dir=None):
    rxcal_path = Path(rxcal_path)
    blob = rxcal_path.read_bytes()
    starts = _v47_find_block_starts(blob)
    if len(starts) < 5:
        raise ValueError(f"Expected 5 RXCAL channel blocks; found {len(starts)}.")

    output_dir = Path(output_dir or (rxcal_path.parent / (rxcal_path.stem + "_V47_RXCAL")))
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["E1", "E2", "H1", "H2", "H3"]

    summary = {
        "resolver_version": "V47.1",
        "rxcal_file": str(rxcal_path),
        "official_schema_supervision": {
            "response_order": {
                "T0": "10 kHz low-pass response",
                "T1": "1 kHz low-pass response",
                "T2": "100 Hz low-pass response",
                "T3": "10 Hz low-pass response",
            },
            "json_fields": ["freq", "magnitude", "phs_deg"],
            "receiver_magnitude_expectation": "normalized near unity",
            "crossover_continuity_constraint": False,
        },
        "channels": {},
    }

    candidate_json = _v47_json_header(rxcal_path)
    report = [
        "V47.1 RXCAL OFFICIAL-JSON-SUPERVISED DECODER",
        "=" * 116,
        f"RXCAL: {rxcal_path}",
        f"Output folder: {output_dir}",
        "",
        "OFFICIAL RESPONSE-STATE MODEL:",
        "  T0 = 10 kHz LP",
        "  T1 = 1 kHz LP",
        "  T2 = 100 Hz LP",
        "  T3 = 10 Hz LP",
        "  T0-T3 are independent response states. NO crossover-equality constraint.",
        "",
    ]

    rows_out = []
    all_gate_ok = True

    for bi, b0 in enumerate(starts[:5]):
        ch = labels[bi]
        state = (channel_states or {}).get(ch, {}) if isinstance(channel_states, dict) else {}
        ch_summary = {
            "block_index": bi,
            "block_start": int(b0),
            "recording_state": state,
            "response_states": {},
        }
        chan_json = {"tag": ch, "num_of_responses": 4, "chan_data": []}
        report.append(f"[{ch}] block={bi} start={b0} state={state}")

        for ti, rel, n, lp_hz, lp_name in V47_TABLES:
            a = _v47_unpack_table(blob, b0, rel, n)
            if not len(a):
                ch_summary["response_states"][str(ti)] = {"status": "NO_DATA", "lp_hz": lp_hz}
                all_gate_ok = False
                continue

            ranked = []
            for name, z in _v47_candidates(a).items():
                score, metrics = _v47_score(a[:, 0], z, lp_hz)
                ranked.append((score, name, z, metrics))
            ranked.sort(key=lambda q: q[0])

            score, name, z, metrics = ranked[0]
            confidence, margin = _v47_confidence(ranked)
            state_ok = confidence in ("HIGH", "MEDIUM")
            all_gate_ok = all_gate_ok and state_ok

            order = np.argsort(a[:, 0])
            f = np.asarray(a[:, 0], float)[order]
            zz = np.asarray(z, complex)[order]
            mag = np.abs(zz)
            phs = np.degrees(np.angle(zz))

            chan_json["chan_data"].append({
                "num_records": int(len(f)),
                "freq": [float(x) for x in f],
                "magnitude": [float(x) for x in mag],
                "phs_deg": [float(x) for x in phs],
            })

            ch_summary["response_states"][str(ti)] = {
                "lp_name": lp_name,
                "lp_hz": lp_hz,
                "n_records": int(len(f)),
                "best_representation": name,
                "confidence": confidence,
                "score": float(score),
                "score_margin": float(margin),
                "metrics": metrics,
                "top_candidates": [
                    {"representation": nm, "score": float(sc), "metrics": met}
                    for sc, nm, _, met in ranked[:6]
                ],
            }

            report.append(
                f"  T{ti} {lp_name}: {name}; confidence={confidence}; "
                f"score={score:.6g}; margin={margin:.6g}; "
                f"|H| median={metrics.get('median_magnitude', float('nan')):.6g}"
            )

            for jj, oi in enumerate(order):
                rows_out.append({
                    "channel": ch,
                    "table": ti,
                    "lp_name": lp_name,
                    "lp_hz": lp_hz,
                    "confidence": confidence,
                    "representation": name,
                    "frequency_hz": float(f[jj]),
                    "magnitude": float(mag[jj]),
                    "phase_deg": float(phs[jj]),
                    "raw_f1": float(a[oi, 1]),
                    "raw_f2": float(a[oi, 2]),
                    "raw_f3": float(a[oi, 3]),
                    "raw_f4": float(a[oi, 4]),
                    "raw_f5": float(a[oi, 5]),
                    "raw_f6": float(a[oi, 6]),
                    "raw_f7": float(a[oi, 7]),
                })

        candidate_json["cal_data"].append(chan_json)
        summary["channels"][ch] = ch_summary
        report.append("")

    csv_path = output_dir / "V47_RXCAL_decoded_response_candidates.csv"
    if rows_out:
        import csv
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    unverified = output_dir / (rxcal_path.stem + "_V47_UNVERIFIED_rxcal.json")
    unverified.write_text(json.dumps(candidate_json, indent=2, ensure_ascii=False), encoding="utf-8")

    candidate_path = ""
    if all_gate_ok:
        p = output_dir / (rxcal_path.stem + "_V47_candidate_rxcal.json")
        p.write_text(json.dumps(candidate_json, indent=2, ensure_ascii=False), encoding="utf-8")
        candidate_path = str(p)

    summary["all_states_passed_confidence_gate"] = bool(all_gate_ok)
    summary["decoded_csv"] = str(csv_path)
    summary["unverified_json"] = str(unverified)
    summary["candidate_json"] = candidate_path

    report += [
        "CONFIDENCE GATE",
        "-" * 116,
        f"All 5 channels x 4 LP states passed MEDIUM/HIGH gate: {all_gate_ok}",
        f"UNVERIFIED JSON: {unverified}",
        f"candidate rxcal.json: {candidate_path if candidate_path else '(not emitted)'}",
        "",
        "RULES:",
        "  * Shared-frequency disagreement between T0/T1/T2/T3 is NOT penalized.",
        "  * Binary-derived JSON remains reverse-engineered until compared against official EMpower export.",
        "  * LOW-confidence states block production-candidate JSON emission.",
        "  * No binary-derived RXCAL is attached automatically to MTH5 in V47.",
    ]

    sp = output_dir / "V47_RXCAL_official_json_supervised_summary.json"
    rp = output_dir / "V47_RXCAL_official_json_supervised_report.txt"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rp.write_text("\n".join(report), encoding="utf-8")
    summary["summary_json"] = str(sp)
    summary["report_txt"] = str(rp)
    return summary


def main():
    app=PhoenixMTU5CApp(); app.mainloop()

if __name__=="__main__":
    main()
