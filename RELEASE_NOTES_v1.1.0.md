# pyTRACMT v1.1.0 — Release Notes

This release packages the V1.1 Python GUI/CLI, V1.1 user manual, automated tests, and licensing/provenance documentation.

## Highlights

- MTH5 input and metadata-aware instrument-response routing.
- Phoenix CTS calibration path for compatible raw-count data.
- No-RR, ordinary RR, and robust RRMS workflows.
- Rotation, uncertainty estimation, visualization, and exports.
- External TRACMT-derived Tukey biweight CSV with pinned SHA-256 verification, replacing the opaque embedded compressed payload.

## Installation

`python -m pip install -r requirements.txt`

`python pyTRACMT.py`

## Testing

Seven included unit tests passed locally on the packaging environment (`python -m pytest -q`). The CI workflow targets Python 3.10 and 3.11. Full field-data calibration and cross-version regression testing are not established by these unit tests.

## Compatibility and attribution

Keep `data/tukey_biweight_parameters.csv` beside `pyTRACMT.py` under the `data/` directory. pyTRACMT code is MIT licensed; see `THIRD_PARTY_NOTICES.md` for the upstream TRACMT-derived table attribution.

## Phoenix MTU-5A and MTU-5C Utilities

Additional utilities are provided for converting Phoenix magnetotelluric (MT) time-series recordings into MTH5 format, supporting both legacy MTU-5A and newer MTU-5C acquisition systems.

### Phoenix MTU-5C to MTH5 Converter

Location: `Phoenix_Utility/MTU5C/`

- Converts Phoenix MTU-5C recordings into MTH5 format.
- Supports multi-rate time-series recordings.
- Provides waveform visualization and recording-continuity verification.
- Supports RXCAL and SCAL instrument-response metadata.
- Preserves voltage-domain time-series data and instrument-response application states.
- Provides verification of exported MTH5 files.
- Includes an English user manual.

### Phoenix MTU-5A to MTH5 Converter

Location: `Phoenix_Utility/MTU5A/`

- Converts legacy Phoenix MTU-5A time-series recordings into MTH5 format.
- Reads TSn time-series files and associated TBL acquisition metadata.
- Supports multiple sampling rates.
- Preserves original ADC-count time-series data.
- Supports associated acquisition and calibration metadata.
- Includes an English user manual.

### Phoenix SysCal GUI v3.1

Location: `Phoenix_Utility/MTU5A/`

- Provides a graphical interface for the manufacturer's SysCal calibration workflow.
- Supports inspection and modification of TBL acquisition parameters.
- Supports selection of receiver and sensor calibration files.
- Executes the manufacturer's SysCal program to generate CTS calibration files.
- Provides execution logs and output verification.
- Includes an English user manual.

### Calibration and Data Compatibility

- MTU-5C time-series data are preserved in voltage units, with the corresponding instrument-response metadata maintained for subsequent processing.
- MTU-5A time-series data are preserved as raw ADC counts, allowing calibration to be applied during subsequent processing.
- Users should verify channel units, acquisition parameters, and instrument-response application states to avoid duplicate gain or response corrections.
- The manufacturer's SysCal executable is required for CTS generation and is not included in this repository.

### Availability

The Phoenix MTU-5A and MTU-5C utilities, together with their English user manuals, are available under the `Phoenix_Utility/` directory in the repository's `main` branch.
