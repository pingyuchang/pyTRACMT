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
