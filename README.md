# pyTRACMT v1.1.0

Open-source Python GUI/CLI for magnetotelluric (MT) time-series processing following the principal TRACMT workflow. The V1.1 release adds MTH5 input/calibration routing, Phoenix full-system CTS handling for appropriate raw-count inputs, horizontal tensor rotation, and No-RR processing in addition to ordinary remote-reference (RR) and robust RRMS options.

> **Validation notice:** Instrument response correction and physical units depend on correct source metadata, calibration files, electrode geometry, and channel orientation. Check results against independently validated reference data before scientific interpretation. The bundled automated tests cover selected numerical components, not end-to-end instrument calibration or all field configurations.

## Install

Recommended: Python 3.10 or 3.11. In a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

On some systems, Tkinter must be installed separately through the operating system or Python distribution.

## Required directory layout (GUI and CLI)

The external Tukey table is a **required release asset**. Download/extract the complete repository or release package and keep the following structure:

```text
pyTRACMT/
├── pyTRACMT.py
└── data/
    └── tukey_biweight_parameters.csv
```

The CSV is resolved relative to `pyTRACMT.py`, **not** the current working directory. It is loaded automatically when needed; no GUI file selection is required. Do not rename, edit, reformat, or move it. A missing file or SHA-256 mismatch raises an error during table loading. This requirement applies to both GUI and CLI usage.

## Run

```bash
python pyTRACMT.py
```

For command-line processing using an existing TRACMT-style parameter file:

```bash
python pyTRACMT.py param.dat
```

## Input and processing

- Separate calibrated single-channel time-series files; choose the correct input-unit mode.
- MTH5 run selection and conditional correction of unapplied instrument response stages.
- Phoenix raw-count MTH5 with a matching full-system CTS response, when applicable.
- No-RR, ordinary RR, and robust RRMS modes; independent remote magnetic channels are required for RR/RRMS.
- Spectral segmentation, optional preprocessing, impedance tensor estimation, uncertainty estimation, plotting, and export.

See [the V1.1 user manual](docs/pyTRACMT_User_Manual_V1.1.docx) for setup, calibration limitations, rotation conventions, and troubleshooting.

## Tukey biweight table

The TRACMT-derived numeric table is stored at `data/tukey_biweight_parameters.csv` and checked against a pinned SHA-256 digest before loading. **Keep the `data/` folder beside `pyTRACMT.py`**. The program does not execute the CSV. See [third-party notices](THIRD_PARTY_NOTICES.md) and [security notes](SECURITY.md).

## Tests

```bash
python -m pytest -q
```

The included tests cover selected Tukey-table and processing components. GitHub Actions runs the tests on Python 3.10 and 3.11. Passing tests is not proof of physical calibration accuracy.

## License and attribution

pyTRACMT code is distributed under the MIT License; the TRACMT-derived Tukey table has upstream BSD-3-Clause attribution documented in `THIRD_PARTY_NOTICES.md`. Original TRACMT: Yoshiya Usui.

## Release history

See [CHANGELOG.md](CHANGELOG.md) and [v1.1.0 release notes](RELEASE_NOTES_v1.1.0.md). Previous v1.0.x releases remain available in repository history/tags.
