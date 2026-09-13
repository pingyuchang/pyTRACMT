# pyTRACMT

**pyTRACMT** is an open-source Python implementation of the principal TRACMT magnetotelluric (MT) processing workflow with an integrated graphical user interface (GUI) and command-line interface. The source-aligned implementation is intended to reproduce the principal numerical behavior, processing philosophy, and file workflow of TRACMT while providing a more accessible and extensible Python environment.

## Main capabilities

- Ordinary remote-reference (RR) transfer-function estimation
- Robust multivariate remote-reference / RRMS processing
- Sequential robust M-estimation
- Standard and robust prewhitening
- Robust time-domain spike filtering
- Squared-coherence-based spectral segment rejection
- High-pass, low-pass, and notch filtering
- Frequency-domain sensor calibration
- Segment-based FFT spectral processing
- Zxx, Zxy, Zyx, and Zyy impedance estimation
- Apparent resistivity and phase calculation
- Parametric, bootstrap, jackknife, robust-bootstrap, and strict/full-refit bootstrap uncertainty estimation
- TRACMT-style `param.dat`, log, and convergence output
- CSV and basic EDI export
- GUI-based configuration, time-series preview, response plotting, uncertainty/error-bar display, and PNG export

## Version

Current SoftwareX reference release: **V1.0.1**

Release: https://github.com/pingyuchang/pyTRACMT/releases/tag/V1.0.1

V1.0.1 adds automated software testing, GitHub Actions continuous
integration, explicit third-party licensing documentation, and a transparent
human-readable representation of the TRACMT-derived Tukey biweight parameter
table with SHA-256 integrity verification.

## Requirements

Recommended environment:

- Python 3.10 or 3.11
- NumPy
- Pandas
- SciPy
- Matplotlib
- Tkinter (normally included with Python)
- OpenPyXL (optional/supporting workflows)

Install Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the GUI

```bash
python pytracmt.py
```

## Command-line execution

```bash
python pytracmt.py path/to/param.dat path/to/output_folder
```

## Reproducibility and SoftwareX validation

The repository contains three TRACMT comparison configurations based on the `sample_Usui` dataset:

- non-robust remote reference
- ordinary robust remote reference
- RRMS

See [`examples/sample_Usui/README.md`](examples/sample_Usui/README.md) for the dataset layout and reproduction procedure. The original TRACMT reference outputs used for comparison are included under `validation/TRACMT_reference_results/`.

The raw validation time series are distributed separately as the GitHub Release asset `sample_Usui_data.zip` because the uncompressed text files exceed ordinary GitHub per-file limits.

## Documentation

Additional installation and GUI documentation is available in the `docs/` directory.

## Scientific references

Please cite the associated SoftwareX article when available and the relevant TRACMT methodology papers:

1. Usui, Y. (2024). Prewhitening of magnetotelluric data using a robust filter and robust PARCOR. *Bulletin of the Earthquake Research Institute, University of Tokyo*, 99, 1–20.
2. Usui, Y., Uyeshima, M., Sakanaka, S., Hashimoto, T., Ichiki, M., Kaida, T., Yamaya, Y., Ogawa, Y., Masuda, M., & Akiyama, T. (2024). New robust remote reference estimator using robust multivariate linear regression. *Geophysical Journal International*, 238, 943–959.
3. Usui, Y. et al. (2025). Application of the fast and robust bootstrap method to the uncertainty analysis of the magnetotelluric transfer function. *Geophysical Journal International*, 242, ggaf162.

## Authors

- Ping-Yu Chang — Department of Earth Sciences, National Central University, Taiwan
- Yoshiya Usui — Earthquake Research Institute, The University of Tokyo, Japan

## License

pyTRACMT V1.0.1 source code is released under the MIT License.
The TRACMT-derived Tukey biweight parameter table is separately documented
in THIRD_PARTY_NOTICES.md with its upstream BSD-3-Clause attribution. See [`LICENSE`](LICENSE).

## Validation-data permission

The `sample_Usui` validation dataset was provided by Yoshiya Usui and is distributed with his permission for pyTRACMT validation and reproducibility. See `NOTICE.md`.
