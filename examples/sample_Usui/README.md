# sample_Usui validation dataset

This dataset is the reproducibility example used to compare pyTRACMT with TRACMT in the associated SoftwareX manuscript. The dataset was provided by Yoshiya Usui and is distributed with his permission for validation and reproducibility of pyTRACMT.

## Channels

The example uses six separate time-series channels sampled at 32 Hz:

- `ex.txt` — local electric Ex
- `ey.txt` — local electric Ey
- `hx.txt` — local magnetic Hx
- `hy.txt` — local magnetic Hy
- `hrx.txt` — remote-reference magnetic Hx
- `hry.txt` — remote-reference magnetic Hy

Each case processes 2,764,800 samples per channel.

## Large dataset download

The six raw text files are too large for ordinary GitHub file storage. For the SoftwareX release, upload `sample_Usui_data.zip` as a GitHub Release asset. Download and extract it so that the local repository contains:

```
examples/sample_Usui/data/ex.txt
examples/sample_Usui/data/ey.txt
examples/sample_Usui/data/hx.txt
examples/sample_Usui/data/hy.txt
examples/sample_Usui/data/hrx.txt
examples/sample_Usui/data/hry.txt
```

The public `param.dat` files use relative paths to this `data` directory.

## Validation cases

1. `NonRobustRemoteReference/param.dat` — ordinary remote-reference calculation without M-estimator weighting; parametric uncertainty setting.
2. `OrdinaryRobustRemoteReference/param.dat` — ordinary remote-reference workflow using the supplied validation configuration and fixed-weight bootstrap uncertainty setting.
3. `RRMS/param.dat` — Remote Reference Multivariate S-estimator workflow using the supplied validation configuration and robust-bootstrap uncertainty setting.

The original TRACMT reference outputs are preserved under `validation/TRACMT_reference_results/`.

## Run

From the repository root, activate the Python environment and run a case, for example:

```bash
python pytracmt.py examples/sample_Usui/RRMS/param.dat outputs/RRMS
```

Compare the generated `response_functions.csv` and `apparent_resistivity_and_phase.csv` with the corresponding files under `validation/TRACMT_reference_results/RRMS/`.
