# pyTRACMT v1.0.1 — SoftwareX transparency and testing revision

This maintenance release directly addresses the SoftwareX editorial concerns regarding
software testing and the provenance/security of the embedded Tukey biweight parameter table.

## Changes since v1.0.0

- Added an automated `pytest` suite (7 tests) covering Tukey-table integrity and loading,
  representative Tukey calculations, MT19937-64 reproducibility, robust filtering,
  spectral segmentation, and Tukey weighting.
- Removed the opaque Base85/zlib-compressed Tukey table from the Python source.
- Moved the Tukey parameter table to the human-readable
  `data/tukey_biweight_parameters.csv`.
- Added a pinned SHA-256 integrity check before parsing the table.
- Added `THIRD_PARTY_NOTICES.md` documenting the table's provenance from TRACMT
  (`TableOfTukeysBiweightParameters.h`) and its upstream BSD-3-Clause license.
- Added `SECURITY.md` explaining that the table is numeric text only and is not executed.
- Added the MIT `LICENSE` for pyTRACMT.
- Added GitHub Actions CI to run the test suite on Python 3.10 and 3.11.

## Verification

Local test result for this release package:

`7 passed`

## Licensing

pyTRACMT source code is distributed under the MIT License. The transcribed Tukey
biweight parameter table is derived from TRACMT by Yoshiya Usui and retains the
upstream BSD-3-Clause attribution described in `THIRD_PARTY_NOTICES.md`.
