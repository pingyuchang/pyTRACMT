# Security note

pyTRACMT does not execute the Tukey parameter data. The table is plain-text CSV and is loaded only as `float64` numeric values using `numpy.loadtxt`. The program verifies a pinned SHA-256 digest before parsing it. No `pickle`, `eval`, dynamic import, decompression, or executable binary payload is used for this table.
