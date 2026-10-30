
## 2. Totals

Added LOC per theme (sum of positive ~LOC; "rows" = feature rows in that theme; cross-report duplicates counted once):

| Theme | Rows | ~LOC added |
|---|---|---|
| repack | 19 | 3751 |
| tp-core | 33 | 2460 |
| pipeline-stages | 20 | 1990 |
| spec-decode | 40 | 1790 |
| context-runtime | 28 | 1532 |
| peer-ar | 10 | 1379 |
| kv-memory | 12 | 966 |
| model-qwen | 15 | 937 |
| model-dsv4 | 19 | 781 |
| debug-trace | 16 | 780 |
| kernels-fusion | 20 | 630 |
| mmq-mmvq | 12 | 532 |
| flashnext-moe | 2 | 250 |
| build-misc | 18 | 241 |
| model-other | 7 | 179 |
| **Total** | 271 | **18198** |

Added LOC per class (primary letter; "other" = mixed or cosmetic):

| Class | Rows | ~LOC added |
|---|---|---|
| A | 58 | 2018 |
| B | 70 | 7648 |
| C | 69 | 5666 |
| E | 57 | 1833 |
| F | 12 | 926 |
| other | 5 | 107 |

Class key (from reports): A essential gfx906/TP compat, B perf, C functional extension, D superseded, E redundant/dead/debug, F unclear.
Rows marked "?" or with 0 added are excluded from sums. Quick-win LOC removed is listed in section 3, not subtracted here.
