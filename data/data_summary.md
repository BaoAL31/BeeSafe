# BeeSafe Data Summary

- Data directory: `data`
- Annotation CSV files scanned: **4**
- CSV files used for dataset total: **1**
- Dataset total samples (deduplicated): **13509**
- Rows with bounding boxes (label > 0): **3947** (29.22%)
- Malformed lines: **0**
- Missing image paths: **0**
- Note: `gt.csv` is the full dataset; `train|test|val/gt_one.csv` are the same samples split into subsets. Global totals use `gt.csv` only so counts match the official 13,509 figure.

## Label Distribution (Overall)

| Label | Count | Ratio |
|---:|---:|---:|
| 0 | 9562 | 70.78% |
| 1 | 3083 | 22.82% |
| 3 | 864 | 6.40% |

## Label Meanings

- `0`: healthy (VarroaDataset class 0)
- `1`: Varroa-infected (VarroaDataset class 1)
- `3`: unknown (present in some exports; not defined in official README)

## Per-File Breakdown

| File | Samples | BBox Samples | BBox Ratio | Malformed | Missing Images | Labels |
|---|---:|---:|---:|---:|---:|---|
| `gt.csv` | 13509 | 3947 | 29.22% | 0 | 0 | 0:9562<br>1:3083<br>3:864 |
| `test/gt_one.csv` | 3408 | 942 | 27.64% | 0 | 0 | 0:2466<br>1:736<br>3:206 |
| `train/gt_one.csv` | 8225 | 2554 | 31.05% | 0 | 0 | 0:5671<br>1:1974<br>3:580 |
| `val/gt_one.csv` | 1876 | 451 | 24.04% | 0 | 0 | 0:1425<br>1:373<br>3:78 |

## Sample images (one per label)

*No sample images were produced (no matching image files on disk, or use `--sample-images 0` to skip this section).*
