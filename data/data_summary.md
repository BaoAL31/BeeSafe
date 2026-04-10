# BeeSafe Data Summary

The annotations are split into three subsets: **train**, **validation**, and **test** (`train/gt_one.csv`: 8225 samples; `val/gt_one.csv`: 1876; `test/gt_one.csv`: 3408). The full list of all samples is in `gt.csv` (13509 samples).

- **Total samples:** 13509
- **Samples with bounding boxes:** 3947 (29.22% of total)

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

### Label 1

![Label 1](summary_samples/sample_01.png)

*`test/videos/2017-10-17_1-39-36/2017-10-17_1-39-36.mp4-bee_id_5908-65985-1.png` - class **1** - 1 box(es)*

### Label 3

![Label 3](summary_samples/sample_02.png)

*`train/videos/2017-08-28_09-30-00-1_500_dirty_glass/2017-08-28_09-30-00-1_500_dirty_glass.mp4-bee_id_7133-32115-1.png` - class **3** - 1 box(es)*

### Label 0

![Label 0](summary_samples/sample_03.png)

*`val/videos/2017-09-01_3-01-01/2017-09-01_3-01-01.mp4-bee_id_2454-60900-1.png` - class **0** - negative (no box)*
