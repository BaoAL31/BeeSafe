# BeeSafe Data Summary

The annotations are split into three subsets: **train**, **validation**, and **test** (`train/gt_one.csv`: 8225 samples; `val/gt_one.csv`: 1876; `test/gt_one.csv`: 3408). The full list of all samples is in `gt.csv` (13509 samples).

## Per-File Breakdown

| File | Samples | BBox Samples | Labels |
|---|---:|---:|---|
| `gt.csv` | 13509 | 3947 | 0:9562<br>1:3083<br>3:864 |
| `test/gt_one.csv` | 3408 | 942 | 0:2466<br>1:736<br>3:206 |
| `train/gt_one.csv` | 8225 | 2554 | 0:5671<br>1:1974<br>3:580 |
| `val/gt_one.csv` | 1876 | 451 | 0:1425<br>1:373<br>3:78 |

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

## Sample images (3 per infected label)

### Label 1

![Label 1](summary_samples/sample_01.png)

*`test/videos/2017-10-17_1-39-36/2017-10-17_1-39-36.mp4-bee_id_5908-65985-1.png` - class **1** - 1 box(es)*

![Label 1](summary_samples/sample_02.png)

*`train/videos/2017-09-20_19-24-55/2017-09-20_19-24-55.mp4-bee_id_3097-8040-1.png` - class **1** - 2 box(es)*

![Label 1](summary_samples/sample_03.png)

*`test/videos/2017-09-01_10-54-26/2017-09-01_10-54-26.mp4-bee_id_7904-24525-1.png` - class **1** - 1 box(es)*

### Label 3

![Label 3](summary_samples/sample_04.png)

*`train/videos/2017-08-28_09-30-00-1_500_dirty_glass/2017-08-28_09-30-00-1_500_dirty_glass.mp4-bee_id_7133-32115-1.png` - class **3** - 1 box(es)*

![Label 3](summary_samples/sample_05.png)

*`train/videos/2017-09-25_16-03-38-2/2017-09-25_16-03-38.mp4-bee_id_438-14745-1.png` - class **3** - 1 box(es)*

![Label 3](summary_samples/sample_06.png)

*`val/videos/2017-09-29_15-31-49/2017-09-29_15-31-49.mp4-bee_id_48-1245-1.png` - class **3** - 1 box(es)*
