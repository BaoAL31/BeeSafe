# BeeSafe Modeling

Binary image-level classification: **healthy (0)** vs **infected (1)**.
Raw CSV labels 1 and 3 (different Varroa infection markers) are combined into a
single "infected" class.

## Setup

### Clone the MCUNet model zoo

The training scripts depend on the
[MCUNet](https://github.com/mit-han-lab/mcunet) model zoo (MIT license, NeurIPS
2020/2021). Clone it into `modeling/mcunet`:

```bash
# From the repository root
git clone https://github.com/mit-han-lab/mcunet.git modeling/mcunet
```

This gives you the `mcunet.model_zoo` package used to build lightweight
architectures (`mcunet-in0`..`in4`, `mbv2-w0.35`, etc.).

### Install dependencies

```bash
# From the repository root
pip install -r requirements.txt
```

This installs MCUNet in editable mode (from `modeling/mcunet`), PyTorch,
TensorBoard, and profiling dependencies (`psutil`, `pynvml`).

### Download the dataset

The dataset is the [VarroaDataset](https://zenodo.org/records/4085044) (CC BY 4.0)
from the Computer Vision Lab, TU Wien. Download these four files:

1. **`gt.csv`** — annotation file (13,509 samples)
2. **`train.zip`** — training images (~703 MB)
3. **`val.zip`** — validation images (~163 MB)
4. **`test.zip`** — test images (~292 MB)

Then extract and arrange them:

```bash
# From the repository root
mkdir -p data
# Move gt.csv into data/
mv gt.csv data/

# Extract each zip into data/ (creates data/train/, data/val/, data/test/)
unzip train.zip -d data/
unzip val.zip -d data/
unzip test.zip -d data/
```

### Data layout

After extraction, the scripts expect the following under `data/` (configurable
via `--data-dir`):

```
data/
  gt.csv              # full annotation file
  train/gt_one.csv
  val/gt_one.csv
  test/gt_one.csv
  train/videos/...    # images referenced by gt_one.csv
  val/videos/...
  test/videos/...
```

Each line in `gt_one.csv`: `<relative_image_path> <label>` (label 0 = healthy,
1 or 3 = infected).

---

## Training

```bash
python -m modeling.training.train_mcunet_classification [OPTIONS]
```

### Key arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root data directory containing train/val/test splits |
| `--net-id` | `mcunet-in3` | MCUNet model zoo architecture (`--list-net-ids` to see all) |
| `--epochs` | `10` | Maximum training epochs |
| `--batch-size` | `64` | Batch size for train and eval |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--no-class-weights` | off | Disable inverse-frequency CE weights (enabled by default) |
| `--pretrained` | off | Load ImageNet-pretrained MCUNet backbone weights |
| `--early-stopping-patience` | `0` | Stop after N epochs without val metric improvement (0 = disabled) |
| `--save-dir` | `modeling/checkpoints/mcunet` | Where to save best checkpoint and metrics JSON |
| `--no-tensorboard` | off | Disable TensorBoard logging |
| `--tensorboard-dir` | `<save-dir>/tensorboard` | Override TensorBoard log directory |
| `--download-tflite` | off | Download matching `.tflite` from MCUNet release after training |

### Example

```bash
python -m modeling.training.train_mcunet_classification \
    --epochs 100 \
    --lr 1e-3 \
    --early-stopping-patience 10 \
    --pretrained
```

### What it produces

- `modeling/checkpoints/mcunet/<net-id>_best.pt` — best checkpoint (by val infected recall)
- `modeling/checkpoints/mcunet/<net-id>_metrics.json` — training summary (metrics, hyperparams, paths)
- `modeling/checkpoints/mcunet/tensorboard/` — TensorBoard event files

### Metrics logged

Per epoch (console + TensorBoard):

- **loss**, **accuracy**, **infected recall** (TP / (TP + FN) on infected class)
- Checkpoint saved when val infected recall improves

After training, the best checkpoint is evaluated on the **test split** and
results are saved to the metrics JSON.

### TensorBoard

```bash
python -m tensorboard --logdir modeling/checkpoints/mcunet/tensorboard
```

Open http://localhost:6006 in a browser.

### GPU

Training automatically uses CUDA if available (`torch.cuda.is_available()`).
No extra flag is needed. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## Evaluation

```bash
python -m modeling.evaluation.evaluate classification [OPTIONS]
python -m modeling.evaluation.evaluate localization [OPTIONS]
```

### Classification evaluation

```bash
python -m modeling.evaluation.evaluate classification \
    --checkpoint modeling/checkpoints/mcunet/mcunet-in3_best.pt \
    --split test \
    --output-json results/classification_eval.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--data-dir` | `data` | Data root |
| `--split` | `test` | `test` or `val` |
| `--batch-size` | `64` | Batch size |
| `--output-json` | none | Save metrics to JSON file |
| `--skip-latency-memory` | off | Skip resource profiling |
| `--latency-warmup` | `2` | Warmup batches before timing |
| `--latency-max-batches` | all | Cap timed batches |

### Localization evaluation

```bash
python -m modeling.evaluation.evaluate localization \
    --checkpoint modeling/checkpoints/localization/localization_best.pt \
    --split test \
    --output-json results/localization_eval.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--data-dir` | `data` | Data root |
| `--split` | `test` | `test` or `val` |
| `--batch-size` | `8` | Batch size |
| `--score-thresh` | from checkpoint | Detection score threshold |
| `--output-json` | none | Save metrics to JSON file |
| `--skip-latency-memory` | off | Skip resource profiling |
| `--latency-warmup` | `2` | Warmup batches |
| `--latency-max-batches` | all | Cap timed batches |

### What it reports

**Accuracy metrics** (console + JSON):

- Loss, accuracy, infected recall (classification)
- Positive-image recall @ score threshold (localization)

**Resource profiling** (unless `--skip-latency-memory`):

| Metric | Description |
|--------|-------------|
| Latency per image | Mean and stdev in ms (CUDA-synced if GPU) |
| Peak CUDA alloc | `torch.cuda.max_memory_allocated` in MB |
| Peak RSS | Process resident memory high-water mark (psutil) |
| RSS before / after | Memory snapshot before and after inference |
| Peak CPU % | Highest CPU utilization during inference (psutil) |
| GPU utilization | GPU compute and VRAM usage via NVIDIA driver (pynvml) |

Example output:

```
classification | test | n=3408 | loss 0.3412 | acc 0.8750 | infected_recall 0.9200
  latency/img 1.234 ms (±0.089)
  peak CUDA alloc 48.2 MB | peak RSS 1204.3 MB | RSS delta +12.1 MB
  peak CPU 78% | NVIDIA GeForce RTX 3060 util 45% VRAM 1024/12288 MB (8%)
```
