# BeeSafe

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

### Download audio dataset (optional)

For acoustic model training, download the Zenodo beehive audio dataset:

```bash
# From the repository root
python data/download_zenodo_bee_audio.py --output-dir data/zenodo_bee_audio
```

This downloads ~3.7 GB of annotated beehive audio recordings with QueenBee/NO_QueenBee labels.

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
  test/videos/
```

Each line in `gt_one.csv`: `<relative_image_path> <label>` (label 0 = healthy,
1 or 3 = infected).

For audio data (if downloaded):
```
data/
  zenodo_bee_audio/
    *.wav             # audio files with labels in filename
    *.lab             # optional label files
```

---

## Training

### Visual Model Training

```bash
python -m modeling.training.train_mcunet_classification [OPTIONS]
```

#### Key arguments

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

#### Example

```bash
python -m modeling.training.train_mcunet_classification \
    --epochs 100 \
    --lr 1e-3 \
    --early-stopping-patience 10 \
    --pretrained
```

#### What it produces

- `modeling/checkpoints/mcunet/<net-id>_best.pt` — best checkpoint (by val infected recall)
- `modeling/checkpoints/mcunet/<net-id>_metrics.json` — training summary (metrics, hyperparams, paths)
- `modeling/checkpoints/mcunet/tensorboard/` — TensorBoard event files

#### Metrics logged

Per epoch (console + TensorBoard):

- **loss**, **accuracy**, **infected recall** (TP / (TP + FN) on infected class)
- Checkpoint saved when val infected recall improves

After training, the best checkpoint is evaluated on the **test split** and
results are saved to the metrics JSON.

#### TensorBoard

```bash
python -m tensorboard --logdir modeling/checkpoints/mcunet/tensorboard
```

Open http://localhost:6006 in a browser.

#### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Class imbalance | Apply weighted loss (healthy: 1.0, infected: 2.42) |
| Single controlled source | Augmentation pipeline: Gaussian blur, brightness jitter, horizontal flip, random rotation |
| Overfitting to Zenodo distribution | Dropout, early stopping, high batch size |
| Metric misalignment | Use infected class recall as primary stopping criterion |
| Model too heavy for real-time Jetson inference | Use small architectures/frameworks like YOLOv8n or MobileNetV3; benchmark latency and peak memory on device |

#### GPU

Training automatically uses CUDA if available (`torch.cuda.is_available()`).
No extra flag is needed. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### Acoustic Model Training

```bash
python -m modeling.training.train_acoustic [OPTIONS]
```

Trains MCUNet on mel-spectrograms extracted from beehive audio for normal vs abnormal classification.

#### Key arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--audio-dir` | `data/zenodo_bee_audio` | Directory containing audio files |
| `--sr` | `16000` | Sample rate for audio processing |
| `--n-mels` | `64` | Number of mel bands for spectrogram |
| `--n-fft` | `2048` | FFT size for spectrogram |
| `--hop-length` | `512` | Hop length for spectrogram |
| `--segment-duration` | `2.0` | Duration of audio segments (seconds) |
| `--net-id` | `mcunet-in3` | MCUNet model ID from model zoo |
| `--epochs` | `30` | Maximum training epochs |
| `--batch-size` | `64` | Batch size for train and eval |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--num-workers` | `2` | Number of data loading workers |
| `--no-class-weights` | off | Disable inverse-frequency CE weights |
| `--save-dir` | `modeling/checkpoints/acoustic` | Where to save best checkpoint and metrics |
| `--no-tensorboard` | off | Disable TensorBoard logging |
| `--early-stopping-patience` | `5` | Stop after N epochs without val metric improvement |
| `--val-ratio` | `0.15` | Fraction of files for validation |
| `--test-ratio` | `0.15` | Fraction of files for test |
| `--seed` | `42` | Random seed for reproducibility |

#### Example

```bash
python -m modeling.training.train_acoustic \
    --epochs 50 \
    --lr 1e-3 \
    --early-stopping-patience 5 \
    --audio-dir data/zenodo_bee_audio
```

#### What it produces

- `modeling/checkpoints/acoustic/mcunet_acoustic_best.pt` — best checkpoint (by val abnormal recall)
- `modeling/checkpoints/acoustic/mcunet_acoustic_metrics.json` — training summary
- `modeling/checkpoints/acoustic/tensorboard/` — TensorBoard event files (if enabled)

#### Metrics logged

Per epoch (console + TensorBoard):

- **loss**, **accuracy**, **abnormal recall** (TP / (TP + FN) on abnormal class)
- Checkpoint saved when val abnormal recall improves

After training, the best checkpoint is evaluated on the **test split** and
results are saved to the metrics JSON.

#### GPU

Training automatically uses CUDA if available (`torch.cuda.is_available()`).
No extra flag is needed.

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

---

## Fusion Logic Example

The following table shows how the acoustic and visual model outputs are fused to determine the final hive state:

| Acoustic model output | Visual model output | Final hive state (fusion) |
|-----------------------|---------------------|---------------------------|
| normal                | Varroa‑negative     | Healthy                   |
| normal                | Varroa‑positive     | Varroa Infestation (early)|
| abnormal              | Varroa‑negative     | Queenless / Stressed      |
| abnormal              | Varroa‑positive     | Critical: Varroa + Stress |