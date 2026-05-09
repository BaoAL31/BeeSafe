# BeeSafe Demo

This folder contains a demo script that runs inference on both acoustic and visual BeeSafe models.

## Usage

```bash
# Run full demo with default settings
python demo/demo.py

# Run with limited samples (faster)
python demo/demo.py --max-audio-samples 5 --max-image-samples 10

# Specify custom checkpoints
python demo/demo.py \
    --acoustic-checkpoint modeling/checkpoints/acoustic/mcunet_acoustic_best.pt \
    --visual-checkpoint modeling/checkpoints/visual/mcunet-in3_best.pt

# Run on CPU
python demo/demo.py --device cpu
```

## Arguments

- `--acoustic-checkpoint`: Path to acoustic model checkpoint (default: `modeling/checkpoints/acoustic/mcunet_acoustic_best.pt`)
- `--visual-checkpoint`: Path to visual model checkpoint (default: `modeling/checkpoints/visual/mcunet-in3_best.pt`)
- `--audio-dir`: Directory containing audio files (default: `data/zenodo_bee_audio`)
- `--data-dir`: Directory containing image data (default: `data`)
- `--output`: Output path for results markdown (default: `demo/results.md`)
- `--max-audio-samples`: Maximum audio segments to process (default: 10)
- `--max-image-samples`: Maximum images to process (default: 20)
- `--device`: Device to run on - `cuda` or `cpu` (default: auto-detect)

## Data Requirements

### Visual Data
The visual demo requires:
- `data/train/gt_one.csv` - Training ground truth CSV
- `data/val/gt_one.csv` - Validation ground truth CSV
- `data/test/gt_one.csv` - Test ground truth CSV

### Acoustic Data
The acoustic demo requires audio files in `data/zenodo_bee_audio/`. To download:
```bash
python data/download_zenodo_bee_audio.py
```

## Output

The demo generates `demo/results.md` with:
- Model information (checkpoint, architecture)
- Accuracy results
- Sample predictions table

## Checkpoints

Trained checkpoints should be placed in:
- `modeling/checkpoints/acoustic/` - Acoustic model checkpoints
- `modeling/checkpoints/visual/` - Visual model checkpoints
