# BeeSafe Demo Results

**Generated:** 2026-05-08 20:56:11
**Device:** cuda

## Acoustic Model Results

- **Checkpoint:** `modeling\checkpoints\acoustic\mcunet_acoustic_best.pt`
- **Model:** mcunet-in3
- **Samples:** 5
- **Accuracy:** 100.00% (5/5)

### Sample Predictions

| File | True Label | Predicted | Correct |
|------|------------|-----------|---------|
| CF003 - Active - Day - (216).wav | normal | normal | ✓ |
| CF003 - Active - Day - (216).wav | normal | normal | ✓ |
| CF003 - Active - Day - (216).wav | normal | normal | ✓ |
| CF003 - Active - Day - (216).wav | normal | normal | ✓ |
| CF003 - Active - Day - (216).wav | normal | normal | ✓ |

## Visual Model Results

- **Checkpoint:** `modeling\checkpoints\visual\mcunet-in3_best.pt`
- **Model:** mcunet-in3
- **Samples:** 10
- **Accuracy:** 80.00% (8/10)
- **Image Size:** 224x224

### Sample Predictions

| File | True Label | Predicted | Correct |
|------|------------|-----------|---------|
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | healthy | healthy | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | infected | infected | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | healthy | healthy | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | infected | healthy | ✗ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | healthy | healthy | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | infected | infected | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | infected | infected | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | healthy | healthy | ✓ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | infected | healthy | ✗ |
| 2017-08-28_09-30-00-1_500_dirty_glass.mp | healthy | healthy | ✓ |
