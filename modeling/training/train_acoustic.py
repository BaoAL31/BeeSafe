"""Train MCUNet on mel-spectrogram segments for beehive audio classification.

Usage:
    python -m modeling.training.train_acoustic [OPTIONS]

Uses MCUNet architecture (same as visual branch) for acoustic spectrogram classification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import librosa
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from mcunet.model_zoo import build_model


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract log-Mel spectrogram from audio array."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize per spectrogram
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def load_audio_segment(
    audio_path: Path,
    offset: float,
    duration: float,
    sr: int,
) -> Tuple[np.ndarray, int]:
    """Load a segment of audio from file."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        file_sr = info.samplerate
        start_frame = int(offset * file_sr)
        frames = int(duration * file_sr)
        audio, _ = sf.read(audio_path, start=start_frame, frames=frames)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        return audio, sr
    except ImportError:
        audio, file_sr = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)
        return audio, sr


class BeeAudioDataset(Dataset):
    """Dataset of mel-spectrogram segments from beehive audio files."""

    def __init__(
        self,
        audio_dir: Path,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_duration: float = 2.0,
        file_filter: Callable[[str], bool] | None = None,
        use_annotations: bool = False,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.samples: List[Tuple[Path, float, int]] = []

        audio_files = sorted(
            p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3")
        )

        for audio_path in audio_files:
            if file_filter and not file_filter(audio_path.name):
                continue

            label = self._parse_label(audio_path.name)

            try:
                import soundfile as sf

                info = sf.info(audio_path)
                duration = info.duration
            except ImportError:
                duration = librosa.get_duration(path=audio_path)

            n_segments = int(duration // segment_duration)
            for i in range(n_segments):
                offset = i * segment_duration
                self.samples.append((audio_path, offset, label))

        if not self.samples:
            raise RuntimeError(f"No usable audio samples found in {audio_dir}")

    @staticmethod
    def _parse_label(filename: str) -> int:
        """Parse label from filename. Returns 0=normal, 1=abnormal."""
        lower = filename.lower()
        if "no_queen" in lower or "queenless" in lower:
            return 1
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, offset, label = self.samples[idx]
        audio, _ = load_audio_segment(
            audio_path, offset, self.segment_duration, self.sr
        )
        spec = extract_mel_spectrogram(
            audio, self.sr, self.n_mels, self.n_fft, self.hop_length
        )
        # Convert to 3-channel by repeating (MCUNet expects 3 channels)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # [1, n_mels, time]
        spec_tensor = spec_tensor.repeat(3, 1, 1)  # [3, n_mels, time]
        return spec_tensor, label


def make_split(
    audio_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split audio files into train/val/test by file (not segment)."""
    import random

    random.seed(seed)

    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3")
    )

    random.shuffle(audio_files)
    n = len(audio_files)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    test_files = audio_files[:n_test]
    val_files = audio_files[n_test : n_test + n_val]
    train_files = audio_files[n_test + n_val :]

    return train_files, val_files, test_files


def count_class_frequencies(dataset: Dataset, num_classes: int) -> List[int]:
    """Count class occurrences in a dataset."""
    counts = [0] * num_classes
    if hasattr(dataset, "samples"):
        for _, _, label in dataset.samples:
            counts[label] += 1
    return counts


def cross_entropy_class_weights(
    counts: List[int], device: torch.device
) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    total = sum(counts)
    if total == 0:
        return torch.ones(len(counts), device=device)
    if len(counts) == 2:
        n0, n1 = counts[0], counts[1]
        if n1 < 1:
            return torch.tensor([1.0, 1.0], device=device)
        return torch.tensor([1.0, n0 / n1], device=device)
    w = torch.tensor(
        [total / (len(counts) * c) if c > 0 else 0.0 for c in counts],
        device=device,
    )
    return w


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run one epoch. Returns (loss, accuracy, abnormal_recall)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        logits = model(specs)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([], dtype=torch.long)
    labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([], dtype=torch.long)

    tp = ((preds_cat == 1) & (labels_cat == 1)).sum().item()
    support = (labels_cat == 1).sum().item()
    abnormal_recall = tp / support if support > 0 else float("nan")

    phase = "train" if is_train else "val  "
    n_pred_0 = (preds_cat == 0).sum().item()
    n_pred_1 = (preds_cat == 1).sum().item()
    n_label_0 = (labels_cat == 0).sum().item()
    n_label_1 = (labels_cat == 1).sum().item()
    print(
        f"  [{phase}] labels: 0={n_label_0} 1={n_label_1} | "
        f"preds: 0={n_pred_0} 1={n_pred_1} | "
        f"TP={tp} FN={support - tp} abnormal_recall={abnormal_recall:.4f}"
    )

    return avg_loss, acc, abnormal_recall


def _json_float(x: float) -> float | None:
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def modify_mcunet_for_spectrogram(model: nn.Module, n_time: int = 32) -> nn.Module:
    """Modify MCUNet to accept spectrogram input and produce correct output size.

    MCUNet expects 3-channel input (RGB). For spectrograms, we repeat the single
    channel 3 times. The model output is adapted via replace_classifier_head.
    """
    return model


def replace_classifier_head(model: nn.Module, num_classes: int) -> None:
    """Replace the classifier head for the number of classes."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return
        if hasattr(model.classifier, "in_features"):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    raise RuntimeError("Could not find classifier layer to replace.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MCUNet on beehive audio spectrograms (normal vs abnormal)."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/zenodo_bee_audio"),
        help="Directory containing audio files",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--n-mels", type=int, default=64, help="Number of mel bands")
    parser.add_argument("--n-time", type=int, default=32, help="Time frames")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument(
        "--segment-duration", type=float, default=2.0, help="Segment duration (s)"
    )
    parser.add_argument(
        "--net-id",
        type=str,
        default="mcunet-in3",
        help="MCUNet model ID from model zoo",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("modeling/checkpoints/acoustic"))
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            "Download the dataset first with: python data/download_zenodo_bee_audio.py"
        )

    num_classes = 2

    print(f"Splitting files from {audio_dir} ...")
    train_files, val_files, test_files = make_split(
        audio_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")

    def file_filter(keep: List[Path]) -> Callable[[str], bool]:
        keep_names = {p.name for p in keep}
        return lambda name: name in keep_names

    train_ds = BeeAudioDataset(
        audio_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        segment_duration=args.segment_duration,
        file_filter=file_filter(train_files),
    )
    val_ds = BeeAudioDataset(
        audio_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        segment_duration=args.segment_duration,
        file_filter=file_filter(val_files),
    )
    test_ds = BeeAudioDataset(
        audio_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        segment_duration=args.segment_duration,
        file_filter=file_filter(test_files),
    )
    print(f"  Train segments: {len(train_ds)}")
    print(f"  Val segments:   {len(val_ds)}")
    print(f"  Test segments:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, input_size, _ = build_model(net_id=args.net_id, pretrained=False)
    replace_classifier_head(model, num_classes=num_classes)
    model = model.to(device)
    print(model)

    train_counts = count_class_frequencies(train_ds, num_classes)
    print(f"Class counts (train): {train_counts}")
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        ce_w = cross_entropy_class_weights(train_counts, device)
        print(f"Class weights: {ce_w.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=ce_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = args.save_dir / "tensorboard"
    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        try:
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"TensorBoard log dir: {tb_log_dir.resolve()}")
        except Exception as e:
            print(f"Warning: TensorBoard not available: {e}", file=sys.stderr)
            writer = None

    best_val_abnormal_recall = -1.0
    best_path = args.save_dir / "mcunet_acoustic_best.pt"
    epochs_without_improve = 0
    last_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            train_loss, train_acc, train_abl_rec = run_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_abl_rec = run_epoch(
                model, val_loader, criterion, None, device
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} abl_rec {train_abl_rec:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} abl_rec {val_abl_rec:.4f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/accuracy", train_acc, epoch)
                if not math.isnan(train_abl_rec):
                    writer.add_scalar("train/abnormal_recall", train_abl_rec, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                if not math.isnan(val_abl_rec):
                    writer.add_scalar("val/abnormal_recall", val_abl_rec, epoch)

            if not math.isnan(val_abl_rec) and val_abl_rec > best_val_abnormal_recall:
                best_val_abnormal_recall = val_abl_rec
                epochs_without_improve = 0
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "net_id": args.net_id,
                    "n_mels": args.n_mels,
                    "n_time": args.n_time,
                    "num_classes": num_classes,
                    "segment_duration": args.segment_duration,
                    "sr": args.sr,
                    "best_val_abnormal_recall": best_val_abnormal_recall,
                    "val_acc_at_best": val_acc,
                    "train_class_counts": train_counts,
                }
                torch.save(ckpt, best_path)
            else:
                if args.early_stopping_patience > 0:
                    epochs_without_improve += 1
                    if epochs_without_improve >= args.early_stopping_patience:
                        print("Early stopping triggered.")
                        break

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, test_abl_rec = run_epoch(
            model, test_loader, criterion, None, device
        )
    finally:
        if writer is not None:
            writer.close()

    metrics = {
        "net_id": args.net_id,
        "n_mels": args.n_mels,
        "n_time": args.n_time,
        "num_classes": num_classes,
        "best_val_abnormal_recall": _json_float(best_val_abnormal_recall),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_abnormal_recall": _json_float(test_abl_rec),
        "checkpoint": str(best_path.as_posix()),
    }
    metrics_path = args.save_dir / "mcunet_acoustic_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nBest checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
