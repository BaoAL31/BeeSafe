"""Demo script for BeeSafe acoustic and visual models.

Runs inference using trained checkpoints and generates a result markdown report.

Usage:
    python demo/demo.py [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

try:
    from mcunet.model_zoo import build_model
except ImportError:
    print("ERROR: mcunet not found. Install with: pip install mcunet", file=sys.stderr)
    sys.exit(1)

# ===================== Acoustic Model Components =====================

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract log-Mel spectrogram from audio array."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def load_audio_segment(
    audio_path: Path,
    offset: float,
    duration: float,
    sr: int,
) -> Tuple[np.ndarray, int]:
    """Load a segment of audio from file."""
    import librosa

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


class DemoAudioDataset(Dataset):
    """Simple dataset for demo audio inference."""

    def __init__(
        self,
        audio_dir: Path,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_duration: float = 2.0,
        max_segments: int = 10,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.samples: List[Tuple[Path, float, int, str]] = []

        audio_files = sorted(
            p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3")
        )

        for audio_path in audio_files:
            label = 1 if "no_queen" in audio_path.name.lower() or "queenless" in audio_path.name.lower() else 0
            label_str = "abnormal" if label == 1 else "normal"

            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
            except ImportError:
                duration = librosa.get_duration(path=audio_path)

            n_segments = min(int(duration // segment_duration), max_segments)
            for i in range(n_segments):
                offset = i * segment_duration
                self.samples.append((audio_path, offset, label, label_str))

            if len(self.samples) >= max_segments:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        audio_path, offset, label, label_str = self.samples[idx]
        audio, _ = load_audio_segment(audio_path, offset, self.segment_duration, self.sr)
        spec = extract_mel_spectrogram(audio, self.sr, self.n_mels, self.n_fft, self.hop_length)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1)
        return spec_tensor, label, label_str, audio_path.name


# ===================== Visual Model Components =====================

def parse_gt_line(line: str) -> Tuple[str, int]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("invalid ground-truth line")
    return parts[0], int(parts[1])


class DemoImageDataset(Dataset):
    """Simple dataset for demo image inference."""

    def __init__(self, csv_path: Path, data_dir: Path, max_samples: int = 20) -> None:
        self.samples: List[Tuple[Path, int, str]] = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.transform = transform

        count = 0
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_rel, raw_label = parse_gt_line(line)
            image_path = csv_path.parent / image_rel
            if not image_path.exists():
                continue
            label = 0 if raw_label == 0 else 1
            label_str = "healthy" if label == 0 else "infected"
            self.samples.append((image_path, label, label_str))
            count += 1
            if count >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        image_path, label, label_str = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.transform(img)
        return image, label, label_str, image_path.name


# ===================== Demo Runner =====================

class BeeSafeDemo:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.results: Dict = {}

    def _load_checkpoint_model(self, checkpoint: Dict, device: torch.device) -> nn.Module:
        """Load model from checkpoint handling classifier head mismatch."""
        net_id = checkpoint.get("net_id", "mcunet-in3")
        num_classes = checkpoint.get("num_classes", 2)

        model, _, _ = build_model(net_id=net_id, pretrained=False)

        # Replace classifier head to match checkpoint
        self._replace_classifier_head(model, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model

    def _replace_classifier_head(self, model: nn.Module, num_classes: int) -> None:
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
            if hasattr(model.classifier, "linear") and isinstance(model.classifier.linear, nn.Linear):
                in_features = model.classifier.linear.in_features
                model.classifier.linear = nn.Linear(in_features, num_classes)
                return

        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return

        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            return

        raise RuntimeError("Could not find classifier layer to replace")

    def run_acoustic_demo(
        self,
        checkpoint_path: Path,
        audio_dir: Path,
        max_segments: int = 10,
    ) -> Dict:
        """Run acoustic model demo."""
        print("\n" + "="*60)
        print("Running Acoustic Model Demo")
        print("="*60)

        if not checkpoint_path.exists():
            return {"error": f"Checkpoint not found: {checkpoint_path}"}

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        net_id = checkpoint.get("net_id", "mcunet-in3")

        model = self._load_checkpoint_model(checkpoint, self.device)
        model = model.to(self.device)
        model.eval()

        dataset = DemoAudioDataset(audio_dir, max_segments=max_segments)
        if len(dataset) == 0:
            return {"error": f"No audio samples found in {audio_dir}"}

        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        all_preds = []
        all_labels = []
        sample_results = []

        with torch.no_grad():
            for specs, labels, label_strs, names in loader:
                specs = specs.to(self.device)
                logits = model(specs)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                for i in range(len(names)):
                    sample_results.append({
                        "file": names[i],
                        "true_label": "abnormal" if labels[i].item() == 1 else "normal",
                        "pred_label": "abnormal" if preds[i].item() == 1 else "normal",
                        "correct": preds[i].item() == labels[i].item(),
                    })

        correct = sum(r["correct"] for r in sample_results)
        total = len(sample_results)
        accuracy = correct / total if total > 0 else 0

        return {
            "model_type": "acoustic",
            "checkpoint": str(checkpoint_path),
            "net_id": net_id,
            "num_samples": total,
            "accuracy": accuracy,
            "correct": correct,
            "samples": sample_results[:10],
        }

    def run_visual_demo(
        self,
        checkpoint_path: Path,
        csv_path: Path,
        data_dir: Path,
        max_samples: int = 20,
    ) -> Dict:
        """Run visual model demo."""
        print("\n" + "="*60)
        print("Running Visual Model Demo")
        print("="*60)

        if not checkpoint_path.exists():
            return {"error": f"Checkpoint not found: {checkpoint_path}"}

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        net_id = checkpoint.get("net_id", "mcunet-in3")

        model = self._load_checkpoint_model(checkpoint, self.device)
        model = model.to(self.device)
        model.eval()
        image_size = 224  # Standard input size

        dataset = DemoImageDataset(csv_path, data_dir, max_samples=max_samples)
        if len(dataset) == 0:
            return {"error": f"No image samples found in {csv_path}"}

        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        sample_results = []

        with torch.no_grad():
            for images, labels, label_strs, names in loader:
                images = images.to(self.device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                for i in range(len(names)):
                    sample_results.append({
                        "file": names[i],
                        "true_label": label_strs[i],
                        "pred_label": "infected" if preds[i].item() == 1 else "healthy",
                        "correct": preds[i].item() == labels[i].item(),
                    })

        correct = sum(r["correct"] for r in sample_results)
        total = len(sample_results)
        accuracy = correct / total if total > 0 else 0

        return {
            "model_type": "visual",
            "checkpoint": str(checkpoint_path),
            "net_id": net_id,
            "image_size": image_size,
            "num_samples": total,
            "accuracy": accuracy,
            "correct": correct,
            "samples": sample_results[:10],
        }

    def generate_report(self, output_path: Path) -> None:
        """Generate markdown report."""
        report = []
        report.append("# BeeSafe Demo Results")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Device:** {self.device}")
        report.append("")

        for key in ["acoustic", "visual"]:
            if key not in self.results:
                continue
            result = self.results[key]
            report.append(f"## {key.title()} Model Results")
            report.append("")

            if "error" in result:
                report.append(f"**Error:** {result['error']}")
                report.append("")
                continue

            report.append(f"- **Checkpoint:** `{result['checkpoint']}`")
            report.append(f"- **Model:** {result['net_id']}")
            report.append(f"- **Samples:** {result['num_samples']}")
            report.append(f"- **Accuracy:** {result['accuracy']:.2%} ({result['correct']}/{result['num_samples']})")
            if "image_size" in result:
                report.append(f"- **Image Size:** {result['image_size']}x{result['image_size']}")
            report.append("")

            report.append("### Sample Predictions")
            report.append("")
            report.append("| File | True Label | Predicted | Correct |")
            report.append("|------|------------|-----------|---------|")
            for s in result.get("samples", []):
                correct_mark = "✓" if s["correct"] else "✗"
                report.append(f"| {s['file'][:40]} | {s['true_label']} | {s['pred_label']} | {correct_mark} |")
            report.append("")

        output_path.write_text("\n".join(report), encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


# ===================== Main =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="BeeSafe Demo - Acoustic & Visual Models")
    parser.add_argument("--acoustic-checkpoint", type=Path, default=Path("modeling/checkpoints/acoustic/mcunet_acoustic_best.pt"))
    parser.add_argument("--visual-checkpoint", type=Path, default=Path("modeling/checkpoints/visual/mcunet-in3_best.pt"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/zenodo_bee_audio"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("demo/results.md"))
    parser.add_argument("--max-audio-samples", type=int, default=10)
    parser.add_argument("--max-image-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")
    args = parser.parse_args()

    demo = BeeSafeDemo(device=args.device)

    # Run acoustic demo
    if args.acoustic_checkpoint.exists() and args.audio_dir.exists():
        demo.results["acoustic"] = demo.run_acoustic_demo(
            args.acoustic_checkpoint, args.audio_dir, args.max_audio_samples
        )
    else:
        print("Skipping acoustic demo (checkpoint or audio dir not found)")

    # Run visual demo
    train_csv = args.data_dir / "train" / "gt_one.csv"
    if args.visual_checkpoint.exists() and train_csv.exists():
        demo.results["visual"] = demo.run_visual_demo(
            args.visual_checkpoint, train_csv, args.data_dir, args.max_image_samples
        )
    else:
        print("Skipping visual demo (checkpoint or data csv not found)")

    # Generate report
    demo.generate_report(args.output)


if __name__ == "__main__":
    main()
