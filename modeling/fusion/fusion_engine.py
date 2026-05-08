"""Late-fusion engine for multimodal hive health monitoring.

Combines acoustic and visual branch predictions using a decision table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class FusionConfig:
    """Configuration for the fusion engine."""
    mapping: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("normal", "Varroa-negative", "Healthy"),
            ("normal", "Varroa-positive", "Varroa Infestation (early)"),
            ("abnormal", "Varroa-negative", "Queenless / Stressed"),
            ("abnormal", "Varroa-positive", "Critical: Varroa + Stress"),
        ]
    )
    min_confidence: float = 0.5
    health_score_weights: Dict[str, float] = field(
        default_factory=lambda: {"acoustic": 0.5, "visual": 0.5}
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FusionConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f)
        fusion_config = config.get("fusion", {})
        return cls(
            mapping=fusion_config.get("mapping", cls().mapping),
            min_confidence=fusion_config.get("min_confidence", 0.5),
            health_score_weights=fusion_config.get(
                "health_score_weights", cls().health_score_weights
            ),
        )


@dataclass
class BranchPrediction:
    """Prediction from a single branch (acoustic or visual)."""
    class_name: str
    confidence: float
    all_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result from the fusion engine."""
    acoustic_pred: BranchPrediction
    visual_pred: BranchPrediction
    final_state: str
    health_score: float
    timestamp: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "acoustic": {
                "class": self.acoustic_pred.class_name,
                "confidence": self.acoustic_pred.confidence,
                "all_probs": self.acoustic_pred.all_probs,
            },
            "visual": {
                "class": self.visual_pred.class_name,
                "confidence": self.visual_pred.confidence,
                "all_probs": self.visual_pred.all_probs,
            },
            "fusion": {
                "final_state": self.final_state,
                "health_score": self.health_score,
            },
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class FusionEngine:
    """Late-fusion engine for hive health assessment."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Build a lookup table from (acoustic_class, visual_class) -> final_state."""
        self.lookup: Dict[Tuple[str, str], str] = {}
        for ac, vc, final in self.config.mapping:
            self.lookup[(ac, vc)] = final

    def fuse(
        self,
        acoustic_pred: BranchPrediction,
        visual_pred: BranchPrediction,
    ) -> FusionResult:
        """Fuse predictions from both branches."""
        ac_name = acoustic_pred.class_name
        vc_name = visual_pred.class_name

        final_state = self.lookup.get(
            (ac_name, vc_name), "Unknown: " + str((ac_name, vc_name))
        )

        health_score = self._compute_health_score(acoustic_pred, visual_pred)

        return FusionResult(
            acoustic_pred=acoustic_pred,
            visual_pred=visual_pred,
            final_state=final_state,
            health_score=health_score,
        )

    def _compute_health_score(
        self,
        acoustic_pred: BranchPrediction,
        visual_pred: BranchPrediction,
    ) -> float:
        """Compute overall health score (1.0 = healthy, 0.0 = critical).

        Based on risk scores from each branch:
        - normal/Varroa-negative -> low risk
        - abnormal/Varroa-positive -> high risk
        """
        w = self.config.health_score_weights

        ac_risk = 0.0
        if acoustic_pred.class_name == "abnormal":
            ac_risk = acoustic_pred.confidence
        else:
            ac_risk = 1.0 - acoustic_pred.confidence

        vc_risk = 0.0
        if visual_pred.class_name == "Varroa-positive":
            vc_risk = visual_pred.confidence
        else:
            vc_risk = 1.0 - visual_pred.confidence

        total_risk = ac_risk * w.get("acoustic", 0.5) + vc_risk * w.get("visual", 0.5)
        health_score = 1.0 - total_risk
        return max(0.0, min(1.0, health_score))


def load_model_checkpoint(checkpoint_path: str | Path) -> Dict:
    """Load a model checkpoint and return the state dict and metadata."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt


def run_inference_on_audio(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path = "config.yaml",
) -> BranchPrediction:
    """Run inference on a single audio file using MCUNet acoustic model."""
    import torch
    import librosa
    import numpy as np
    from mcunet.model_zoo import build_model

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ac_config = config["acoustic"]
    sr = ac_config["sr"]
    n_mels = ac_config["n_mels"]
    n_time = ac_config["n_time"]
    net_id = ac_config["net_id"]

    ckpt = load_model_checkpoint(checkpoint_path)
    model, _, _ = build_model(net_id=net_id, pretrained=False)

    num_classes = ckpt.get("num_classes", 2)
    from modeling.training.train_acoustic import replace_classifier_head

    replace_classifier_head(model, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    audio, _ = librosa.load(str(audio_path), sr=sr, duration=ac_config["segment_duration"])
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=ac_config["n_fft"], hop_length=ac_config["hop_length"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    spec_tensor = torch.from_numpy(mel_db).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    with torch.no_grad():
        logits = model(spec_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    class_names = ac_config["class_names"]
    pred_class = class_names[pred_idx]
    all_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}

    return BranchPrediction(
        class_name=pred_class, confidence=confidence, all_probs=all_probs
    )


def run_inference_on_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path = "config.yaml",
) -> BranchPrediction:
    """Run inference on a single image using MCUNet visual model."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from mcunet.model_zoo import build_model

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vis_config = config["visual"]
    image_size = vis_config["image_size"]
    net_id = vis_config["net_id"]

    ckpt = load_model_checkpoint(checkpoint_path)
    model, _, _ = build_model(net_id=net_id, pretrained=False)

    num_classes = ckpt.get("num_classes", 2)
    from modeling.training.train_visual_classification import replace_classifier_head

    replace_classifier_head(model, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    class_names = vis_config["class_names"]
    pred_class = class_names[pred_idx]
    all_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}

    return BranchPrediction(
        class_name=pred_class, confidence=confidence, all_probs=all_probs
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python fusion_engine.py <audio_file> <image_file> [config.yaml]")
        sys.exit(1)

    audio_path = sys.argv[1]
    image_path = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "config.yaml"

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ac_ckpt = config["paths"]["acoustic_checkpoint"]
    vis_ckpt = config["paths"]["visual_checkpoint"]

    print(f"Running acoustic inference on: {audio_path}")
    ac_pred = run_inference_on_audio(audio_path, ac_ckpt, config_path)
    print(f"  Prediction: {ac_pred.class_name} (confidence: {ac_pred.confidence:.4f})")

    print(f"Running visual inference on: {image_path}")
    vis_pred = run_inference_on_image(image_path, vis_ckpt, config_path)
    print(f"  Prediction: {vis_pred.class_name} (confidence: {vis_pred.confidence:.4f})")

    engine = FusionEngine(FusionConfig.from_yaml(config_path))
    result = engine.fuse(ac_pred, vis_pred)

    print(f"\nFusion Result:")
    print(f"  Final State: {result.final_state}")
    print(f"  Health Score: {result.health_score:.4f}")
    print(f"\nJSON output:\n{result.to_json()}")
