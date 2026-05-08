"""Jetson deployment runner for real-time multimodal hive health monitoring.

Continuously monitors audio, triggers camera on abnormal detection,
fuses results, and logs hive health state.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml


class AudioCapture:
    """Continuous audio capture with sliding window."""

    def __init__(self, sample_rate: int = 16000, window_size: float = 2.0, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = []

    def start(self):
        """Start audio capture (placeholder for Jetson implementation)."""
        print(f"Starting audio capture at {self.sample_rate}Hz...")
        print("  (On Jetson: use PyAudio or similar for real audio input)")

    def get_window(self) -> Optional[bytes]:
        """Get next audio window (placeholder)."""
        return None

    def stop(self):
        """Stop audio capture."""
        print("Audio capture stopped.")


class CameraTrigger:
    """Camera capture triggered by acoustic anomaly or schedule."""

    def __init__(self, image_size: tuple = (96, 96), trigger_on_abnormal: bool = True):
        self.image_size = image_size
        self.trigger_on_abnormal = trigger_on_abnormal
        self.last_capture_time = 0
        self.capture_interval = 300

    def capture(self, force: bool = False) -> Optional[Path]:
        """Capture an image (placeholder for Jetson implementation)."""
        now = time.time()
        if not force and (now - self.last_capture_time) < self.capture_interval:
            return None

        print("  [Camera] Capturing image...")
        print("  (On Jetson: use OpenCV/CSI camera for actual capture)")
        self.last_capture_time = now
        return Path("captured_image.jpg")


class MCUNetInference:
    """Run inference using MCUNet models (TensorRT/ONNX optimized)."""

    def __init__(self, checkpoint_path: Path, config_path: Path, model_type: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model_type = model_type
        self.model = None
        self.config = None
        self.device = "cuda" if self._cuda_available() else "cpu"

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self):
        """Load model (placeholder for TensorRT/ONNX loading)."""
        print(f"Loading {self.model_type} model from {self.checkpoint_path}...")
        print(f"  Device: {self.device}")
        print("  (On Jetson: load TensorRT engine for optimal performance)")

        import yaml
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def predict(self, input_data) -> Dict:
        """Run inference (placeholder)."""
        if self.model_type == "acoustic":
            return {"class": "normal", "confidence": 0.95, "all_probs": {"normal": 0.95, "abnormal": 0.05}}
        else:
            return {"class": "Varroa-negative", "confidence": 0.90, "all_probs": {"Varroa-negative": 0.90, "Varroa-positive": 0.10}}


class HealthLogger:
    """Log hive health assessments."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: Dict):
        """Append result to JSON log."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"  [Log] {result['fusion']['final_state']} (score={result['fusion']['health_score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Jetson hive health monitor")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    deployment = config.get("deployment", {})
    audio_config = deployment.get("audio", {})
    camera_config = deployment.get("camera", {})

    print("="*60)
    print("BeeSafe Jetson Deployment Runner")
    print("="*60)

    audio_cap = AudioCapture(
        sample_rate=audio_config.get("sample_rate", 16000),
        window_size=audio_config.get("window_size", 2.0),
        overlap=audio_config.get("overlap", 0.5),
    )

    camera = CameraTrigger(
        image_size=tuple(camera_config.get("image_size", [96, 96])),
        trigger_on_abnormal=camera_config.get("trigger_on_abnormal", True),
    )

    acoustic_inf = MCUNetInference(
        Path(config["paths"]["acoustic_checkpoint"]), args.config, "acoustic"
    )
    visual_inf = MCUNetInference(
        Path(config["paths"]["visual_checkpoint"]), args.config, "visual"
    )

    from modeling.fusion.fusion_engine import FusionEngine, FusionConfig
    engine = FusionEngine(FusionConfig.from_yaml(args.config))

    logger = HealthLogger(Path(deployment.get("log_file", "logs/hive_health_log.json")))

    acoustic_inf.load()
    visual_inf.load()
    audio_cap.start()

    try:
        print("\nMonitoring started... (Ctrl+C to stop)\n")
        while True:
            audio_window = audio_cap.get_window()

            if audio_window or args.simulate:
                if args.simulate:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Simulating...")

                ac_result = acoustic_inf.predict(audio_window)
                print(f"  Acoustic: {ac_result['class']} ({ac_result['confidence']:.3f})")

                should_capture = (
                    camera.trigger_on_abnormal and ac_result['class'] == 'abnormal'
                ) or args.simulate

                if should_capture:
                    img_path = camera.capture(force=args.simulate)
                    if img_path or args.simulate:
                        vis_result = visual_inf.predict(img_path)
                        print(f"  Visual:   {vis_result['class']} ({vis_result['confidence']:.3f})")

                        from modeling.fusion.fusion_engine import BranchPrediction
                        ac_pred = BranchPrediction(
                            class_name=ac_result['class'],
                            confidence=ac_result['confidence'],
                            all_probs=ac_result.get('all_probs', {}),
                        )
                        vis_pred = BranchPrediction(
                            class_name=vis_result['class'],
                            confidence=vis_result['confidence'],
                            all_probs=vis_result.get('all_probs', {}),
                        )

                        result = engine.fuse(ac_pred, vis_pred)
                        log_entry = result.to_dict()
                        logger.log(log_entry)
                        print(f"  Final:    {result.final_state}")
                else:
                    print("  Camera:   (no trigger)")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        audio_cap.stop()
        print("Done.")


if __name__ == "__main__":
    main()
