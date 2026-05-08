"""Verification runner for multimodal hive health monitoring.

Runs inference on test audio and image pairs, fuses results, and outputs JSON logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

from modeling.fusion.fusion_engine import (
    FusionEngine,
    FusionConfig,
    BranchPrediction,
    run_inference_on_audio,
    run_inference_on_image,
)


def verify_pair(
    audio_path: Path,
    image_path: Path,
    acoustic_ckpt: Path,
    visual_ckpt: Path,
    config_path: Path,
    engine: FusionEngine,
) -> Dict:
    """Verify a single audio-image pair."""
    print(f"\n{'='*60}")
    print(f"Audio: {audio_path.name}")
    print(f"Image: {image_path.name}")
    print(f"{'='*60}")

    ac_pred = run_inference_on_audio(audio_path, acoustic_ckpt, config_path)
    print(f"  Acoustic: {ac_pred.class_name} (conf={ac_pred.confidence:.4f})")

    vis_pred = run_inference_on_image(image_path, visual_ckpt, config_path)
    print(f"  Visual:   {vis_pred.class_name} (conf={vis_pred.confidence:.4f})")

    result = engine.fuse(ac_pred, vis_pred)
    print(f"  Final:    {result.final_state}")
    print(f"  Health:   {result.health_score:.4f}")

    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify multimodal hive health fusion on test pairs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory with test audio files (overrides config)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory with test image files (overrides config)",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="JSON file with [[audio_path, image_path], ...] pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/verification_results.json"),
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    acoustic_ckpt = Path(config["paths"]["acoustic_checkpoint"])
    visual_ckpt = Path(config["paths"]["visual_checkpoint"])

    if not acoustic_ckpt.exists():
        print(f"Warning: Acoustic checkpoint not found: {acoustic_ckpt}")
        print("  Run train_mcunet_acoustic.py first.")
    if not visual_ckpt.exists():
        print(f"Warning: Visual checkpoint not found: {visual_ckpt}")
        print("  Run train_mcunet_classification.py first.")

    engine = FusionEngine(FusionConfig.from_yaml(args.config))

    results: List[Dict] = []

    if args.pairs:
        pairs_data = json.loads(args.pairs.read_text())
        for audio_str, image_str in pairs_data:
            audio_path = Path(audio_str)
            image_path = Path(image_str)
            if audio_path.exists() and image_path.exists():
                result_dict = verify_pair(
                    audio_path,
                    image_path,
                    acoustic_ckpt,
                    visual_ckpt,
                    args.config,
                    engine,
                )
                results.append(result_dict)
    else:
        audio_dir = args.audio_dir or Path(config["acoustic"]["audio_dir"]) / "test"
        image_dir = args.image_dir or Path(config["visual"]["data_dir"]) / "test"
        print(f"Audio dir: {audio_dir}, exists: {audio_dir.exists()}")
        print(f"Image dir: {image_dir}, exists: {image_dir.exists()}")

        if audio_dir.exists() and image_dir.exists():
            audio_files = sorted(audio_dir.glob("*.wav"))[:5]
            # Search for image files recursively in image_dir
            image_files = sorted(list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png")))[:5]
            print(f"Found {len(audio_files)} audio files and {len(image_files)} image files")

            for af in audio_files[: min(len(audio_files), len(image_files))]:
                for imf in image_files[:1]:
                    print(f"Processing pair: {af.name} and {imf.name}")
                    result_dict = verify_pair(
                        af, imf, acoustic_ckpt, visual_ckpt, args.config, engine
                    )
                    results.append(result_dict)
        else:
            print("Either audio_dir or image_dir does not exist")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n\nResults saved to: {args.output}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  {r['fusion']['final_state']:30s} | health={r['fusion']['health_score']:.3f}"
        )


if __name__ == "__main__":
    main()
