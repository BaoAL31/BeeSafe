"""
Evaluate trained BeeSafe checkpoints.

Run from the repository root, for example:

  python -m modeling.evaluation.evaluate classification --checkpoint modeling/checkpoints/mcunet/mcunet-in3_best.pt
  python -m modeling.evaluation.evaluate localization --checkpoint modeling/checkpoints/localization/localization_best.pt

Latency uses CUDA synchronize + perf_counter; peak memory uses torch.cuda.max_memory_allocated
after warmup. CPU runs optionally report process RSS after inference if psutil is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from mcunet.model_zoo import build_model as mcunet_build

from modeling.training.train_localization import (
    BeeSafeDetectionDataset,
    build_model as build_localization_model,
    collate_fn,
    eval_pos_recall,
)
from modeling.training.train_mcunet_classification import (
    BeeSafeDataset,
    make_label_mapper,
    replace_classifier_head,
    run_epoch,
)


def _split_csv(data_dir: Path, split: str) -> Path:
    p = data_dir / split / "gt_one.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    return p


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _cuda_peak_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def _try_cpu_rss_mb() -> Optional[float]:
    try:
        import os

        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


def measure_classification_latency_memory(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    max_timed_batches: Optional[int],
) -> Dict[str, Any]:
    """
    Mean batch / per-image latency (ms) and peak CUDA memory during timed inference.
    CUDA peak is reset after warmup so it reflects steady-state forward passes only.
    """
    model.eval()
    batch_ms: List[float] = []
    per_image_ms: List[float] = []
    n_warmup_done = 0
    n_timed = 0
    cuda_peak_reset = False

    for _batch_idx, (images, _labels) in enumerate(loader):
        images = images.to(device)
        if n_warmup_done < warmup_batches:
            with torch.no_grad():
                model(images)
            _sync_if_cuda(device)
            n_warmup_done += 1
            continue

        if max_timed_batches is not None and n_timed >= max_timed_batches:
            break

        if not cuda_peak_reset:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                _sync_if_cuda(device)
            cuda_peak_reset = True

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(images)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        bs = images.size(0)
        batch_ms.append(elapsed_ms)
        per_image_ms.append(elapsed_ms / max(bs, 1))
        n_timed += 1

    out: Dict[str, Any] = {
        "warmup_batches_run": n_warmup_done,
        "timed_batches": n_timed,
        "latency_ms_per_batch_mean": statistics.mean(batch_ms) if batch_ms else None,
        "latency_ms_per_batch_stdev": statistics.stdev(batch_ms) if len(batch_ms) > 1 else None,
        "latency_ms_per_image_mean": statistics.mean(per_image_ms) if per_image_ms else None,
        "latency_ms_per_image_stdev": statistics.stdev(per_image_ms) if len(per_image_ms) > 1 else None,
        "peak_memory_cuda_mb": _cuda_peak_mb() if device.type == "cuda" and n_timed > 0 else None,
        "process_rss_mb_before": None,
        "process_rss_mb_after": None,
    }
    if device.type != "cuda" and n_timed > 0:
        out["process_rss_mb_after"] = _try_cpu_rss_mb()
    return out


def measure_localization_latency_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
    max_timed_batches: Optional[int],
) -> Dict[str, Any]:
    model.eval()
    batch_ms: List[float] = []
    per_image_ms: List[float] = []
    n_warmup_done = 0
    n_timed = 0
    cuda_peak_reset = False

    for _batch_idx, (images, _targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        if n_warmup_done < warmup_batches:
            with torch.no_grad():
                model(images)
            _sync_if_cuda(device)
            n_warmup_done += 1
            continue

        if max_timed_batches is not None and n_timed >= max_timed_batches:
            break

        if not cuda_peak_reset:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                _sync_if_cuda(device)
            cuda_peak_reset = True

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(images)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        bs = len(images)
        batch_ms.append(elapsed_ms)
        per_image_ms.append(elapsed_ms / max(bs, 1))
        n_timed += 1

    out: Dict[str, Any] = {
        "warmup_batches_run": n_warmup_done,
        "timed_batches": n_timed,
        "latency_ms_per_batch_mean": statistics.mean(batch_ms) if batch_ms else None,
        "latency_ms_per_batch_stdev": statistics.stdev(batch_ms) if len(batch_ms) > 1 else None,
        "latency_ms_per_image_mean": statistics.mean(per_image_ms) if per_image_ms else None,
        "latency_ms_per_image_stdev": statistics.stdev(per_image_ms) if len(per_image_ms) > 1 else None,
        "peak_memory_cuda_mb": _cuda_peak_mb() if device.type == "cuda" and n_timed > 0 else None,
        "process_rss_mb_before": None,
        "process_rss_mb_after": None,
    }
    if device.type != "cuda" and n_timed > 0:
        out["process_rss_mb_after"] = _try_cpu_rss_mb()
    return out


def eval_classification(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = Path(args.data_dir).resolve()
    csv_path = _split_csv(data_dir, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    net_id = ckpt["net_id"]
    image_size = int(ckpt["image_size"])
    num_classes = int(ckpt["num_classes"])
    binary_infected = bool(ckpt["binary_infected"])

    model, _, _ = mcunet_build(net_id=net_id, pretrained=False)
    replace_classifier_head(model, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    transform_eval = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    label_mapper = make_label_mapper(binary_infected)
    ds = BeeSafeDataset(csv_path, data_dir, transform_eval, label_mapper)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Unweighted CE for a comparable scalar loss on held-out data
    criterion = nn.CrossEntropyLoss()
    loss, acc, infected_recall = run_epoch(
        model, loader, criterion, None, device, binary_infected
    )

    metrics: Dict[str, Any] = {
        "task": "classification",
        "checkpoint": str(ckpt_path.as_posix()),
        "split": args.split,
        "net_id": net_id,
        "num_classes": num_classes,
        "binary_infected": binary_infected,
        "n_samples": len(ds),
        "loss": loss,
        "accuracy": acc,
        "infected_recall": infected_recall,
    }

    print(
        f"classification | {args.split} | n={len(ds)} | "
        f"loss {loss:.4f} | acc {acc:.4f} | infected_recall {infected_recall:.4f}"
    )

    if not getattr(args, "skip_latency_memory", False):
        lm = measure_classification_latency_memory(
            model,
            loader,
            device,
            warmup_batches=args.latency_warmup,
            max_timed_batches=args.latency_max_batches,
        )
        metrics["latency_memory"] = lm
        lat_m = lm.get("latency_ms_per_image_mean")
        peak = lm.get("peak_memory_cuda_mb")
        rss = lm.get("process_rss_mb_after")
        extra = []
        if lat_m is not None:
            extra.append(f"latency/img ~{lat_m:.3f} ms")
        if peak is not None:
            extra.append(f"peak CUDA alloc {peak:.1f} MB")
        elif rss is not None:
            extra.append(f"RSS ~{rss:.1f} MB (after, psutil)")
        if extra:
            print("  " + " | ".join(extra))
    else:
        metrics["latency_memory"] = None
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return metrics


def eval_localization(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = Path(args.data_dir).resolve()
    csv_path = _split_csv(data_dir, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = int(ckpt["num_classes"])
    binary_labels = bool(ckpt["binary_labels"])
    score_thresh = float(
        args.score_thresh
        if args.score_thresh is not None
        else ckpt.get("score_thresh", 0.3)
    )

    model = build_localization_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    ds = BeeSafeDetectionDataset(csv_path, binary_labels=binary_labels)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    recall = eval_pos_recall(model, loader, device, score_thresh=score_thresh)

    metrics: Dict[str, Any] = {
        "task": "localization",
        "checkpoint": str(ckpt_path.as_posix()),
        "split": args.split,
        "num_classes": num_classes,
        "binary_labels": binary_labels,
        "score_thresh": score_thresh,
        "positive_image_recall": recall,
    }
    print(
        f"localization | {args.split} | pos_image_recall@{score_thresh:.2f} {recall:.4f}"
    )

    if not getattr(args, "skip_latency_memory", False):
        lm = measure_localization_latency_memory(
            model,
            loader,
            device,
            warmup_batches=args.latency_warmup,
            max_timed_batches=args.latency_max_batches,
        )
        metrics["latency_memory"] = lm
        lat_m = lm.get("latency_ms_per_image_mean")
        peak = lm.get("peak_memory_cuda_mb")
        rss = lm.get("process_rss_mb_after")
        extra = []
        if lat_m is not None:
            extra.append(f"latency/img ~{lat_m:.3f} ms")
        if peak is not None:
            extra.append(f"peak CUDA alloc {peak:.1f} MB")
        elif rss is not None:
            extra.append(f"RSS ~{rss:.1f} MB (after, psutil)")
        if extra:
            print("  " + " | ".join(extra))
    else:
        metrics["latency_memory"] = None
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BeeSafe classification or localization checkpoints."
    )
    sub = parser.add_subparsers(dest="task", required=True)

    p_cls = sub.add_parser(
        "classification",
        help="MCUNet image-level classifier (checkpoint from train_mcunet_classification).",
    )
    p_cls.add_argument("--checkpoint", type=Path, required=True)
    p_cls.add_argument("--data-dir", type=Path, default=Path("data"))
    p_cls.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Which gt_one split to evaluate (default: test).",
    )
    p_cls.add_argument("--batch-size", type=int, default=64)
    p_cls.add_argument("--num-workers", type=int, default=2)
    p_cls.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON.",
    )
    p_cls.add_argument(
        "--skip-latency-memory",
        action="store_true",
        help="Skip per-batch latency and peak memory measurement.",
    )
    p_cls.add_argument(
        "--latency-warmup",
        type=int,
        default=2,
        help="Batches to run before timing (default: 2).",
    )
    p_cls.add_argument(
        "--latency-max-batches",
        type=int,
        default=None,
        help="Max batches to time (default: all batches after warmup).",
    )
    p_cls.set_defaults(func=eval_classification)

    p_loc = sub.add_parser(
        "localization",
        help="Faster R-CNN detector (checkpoint from train_localization).",
    )
    p_loc.add_argument("--checkpoint", type=Path, required=True)
    p_loc.add_argument("--data-dir", type=Path, default=Path("data"))
    p_loc.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
    )
    p_loc.add_argument("--batch-size", type=int, default=8)
    p_loc.add_argument("--num-workers", type=int, default=2)
    p_loc.add_argument(
        "--score-thresh",
        type=float,
        default=None,
        help="Override score threshold (default: value stored in checkpoint, else 0.3).",
    )
    p_loc.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    p_loc.add_argument(
        "--skip-latency-memory",
        action="store_true",
        help="Skip per-batch latency and peak memory measurement.",
    )
    p_loc.add_argument("--latency-warmup", type=int, default=2)
    p_loc.add_argument(
        "--latency-max-batches",
        type=int,
        default=None,
        help="Max batches to time (default: all after warmup).",
    )
    p_loc.set_defaults(func=eval_localization)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
