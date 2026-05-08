"""
Evaluate trained BeeSafe checkpoints.

Run from the repository root, for example:

  python -m modeling.evaluation.evaluate classification --checkpoint modeling/checkpoints/visual/mcunet-in3_best.pt
  python -m modeling.evaluation.evaluate localization --checkpoint modeling/checkpoints/localization/localization_best.pt

Latency uses CUDA synchronize + perf_counter (wall time — comparable across machines).
Peak memory uses torch.cuda.max_memory_allocated after warmup and optional process RSS (psutil).
We do not report CPU/GPU utilization %: those are machine-relative; use latency and throughput instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVALUATION_DIR = Path(__file__).resolve().parent  # modeling/evaluation/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import math
import os
import statistics
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from mcunet.model_zoo import build_model as mcunet_build

from modeling.training.train_localization import (
    BeeSafeDetectionDataset,
    build_model as build_localization_model,
    collate_fn,
    eval_pos_recall,
)
from modeling.training.train_visual_classification import (
    BeeSafeDataset,
    make_label_mapper,
    replace_classifier_head,
    run_epoch,
)
from modeling.training.train_acoustic import (
    BeeAudioDataset,
    count_class_frequencies,
    cross_entropy_class_weights,
    run_epoch as run_epoch_acoustic,
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
    if not _HAS_PSUTIL:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return None


def _throughput_images_per_sec(latency_ms_per_image_mean: Optional[float]) -> Optional[float]:
    """Images per second from mean latency; comparable across machines (inverse of wall time)."""
    if latency_ms_per_image_mean is None or latency_ms_per_image_mean <= 0:
        return None
    return 1000.0 / latency_ms_per_image_mean


class _ResourceMonitor:
    """Background thread that samples process RSS at a fixed interval (peak footprint)."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self._interval = interval_s
        self._peak_rss_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not _HAS_PSUTIL:
            return
        self._proc = psutil.Process(os.getpid())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / (1024**2)
                if rss > self._peak_rss_mb:
                    self._peak_rss_mb = rss
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> Dict[str, Optional[float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "peak_rss_mb": self._peak_rss_mb if _HAS_PSUTIL else None,
        }


def _try_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """GPU name and VRAM (absolute MB). No utilization % — not comparable across GPUs."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        pynvml.nvmlShutdown()
        return {
            "gpu_name": name,
            "gpu_mem_used_mb": mem.used / (1024**2),
            "gpu_mem_total_mb": mem.total / (1024**2),
        }
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
    Wall-clock latency (ms), throughput (images/s), peak CUDA/RSS memory, optional GPU VRAM (MB).
    """
    model.eval()
    batch_ms: List[float] = []
    per_image_ms: List[float] = []
    n_warmup_done = 0
    n_timed = 0
    cuda_peak_reset = False

    rss_before = _try_cpu_rss_mb()
    monitor = _ResourceMonitor(interval_s=0.05)

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
            monitor.start()
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

    resource_stats = monitor.stop()
    rss_after = _try_cpu_rss_mb()
    gpu_info = _try_gpu_memory_info() if device.type == "cuda" else None

    lat_img = statistics.mean(per_image_ms) if per_image_ms else None
    out: Dict[str, Any] = {
        "warmup_batches_run": n_warmup_done,
        "timed_batches": n_timed,
        "latency_ms_per_batch_mean": statistics.mean(batch_ms) if batch_ms else None,
        "latency_ms_per_batch_stdev": statistics.stdev(batch_ms) if len(batch_ms) > 1 else None,
        "latency_ms_per_image_mean": lat_img,
        "latency_ms_per_image_stdev": statistics.stdev(per_image_ms) if len(per_image_ms) > 1 else None,
        "throughput_images_per_sec_mean": _throughput_images_per_sec(lat_img),
        "peak_memory_cuda_mb": _cuda_peak_mb() if device.type == "cuda" and n_timed > 0 else None,
        "process_rss_mb_before": rss_before,
        "process_rss_mb_after": rss_after,
        "peak_rss_mb": resource_stats.get("peak_rss_mb"),
        "gpu_memory": gpu_info,
    }
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

    rss_before = _try_cpu_rss_mb()
    monitor = _ResourceMonitor(interval_s=0.05)

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
            monitor.start()
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

    resource_stats = monitor.stop()
    rss_after = _try_cpu_rss_mb()
    gpu_info = _try_gpu_memory_info() if device.type == "cuda" else None

    lat_img = statistics.mean(per_image_ms) if per_image_ms else None
    out: Dict[str, Any] = {
        "warmup_batches_run": n_warmup_done,
        "timed_batches": n_timed,
        "latency_ms_per_batch_mean": statistics.mean(batch_ms) if batch_ms else None,
        "latency_ms_per_batch_stdev": statistics.stdev(batch_ms) if len(batch_ms) > 1 else None,
        "latency_ms_per_image_mean": lat_img,
        "latency_ms_per_image_stdev": statistics.stdev(per_image_ms) if len(per_image_ms) > 1 else None,
        "throughput_images_per_sec_mean": _throughput_images_per_sec(lat_img),
        "peak_memory_cuda_mb": _cuda_peak_mb() if device.type == "cuda" and n_timed > 0 else None,
        "process_rss_mb_before": rss_before,
        "process_rss_mb_after": rss_after,
        "peak_rss_mb": resource_stats.get("peak_rss_mb"),
        "gpu_memory": gpu_info,
    }
    return out


def _print_resource_summary(lm: Dict[str, Any]) -> None:
    """Pretty-print wall-clock latency, throughput, and memory from a measure_ result."""
    lat_m = lm.get("latency_ms_per_image_mean")
    lat_s = lm.get("latency_ms_per_image_stdev")
    tput = lm.get("throughput_images_per_sec_mean")
    peak_cuda = lm.get("peak_memory_cuda_mb")
    peak_rss = lm.get("peak_rss_mb")
    rss_before = lm.get("process_rss_mb_before")
    rss_after = lm.get("process_rss_mb_after")
    gpu = lm.get("gpu_memory")

    lines: List[str] = []
    if lat_m is not None:
        s = f"latency/img {lat_m:.3f} ms"
        if lat_s is not None:
            s += f" (±{lat_s:.3f})"
        lines.append(s)
    if tput is not None:
        lines.append(f"throughput ~{tput:.1f} images/s (from mean latency)")

    mem_parts: List[str] = []
    if peak_cuda is not None:
        mem_parts.append(f"peak CUDA alloc {peak_cuda:.1f} MB")
    if peak_rss is not None:
        mem_parts.append(f"peak RSS {peak_rss:.1f} MB")
    elif rss_after is not None:
        mem_parts.append(f"RSS {rss_after:.1f} MB (after)")
    if rss_before is not None and rss_after is not None:
        mem_parts.append(f"RSS delta {rss_after - rss_before:+.1f} MB")
    if mem_parts:
        lines.append(" | ".join(mem_parts))

    if gpu is not None:
        name = gpu.get("gpu_name", "GPU")
        mem_used = gpu.get("gpu_mem_used_mb")
        mem_total = gpu.get("gpu_mem_total_mb")
        if mem_used is not None and mem_total is not None:
            lines.append(f"{name} | VRAM snapshot {mem_used:.0f}/{mem_total:.0f} MB")
        else:
            lines.append(name)

    for line in lines:
        print(f"  {line}")


_EVAL_RESULTS: Dict[str, Any] = {}
_EVAL_SKIP_WRITE = False


def _write_evaluation_results_md(
    metrics: Dict[str, Any], output_dir: Path
) -> Path:
    """Write evaluation markdown files with metrics and explanations."""
    global _EVAL_RESULTS
    task = metrics.get("task")
    if task:
        _EVAL_RESULTS[task] = metrics

    if _EVAL_SKIP_WRITE:
        return Path("")

    eval_dir = Path(__file__).resolve().parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = eval_dir / "evaluation_latest.md"
    
    if not _EVAL_RESULTS:
        return Path("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# BeeSafe Evaluation Results")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    
    lines.append("## Metrics Explanation")
    lines.append("")
    lines.append("### Classification & Acoustic Metrics")
    lines.append("- **Accuracy**: Percentage of correct predictions across all classes")
    lines.append("- **Recall**: Percentage of actual positive cases correctly identified (Infected/Abnormal recall)")
    lines.append("- **Loss**: Cross-entropy loss value (lower is better)")
    lines.append("")
    lines.append("### Performance Metrics")
    lines.append("- **Latency**: Average inference time per image/audio sample in milliseconds")
    lines.append("- **Throughput**: Number of images/audio samples processed per second")
    lines.append("- **Peak CUDA Memory**: Maximum GPU memory allocated during inference (MB)")
    lines.append("- **Peak RSS**: Maximum Resident Set Size (process memory usage) in MB")
    lines.append("- **GPU Memory**: GPU VRAM usage snapshot (used/total MB)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for task_name, m in _EVAL_RESULTS.items():
        lines.append(f"## {task_name.title()}")
        lines.append("")

        if task_name == "classification":
            lines.append("### Visual (Classification) Results")
            lines.append("")
            lines.append(f"- **Accuracy**: {m.get('accuracy', 0):.1%}")
            lines.append(f"- **Infected Recall**: {m.get('infected_recall', 0):.1%}")
            lines.append(f"- **Loss**: {m.get('loss', 0):.4f}")
            lines.append(f"- **Samples**: {m.get('n_samples', 'n/a')}")
            lm = m.get("latency_memory")
            if lm:
                lines.append("")
                lines.append("**Performance Metrics:**")
                lines.append(f"- Latency: {lm.get('latency_ms_per_image_mean', 0):.3f} ms/img")
                lines.append(f"- Throughput: ~{lm.get('throughput_images_per_sec_mean', 0):.0f} img/s")
                if lm.get('peak_memory_cuda_mb'):
                    lines.append(f"- Peak CUDA Memory: {lm.get('peak_memory_cuda_mb', 0):.1f} MB")
                if lm.get('peak_rss_mb'):
                    lines.append(f"- Peak RSS: {lm.get('peak_rss_mb', 0):.1f} MB")
                gpu = lm.get('gpu_memory')
                if gpu and gpu.get('gpu_mem_used_mb'):
                    lines.append(f"- GPU VRAM: {gpu.get('gpu_mem_used_mb', 0):.0f}/{gpu.get('gpu_mem_total_mb', 0):.0f} MB")
        elif task_name == "acoustic":
            lines.append("### Acoustic Results")
            lines.append("")
            lines.append(f"- **Accuracy**: {m.get('accuracy', 0):.1%}")
            lines.append(f"- **Abnormal Recall**: {m.get('abnormal_recall', 0):.1%}")
            lines.append(f"- **Loss**: {m.get('loss', 0):.4f}")
            lines.append(f"- **Samples**: {m.get('n_samples', 'n/a')}")
            lm = m.get("latency_memory")
            if lm:
                lines.append("")
                lines.append("**Performance Metrics:**")
                lines.append(f"- Latency: {lm.get('latency_ms_per_image_mean', 0):.3f} ms/sample")
                lines.append(f"- Throughput: ~{lm.get('throughput_images_per_sec_mean', 0):.0f} samples/s")
                if lm.get('peak_memory_cuda_mb'):
                    lines.append(f"- Peak CUDA Memory: {lm.get('peak_memory_cuda_mb', 0):.1f} MB")
                if lm.get('peak_rss_mb'):
                    lines.append(f"- Peak RSS: {lm.get('peak_rss_mb', 0):.1f} MB")
                gpu = lm.get('gpu_memory')
                if gpu and gpu.get('gpu_mem_used_mb'):
                    lines.append(f"- GPU VRAM: {gpu.get('gpu_mem_used_mb', 0):.0f}/{gpu.get('gpu_mem_total_mb', 0):.0f} MB")
        lines.append("")
        lines.append("---")
        lines.append("")

    md_content = "\n".join(lines)
    md_path = eval_dir / f"evaluation_{timestamp}.md"
    md_path.write_text(md_content, encoding="utf-8")
    latest_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Updated {latest_path}")
    return md_path


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
        "infected_recall": (
            None if isinstance(infected_recall, float) and math.isnan(infected_recall)
            else infected_recall
        ),
    }

    ir_s = (
        "n/a"
        if isinstance(infected_recall, float) and math.isnan(infected_recall)
        else f"{infected_recall:.4f}"
    )
    print(
        f"classification | {args.split} | n={len(ds)} | "
        f"loss {loss:.4f} | acc {acc:.4f} | infected_recall {ir_s}"
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
        _print_resource_summary(lm)
    else:
        metrics["latency_memory"] = None
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    _write_evaluation_results_md(metrics, _EVALUATION_DIR)

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
        _print_resource_summary(lm)
    else:
        metrics["latency_memory"] = None
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    _write_evaluation_results_md(metrics, _EVALUATION_DIR)

    return metrics


def eval_acoustic(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate acoustic MCUNet model."""
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    net_id = ckpt.get("net_id", "mcunet-in3")
    n_mels = int(ckpt.get("n_mels", 64))
    n_time = int(ckpt.get("n_time", 32))
    num_classes = int(ckpt.get("num_classes", 2))
    segment_duration = float(ckpt.get("segment_duration", 2.0))
    sr = int(ckpt.get("sr", 16000))

    from mcunet.model_zoo import build_model as mcunet_build
    model, _, _ = mcunet_build(net_id=net_id, pretrained=False)
    from modeling.training.train_acoustic import replace_classifier_head
    replace_classifier_head(model, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    audio_dir = Path(args.audio_dir).resolve()
    from modeling.training.train_acoustic import BeeAudioDataset
    ds = BeeAudioDataset(
        audio_dir,
        sr=sr,
        n_mels=n_mels,
        n_fft=int(ckpt.get("n_fft", 2048)),
        hop_length=int(ckpt.get("hop_length", 512)),
        segment_duration=segment_duration,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    loss, acc, abnormal_recall = run_epoch_acoustic(
        model, loader, criterion, None, device
    )

    metrics: Dict[str, Any] = {
        "task": "acoustic",
        "checkpoint": str(ckpt_path.as_posix()),
        "net_id": net_id,
        "num_classes": num_classes,
        "n_samples": len(ds),
        "loss": loss,
        "accuracy": acc,
        "abnormal_recall": (
            None if isinstance(abnormal_recall, float) and math.isnan(abnormal_recall)
            else abnormal_recall
        ),
    }

    ar_s = (
        "n/a"
        if isinstance(abnormal_recall, float) and math.isnan(abnormal_recall)
        else f"{abnormal_recall:.4f}"
    )
    print(
        f"acoustic | n={len(ds)} | "
        f"loss {loss:.4f} | acc {acc:.4f} | abnormal_recall {ar_s}"
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
        _print_resource_summary(lm)
    else:
        metrics["latency_memory"] = None

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    _write_evaluation_results_md(metrics, _EVALUATION_DIR)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BeeSafe classification, localization, or acoustic checkpoints."
    )
    sub = parser.add_subparsers(dest="task", required=True)

    p_cls = sub.add_parser(
        "classification",
        help="MCUNet image-level classifier (checkpoint from train_visual_classification).",
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

    p_aud = sub.add_parser(
        "acoustic",
        help="MCUNet acoustic classifier (checkpoint from train_acoustic).",
    )
    p_aud.add_argument("--checkpoint", type=Path, required=True)
    p_aud.add_argument("--audio-dir", type=Path, default=Path("data/zenodo_bee_audio_test"))
    p_aud.add_argument("--batch-size", type=int, default=64)
    p_aud.add_argument("--num-workers", type=int, default=2)
    p_aud.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    p_aud.add_argument(
        "--skip-latency-memory",
        action="store_true",
    )
    p_aud.add_argument("--latency-warmup", type=int, default=2)
    p_aud.add_argument(
        "--latency-max-batches",
        type=int,
        default=None,
    )
    p_aud.set_defaults(func=eval_acoustic)

    def eval_all(args: argparse.Namespace) -> Dict[str, Any]:
        global _EVAL_RESULTS, _EVAL_SKIP_WRITE
        _EVAL_RESULTS = {}
        _EVAL_SKIP_WRITE = True

        ac_ckpt = Path(args.acoustic_checkpoint).resolve()
        vis_ckpt = Path(args.visual_checkpoint).resolve()
        config = Path(args.config).resolve()
        audio_dir = Path(args.audio_dir).resolve()
        image_data_dir = Path(args.image_data_dir).resolve()

        class Args:
            pass

        cls_args = Args()
        cls_args.checkpoint = vis_ckpt
        cls_args.data_dir = image_data_dir
        cls_args.split = args.split
        cls_args.batch_size = args.batch_size
        cls_args.num_workers = args.num_workers
        cls_args.skip_latency_memory = True
        cls_args.output_json = None
        cls_args.latency_warmup = 2
        cls_args.latency_max_batches = None

        print("=== Running classification ===")
        eval_classification(cls_args)

        aud_args = Args()
        aud_args.checkpoint = ac_ckpt
        aud_args.audio_dir = audio_dir
        aud_args.batch_size = args.batch_size
        aud_args.num_workers = args.num_workers
        aud_args.skip_latency_memory = True
        aud_args.output_json = None
        aud_args.latency_warmup = 2
        aud_args.latency_max_batches = None

        print("\n=== Running acoustic ===")
        eval_acoustic(aud_args)

        eval_dir = Path("modeling/evaluation")
        _EVAL_SKIP_WRITE = False
        _write_evaluation_results_md({}, eval_dir)
        return {}

    p_all = sub.add_parser(
        "all",
        help="Run all evaluations (classification, acoustic) in one go.",
    )
    p_all.add_argument("--acoustic-checkpoint", type=Path, required=True)
    p_all.add_argument("--visual-checkpoint", type=Path, required=True)
    p_all.add_argument("--audio-dir", type=Path, default=Path("data/zenodo_bee_audio_test"))
    p_all.add_argument("--image-data-dir", type=Path, default=Path("data"))
    p_all.add_argument("--split", choices=("test", "val"), default="test")
    p_all.add_argument("--batch-size", type=int, default=32)
    p_all.add_argument("--num-workers", type=int, default=2)
    p_all.add_argument("--output-json", type=Path, default=None)
    p_all.set_defaults(func=eval_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
