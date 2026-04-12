from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


BBox = Tuple[float, float, float, float]


def parse_gt_line(line: str) -> Tuple[str, int, List[BBox]]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("invalid ground-truth line")
    image_rel = parts[0]
    label = int(parts[1])
    coords = parts[2:]
    boxes: List[BBox] = []
    if coords:
        if len(coords) % 4 != 0:
            raise ValueError("bbox coordinates must be in groups of 4")
        for i in range(0, len(coords), 4):
            x1, y1, x2, y2 = map(float, coords[i : i + 4])
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return image_rel, label, boxes


class BeeSafeDetectionDataset(Dataset):
    def __init__(self, csv_path: Path, binary_labels: bool = True) -> None:
        self.samples: List[Tuple[Path, int, List[BBox]]] = []
        self.binary_labels = binary_labels
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_rel, raw_label, boxes = parse_gt_line(line)
            image_path = csv_path.parent / image_rel
            if not image_path.exists():
                continue
            self.samples.append((image_path, raw_label, boxes))
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _map_label(self, raw_label: int) -> int:
        if self.binary_labels:
            # One foreground class: infected spot
            return 1
        # Two foreground classes: 1 and 3
        if raw_label == 1:
            return 1
        if raw_label == 3:
            return 2
        return 1

    def __getitem__(self, idx: int):
        image_path, raw_label, boxes = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = F.to_tensor(img)

        if boxes:
            labels = torch.full(
                (len(boxes),),
                fill_value=self._map_label(raw_label),
                dtype=torch.int64,
            )
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            area = (box_tensor[:, 2] - box_tensor[:, 0]) * (
                box_tensor[:, 3] - box_tensor[:, 1]
            )
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            labels = torch.zeros((0,), dtype=torch.int64)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": box_tensor,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def count_image_level_healthy_infected(csv_path: Path) -> Tuple[int, int]:
    """Rows with label 0 vs non-zero (train split), for loss weighting."""
    healthy = 0
    infected = 0
    for raw in csv_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue
        try:
            lab = int(parts[1])
        except ValueError:
            continue
        if lab == 0:
            healthy += 1
        else:
            infected += 1
    return healthy, infected


def localization_classifier_loss_weight(healthy: int, infected: int) -> float:
    """
    Scale ROI classification loss when infected images are rarer than healthy
    (same ratio idea as classification CE: weight ~ n_healthy / n_infected).
    """
    if infected < 1:
        return 1.0
    return float(healthy) / float(infected)


def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float] | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses: Dict[str, torch.Tensor] = model(images, targets)  # type: ignore[assignment]
        if loss_weights:
            loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
        else:
            loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def eval_pos_recall(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.3,
) -> float:
    """
    Lightweight validation metric:
    among images that have at least one GT box, how many have
    at least one predicted box over score threshold.
    """
    model.eval()
    positive_images = 0
    hit_images = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            has_gt = tgt["boxes"].shape[0] > 0
            if not has_gt:
                continue
            positive_images += 1
            scores = out["scores"].detach().cpu()
            if (scores >= score_thresh).any():
                hit_images += 1
    if positive_images == 0:
        return 0.0
    return hit_images / positive_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Faster R-CNN (MobileNetV3) for bounding-box localization on BeeSafe "
            "gt_one.csv splits."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--binary-labels", action="store_true", default=True)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use torchvision COCO-pretrained Faster R-CNN weights.",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/localization"))
    parser.add_argument(
        "--no-loss-weights",
        action="store_true",
        help="Disable inverse-frequency weighting of loss_classifier (default: enabled).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train" / "gt_one.csv"
    val_csv = data_dir / "val" / "gt_one.csv"
    test_csv = data_dir / "test" / "gt_one.csv"
    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    train_ds = BeeSafeDetectionDataset(train_csv, binary_labels=args.binary_labels)
    val_ds = BeeSafeDetectionDataset(val_csv, binary_labels=args.binary_labels)
    test_ds = BeeSafeDetectionDataset(test_csv, binary_labels=args.binary_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_classes = 2 if args.binary_labels else 3  # background + foreground classes
    model = build_model(num_classes=num_classes, pretrained=args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_healthy, n_infected = count_image_level_healthy_infected(train_csv)
    if args.no_loss_weights:
        loss_weights: Dict[str, float] | None = None
        print("Training loss: uniform (no loss_classifier reweighting).")
    else:
        w_cls = localization_classifier_loss_weight(n_healthy, n_infected)
        loss_weights = {"loss_classifier": w_cls}
        print(
            "Training loss: weighted (loss_classifier × "
            f"{w_cls:.4f} from train rows healthy={n_healthy}, infected={n_infected})"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "localization_best.pt"
    best_val_recall = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_weights)
        val_recall = eval_pos_recall(
            model, val_loader, device, score_thresh=args.score_thresh
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | val_pos_recall@{args.score_thresh:.2f} {val_recall:.4f}"
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "binary_labels": args.binary_labels,
                    "best_val_pos_recall": best_val_recall,
                    "score_thresh": args.score_thresh,
                    "train_healthy_rows": n_healthy,
                    "train_infected_rows": n_infected,
                    "loss_weights_enabled": not args.no_loss_weights,
                    "loss_classifier_weight": (
                        localization_classifier_loss_weight(n_healthy, n_infected)
                        if not args.no_loss_weights
                        else None
                    ),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_recall = eval_pos_recall(
        model, test_loader, device, score_thresh=args.score_thresh
    )
    metrics = {
        "best_val_pos_recall": best_val_recall,
        "test_pos_recall": test_recall,
        "score_thresh": args.score_thresh,
        "num_classes": num_classes,
        "binary_labels": args.binary_labels,
        "train_healthy_rows": n_healthy,
        "train_infected_rows": n_infected,
        "loss_weights_enabled": not args.no_loss_weights,
        "loss_classifier_weight": (
            localization_classifier_loss_weight(n_healthy, n_infected)
            if not args.no_loss_weights
            else None
        ),
        "checkpoint": str(best_path.as_posix()),
    }
    metrics_path = args.save_dir / "localization_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_path}")
    print(f"Test positive-image recall: {test_recall:.4f}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
