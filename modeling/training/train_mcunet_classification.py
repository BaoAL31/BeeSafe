from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modeling.mcunet_patch import apply_mcunet_download_patch

apply_mcunet_download_patch()

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mcunet.model_zoo import build_model


def parse_gt_line(line: str) -> Tuple[str, int]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("invalid ground-truth line")
    return parts[0], int(parts[1])


def make_label_mapper(binary_infected: bool) -> Callable[[int], int]:
    if binary_infected:
        return lambda label: 0 if label == 0 else 1
    label_to_idx = {0: 0, 1: 1, 3: 2}
    return lambda label: label_to_idx[label]


class BeeSafeDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_dir: Path,
        transform: transforms.Compose,
        label_mapper: Callable[[int], int],
    ) -> None:
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform

        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_rel, raw_label = parse_gt_line(line)
            image_path = csv_path.parent / image_rel
            if not image_path.exists():
                continue
            self.samples.append((image_path, label_mapper(raw_label)))

        if not self.samples:
            raise RuntimeError(f"No usable samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.transform(img)
        return image, label


def replace_classifier_head(model: nn.Module, num_classes: int) -> None:
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


def count_class_frequencies(samples: Sequence[Tuple[Path, int]], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for _, y in samples:
        counts[y] += 1
    return counts


def cross_entropy_class_weights(counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weights for CrossEntropyLoss. For binary healthy vs infected,
    healthy=1.0 and infected=n_healthy/n_infected (e.g. ~2.42 for 9562:3947).
    For three classes, uses weight[c] = N / (C * n_c).
    """
    num_classes = len(counts)
    total = sum(counts)
    if total == 0:
        return torch.ones(num_classes, device=device)

    if num_classes == 2:
        n0, n1 = counts[0], counts[1]
        if n1 < 1:
            w = torch.tensor([1.0, 1.0], dtype=torch.float32)
        else:
            w = torch.tensor([1.0, n0 / n1], dtype=torch.float32)
    else:
        w = torch.tensor(
            [total / (num_classes * c) if c > 0 else 0.0 for c in counts],
            dtype=torch.float32,
        )
    return w.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MCUNet for BeeSafe image-level classification (healthy vs infected)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--net-id",
        type=str,
        default="mcunet-320kB",
        help="MCUNet zoo id (e.g. mcunet-320kB, mcunet-5fps, mcunet-512kB, mbv2-320kB). "
        "Run mcunet.model_zoo.NET_INFO or see mcunet docs for valid names.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--binary-infected", action="store_true")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency weights in CrossEntropyLoss (default: enabled).",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints/mcunet"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train" / "gt_one.csv"
    val_csv = data_dir / "val" / "gt_one.csv"
    test_csv = data_dir / "test" / "gt_one.csv"
    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    model, image_size, _ = build_model(net_id=args.net_id, pretrained=args.pretrained)
    num_classes = 2 if args.binary_infected else 3
    replace_classifier_head(model, num_classes=num_classes)

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
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

    label_mapper = make_label_mapper(args.binary_infected)
    train_ds = BeeSafeDataset(train_csv, data_dir, transform_train, label_mapper)
    val_ds = BeeSafeDataset(val_csv, data_dir, transform_eval, label_mapper)
    test_ds = BeeSafeDataset(test_csv, data_dir, transform_eval, label_mapper)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_counts = count_class_frequencies(train_ds.samples, num_classes)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("CrossEntropyLoss: uniform class weights (no reweighting).")
    else:
        ce_w = cross_entropy_class_weights(train_counts, device)
        criterion = nn.CrossEntropyLoss(weight=ce_w)
        print(
            "CrossEntropyLoss: inverse-frequency class weights "
            f"(from train split counts {train_counts}): {ce_w.detach().cpu().tolist()}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_path = args.save_dir / f"{args.net_id}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt: Dict = {
                "model_state_dict": model.state_dict(),
                "net_id": args.net_id,
                "image_size": image_size,
                "num_classes": num_classes,
                "binary_infected": args.binary_infected,
                "best_val_acc": best_val_acc,
                "train_class_counts": train_counts,
                "class_weights_enabled": not args.no_class_weights,
            }
            if not args.no_class_weights:
                ckpt["ce_class_weights"] = cross_entropy_class_weights(
                    train_counts, torch.device("cpu")
                ).tolist()
            torch.save(ckpt, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)

    metrics = {
        "net_id": args.net_id,
        "image_size": image_size,
        "num_classes": num_classes,
        "binary_infected": args.binary_infected,
        "train_class_counts": train_counts,
        "class_weights_enabled": not args.no_class_weights,
        "ce_class_weights": (
            cross_entropy_class_weights(train_counts, torch.device("cpu")).tolist()
            if not args.no_class_weights
            else None
        ),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "checkpoint": str(best_path.as_posix()),
    }
    metrics_path = args.save_dir / f"{args.net_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path}")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
