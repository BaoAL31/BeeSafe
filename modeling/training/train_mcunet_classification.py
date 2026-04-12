from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from mcunet.model_zoo import build_model, download_tflite, net_id_list

from modeling.training.classification_metrics import infected_recall


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
    binary_infected: bool,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
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
    inf_rec = infected_recall(preds_cat, labels_cat, binary_infected)
    return avg_loss, acc, inf_rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train MCUNet for BeeSafe binary image-level classification (healthy vs infected; "
            "raw labels 1 and 3 are combined as infected). "
            "Checkpoints are selected by validation infected recall (catch all infected bees)."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--net-id",
        type=str,
        default="mcunet-in3",
        help=(
            "MCUNet zoo id (see --list-net-ids). Upstream clone uses mcunet-in0..in4, "
            "mbv2-w0.35, proxyless-w0.3, mcunet-vww0..vww2, person-det. Default mcunet-in3 ≈ 320KB/1MB."
        ),
    )
    parser.add_argument(
        "--list-net-ids",
        action="store_true",
        help="Print mcunet.model_zoo.net_id_list and exit.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency weights in CrossEntropyLoss (default: enabled).",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--download-tflite",
        action="store_true",
        help="After evaluation, download the matching .tflite from the MCUNet release (same net_id).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("modeling/checkpoints/mcunet"),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (default: <save-dir>/tensorboard).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many epochs with no improvement in val infected recall "
            "(0 = run all epochs)."
        ),
    )
    args = parser.parse_args()
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0")

    if args.list_net_ids:
        print(net_id_list)
        return

    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train" / "gt_one.csv"
    val_csv = data_dir / "val" / "gt_one.csv"
    test_csv = data_dir / "test" / "gt_one.csv"
    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    model, image_size, _ = build_model(net_id=args.net_id, pretrained=args.pretrained)
    # Always binary: healthy (0) vs infected; raw labels 1 and 3 map to class 1.
    num_classes = 2
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

    label_mapper = make_label_mapper(True)
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
    print(f"Using device: {device}")
    model = model.to(device)

    train_counts = count_class_frequencies(train_ds.samples, num_classes)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        ce_w = cross_entropy_class_weights(train_counts, device)
        criterion = nn.CrossEntropyLoss(weight=ce_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = (
        args.tensorboard_dir
        if args.tensorboard_dir is not None
        else args.save_dir / "tensorboard"
    )
    writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard log dir: {tb_log_dir.resolve()}")

    best_val_inf_rec = -1.0
    best_path = args.save_dir / f"{args.net_id}_best.pt"
    epochs_without_improve = 0
    last_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            train_loss, train_acc, train_inf_rec = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                True,
            )
            val_loss, val_acc, val_inf_rec = run_epoch(
                model, val_loader, criterion, None, device, True
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} inf_rec {train_inf_rec:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} inf_rec {val_inf_rec:.4f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/accuracy", train_acc, epoch)
                writer.add_scalar("train/infected_recall", train_inf_rec, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                writer.add_scalar("val/infected_recall", val_inf_rec, epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if val_inf_rec > best_val_inf_rec:
                best_val_inf_rec = val_inf_rec
                epochs_without_improve = 0
                ckpt: Dict = {
                    "model_state_dict": model.state_dict(),
                    "net_id": args.net_id,
                    "image_size": image_size,
                    "num_classes": num_classes,
                    "binary_infected": True,
                    "best_val_infected_recall": best_val_inf_rec,
                    "val_acc_at_best": val_acc,
                    "train_class_counts": train_counts,
                    "class_weights_enabled": not args.no_class_weights,
                }
                if not args.no_class_weights:
                    ckpt["ce_class_weights"] = cross_entropy_class_weights(
                        train_counts, torch.device("cpu")
                    ).tolist()
                torch.save(ckpt, best_path)
                if writer is not None:
                    writer.add_scalar("best/val_infected_recall", best_val_inf_rec, epoch)
            else:
                if args.early_stopping_patience > 0:
                    epochs_without_improve += 1
                    if epochs_without_improve >= args.early_stopping_patience:
                        print(
                            "Early stopping: no improvement in val infected recall for "
                            f"{args.early_stopping_patience} epoch(s)."
                        )
                        break

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, test_inf_rec = run_epoch(
            model, test_loader, criterion, None, device, True
        )

        if writer is not None:
            writer.add_scalar("test/loss", test_loss, last_epoch)
            writer.add_scalar("test/accuracy", test_acc, last_epoch)
            writer.add_scalar("test/infected_recall", test_inf_rec, last_epoch)
    finally:
        if writer is not None:
            writer.close()

    metrics = {
        "net_id": args.net_id,
        "image_size": image_size,
        "num_classes": num_classes,
        "binary_infected": True,
        "train_class_counts": train_counts,
        "class_weights_enabled": not args.no_class_weights,
        "ce_class_weights": (
            cross_entropy_class_weights(train_counts, torch.device("cpu")).tolist()
            if not args.no_class_weights
            else None
        ),
        "best_val_infected_recall": best_val_inf_rec,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_infected_recall": test_inf_rec,
        "checkpoint": str(best_path.as_posix()),
        "tensorboard_dir": (
            str(tb_log_dir.resolve().as_posix()) if not args.no_tensorboard else None
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "epochs_trained": last_epoch,
        "stopped_early": last_epoch < args.epochs,
    }
    if args.download_tflite:
        tflite_path = download_tflite(net_id=args.net_id)
        metrics["tflite_path"] = tflite_path
        if tflite_path:
            print(f"TFLite: {tflite_path}")
        else:
            print("TFLite download failed (see stderr above).", file=sys.stderr)

    metrics_path = args.save_dir / f"{args.net_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if last_epoch < args.epochs:
        print(f"Stopped after {last_epoch} epoch(s) (max {args.epochs}).")
    print(f"Best checkpoint: {best_path} (val infected recall {best_val_inf_rec:.4f})")
    print(
        f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | "
        f"Test infected recall: {test_inf_rec:.4f}"
    )
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
