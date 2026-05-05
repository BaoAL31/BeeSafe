from pathlib import Path
import shutil


def label_to_class(label: int):
    return "healthy" if label == 0 else "infected"


def convert(split):
    base = Path(f"data/{split}")
    videos_root = base / "videos"
    labels_root = base / "labels"

    out_root = Path("data_yolo_cls")
    split_root = out_root / split

    (split_root / "healthy").mkdir(parents=True, exist_ok=True)
    (split_root / "infected").mkdir(parents=True, exist_ok=True)

    total = 0

    for video_folder in videos_root.iterdir():
        if not video_folder.is_dir():
            continue

        label_folder = labels_root / video_folder.name
        if not label_folder.exists():
            continue

        images = list(video_folder.rglob("*.png"))

        for img in images:
            label_file = label_folder / (img.stem + ".txt")

            if not label_file.exists():
                continue

            lines = label_file.read_text().strip().splitlines()
            if not lines:
                continue

            parts = lines[0].split()
            label = int(parts[0])  # Use only first number for class

            cls = label_to_class(label)

            out_path = split_root / cls / img.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(img, out_path)
            total += 1

    print(f"{split}: {total} images copied")


if __name__ == "__main__":
    convert("train")
    convert("val")
    convert("test")