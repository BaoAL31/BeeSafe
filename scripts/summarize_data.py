from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError  # type: ignore[import-untyped]
except ImportError:
    Image = None  # type: ignore[misc, assignment]
    ImageDraw = None  # type: ignore[misc, assignment]
    ImageFont = None  # type: ignore[misc, assignment]
    UnidentifiedImageError = OSError  # type: ignore[misc, assignment]


BBox = Tuple[int, int, int, int]


def _text_pixel_size(text: str, font: object) -> Tuple[int, int]:
    """Bitmap default fonts do not support ImageDraw.textbbox on some Pillow builds."""
    if hasattr(font, "getbbox"):
        x0, y0, x1, y1 = font.getbbox(text)
        return x1 - x0, y1 - y0
    return len(text) * 6, 11


def parse_gt_line(parts: Sequence[str]) -> Tuple[str, int, List[BBox]]:
    if len(parts) < 2:
        raise ValueError("too few tokens")
    image_path = parts[0]
    label = int(parts[1])
    nums = parts[2:]
    boxes: List[BBox] = []
    if nums:
        if len(nums) % 4 != 0:
            raise ValueError("coordinate count not a multiple of 4")
        for i in range(0, len(nums), 4):
            boxes.append(
                (int(nums[i]), int(nums[i + 1]), int(nums[i + 2]), int(nums[i + 3]))
            )
    return image_path, label, boxes


def read_annotation_csv(csv_path: Path, data_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                image_rel, label, boxes = parse_gt_line(parts)
            except (ValueError, IndexError):
                continue
            abs_img = csv_path.parent / image_rel
            csv_rel = str(csv_path.relative_to(data_dir).as_posix())
            rows.append(
                {
                    "csv_rel": csv_rel,
                    "image_rel": image_rel,
                    "label": label,
                    "boxes": [list(b) for b in boxes],
                    "abs_image": abs_img,
                }
            )
    return rows


def iter_annotation_rows(
    data_dir: Path, gt_paths: Sequence[Path]
) -> List[Dict]:
    rows: List[Dict] = []
    for csv_path in gt_paths:
        rows.extend(read_annotation_csv(csv_path, data_dir))
    return rows


def rows_for_visual_sampling(data_dir: Path, gt_paths: Sequence[Path]) -> List[Dict]:
    """
    Use only data/gt.csv when present so each image path resolves once as
    data/<path-in-csv>. Split CSVs duplicate the same samples.
    """
    data_dir = data_dir.resolve()
    master = data_dir / "gt.csv"
    if master.is_file():
        return read_annotation_csv(master, data_dir)
    return iter_annotation_rows(data_dir, gt_paths)


def dedupe_rows_by_image_path(rows: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    out: List[Dict] = []
    for r in rows:
        key = str(r["abs_image"].resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def draw_sample_image(
    row: Dict,
    out_path: Path,
) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for sample images")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(row["abs_image"]) as im:
        im.load()
        img = im.convert("RGB")
        draw = ImageDraw.Draw(img)
        label = row["label"]
        boxes: List[BBox] = [tuple(b) for b in row["boxes"]]
        palette = {0: "#64748b", 1: "#16a34a", 3: "#ea580c"}
        color = palette.get(label, "#2563eb")

        font = ImageFont.load_default()

        if boxes:
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                text = f"label {label}"
                tw, th = _text_pixel_size(text, font)
                tx = x1 + bi * 2
                ty = max(0, y1 - th - 4)
                draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=color)
                draw.text((tx + 2, ty + 1), text, fill="white", font=font)
        else:
            msg = f"label {label} (no box)"
            draw.rectangle([4, 4, 220, 28], fill="#1e293b")
            draw.text((8, 8), msg, fill="white", font=font)

        img.save(out_path)


_READ_ERRORS = (OSError, ValueError, UnidentifiedImageError)


def build_markdown_report(result: Dict) -> str:
    totals = result["totals"]
    label_meanings = result.get("label_meanings", {})
    scanned = result.get("annotation_csv_files_scanned", totals.get("csv_files_used_for_total", 0))
    dedup_note = result.get("totals_dedup_note")
    lines = [
        "# BeeSafe Data Summary",
        "",
        f"- Data directory: `{result['data_dir']}`",
        f"- Annotation CSV files scanned: **{scanned}**",
        f"- CSV files used for dataset total: **{totals['csv_files_used_for_total']}**",
        f"- Dataset total samples (deduplicated): **{totals['samples']}**",
        f"- Rows with bounding boxes (label > 0): **{totals['bbox_samples']}** ({totals['bbox_ratio']:.2%})",
        f"- Malformed lines: **{totals['malformed_lines']}**",
        f"- Missing image paths: **{totals['missing_image_paths']}**",
    ]
    if dedup_note:
        lines.append(f"- Note: {dedup_note}")
    lines.extend(
        [
        "",
        "## Label Distribution (Overall)",
        "",
        "| Label | Count | Ratio |",
        "|---:|---:|---:|",
        ]
    )

    total_samples = totals["samples"] or 1
    for label, count in totals["labels"].items():
        ratio = count / total_samples
        lines.append(f"| {label} | {count} | {ratio:.2%} |")

    lines.extend(["", "## Label Meanings", ""])
    if label_meanings:
        for label in sorted(int(k) for k in label_meanings.keys()):
            lines.append(f"- `{label}`: {label_meanings[str(label)]}")
    else:
        lines.append("- No label meanings provided.")

    lines.extend(
        [
            "",
            "## Per-File Breakdown",
            "",
            "| File | Samples | BBox Samples | BBox Ratio | Malformed | Missing Images | Labels |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for item in result["files"]:
        labels_text = "<br>".join(
            f"{label}:{count}" for label, count in item["labels"].items()
        )
        lines.append(
            f"| `{item['file']}` | {item['samples']} | {item['bbox_samples']} | "
            f"{item['bbox_ratio']:.2%} | {item['malformed_lines']} | "
            f"{item['missing_image_paths']} | {labels_text} |"
        )

    lines.extend(["", "## Sample images (one per label)", ""])

    visuals = result.get("sample_visuals")
    missing_labs = result.get("sample_visuals_missing_labels") or []
    failed_labs = result.get("sample_visuals_failed_labels") or []
    if visuals is None:
        lines.append("*Sample images were not generated.*")
    elif any("error" in v for v in visuals):
        err = next(v["error"] for v in visuals if "error" in v)
        lines.append(f"*Could not render sample images: {err}*")
    elif not visuals:
        lines.append(
            "*No sample images were produced (no matching image files on disk, "
            "or use `--sample-images 0` to skip this section).*"
        )
    else:
        if missing_labs:
            labs = ", ".join(str(x) for x in missing_labs)
            lines.append(
                f"*Note: label(s) **{labs}** appear in annotations but have no "
                "image file on disk for any row; those labels are omitted below.*"
            )
            lines.append("")
        if failed_labs:
            labs = ", ".join(str(x) for x in failed_labs)
            lines.append(
                f"*Note: could not render label(s) **{labs}** (all candidate "
                "images failed to open or draw); those labels are omitted below.*"
            )
            lines.append("")
        for v in visuals:
            if "error" in v:
                continue
            cap = v["caption"]
            rel = v["file"]
            lb = v["label"]
            lines.append(f"### Label {lb}")
            lines.append("")
            lines.append(f"![Label {lb}]({rel})")
            lines.append("")
            lines.append(f"*{cap}*")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_label_meanings(raw: str) -> Dict[str, str]:
    """
    Parse label meanings from: "0=negative,1=class one,3=class three"
    """
    mappings: Dict[str, str] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            label = int(key)
        except ValueError:
            continue
        mappings[str(label)] = value
    return mappings


def clear_sample_dir(samples_dir: Path) -> None:
    if not samples_dir.exists():
        return
    for p in samples_dir.glob("sample_*.png"):
        p.unlink()


def render_sample_visuals(
    data_dir: Path,
    gt_paths: Sequence[Path],
    md_output_path: Path,
    samples_dir: Path,
    enabled: bool,
    seed: int,
) -> tuple[List[Dict], List[int], List[int]]:
    if not enabled:
        return [], [], []

    # One manifest avoids duplicate rows; paths are always data_dir / <path in CSV>
    rows = rows_for_visual_sampling(data_dir, gt_paths)
    labels_in_ann = {r["label"] for r in rows}
    by_label_rows: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label_rows[r["label"]].append(r)
    missing_labels = sorted(
        lb
        for lb in labels_in_ann
        if not any(x["abs_image"].exists() for x in by_label_rows[lb])
    )

    clear_sample_dir(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    md_parent = md_output_path.parent.resolve()

    if Image is None or ImageDraw is None:
        return (
            [
                {
                    "error": "Pillow is not installed; run: pip install -r requirements.txt"
                }
            ],
            missing_labels,
            [],
        )

    rng = random.Random(seed)
    preferred_order = (1, 3, 0)
    target_labels = [lb for lb in preferred_order if lb in labels_in_ann]
    for lb in sorted(labels_in_ann):
        if lb not in target_labels:
            target_labels.append(lb)

    out: List[Dict] = []
    failed_labels: List[int] = []

    for label in target_labels:
        pool = [r for r in rows if r["label"] == label]
        if label in (1, 3):
            pool = [r for r in pool if r["boxes"] and r["abs_image"].exists()]
        else:
            pool = [r for r in pool if r["abs_image"].exists()]
        pool = dedupe_rows_by_image_path(pool)
        rng.shuffle(pool)

        if not pool:
            if label not in missing_labels:
                failed_labels.append(label)
                if label in (1, 3):
                    print(
                        f"  Label {label}: no row with image on disk and at least one bbox.",
                        file=sys.stderr,
                    )
            continue

        out_path = samples_dir / f"sample_{len(out) + 1:02d}.png"
        row_ok: Optional[Dict] = None
        first_err: Optional[BaseException] = None
        for row in pool:
            try:
                draw_sample_image(row, out_path)
                row_ok = row
                break
            except _READ_ERRORS as e:
                if first_err is None:
                    first_err = e
                continue
        if row_ok is None:
            failed_labels.append(label)
            print(
                f"  Label {label}: tried {len(pool)} image(s); "
                f"example error: {type(first_err).__name__}: {first_err}",
                file=sys.stderr,
            )
            continue

        row = row_ok
        rel = str(out_path.resolve().relative_to(md_parent).as_posix())
        nbox = len(row["boxes"])
        if nbox:
            cap = (
                f"`{row['image_rel']}` — class **{row['label']}** — "
                f"{nbox} box(es)"
            )
        else:
            cap = (
                f"`{row['image_rel']}` — class **{row['label']}** — "
                "negative (no box)"
            )
        out.append(
            {
                "file": rel,
                "caption": cap,
                "label": row["label"],
                "boxes": row["boxes"],
                "source_csv": row["csv_rel"],
                "image_rel": row["image_rel"],
            }
        )

    return out, missing_labels, failed_labels


def summarize_file(file_path: Path, data_dir: Path) -> Dict:
    line_count = 0
    labels = Counter()
    bbox_lines = 0
    malformed_lines = 0
    missing_image_paths = 0

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            line_count += 1
            parts = line.split()

            if len(parts) < 2:
                malformed_lines += 1
                continue

            image_path, label_token = parts[0], parts[1]

            try:
                label = int(label_token)
                labels[label] += 1
            except ValueError:
                malformed_lines += 1
                continue

            if label > 0:
                bbox_lines += 1

            expected_image = file_path.parent / image_path
            if not expected_image.exists():
                missing_image_paths += 1

    return {
        "file": str(file_path.relative_to(data_dir).as_posix()),
        "samples": line_count,
        "bbox_samples": bbox_lines,
        "bbox_ratio": (bbox_lines / line_count) if line_count else 0.0,
        "labels": dict(sorted(labels.items())),
        "malformed_lines": malformed_lines,
        "missing_image_paths": missing_image_paths,
    }


def collect_data_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.rglob("gt*.csv"))
    return [p for p in files if p.is_file()]


def files_for_dataset_totals(
    data_dir: Path, all_files: List[Path]
) -> tuple[List[Path], str]:
    """
    Avoid double-counting when both the full manifest (gt.csv) and the
    train/test/val split files (gt_one.csv) are present: they describe the
    same samples (VarroaDataset: 8,225 + 3,408 + 1,876 = 13,509).
    """
    data_dir = data_dir.resolve()
    by_resolved = {p.resolve(): p for p in all_files}
    root_gt = (data_dir / "gt.csv").resolve()
    split_paths = [
        (data_dir / "train" / "gt_one.csv").resolve(),
        (data_dir / "test" / "gt_one.csv").resolve(),
        (data_dir / "val" / "gt_one.csv").resolve(),
    ]
    has_root = root_gt in by_resolved
    has_all_splits = all(sp in by_resolved for sp in split_paths)

    if has_root and has_all_splits:
        return [by_resolved[root_gt]], (
            "`gt.csv` is the full dataset; `train|test|val/gt_one.csv` are the same "
            "samples split into subsets. Global totals use `gt.csv` only so counts "
            "match the official 13,509 figure."
        )
    if has_root:
        return [by_resolved[root_gt]], "Global totals use `gt.csv`."
    if has_all_splits:
        return [by_resolved[sp] for sp in split_paths], (
            "No `gt.csv`; global totals sum `train`, `test`, and `val` `gt_one.csv`."
        )
    return all_files, "Global totals sum all matched `gt*.csv` files."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize BeeSafe ground-truth data files and write a report."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("data/data_summary.md"),
        help="Where to write summary Markdown (default: data/data_summary.md)",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=1,
        help="0 to skip sample figures; any positive value renders one random sample "
        "per distinct label (default: 1).",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Directory for rendered PNGs (default: next to the Markdown file, "
        "as summary_samples/)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for choosing sample images (default: 42)",
    )
    parser.add_argument(
        "--label-meanings",
        type=str,
        default=(
            "0=healthy (VarroaDataset class 0),"
            "1=Varroa-infected (VarroaDataset class 1),"
            "3=unknown (present in some exports; not defined in official README)"
        ),
        help="Comma-separated label meanings. Example: "
        '"0=healthy,1=Varroa-infected,3=unknown"',
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    md_output_path = args.md_output.resolve()
    samples_dir = (
        args.samples_dir.resolve()
        if args.samples_dir is not None
        else md_output_path.parent / "summary_samples"
    )

    if md_output_path.is_dir():
        md_output_path = md_output_path / "data_summary.md"

    md_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = collect_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files matching gt*.csv found in: {data_dir}")

    per_file = [summarize_file(file_path, data_dir) for file_path in files]

    canonical_files, totals_dedup_note = files_for_dataset_totals(data_dir, files)
    per_canonical = [summarize_file(file_path, data_dir) for file_path in canonical_files]

    totals = {
        "csv_files_used_for_total": len(canonical_files),
        "samples": sum(item["samples"] for item in per_canonical),
        "bbox_samples": sum(item["bbox_samples"] for item in per_canonical),
        "malformed_lines": sum(item["malformed_lines"] for item in per_canonical),
        "missing_image_paths": sum(item["missing_image_paths"] for item in per_canonical),
    }

    all_labels = Counter()
    for item in per_canonical:
        for label, count in item["labels"].items():
            all_labels[int(label)] += int(count)

    totals["labels"] = dict(sorted(all_labels.items()))
    totals["bbox_ratio"] = (
        totals["bbox_samples"] / totals["samples"] if totals["samples"] else 0.0
    )
    label_meanings = parse_label_meanings(args.label_meanings)

    sample_visuals, missing_label_list, failed_label_list = render_sample_visuals(
        data_dir,
        files,
        md_output_path,
        samples_dir,
        args.sample_images > 0,
        args.sample_seed,
    )

    result = {
        "data_dir": str(data_dir.name),
        "annotation_csv_files_scanned": len(files),
        "totals_dedup_note": totals_dedup_note,
        "totals": totals,
        "label_meanings": label_meanings,
        "files": per_file,
        "sample_visuals_mode": "one_per_label",
        "sample_visuals_enabled": args.sample_images > 0,
        "sample_visuals_missing_labels": missing_label_list,
        "sample_visuals_failed_labels": failed_label_list,
        "sample_visuals": sample_visuals,
    }

    md_output_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(f"Markdown report written to: {md_output_path}")
    print(
        f"CSV files scanned: {len(files)} | "
        f"Dataset total samples: {totals['samples']} | "
        f"BBox-positive rows: {totals['bbox_samples']} ({totals['bbox_ratio']:.2%})"
    )
    if args.sample_images > 0:
        n_ok = len([v for v in sample_visuals if "error" not in v])
        labs = ", ".join(str(v["label"]) for v in sample_visuals if "error" not in v)
        print(f"Sample images (one per label): {n_ok} rendered [{labs}] -> {samples_dir}")
        if missing_label_list:
            ml = ", ".join(str(x) for x in missing_label_list)
            print(f"  Labels with no on-disk image in any row: {ml}")
        if failed_label_list:
            fl = ", ".join(str(x) for x in failed_label_list)
            print(f"  Labels skipped (render failed for all candidates): {fl}")


if __name__ == "__main__":
    main()
