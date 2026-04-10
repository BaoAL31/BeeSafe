from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]
except ImportError:
    Image = None  # type: ignore[misc, assignment]
    ImageDraw = None  # type: ignore[misc, assignment]
    ImageFont = None  # type: ignore[misc, assignment]


BBox = Tuple[int, int, int, int]


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


def iter_annotation_rows(
    data_dir: Path, gt_paths: Sequence[Path]
) -> List[Dict]:
    rows: List[Dict] = []
    for csv_path in gt_paths:
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


def _row_key(row: Dict) -> str:
    return str(row["abs_image"].resolve())


def pick_visual_samples(
    rows: List[Dict],
    max_samples: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    existing = [r for r in rows if r["abs_image"].exists()]
    with_bbox = [r for r in existing if r["boxes"]]
    negatives = [r for r in existing if r["label"] == 0 and not r["boxes"]]

    if not with_bbox and not negatives:
        return []

    n_neg = min(2, len(negatives), max(0, max_samples // 4))
    n_pos = max(0, min(len(with_bbox), max_samples - n_neg))

    seen: set[str] = set()
    chosen: List[Dict] = []

    by_label: Dict[int, List[Dict]] = defaultdict(list)
    for r in with_bbox:
        by_label[r["label"]].append(r)
    for lb in by_label:
        rng.shuffle(by_label[lb])
    labels = sorted(by_label.keys())

    if labels and n_pos > 0:
        per = max(1, n_pos // len(labels))
        for lb in labels:
            for r in by_label[lb][:per]:
                if len(chosen) >= n_pos:
                    break
                k = _row_key(r)
                if k in seen:
                    continue
                seen.add(k)
                chosen.append(r)
            if len(chosen) >= n_pos:
                break

        pool = [r for r in with_bbox if _row_key(r) not in seen]
        rng.shuffle(pool)
        for r in pool:
            if len(chosen) >= n_pos:
                break
            seen.add(_row_key(r))
            chosen.append(r)

    rng.shuffle(negatives)
    for r in negatives:
        if len(chosen) >= max_samples:
            break
        k = _row_key(r)
        if k in seen:
            continue
        seen.add(k)
        chosen.append(r)

    while len(chosen) < max_samples:
        pool = [r for r in with_bbox if _row_key(r) not in seen]
        if not pool:
            break
        r = pool.pop(rng.randrange(len(pool)))
        seen.add(_row_key(r))
        chosen.append(r)

    return chosen[:max_samples]


def draw_sample_image(
    row: Dict,
    out_path: Path,
) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for sample images")
    img = Image.open(row["abs_image"]).convert("RGB")
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
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x1 + bi * 2
            ty = max(0, y1 - th - 4)
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=color)
            draw.text((tx + 2, ty + 1), text, fill="white", font=font)
    else:
        msg = f"label {label} (no box)"
        draw.rectangle([4, 4, 220, 28], fill="#1e293b")
        draw.text((8, 8), msg, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def build_markdown_report(result: Dict) -> str:
    totals = result["totals"]
    lines = [
        "# BeeSafe Data Summary",
        "",
        f"- Data directory: `{result['data_dir']}`",
        f"- Files: **{totals['files']}**",
        f"- Samples: **{totals['samples']}**",
        f"- BBox samples: **{totals['bbox_samples']}** ({totals['bbox_ratio']:.2%})",
        f"- Malformed lines: **{totals['malformed_lines']}**",
        f"- Missing image paths: **{totals['missing_image_paths']}**",
        "",
        "## Label Distribution (Overall)",
        "",
        "| Label | Count | Ratio |",
        "|---:|---:|---:|",
    ]

    total_samples = totals["samples"] or 1
    for label, count in totals["labels"].items():
        ratio = count / total_samples
        lines.append(f"| {label} | {count} | {ratio:.2%} |")

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

    lines.extend(["", "## Sample images (boxes + labels)", ""])

    req = int(result.get("sample_visuals_requested") or 0)
    visuals = result.get("sample_visuals")
    if visuals is None:
        lines.append("*Sample images were not generated.*")
    elif any("error" in v for v in visuals):
        err = next(v["error"] for v in visuals if "error" in v)
        lines.append(f"*Could not render sample images: {err}*")
    elif not visuals:
        lines.append(
            "*No sample images were produced (no matching image files on disk, "
            "or use `--sample-images N` with N > 0).*"
        )
    else:
        ok_count = len([v for v in visuals if "error" not in v])
        if req > 0 and ok_count < req:
            lines.append(
                f"*Note: requested **{req}** sample(s); **{ok_count}** could be "
                "rendered (fewer image files exist locally than annotations reference).*"
            )
            lines.append("")
        for i, v in enumerate(visuals, start=1):
            cap = v["caption"]
            rel = v["file"]
            lines.append(f"### Sample {i}")
            lines.append("")
            lines.append(f"![Sample {i}]({rel})")
            lines.append("")
            lines.append(f"*{cap}*")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
    max_samples: int,
    seed: int,
) -> List[Dict]:
    if max_samples <= 0:
        return []

    rows = iter_annotation_rows(data_dir, gt_paths)
    picked = pick_visual_samples(rows, max_samples, seed)
    clear_sample_dir(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    md_parent = md_output_path.parent.resolve()

    if Image is None or ImageDraw is None:
        return [
            {
                "error": "Pillow is not installed; run: pip install -r requirements.txt"
            }
        ]

    out: List[Dict] = []
    for row in picked:
        out_path = samples_dir / f"sample_{len(out) + 1:02d}.png"
        try:
            draw_sample_image(row, out_path)
        except (OSError, ValueError):
            continue

        rel = str(out_path.resolve().relative_to(md_parent).as_posix())
        nbox = len(row["boxes"])
        if nbox:
            cap = (
                f"`{row['csv_rel']}` — class **{row['label']}** — "
                f"{nbox} box(es)"
            )
        else:
            cap = (
                f"`{row['csv_rel']}` — class **{row['label']}** — "
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

    return out


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
        "--output",
        type=Path,
        default=Path("data/data_summary.json"),
        help="Where to write summary JSON (default: data/data_summary.json)",
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
        default=8,
        help="Number of annotated sample images to render (0 to skip). Default: 8",
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

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    md_output_path = args.md_output.resolve()
    samples_dir = (
        args.samples_dir.resolve()
        if args.samples_dir is not None
        else md_output_path.parent / "summary_samples"
    )

    if output_path.is_dir():
        output_path = output_path / "data_summary.json"
    if md_output_path.is_dir():
        md_output_path = md_output_path / "data_summary.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = collect_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files matching gt*.csv found in: {data_dir}")

    per_file = [summarize_file(file_path, data_dir) for file_path in files]

    totals = {
        "files": len(per_file),
        "samples": sum(item["samples"] for item in per_file),
        "bbox_samples": sum(item["bbox_samples"] for item in per_file),
        "malformed_lines": sum(item["malformed_lines"] for item in per_file),
        "missing_image_paths": sum(item["missing_image_paths"] for item in per_file),
    }

    all_labels = Counter()
    for item in per_file:
        for label, count in item["labels"].items():
            all_labels[int(label)] += int(count)

    totals["labels"] = dict(sorted(all_labels.items()))
    totals["bbox_ratio"] = (
        totals["bbox_samples"] / totals["samples"] if totals["samples"] else 0.0
    )

    sample_visuals = render_sample_visuals(
        data_dir,
        files,
        md_output_path,
        samples_dir,
        args.sample_images,
        args.sample_seed,
    )

    result = {
        "data_dir": str(data_dir.name),
        "totals": totals,
        "files": per_file,
        "sample_visuals_requested": args.sample_images,
        "sample_visuals": sample_visuals,
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_output_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(f"Summary written to: {output_path}")
    print(f"Markdown report written to: {md_output_path}")
    print(
        f"Files: {totals['files']} | Samples: {totals['samples']} | "
        f"BBox samples: {totals['bbox_samples']} ({totals['bbox_ratio']:.2%})"
    )
    if args.sample_images > 0:
        n_ok = len([v for v in sample_visuals if "error" not in v])
        print(
            f"Sample images: {n_ok}/{args.sample_images} rendered -> {samples_dir}"
        )


if __name__ == "__main__":
    main()
