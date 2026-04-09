from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


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
        labels_text = ", ".join(
            f"{label}:{count}" for label, count in item["labels"].items()
        )
        lines.append(
            f"| `{item['file']}` | {item['samples']} | {item['bbox_samples']} | "
            f"{item['bbox_ratio']:.2%} | {item['malformed_lines']} | "
            f"{item['missing_image_paths']} | {labels_text} |"
        )

    return "\n".join(lines) + "\n"


def summarize_file(file_path: Path) -> Dict:
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
        "file": str(file_path.as_posix()),
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
        default=Path("data_summary.json"),
        help="Where to write summary JSON (default: data_summary.json)",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("data_summary.md"),
        help="Where to write Markdown report (default: data_summary.md)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    md_output_path = args.md_output.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = collect_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files matching gt*.csv found in: {data_dir}")

    per_file = [summarize_file(file_path) for file_path in files]

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

    result = {
        "data_dir": str(data_dir.as_posix()),
        "totals": totals,
        "files": per_file,
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_output_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(f"Summary written to: {output_path}")
    print(f"Markdown report written to: {md_output_path}")
    print(
        f"Files: {totals['files']} | Samples: {totals['samples']} | "
        f"BBox samples: {totals['bbox_samples']} ({totals['bbox_ratio']:.2%})"
    )


if __name__ == "__main__":
    main()
