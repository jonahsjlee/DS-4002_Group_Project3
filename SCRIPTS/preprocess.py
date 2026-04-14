#!/usr/bin/env python3
"""
Preprocess Roboflow YOLOv4-style datasets that use `_annotations.txt`.

Input layout (expected):
    dataset/
      train/
        *.jpg
        _annotations.txt
        _classes.txt
      valid/
        ...
      test/
        ...

Output layout:
    <output_dir>/
      train/
      valid/
      test/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


def parse_annotation_line(line: str) -> Tuple[str, List[Box]]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("Empty annotation line.")

    filename = parts[0]
    boxes: List[Box] = []
    for token in parts[1:]:
        x1, y1, x2, y2, cls = map(int, token.split(","))
        boxes.append(Box(x1, y1, x2, y2, cls))
    return filename, boxes


def format_annotation_line(filename: str, boxes: Iterable[Box]) -> str:
    box_tokens = [f"{b.x1},{b.y1},{b.x2},{b.y2},{b.class_id}" for b in boxes]
    if not box_tokens:
        return filename
    return f"{filename} {' '.join(box_tokens)}"


def preprocess_image(
    img,
    width: int,
    height: int,
    denoise_strength: int,
    clip_limit: float,
) -> Tuple:
    original_h, original_w = img.shape[:2]

    # Mild denoise that generally preserves object boundaries.
    if denoise_strength > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)

    # CLAHE on luminance channel for underwater/low-contrast scenes.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge((l_chan, a_chan, b_chan))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    if width > 0 and height > 0:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    new_h, new_w = img.shape[:2]
    sx = new_w / original_w
    sy = new_h / original_h
    return img, sx, sy


def scale_boxes(boxes: Iterable[Box], sx: float, sy: float, max_w: int, max_h: int) -> List[Box]:
    scaled: List[Box] = []
    for b in boxes:
        x1 = max(0, min(max_w - 1, int(round(b.x1 * sx))))
        y1 = max(0, min(max_h - 1, int(round(b.y1 * sy))))
        x2 = max(0, min(max_w - 1, int(round(b.x2 * sx))))
        y2 = max(0, min(max_h - 1, int(round(b.y2 * sy))))
        if x2 <= x1 or y2 <= y1:
            continue
        scaled.append(Box(x1, y1, x2, y2, b.class_id))
    return scaled


def blur_score(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_split(
    split_dir: Path,
    out_split_dir: Path,
    width: int,
    height: int,
    denoise_strength: int,
    clip_limit: float,
    min_blur_score: float,
) -> Tuple[int, int]:
    annotations_path = split_dir / "_annotations.txt"
    if not annotations_path.exists():
        print(f"[skip] {split_dir.name}: missing _annotations.txt")
        return 0, 0

    out_split_dir.mkdir(parents=True, exist_ok=True)
    kept_lines: List[str] = []
    processed = 0
    dropped = 0

    for raw_line in annotations_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        try:
            filename, boxes = parse_annotation_line(raw_line)
        except ValueError:
            print(f"[warn] Could not parse line in {annotations_path.name}: {raw_line[:120]}")
            dropped += 1
            continue

        image_in = split_dir / filename
        if image_in.suffix.lower() not in IMAGE_EXTS or not image_in.exists():
            dropped += 1
            continue

        img = cv2.imread(str(image_in))
        if img is None:
            dropped += 1
            continue

        score = blur_score(img)
        if score < min_blur_score:
            dropped += 1
            continue

        img, sx, sy = preprocess_image(
            img=img,
            width=width,
            height=height,
            denoise_strength=denoise_strength,
            clip_limit=clip_limit,
        )

        h, w = img.shape[:2]
        new_boxes = scale_boxes(boxes, sx=sx, sy=sy, max_w=w, max_h=h)

        image_out = out_split_dir / filename
        ok = cv2.imwrite(str(image_out), img)
        if not ok:
            dropped += 1
            continue

        kept_lines.append(format_annotation_line(filename, new_boxes))
        processed += 1

    (out_split_dir / "_annotations.txt").write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )

    classes_src = split_dir / "_classes.txt"
    if classes_src.exists():
        (out_split_dir / "_classes.txt").write_text(classes_src.read_text(encoding="utf-8"), encoding="utf-8")

    return processed, dropped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Roboflow YOLOv4-style image dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Dataset root containing train/valid/test folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for processed dataset.",
    )
    parser.add_argument("--width", type=int, default=640, help="Resize width. Use <=0 to keep original.")
    parser.add_argument("--height", type=int, default=640, help="Resize height. Use <=0 to keep original.")
    parser.add_argument(
        "--denoise-strength",
        type=int,
        default=6,
        help="fastNlMeans denoise strength (0 disables denoise).",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for local contrast enhancement.",
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=50.0,
        help="Drop images with blur score below this threshold.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    total_kept = 0
    total_dropped = 0
    for split in SPLITS:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        kept, dropped = process_split(
            split_dir=split_dir,
            out_split_dir=output_dir / split,
            width=args.width,
            height=args.height,
            denoise_strength=args.denoise_strength,
            clip_limit=args.clip_limit,
            min_blur_score=args.min_blur_score,
        )
        total_kept += kept
        total_dropped += dropped
        print(f"[{split}] kept={kept} dropped={dropped}")

    print(f"Done. Total kept={total_kept}, dropped={total_dropped}")
    print(f"Processed dataset written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
