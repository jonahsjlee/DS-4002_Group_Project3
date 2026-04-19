#!/usr/bin/env python3
# Preprocess the Roboflow Aquarium export (_annotations.txt) for YOLOv4, Faster R-CNN, and SSD.
# Input: train|valid|test with images, _annotations.txt, _classes.txt.
# Output per split: resized images, _annotations.txt, labels_yolo/*.txt, labels_voc/*.xml.
# Train split can get flip + brightness augmentation.

from __future__ import annotations

import argparse
import hashlib
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
LABELS_YOLO = "labels_yolo"
LABELS_VOC = "labels_voc"


# One bounding box in pixels plus Roboflow class id.
@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int


# Split one Roboflow line into image filename and list of boxes.
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


# Build one Roboflow-style line for the output _annotations.txt.
def format_annotation_line(filename: str, boxes: Iterable[Box]) -> str:
    box_tokens = [f"{b.x1},{b.y1},{b.x2},{b.y2},{b.class_id}" for b in boxes]
    if not box_tokens:
        return filename
    return f"{filename} {' '.join(box_tokens)}"


# Read class names from _classes.txt (line order = class id).
def load_class_names(classes_path: Path) -> List[str]:
    lines = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


# Flip image left-right and update box coordinates.
def horizontal_flip_image_and_boxes(img: Any, boxes: List[Box]) -> Tuple[Any, List[Box]]:
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)
    new_boxes: List[Box] = []
    for b in boxes:
        nx1 = w - 1 - b.x2
        nx2 = w - 1 - b.x1
        new_boxes.append(Box(nx1, b.y1, nx2, b.y2, b.class_id))
    return flipped, new_boxes


# Random multiplicative brightness (underwater lighting jitter).
def random_brightness(img, low: float, high: float, rng: random.Random):
    factor = rng.uniform(low, high)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


# Denoise, CLAHE contrast, optional resize; return image and x/y scale vs original.
def preprocess_image(
    img: Any,
    width: int,
    height: int,
    denoise_strength: int,
    clip_limit: float,
) -> Tuple[Any, float, float]:
    original_h, original_w = img.shape[:2]

    if denoise_strength > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)

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


# Scale boxes after resize; clip to image and drop invalid boxes.
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


# YOLO label text: class id + box center and size (normalized 0–1).
def boxes_to_yolo_txt(boxes: Iterable[Box], img_w: int, img_h: int) -> str:
    lines: List[str] = []
    iw, ih = float(img_w), float(img_h)
    for b in boxes:
        bw = b.x2 - b.x1
        bh = b.y2 - b.y1
        cx = (b.x1 + b.x2) / 2.0
        cy = (b.y1 + b.y2) / 2.0
        lines.append(
            f"{b.class_id} {cx / iw:.6f} {cy / ih:.6f} {bw / iw:.6f} {bh / ih:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


# Helper: add a child XML element with text (VOC writer).
def _sub_el(parent: ET.Element, name: str, text: str) -> None:
    el = ET.SubElement(parent, name)
    el.text = text


# Write one Pascal VOC XML for an image and its objects.
def write_voc_xml(
    path: Path,
    *,
    folder: str,
    filename: str,
    path_rel: str,
    width: int,
    height: int,
    depth: int,
    class_names: Sequence[str],
    boxes: Iterable[Box],
) -> None:
    annotation = ET.Element("annotation")
    _sub_el(annotation, "folder", folder)
    _sub_el(annotation, "filename", filename)
    _sub_el(annotation, "path", path_rel)

    source = ET.SubElement(annotation, "source")
    _sub_el(source, "database", "Aquarium Combined (Roboflow)")

    size_el = ET.SubElement(annotation, "size")
    _sub_el(size_el, "width", str(width))
    _sub_el(size_el, "height", str(height))
    _sub_el(size_el, "depth", str(depth))

    _sub_el(annotation, "segmented", "0")

    for b in boxes:
        if b.class_id < 0 or b.class_id >= len(class_names):
            continue
        obj = ET.SubElement(annotation, "object")
        _sub_el(obj, "name", class_names[b.class_id])
        _sub_el(obj, "pose", "Unspecified")
        _sub_el(obj, "truncated", "0")
        _sub_el(obj, "difficult", "0")
        bb = ET.SubElement(obj, "bndbox")
        _sub_el(bb, "xmin", str(b.x1))
        _sub_el(bb, "ymin", str(b.y1))
        _sub_el(bb, "xmax", str(b.x2))
        _sub_el(bb, "ymax", str(b.y2))

    tree = ET.ElementTree(annotation)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


# Blur proxy: higher Laplacian variance usually means sharper.
def blur_score(img: Any) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Process one split folder into output images + YOLO + VOC + updated annotations.
def process_split(
    split_dir: Path,
    out_split_dir: Path,
    class_names: List[str],
    width: int,
    height: int,
    denoise_strength: int,
    clip_limit: float,
    min_blur_score: float,
    *,
    augment: bool,
    hflip_prob: float,
    brightness_low: float,
    brightness_high: float,
    seed: int,
) -> Tuple[int, int]:
    annotations_path = split_dir / "_annotations.txt"
    if not annotations_path.exists():
        print(f"[skip] {split_dir.name}: missing _annotations.txt")
        return 0, 0

    out_split_dir.mkdir(parents=True, exist_ok=True)
    (out_split_dir / LABELS_YOLO).mkdir(parents=True, exist_ok=True)
    (out_split_dir / LABELS_VOC).mkdir(parents=True, exist_ok=True)

    split_mix = int(hashlib.md5(split_dir.name.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    rng = random.Random(seed + split_mix)
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
        if min_blur_score > 0 and score < min_blur_score:
            dropped += 1
            continue

        work_boxes = [Box(b.x1, b.y1, b.x2, b.y2, b.class_id) for b in boxes]

        if augment:
            if rng.random() < hflip_prob:
                img, work_boxes = horizontal_flip_image_and_boxes(img, work_boxes)
            img = random_brightness(img, brightness_low, brightness_high, rng)

        img, sx, sy = preprocess_image(
            img=img,
            width=width,
            height=height,
            denoise_strength=denoise_strength,
            clip_limit=clip_limit,
        )

        h, w = img.shape[:2]
        new_boxes = scale_boxes(work_boxes, sx=sx, sy=sy, max_w=w, max_h=h)

        image_out = out_split_dir / filename
        ok = cv2.imwrite(str(image_out), img)
        if not ok:
            dropped += 1
            continue

        stem = Path(filename).stem
        yolo_path = out_split_dir / LABELS_YOLO / f"{stem}.txt"
        yolo_path.write_text(boxes_to_yolo_txt(new_boxes, w, h), encoding="utf-8")

        voc_path = out_split_dir / LABELS_VOC / f"{stem}.xml"
        write_voc_xml(
            voc_path,
            folder=split_dir.name,
            filename=filename,
            path_rel=str(image_out.resolve()),
            width=w,
            height=h,
            depth=3,
            class_names=class_names,
            boxes=new_boxes,
        )

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


# CLI: paths, resize, denoise, blur filter, augmentation knobs.
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess Roboflow object-detection export: resize, optional "
        "augmentation, YOLO + VOC annotation exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Dataset root containing train/, valid/, test/ (Roboflow export).",
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
        default=0.0,
        help="Drop images with blur score below this threshold (0 keeps all images).",
    )
    parser.add_argument(
        "--augment-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply flip/brightness augmentation on the train split only (default: on).",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Probability of horizontal flip when augmentation is enabled.",
    )
    parser.add_argument(
        "--brightness-min",
        type=float,
        default=0.85,
        help="Lower bound of random brightness multiplier (augmented splits).",
    )
    parser.add_argument(
        "--brightness-max",
        type=float,
        default=1.15,
        help="Upper bound of random brightness multiplier (augmented splits).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for augmentation.")
    return parser


# Entry point: preprocess train/valid/test and print counts.
def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    train_classes = input_dir / "train" / "_classes.txt"
    if not train_classes.exists():
        print(f"[error] Missing {train_classes}. Is --input-dir the Roboflow dataset root?")
        raise SystemExit(1)
    class_names = load_class_names(train_classes)

    total_kept = 0
    total_dropped = 0
    for split in SPLITS:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        augment = bool(args.augment_train and split == "train")
        kept, dropped = process_split(
            split_dir=split_dir,
            out_split_dir=output_dir / split,
            class_names=class_names,
            width=args.width,
            height=args.height,
            denoise_strength=args.denoise_strength,
            clip_limit=args.clip_limit,
            min_blur_score=args.min_blur_score,
            augment=augment,
            hflip_prob=args.hflip_prob,
            brightness_low=args.brightness_min,
            brightness_high=args.brightness_max,
            seed=args.seed,
        )
        total_kept += kept
        total_dropped += dropped
        print(f"[{split}] kept={kept} dropped={dropped}")

    print(f"Done. Total kept={total_kept}, dropped={total_dropped}")
    print(f"Processed dataset written to: {output_dir.resolve()}")
    print(
        "Per split: resized images, _annotations.txt (xyxy), labels_yolo/*.txt, labels_voc/*.xml"
    )


if __name__ == "__main__":
    main()
