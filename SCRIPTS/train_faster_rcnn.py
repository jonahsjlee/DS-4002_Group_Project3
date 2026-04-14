#!/usr/bin/env python3
"""
Fine-tune Faster R-CNN (ResNet-50 FPN) on the preprocessed Aquarium VOC labels.

Walkthrough:
  1. Load class names from <data_root>/train/_classes.txt (7 marine classes).
  2. Build a torch Dataset that pairs each train/valid image with its
     labels_voc/*.xml: boxes are xyxy floats; labels are integers 1..7
     (torchvision reserves 0 for background).
  3. Load torchvision.models.detection.fasterrcnn_resnet50_fpn with pretrained
     weights, then replace the classification head so num_classes = 8.
  4. Train with the detection loss (classification + box regression + RPN).
  5. Periodically run the model on the validation loader (loss only) and
     save checkpoints under --output-dir.

Run from repo root (example):
  python3 SCRIPTS/train_faster_rcnn.py \\
    --data-root DATA/processed_aquarium \\
    --epochs 10 \\
    --batch-size 4

Requires: torch, torchvision, pillow
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


def load_class_names(data_root: Path) -> List[str]:
    p = data_root / "train" / "_classes.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def parse_voc_annotation(
    xml_path: Path, class_to_idx: Dict[str, int]
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename_el = root.find("filename")
    if filename_el is None or not filename_el.text:
        raise ValueError(f"No filename in {xml_path}")
    filename = filename_el.text

    boxes: List[List[float]] = []
    labels: List[int] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        if name not in class_to_idx:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", default="0"))
        ymin = float(bb.findtext("ymin", default="0"))
        xmax = float(bb.findtext("xmax", default="0"))
        ymax = float(bb.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[name])

    if not boxes:
        return filename, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    return filename, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class AquariumVocDataset(Dataset):
    """One processed split: images in split_root, VOC XML in split_root/labels_voc."""

    def __init__(self, split_root: Path, class_names: List[str]):
        self.split_root = Path(split_root)
        self.voc_dir = self.split_root / "labels_voc"
        if not self.voc_dir.is_dir():
            raise FileNotFoundError(f"Missing VOC labels dir: {self.voc_dir}")

        self.class_to_idx = {n: i + 1 for i, n in enumerate(class_names)}
        self._items: List[Tuple[Path, torch.Tensor, torch.Tensor]] = []

        for xml_path in sorted(self.voc_dir.glob("*.xml")):
            fname, boxes, labels = parse_voc_annotation(xml_path, self.class_to_idx)
            if labels.numel() == 0:
                continue
            img_path = self.split_root / fname
            if not img_path.is_file():
                continue
            self._items.append((img_path, boxes, labels))

        if not self._items:
            raise RuntimeError(f"No usable images with boxes under {self.split_root}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, boxes, labels = self._items[idx]
        image = Image.open(img_path).convert("RGB")
        image_t = TF.to_tensor(image)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target: Dict[str, Any] = {
            "boxes": boxes.clone(),
            "labels": labels.clone(),
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return image_t, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _move_targets(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


@torch.no_grad()
def mean_forward_loss(model, data_loader, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, targets in data_loader:
        images = [im.to(device) for im in images]
        targets = _move_targets(list(targets), device)
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()))
        n += 1
    return total / max(n, 1)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    max_norm: float,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for images, targets in data_loader:
        images = [im.to(device) for im in images]
        targets = _move_targets(list(targets), device)

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        running += float(losses.detach())
        n += 1
    return running / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Aquarium VOC data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "DATA" / "processed_aquarium",
        help="Root with train/, valid/, test/ subfolders from preprocess.py.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-norm", type=float, default=1.0, help="Gradient clip (0 disables).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "DATA" / "checkpoints_faster_rcnn",
        help="Where to save faster_rcnn_epochXX.pt",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    train_root = data_root / "train"
    valid_root = data_root / "valid"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train split: {train_root}")

    class_names = load_class_names(data_root)
    num_classes = len(class_names) + 1

    train_ds = AquariumVocDataset(train_root, class_names)
    val_ds = AquariumVocDataset(valid_root, class_names) if valid_root.is_dir() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        if val_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"num_classes (incl. background)={num_classes}")
    print(f"Train samples: {len(train_ds)} | device={device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, max_norm=args.max_norm
        )
        msg = f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}"
        if val_loader is not None:
            val_loss = mean_forward_loss(model, val_loader, device)
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

        ckpt = args.output_dir / f"faster_rcnn_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
            },
            ckpt,
        )
        print(f"  saved {ckpt}")


if __name__ == "__main__":
    main()
