#!/usr/bin/env python3
# Train or evaluate Faster R-CNN (ResNet-50 FPN) on VOC-style aquarium data.
# Writes mAP / MAR / micro PRF, FPS, and checkpoints. Needs torch, torchvision, torchmetrics, Pillow.
# Example: python3 SCRIPTS/train_faster_rcnn.py --data-root DATA/processed_aquarium --epochs 5

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "torchmetrics is required for mAP / recall metrics. "
        "Install with: pip install -r requirements-detection.txt"
    ) from e


# Read class names from train/_classes.txt (one name per line).
def load_class_names(data_root: Path) -> List[str]:
    p = data_root / "train" / "_classes.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# Parse one Pascal VOC XML into filename, box tensor, and label tensor (1-based labels).
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


# PyTorch dataset: images in split_root, one VOC XML per image under labels_voc/.
class AquariumVocDataset(Dataset):
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

    # Number of usable (image, boxes) pairs in this split.
    def __len__(self) -> int:
        return len(self._items)

    # Load one image as a tensor and return the detection target dict for the model.
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


# Batch images and targets as lists (required for detection models).
def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


# Faster R-CNN with COCO-pretrained backbone and a new box head for our class count.
def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Move every tensor in each target dict to GPU or CPU.
def _move_targets(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


# Map labels 1..K (model) to 0..K-1 (torchmetrics).
def _to_zero_indexed_labels(labels: torch.Tensor) -> torch.Tensor:
    return (labels - 1).to(dtype=torch.int64)


# Average total detection loss over batches (model stays in train mode).
@torch.no_grad()
def mean_forward_loss(
    model, data_loader, device, max_batches: Optional[int] = None
) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, targets in data_loader:
        images = [im.to(device) for im in images]
        targets = _move_targets(list(targets), device)
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()))
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return total / max(n, 1)


# One training epoch: forward, backward, optional grad clip, optimizer step.
def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    max_norm: float,
    max_batches: Optional[int] = None,
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
        if max_batches is not None and n >= max_batches:
            break
    return running / max(n, 1)


# Filter predictions by score and align label ids for torchmetrics.
def _preds_targets_for_metric(
    raw_preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    score_threshold: float,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    preds_out: List[Dict[str, torch.Tensor]] = []
    for p in raw_preds:
        mask = p["scores"] > score_threshold
        preds_out.append(
            {
                "boxes": p["boxes"][mask].detach().cpu(),
                "scores": p["scores"][mask].detach().cpu(),
                "labels": _to_zero_indexed_labels(p["labels"][mask]).detach().cpu(),
            }
        )
    targets_out = []
    for t in targets:
        targets_out.append(
            {
                "boxes": t["boxes"].detach().cpu(),
                "labels": _to_zero_indexed_labels(t["labels"]).detach().cpu(),
            }
        )
    return preds_out, targets_out


# Turn nested tensors into plain Python floats and lists for JSON.
def _tensor_to_python(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_python(v) for v in obj]
    return obj


# Micro precision, recall, F1 at IoU 0.5 by greedy score-sorted matching to GT.
@torch.no_grad()
def micro_precision_recall_f1_iou50(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
) -> Tuple[float, float, float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for p, t in zip(preds, targets):
        pb, pl = p["boxes"], p["labels"]
        tb, tl = t["boxes"], t["labels"]
        if pb.shape[0] == 0:
            total_fn += int(tl.shape[0])
            continue
        if tb.shape[0] == 0:
            total_fp += int(pl.shape[0])
            continue
        ious = box_iou(pb, tb)
        matched_gt = torch.zeros(tb.shape[0], dtype=torch.bool)
        tp = fp = 0
        order = torch.argsort(p["scores"], descending=True)
        for idx in order.tolist():
            best_iou = 0.0
            best_j = -1
            for j in range(tb.shape[0]):
                if matched_gt[j]:
                    continue
                if pl[idx] != tl[j]:
                    continue
                iou = float(ious[idx, j].item())
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                matched_gt[best_j] = True
                tp += 1
            else:
                fp += 1
        fn = int((~matched_gt).sum().item())
        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    prec = total_tp / denom_p if denom_p > 0 else 0.0
    rec = total_tp / denom_r if denom_r > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# Full test pass: timed inference, torchmetrics mAP/MAR, micro PRF, parameter count.
@torch.no_grad()
def evaluate_detection_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float,
    warmup_batches: int = 2,
) -> Dict[str, Any]:
    model.eval()

    # Warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    it = iter(data_loader)
    for _ in range(warmup_batches):
        try:
            images, _targets = next(it)
        except StopIteration:
            it = iter(data_loader)
            images, _targets = next(it)
        images = [im.to(device) for im in images]
        _ = model(images)

    # Timed pass: forward only (throughput comparable across detectors)
    n_images = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for images, _targets in data_loader:
        images_gpu = [im.to(device) for im in images]
        _ = model(images_gpu)
        n_images += len(images_gpu)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_images / elapsed if elapsed > 0 else 0.0
    ms_per_image = (elapsed / n_images) * 1000.0 if n_images > 0 else 0.0

    # mAP / MAR (torchmetrics) + micro PRF (second pass; small test set)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True, iou_type="bbox")
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []
    for images, targets in data_loader:
        images_gpu = [im.to(device) for im in images]
        raw = model(images_gpu)
        preds_cpu, targets_cpu = _preds_targets_for_metric(raw, list(targets), score_threshold)
        metric.update(preds_cpu, targets_cpu)
        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)

    computed = metric.compute()
    flat = _tensor_to_python(computed)
    prec50, rec50, f1_50 = micro_precision_recall_f1_iou50(all_preds, all_targets)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out: Dict[str, Any] = {
        "n_images_evaluated": n_images,
        "score_threshold": score_threshold,
        "fps_images_per_sec": round(fps, 4),
        "latency_ms_per_image_mean": round(ms_per_image, 4),
        "trainable_parameters": int(trainable),
        "micro_precision_iou50": round(prec50, 6),
        "micro_recall_iou50": round(rec50, 6),
        "micro_f1_iou50": round(f1_50, 6),
        "torchmetrics": flat,
    }
    # Promote common keys for easy comparison tables
    if isinstance(flat, dict):
        for k in ("map", "map_50", "map_75", "map_large", "map_medium", "map_small"):
            if k in flat:
                out[k] = flat[k]
        for k in ("mar_1", "mar_10", "mar_100"):
            if k in flat:
                out[k] = flat[k]
    return out


# CLI: load data, train or resume, evaluate test set, save metrics JSON.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train / eval Faster R-CNN with detection metrics.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "DATA" / "processed_aquarium",
        help="Root with train/, valid/, test/ from preprocess.py.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "OUTPUT" / "checkpoints_faster_rcnn",
        help="Checkpoints (faster_rcnn_epoch*.pt, faster_rcnn_last.pt).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "OUTPUT" / "metrics_faster_rcnn_test.json",
        help="Write test-set metrics JSON here after training (or eval-only).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Filter low-confidence predictions before mAP / PRF matching.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load --resume and run test metrics only.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint from train_faster_rcnn (.pt). Used with --eval-only or to init weights.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each train epoch after this many batches (CPU smoke tests). Default: full epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Cap validation loss batches each epoch. Default: full valid loader.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    train_root = data_root / "train"
    valid_root = data_root / "valid"
    test_root = data_root / "test"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train split: {train_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test split: {test_root}")

    class_names = load_class_names(data_root)
    num_classes = len(class_names) + 1

    train_ds = AquariumVocDataset(train_root, class_names)
    val_ds = AquariumVocDataset(valid_root, class_names) if valid_root.is_dir() else None
    test_ds = AquariumVocDataset(test_root, class_names)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    resume_path: Optional[Path] = args.resume
    if args.eval_only and resume_path is None:
        resume_path = args.output_dir / "faster_rcnn_last.pt"
    if resume_path is not None and resume_path.is_file():
        load_kw: Dict[str, Any] = {"map_location": device}
        try:
            ck = torch.load(resume_path, **load_kw, weights_only=False)
        except TypeError:
            ck = torch.load(resume_path, **load_kw)
        model.load_state_dict(ck["model_state_dict"])
        print(f"Loaded weights from {resume_path}")
        if not args.eval_only and isinstance(ck.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            print("Restored optimizer state (resume training).")
    elif args.eval_only:
        raise FileNotFoundError("--eval-only requires a checkpoint; pass --resume or train first.")

    if not args.eval_only:
        print(f"Classes ({len(class_names)}): {class_names}")
        print(f"num_classes (incl. background)={num_classes}")
        print(f"Train={len(train_ds)} valid={len(val_ds) if val_ds else 0} test={len(test_ds)} | device={device}")
        if args.max_train_batches is not None:
            print(f"[limit] max train batches per epoch: {args.max_train_batches}")
        if args.max_val_batches is not None:
            print(f"[limit] max val batches per epoch: {args.max_val_batches}")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                max_norm=args.max_norm,
                max_batches=args.max_train_batches,
            )
            msg = f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}"
            if val_loader is not None:
                val_loss = mean_forward_loss(
                    model, val_loader, device, max_batches=args.max_val_batches
                )
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

        last_path = args.output_dir / "faster_rcnn_last.pt"
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
            },
            last_path,
        )
        print(f"saved {last_path}")

    print("Evaluating on test set (mAP, recall, FPS, micro PRF @ IoU 0.5)...")
    metrics = evaluate_detection_metrics(
        model,
        test_loader,
        device,
        score_threshold=args.score_threshold,
    )
    metrics["model"] = "fasterrcnn_resnet50_fpn"
    metrics["split"] = "test"
    metrics["class_names_0indexed"] = class_names

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics if k != "torchmetrics"}, indent=2))
    print(f"Wrote full metrics (including per-class torchmetrics) to {args.metrics_json}")


if __name__ == "__main__":
    main()
