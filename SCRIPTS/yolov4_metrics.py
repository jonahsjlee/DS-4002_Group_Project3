
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import shutil
import gdown
import torch
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# below is code copied and pasted from training to clone and build darknet and prepare dataset and cfg

# uncomment below if you need to rerun the code
# resets Colab folders to prevent unecessary copies and loops
os.chdir('/content')
!rm -rf /content/darknet
!rm -rf /content/DS-4002_Group_Project3

# clone needed repositories (YOLOv4 implementation and the project repo)
!git clone https://github.com/AlexeyAB/darknet /content/darknet
!git clone https://github.com/jonahsjlee/DS-4002_Group_Project3.git /content/DS-4002_Group_Project3

# install OpenCV for Darknet to work
!apt-get update
!apt-get install -y libopencv-dev

makefile = Path('/content/darknet/Makefile')
text = makefile.read_text()

# turn off GPU
text = text.replace('GPU=1', 'GPU=0')
text = text.replace('CUDNN=1', 'CUDNN=0')
text = text.replace('CUDNN_HALF=1', 'CUDNN_HALF=0')

# keep OpenCV on
text = text.replace('OPENCV=0', 'OPENCV=1')

makefile.write_text(text)

# compile Darknet
%cd /content/darknet
!make clean
!make -j8

# define classes in the aquarium dataset
classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
num_classes = len(classes)

# set source to the aquarium dataset
# set destination to the adjusted aquarium dataset for darknet
src_root = Path('/content/DS-4002_Group_Project3/DATA/processed_aquarium')
dst_root = Path('/content/darknet/data/aquarium')

# creates output folders in darknet format
for split in ['train', 'valid', 'test']:
    (dst_root / split).mkdir(parents=True, exist_ok=True)

def copy_split(split):
    # defining split folders
    split_dir = src_root / split
    img_dir = split_dir
    lbl_dir = split_dir / 'labels_yolo'

    # initializing counters
    copied = 0
    skipped_missing = 0
    skipped_empty = 0
    skipped_invalid = 0

    # collecting images
    image_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    # matching each image to label
    for img_path in image_files:
        label_path = lbl_dir / f'{img_path.stem}.txt'
        # skipping missing labels
        if not label_path.exists():
            skipped_missing += 1
            continue

        # reading label files
        raw = label_path.read_text(encoding='utf-8', errors='ignore').strip()
        # skipping empty labels
        if not raw:
            skipped_empty += 1
            continue

        # checking label lines to prevent crashes
        valid_lines = []
        ok = True
        for line in raw.splitlines():
            parts = line.strip().split()
            # making sure each line has 5 values
            if len(parts) != 5:
                ok = False
                break
            # making sure values are numeric
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except Exception:
                ok = False
                break
            # making sure classes are 0-7
            if not (0 <= cls < num_classes):
                ok = False
                break
            # checking that values are normalized between 0 and 1
            if not all(0.0 <= v <= 1.0 for v in [x, y, w, h]):
                ok = False
                break
            # saving checked lines that pass the above conditions
            valid_lines.append(f'{cls} {x} {y} {w} {h}')

        # skipping lines that fail inspection
        if not ok or len(valid_lines) == 0:
            skipped_invalid += 1
            continue

        # copies image into darknet folder and writes a label file for the image
        dst_img = dst_root / split / img_path.name
        dst_lbl = dst_root / split / f'{img_path.stem}.txt'

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text('\n'.join(valid_lines) + '\n')

        copied += 1

    print(
        f'{split}: copied={copied}, '
        f'skipped_missing={skipped_missing}, '
        f'skipped_empty={skipped_empty}, '
        f'skipped_invalid={skipped_invalid}'
    )

copy_split('train')
copy_split('valid')
copy_split('test')

# create obj names for labeling predictions
obj_names = Path('/content/darknet/data/obj.names')
obj_names.write_text('\n'.join(classes) + '\n')

# create train.txt and valid.txt so darknet can differentiate these sets
def write_image_list(split, out_path):
    image_dir = dst_root / split
    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    lines = [str(p).replace('/content/darknet/', '') for p in image_files]
    Path(out_path).write_text('\n'.join(lines) + '\n')

write_image_list('train', '/content/darknet/data/train.txt')
write_image_list('valid', '/content/darknet/data/valid.txt')

#  create obj data (gives darknet necessary information)
obj_data = Path('/content/darknet/data/obj.data')
obj_data.write_text(
f'''classes = {num_classes}
train = data/train.txt
valid = data/valid.txt
names = data/obj.names
backup = backup/
''')

print(obj_data.read_text())

# create custom cfg for the aquarium dataset from provided cfg template
cfg_src = Path('/content/darknet/cfg/yolov4-custom.cfg')
cfg_dst = Path('/content/darknet/cfg/yolov4-aquarium.cfg')

cfg_text = cfg_src.read_text()

# set training paramenters
max_batches = 3000  # training iterations
step1 = 2400
step2 = 2700
filters = (num_classes+5)*3

# change defaults to safer options for Colab T4
cfg_text = cfg_text.replace('batch=64', 'batch=64', 1)
cfg_text = cfg_text.replace('subdivisions=16', 'subdivisions=32', 1)
cfg_text = cfg_text.replace('width=608', 'width=416')
cfg_text = cfg_text.replace('height=608', 'height=416')
cfg_text = cfg_text.replace('max_batches = 500500', f'max_batches = {max_batches}')
cfg_text = cfg_text.replace('steps=400000,450000', f'steps={step1},{step2}')
cfg_text = cfg_text.replace('classes=80', f'classes={num_classes}')

# replace first 3 occurrences of filters=255
# for YOLOv4 to output the correct number of predictions
# since we only have 7 classes instead of 80
count = 0
while 'filters=255' in cfg_text and count < 3:
    cfg_text = cfg_text.replace('filters=255', f'filters={filters}', 1)
    count += 1

# saving custom cfg
cfg_dst.write_text(cfg_text)

# below is code that takes the weights and calculates metrics

# installs dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown', 'torchmetrics', 'pycocotools', 'pillow'], check=False)

# settings
FILE_ID = '1c2Qk91GtVNHjkUGj6QiF57zMWlOdpVAr'   
MAX_TEST_IMAGES = None  # None for full test set
SCORE_THRESH = 0.5
IOU_THRESH = 0.5

# paths
DARKNET = Path("/content/darknet")
TEST_DIR = DARKNET / "data" / "aquarium" / "test"
OBJ_DATA = DARKNET / "data" / "obj.data"
CFG = DARKNET / "cfg" / "yolov4-aquarium.cfg"
NAMES = DARKNET / "data" / "obj.names"
BACKUP_DIR = DARKNET / "backup"
WEIGHTS = BACKUP_DIR / "yolov4-aquarium_best.weights"

# check required paths
required_paths = [DARKNET, TEST_DIR, OBJ_DATA, CFG, NAMES]
for p in required_paths:
    if not p.exists():
        raise FileNotFoundError(f'Missing required path: {p}\n')

# download weights from Google Drive
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

if not WEIGHTS.exists():
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    print(f'Downloading weights from Google Drive to: {WEIGHTS}')
    gdown.download(url, str(WEIGHTS), quiet=False)
else:
    print(f'Weights already exist: {WEIGHTS}')

# load class names and test images
class_names = [x.strip() for x in NAMES.read_text().splitlines() if x.strip()]
class_to_idx = {name: i for i, name in enumerate(class_names)}

image_paths = sorted([
    *TEST_DIR.glob('*.jpg'),
    *TEST_DIR.glob('*.jpeg'),
    *TEST_DIR.glob('*.png'),
])

if MAX_TEST_IMAGES is not None:
    image_paths = image_paths[:MAX_TEST_IMAGES]

if len(image_paths) == 0:
    raise ValueError(f'No test images found in {TEST_DIR}')

print(f'Using weights: {WEIGHTS}')
print(f'Found {len(image_paths)} test images')
print(f'Classes: {class_names}')

# helpers
det_pattern = re.compile(
    r'^\s*(.+?):\s*([0-9]+)%\s*\(left_x:\s*(-?\d+)\s+top_y:\s*(-?\d+)\s+width:\s*(\d+)\s+height:\s*(\d+)\)',
    re.MULTILINE)

def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [x1, y1, x2, y2]

def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def load_gt(label_path: Path, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        c, xc, yc, w, h = line.split()
        c = int(c)
        xc, yc, w, h = map(float, (xc, yc, w, h))
        boxes.append(yolo_to_xyxy(xc, yc, w, h, img_w, img_h))
        labels.append(c)

    return boxes, labels

def to_py(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    return x

def count_yolo_params(cfg_path: Path):
    try:
        lines = [ln.strip() for ln in cfg_path.read_text().splitlines()]
    except Exception:
        return None

    sections = []
    current = None
    for line in lines:
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            if current is not None:
                sections.append(current)
            current = {'type': line[1:-1], 'kv': {}}
        elif current is not None and '=' in line:
            k, v = [x.strip() for x in line.split('=', 1)]
            current['kv'][k] = v
    if current is not None:
        sections.append(current)

    outputs = []
    params = 0
    c = 3

    for sec in sections:
        t = sec['type']
        kv = sec['kv']
        if t == 'net':
            if 'channels' in kv:
                c = int(kv['channels'])
            outputs.append(c)
        elif t == 'convolutional':
            filters = int(kv['filters'])
            size = int(kv.get('size', '1'))
            groups = int(kv.get('groups', '1'))
            batch_normalize = int(kv.get('batch_normalize', '0'))
            cin = c
            params += filters * (cin // groups) * size * size
            if batch_normalize:
                params += 4 * filters
            else:
                params += filters
            c = filters
            outputs.append(c)
        elif t in {'maxpool', 'avgpool', 'upsample', 'dropout', 'sam', 'scale_channels', 'gaussian_yolo', 'yolo'}:
            outputs.append(c)
        elif t == 'shortcut':
            outputs.append(c)
        elif t == 'route':
            layers = [int(x.strip()) for x in kv['layers'].split(',')]
            total = 0
            for li in layers:
                idx = li if li >= 0 else len(outputs) + li
                total += outputs[idx]
            c = total
            outputs.append(c)
        elif t == 'connected':
            out = int(kv['output'])
            params += c * out + out
            c = out
            outputs.append(c)
        else:
            outputs.append(c)

    return params

def run_darknet_on_image(img_path: Path):
    cmd = [
        './darknet', 'detector', 'test',
        str(OBJ_DATA), str(CFG), str(WEIGHTS), str(img_path),
        '-dont_show', '-ext_output', '-thresh', str(SCORE_THRESH)
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=DARKNET, capture_output=True, text=True)
    dt = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(
            f'Darknet failed on {img_path}\n'
            f'STDOUT:\n{res.stdout}\n'
            f'STDERR:\n{res.stderr}')

    return res.stdout + '\n' + res.stderr, dt

# evaluation loop
preds_tm = []
targets_tm = []
tp = 0
fp = 0
fn = 0
latencies = []

for idx, img_path in enumerate(image_paths, 1):
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    gt_boxes, gt_labels = load_gt(img_path.with_suffix(".txt"), img_w, img_h)

    output, dt = run_darknet_on_image(img_path)
    latencies.append(dt)

    pred_boxes = []
    pred_scores = []
    pred_labels = []

    for m in det_pattern.finditer(output):
        cls_name = m.group(1).strip()
        score = int(m.group(2)) / 100.0
        x = float(m.group(3))
        y = float(m.group(4))
        w = float(m.group(5))
        h = float(m.group(6))

        if cls_name not in class_to_idx:
            continue
        if score < SCORE_THRESH:
            continue

        pred_boxes.append([x, y, x + w, y + h])
        pred_scores.append(score)
        pred_labels.append(class_to_idx[cls_name])

    preds_tm.append({
        'boxes': torch.tensor(pred_boxes, dtype=torch.float32) if pred_boxes else torch.zeros((0, 4), dtype=torch.float32),
        'scores': torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.zeros((0,), dtype=torch.float32),
        'labels': torch.tensor(pred_labels, dtype=torch.int64) if pred_labels else torch.zeros((0,), dtype=torch.int64),
    })

    targets_tm.append({
        'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32),
        'labels': torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros((0,), dtype=torch.int64),
    })

    used_gt = set()
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)

    for pi in order:
        p_box = pred_boxes[pi]
        p_cls = pred_labels[pi]
        best_iou = 0.0
        best_gi = None
        for gi, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_labels)):
            if gi in used_gt or g_cls != p_cls:
                continue
            iou = box_iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= IOU_THRESH:
            tp += 1
            used_gt.add(best_gi)
        else:
            fp += 1
    fn += (len(gt_boxes) - len(used_gt))
    if idx % 10 == 0 or idx == len(image_paths):
        print(f"Processed {idx}/{len(image_paths)}")

# compute metrics
metric = MeanAveragePrecision(class_metrics=True)
metric.update(preds_tm, targets_tm)
tm = metric.compute()

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
fps = (1.0 / mean_latency) if mean_latency > 0 else 0.0
trainable_parameters = count_yolo_params(CFG)

results = {
    'n_images_evaluated': len(image_paths),
    'score_threshold': SCORE_THRESH,
    'iou_threshold': IOU_THRESH,
    'fps_images_per_sec': fps,
    'latency_ms_per_image_mean': mean_latency * 1000.0,
    'trainable_parameters': trainable_parameters,
    'micro_precision_iou50': precision,
    'micro_recall_iou50': recall,
    'micro_f1_iou50': f1,
    'torchmetrics': {k: to_py(v) for k, v in tm.items()},
    'map': to_py(tm['map']),
    'map_50': to_py(tm['map_50']),
    'map_75': to_py(tm['map_75']),
    'map_large': to_py(tm['map_large']),
    'map_medium': to_py(tm['map_medium']),
    'map_small': to_py(tm['map_small']),
    'mar_1': to_py(tm['mar_1']),
    'mar_10': to_py(tm['mar_10']),
    'mar_100': to_py(tm['mar_100']),
    'model': 'yolov4_darknet',
    'split': 'test',
    'weights_used': str(WEIGHTS),
    'class_names_0indexed': class_names,
}

out_path = DARKNET / 'test_metrics_yolov4.json'
out_path.write_text(json.dumps(results, indent=2))

print(json.dumps(results, indent=2))
print(f'\nSaved to: {out_path}')
