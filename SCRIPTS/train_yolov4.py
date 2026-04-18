
# import packages that help change folders and work with files
import os
from pathlib import Path
import shutil

# uncomment below if you need to rerun the code
# resets Colab folders to prevent unecessary copies and loops
#os.chdir('/content')
#!rm -rf /content/darknet
#!rm -rf /content/DS-4002_Group_Project3

# clone needed repositories (YOLOv4 implementation and the project repo)
!git clone https://github.com/AlexeyAB/darknet /content/darknet
!git clone https://github.com/jonahsjlee/DS-4002_Group_Project3.git /content/DS-4002_Group_Project3

# install OpenCV for Darknet to work
!apt-get update
!apt-get install -y libopencv-dev

# turn on GPU/CUDNN/OpenCV and edit settings to make things run faster
makefile = Path('/content/darknet/Makefile')
text = makefile.read_text()
# use GPU
text = text.replace('GPU=0', 'GPU=1')
# make process faster
text = text.replace('CUDNN=0', 'CUDNN=1')
# half-precision for increasing speed and using less memory
text = text.replace('CUDNN_HALF=0', 'CUDNN_HALF=1')
# compiling with OpenCV
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

# download pretrained backbone
%cd /content/darknet
!wget -q https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.conv.137 -O /content/darknet/yolov4.conv.137
!ls -lh /content/darknet/yolov4.conv.137

# start training
!./darknet detector train data/obj.data cfg/yolov4-aquarium.cfg yolov4.conv.137 -dont_show -map
