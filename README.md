# Using Object Detection to Classify Marine Animals
- DS 4002 Group 5, Project 3
- Group Leader: Jonah Lee
- Group Members: Jonathan Sutkus, Siwen Liao, Jonah Lee
- Apr 19, 2026

## Repository Contents
The goal of this project is to deploy three separate object detection models (YOLOv4, Faster R-CNN, and SSD) with the aim of determining which model can most accurately classify different marine animals through underwater images collected from the Henry Doorly Zoo in Omaha and the National Aquarium in Baltimore. The DS-4002_Group5_Project3 repository contains the DATA folder (includes the original image data, the preprocessed dataset, and our data appendix), the SCRIPTS folder (includes code for preprocessing/cleaning the dataset and the scripts for each of the three different object detection models), the OUTPUTS folder (), and the LICENSE.md and README.md files.

## Section 1: Software and Platform

### Software/Platform
This project was developed and run using: 
- Google Colab/Jupyter Notebook on a Mac

### Packages
The following Python packages are required:
- annotations from __future__
- argparse
- hashlib
- random
- xml.etree.ElementTree
- dataclass from dataclasses
- Path from pathlib
- Any, Iterable, List, Sequence, Tuple from typing
- torch
- Image from PIL
- DataLoader, Dataset from torch.utils.data
- fasterrcnn_resnet50_fpn from torchvision.models.detection
- FastRCNNPredictor from torchvision.models.detection.faster_rcnn
- box_iou from torchvision.ops
- functional from torchvision.transforms
- ssd300_vgg16 from torchvision.models.detection
- os
- shutil
- json
- re
- subprocess
- sys
- time
- gdown
- MeanAveragePrecision from torchmetrics.detection.mean_ap

## Section 2: Documentation Map
```text
DS-4002_Group5_Project3/
│
├── DATA/
│ ├── preprocessed_aquarium
| | ├── test
| | ├── train
| | ├── valid
│ ├── processed_aquarium
| | ├── test
| | ├── train
| | ├── valid
│ ├── P3_Data_Appendix.pdf
│
├── OUTPUT/
│ ├── metrics_faster_rcnn_test.json
│ ├── metrics_ssd_test.json
│ ├── metrics_yolov4.json
│ ├── train_faster_rcnn_log.txt
│ ├── train_ssd_resume_eval.log
│ ├── train_ssd_run.err
│ ├── train_ssd_run.log
│
├── SCRIPTS/
│ ├── preprocess.py
│ ├── train_faster_rcnn.py
│ ├── train_ssd.py
│ ├── train_yolov4.py
│ ├── yolov4_metrics.py
│
├── .gitignore
├── DS-4002_Group_Project3.code-workspace
├── LICENSE
├── README.md
└──requirements-detection.txt
```

### Folder Descriptions
- **DATA**:
  - Contains our original dataset with the 638 images split into train, valid, and test sets.
  - Cleaned dataset after preprocessing.
  - Data appendix taking you through the dataset splits for how we will train, improve, and test our model. 
- **SCRIPTS**: Contains python scripts for data preprocessing/cleaning, scripts running our object detection models, and scripts for model performance plots.
- **OUTPUT**: Contains JSON, txt, log, and err files for the evaluation metrics of each model and any additional logs when running recorded when running the models. 
- **LICENSE**: MIT license was selected based on recommendation from the DS 4002 Ml3 Rubric.
- **README.md**: Instructions, documentation, and respository overview.

## Section 3: Instructions for Reproduction

- **Step 1**: Clone the repository. Cloning creates a complete local copy of the repository, including all files and branches. Make sure that you can see the DATA, OUTPUT, and SCRIPTS folders. Confirm that the preprocessed and processed folders exist in the DATA folder and contain the train, valid, and test images.
- **Step 2**: Run the preprocess.py script in the SCRIPTS folder. This will process the Roboflow Aquarium Combined export (YOLOv4 PyTorch layout with `_annotations.txt`) for training YOLOv4, Faster R-CNN, and SSD. You can also directly access the preprocessed data from this step in the processed_aquarium folder in the DATA folder. 
- **Step 3**: Run the train_faster_rcnn.py script in the SCRIPTS folder. Confirm that the resulting metrics correspond with those in the metrics_faster_rcnn_test.json file in the OUTPUT folder. 
- **Step 4**: Run the train_ssd.py script in the SCRIPTS folder. Confirm that the resulting metrics correspond with those in the metrics_ssd_test.json file in the OUTPUT folder. 
- **Step 5**: Run the train_yolov4.py script in the SCRIPTS folder with the T4 GPU Hardware Accelerator in Colab. Then, run the yolov4_metrics.py script with the CPU Hardware Accelerator in Colab. Confirm that the resulting metrics correspond with those in the metrics_yolov4.json file in the OUTPUT folder. 
  - Some notes for Step 5:
    - The train_yolov4.py script may require more resources than the free version of Colab offers. The        yolov4_metrics.py script is ran on CPU to limit the amount of GPU resources used. 

## References: 

[1] Fritz. (n.d.). Object detection guide – Everything you need to know. Object detection guide – Everything you need to know 

[2] K. Sarma et al., “A comparative study on Faster R-CNN, YOLO and SSD,” AIP Conference Proceedings, vol. 2971, 060044, 2024. Available: https://pubs.aip.org/aip/acp/article/2971/1/060044/3296342/A-comparative-study-on-faster-R-CNN-YOLO-and-SSD

[3] P. Tsirtsakis, G. Zacharis, G. S. Maraslidis, and G. F. Fragulis, “Deep learning for object recognition: A comprehensive review of models and algorithms,” International Journal of Cognitive Computing in Engineering, vol. 6, pp. 298–312, 2025.

[4] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017. 

[5] W. Liu et al., “SSD: Single Shot MultiBox Detector,” arXiv preprint arXiv:1905.016014, 2019. 
