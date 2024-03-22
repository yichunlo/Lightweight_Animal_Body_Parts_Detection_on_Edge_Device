# Lightweight ML Model for Animal Tracking Edge AI: Hierarchical Bounding Boxes

CS211 Group Project

Yi-chun Lo, Taichang Zhao, Gian Zignago

University of California, Los Angeles

## Project Description

### Context
In the dynamic world of IoT, we're bridging the gap between AI and edge computing, transforming technology interaction. Despite edge devices often have limitations in power and computation capabilities, many interdisciplinary research still call for the need of edge AI. One workable solution is to deploy lightweight machine learning models on the edge, and perform heavy duty machine learning tasks in a centralized server.
In our context, we focus on the interdisciplinary research of AI-driven ethology research. Non-trivial effort is required to deploy efficient, lightweight ML models on the edge to perform object detection and tracking tasks and deliver accurate, reliable results.

### Problem Statement
In this project, we aim to train a lightweight model for object detection and tracking, specifically for animal and animal bodypart tracking. For this project, the animals of interest are in the Panthera lineage of the Felidae family, namely lions (Panthera leo) and tigers (Panthera tigris). Given a video, the model should detect and track the animal of interest. More importantly, it is useful to further track the animal’s separate body parts (e.g. torso, head, limbs, and tails) for behavior analysis purposes.

### Goal
The goal is to train and evaluate a lightweight model to detect and track the animal (lion/tiger) and its separate body parts.

### Tasks
- Track the entire animal (lion/tiger) with a single bounding box, train and evaluate this model accuracy (High Priority)
- Given the above model for the entire animal, track the different body parts of the animal (say, five parts of head, tail, front legs, rear legs, and body trunk) using more bounding boxes, train and evaluate the accuracy of these detailed models (High Priority)
- Use spatial information among the different body parts to improve accuracy of the model (e.g. Inverse Kinematics) (Medium Priority)
- Bonus: Explore background object detection and exclusion (low priority).

### Expected outcome
Two-level bounding box models to track the entire animal and various animal body parts; evaluation metric and workflow.

## Detail of the Project
### Dataset
The first part of dataset is in Pascal VOC format. The structure is like this:
```
|---- 211_datasets
|     |---- test
|     |     |---- images
|     |     |     |---- 0001.jpg
|     |     |     |---- ...
|     |---- train
|     |     |---- images
|     |     |     |---- 0002.jpg
|     |     |     |---- ...
|     |     |---- Annotations
|     |     |     |----
|     |     |     |----
|     |---- val
|     |     |---- images
|     |     |     |---- 0000.jpg
|     |     |     |---- ...
|     |     |---- Annotations
|     |     |     |----
|     |     |     |----
```
We use the dataloader provided by mediapipe to set up the training and validation datasets.
There are total 2485 annotated training images and 277 validation images. For this model, we will only detect the whole tiger (not the body parts).

The second part of dataset is in COCO format. The structure is like this:
```
|---- 211_datasets
|     |---- test
|     |     |---- images
|     |     |     |---- 000000.jpg
|     |     |     |---- ...
|     |     |---- labels.json
|     |---- train
|     |     |---- images
|     |     |     |---- 000002.jpg
|     |     |     |---- ...
|     |     |---- labels.json
|     |---- val
|     |     |---- images
|     |     |     |---- 000004.jpg
|     |     |     |---- ...
|     |     |---- labels.json
```

There are total 2187 training images and 342 validation images, with 4 categories: head, torso, leg, tail.

## Result

<img width="367" alt="image" src="https://github.com/yichunlo/Lightweight_Animal_Body_Parts_Detection_on_Edge_Device/assets/45264573/cb660c0c-2a7d-4b98-b121-01a4d4ffaa00">
Our result is not satisfying due to several reasons:
1. **Not Enough Data**: We only have around 2000 images for training, which is too small for precise body parts detection.
2. **Bias In Data**: The dataset of the second part contains some bias, as we generate the dataset by transform the keypoints into bounding box. Some images would be skipped by us.
3. **Low Quality of the Images**: Data quality is a problem. The dataset contains some images which is too dark or too vague, which makes the detection more difficult.


## Group Members
- Yi-Chun Lo
  - Email: yichunlo0919@g.ucla.edu
  - Github: https://github.com/yichunlo
- Tai-Chang Chou
  - Email: tzhouam@gmail.com
  - Github: https://github.com/tzhouam
- Gian:
  - Email: grz@cs.ucla.edu
  - Github: https://github.com/zignago
