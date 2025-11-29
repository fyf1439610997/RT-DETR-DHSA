# Object Detection Model for "Utilizing Multimodal Learning Analytics with LLMs to Improve Instructional Design in Real-time Classroom Teaching"

This archive contains the implementation details, configuration files, and training logs for the object detection model (DHSA-enhanced RT-DETR) described in the manuscript.

## Overview

The object detection module is designed to recognize student behaviors in real-time classroom scenarios. This repository includes the modified model architecture and the training records to validate the experimental results presented in the paper.

## Directory Structure

Please refer to the following directories to examine the method details and training evidence:

### 1. Model Configuration
* **Path:** `ultralytics/cfg/models/rt-detr`
* **Description:** This folder contains the model configuration files (`.yaml`).
    * You can inspect the definition of the network architecture here.
    * This includes the specific modifications where the DHSA mechanism is integrated into the RT-DETR backbone.

### 2. Dataset Configuration
* **Path:** `dataset`
* **Description:** This folder contains the dataset configuration files used for training.
    * It defines the classes (behavior categories) and data paths.
    * *Note: In compliance with ethical standards and privacy protection for minor students, the raw video frames and annotation files are not included in this public release.*

### 3. Training Results
* **Path:** `run/train`
* **Description:** This folder contains the logs and results from the model training process.
    * **Metrics:** Visualization of loss curves, precision, recall, and mAP scores.
    * **Logs:** Training execution logs demonstrating the convergence of the model.

---
**Note to Reviewers:**
This code is based on the Ultralytics framework. The provided files focus on the specific architectural modifications and experimental validation relevant to the submitted manuscript.
