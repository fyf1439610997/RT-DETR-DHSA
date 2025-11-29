# Object Detection Model for "Utilizing Multimodal Learning Analytics with LLMs to Improve Instructional Design in Real-time Classroom Teaching"

This archive contains the implementation details, configuration files, and training logs for the object detection model (DHSA-enhanced RT-DETR) described in the manuscript.

## Overview

The object detection module is designed to recognize student behaviors in real-time classroom scenarios. It integrates the **Dual-Head Self-Attention (DHSA)** mechanism into the RT-DETR architecture to enhance feature extraction capabilities. 

This repository includes the source code for the modified architecture and the training records to validate the experimental results presented in the paper.

## 🔎 Key Implementation Details (DHSA)

The core technical contribution, the DHSA mechanism, is adapted from the **Histoformer** project and integrated into the RT-DETR backbone.

* **Original DHSA Source Code:** [https://github.com/sunshangquan/Histoformer](https://github.com/sunshangquan/Histoformer)
* **Implementation in This Project:**
    The specific class `TransformerEncoderLayer_DHSA` can be found in the following file:
    > `ultralytics/nn/extra_modules/transformer.py`

## Directory Structure

Please refer to the following directories to examine the method details and training evidence:

### 1. Model Architecture & Configuration
* **DHSA Module Implementation:** `ultralytics/nn/extra_modules/transformer.py`
    * *Check the `TransformerEncoderLayer_DHSA` class to see how the attention mechanism is implemented.*
* **Model Config:** `ultralytics/cfg/models/rt-detr`
    * Contains the `.yaml` configuration files defining the network structure and how the DHSA module is inserted.

### 2. Dataset Configuration
* **Path:** `dataset`
* **Description:** Contains configuration files defining behavior categories and data paths.
    * *Note: In compliance with ethical standards and privacy protection for minor students, the raw video frames and annotation files are not included in this public release.*

### 3. Training Results
* **Path:** `run/train`
* **Description:** Contains logs and results from the model training process.
    * **Metrics:** Visualization of loss curves, precision, recall, and mAP scores.
    * **Logs:** Training execution logs demonstrating model convergence.

---
**Note to Reviewers:**
This code is built upon the Ultralytics framework. We provide the full system code (frontend & backend) to demonstrate the system's integrity. While the raw training data is private, the included model definition files allow for a thorough examination of the proposed technical method.
