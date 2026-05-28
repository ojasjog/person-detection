# Pedestrian Detection and Tracking Analytics

This project evaluates and compares different object detection models for real-time person tracking and line-crossing analytics on video streams. It uses a custom-defined diagonal tripwire threshold to log entries and exits.

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Dependencies

Install the required packages using pip:

```bash
pip install opencv-python ultralytics pandas kagglehub matplotlib numpy

```

## Dataset Download Instructions

The Oxford Town Centre dataset is hosted on Kaggle. To download the dataset automatically via the terminal without using a web browser, utilize the kagglehub utility package.

Run this Python script to download the data to your local machine's unified cache:

```python
import kagglehub

# Download latest version of the Oxford Town Centre dataset
path = kagglehub.dataset_download("almightyj/oxford-town-centre")
print("Dataset downloaded to:", path)

```

## Moving the Data to Your Workspace

1. Navigate to the path printed by the script above.
2. Locate the folder named `versions/1` and find the file named `TownCentreXVID.avi`.
3. Copy or move `TownCentreXVID.avi` directly into the root folder of this cloned repository, or update the `VIDEO_PATH` variable in the script to match this cache folder location.

## Benchmark Report

The framework evaluates three models across distinct structural archetypes over a 500-frame sequence using an identical diagonal tripwire threshold ($P_1$: 1798, 747 to $P_2$: 169, 456).

### Performance Metrics Summary

| Evaluation Metric | YOLOv8-Nano | YOLO11-Nano | RT-DETR-Large |
| --- | --- | --- | --- |
| **Architecture Family** | Convolutional (CNN) | Convolutional + Attention | Vision Transformer (ViT) |
| **Model Size** | 6.2 MB | 5.4 MB | 136.0 MB |
| **Parameters** | 3.2 M | 2.6 M | 32.9 M |
| **Average Latency** | 58.41 ms | 57.51 ms | 487.35 ms |
| **Processing Speed** | 13.91 FPS | 14.0 FPS | 2.0 FPS |
| **mAP@0.5** | 52.3% | 54.1% | 67.2% |
| **mAP@0.5:0.95** | 37.1% | 39.5% | 53.4% |
| **Precision** | 0.81 | 0.83 | 0.91 |
| **Recall** | 0.74 | 0.76 | 0.87 |
| **Unique Line Crossings** | 17 | 18 | 18 |

### Key Analytical Findings

* **Edge Deployment Feasibility:** YOLO11-Nano demonstrates the highest operational efficiency, executing at 14.0 FPS on CPU with a minimized footprint of 2.6M parameters. It completely replaces the legacy YOLOv8-Nano baseline by improving processing speed and accuracy concurrently.
* **Transformer Core Bottleneck:** RT-DETR-Large delivers superior raw precision (0.91) and structural accuracy (67.2% mAP@0.5), yet experiences a critical performance penalty on standard sequential hardware, dropping to 2.0 FPS with a latency spike of 487.35 ms. This showcases the severe hardware constraints introduced by global self-attention mechanisms when run without dedicated GPU acceleration.
* **Downstream Tracking Saturation:** Despite the massive scale discrepancy between the 2.6M parameter YOLO11-Nano and the 32.9M parameter RT-DETR-Large, both architectures converged on almost identical line-crossing tracking counts (18 vs 17 unique crossings). This confirms that under clear spatial visibility conditions, localized architectural optimizations saturate tracking performance, rendering the heavy transformer pipeline computationally cost-prohibitive for generic surveillance scenarios.
