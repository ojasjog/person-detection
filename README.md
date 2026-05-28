Here is a clean, concise README for your project without any emojis.

---

# Pedestrian Detection and Tracking Analytics

This project evaluates and compares different object detection models for real-time person tracking and line-crossing analytics on video streams. It uses a custom-defined diagonal tripwire threshold to log entries and exits.

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Dependencies

Install the required packages using pip:

```bash
pip install opencv-python ultralytics pandas

```

*Note: The `ultralytics` library automatically handles the downloading of the pre-trained model weights (`yolov8n.pt`, `yolo11n.pt`, and `rtdetr-l.pt`) upon first execution.*

## Dataset Used

The project uses the **Oxford Town Centre Dataset**, a benchmark dataset widely used for multiple object tracking evaluation.

* **Source:** University of Oxford / Kaggle
* **Resolution:** 1920x1080 (Full HD) at 25 frames per second
* **Scene Description:** A high-resolution, static overhead view of a busy pedestrian street corner (Cornmarket Street, Oxford, UK).
* **Video File Name:** `TownCentreXVID.avi`

Ensure the downloaded video file path corresponds to the `VIDEO_PATH` variable defined at the top of your execution scripts.
