# Pedestrian Detection and Tracking Analytics

This project evaluates and compares different object detection models for real-time person tracking and line-crossing analytics on video streams. It uses a custom-defined diagonal tripwire threshold to log entries and exits.

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Dependencies

Install the required packages using pip:

```bash
pip install opencv-python ultralytics pandas kagglehub

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
