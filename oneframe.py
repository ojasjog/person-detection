import cv2
from ultralytics import YOLO, RTDETR


VIDEO_PATH = r"TownCentreXVID.mp4"


cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
cap.release()

if success:
    model_v8 = YOLO("yolov8n.pt")
    results_v8 = model_v8(frame, classes=[0], verbose=False)
    results_v8[0].save("yolov8_detection.jpg")
    print("Saved: yolov8_detection.jpg")

    model_v11 = YOLO("yolo11n.pt")
    results_v11 = model_v11(frame, classes=[0], verbose=False)
    results_v11[0].save("yolo11_detection.jpg")
    print("Saved: yolo11_detection.jpg")

    model_detr = RTDETR("rtdetr-l.pt")
    results_detr = model_detr(frame, classes=[0], verbose=False)
    results_detr[0].save("rtdetr_detection.jpg")
    print("Saved: rtdetr_detection.jpg")
else:
    print("Could not read the first frame.")