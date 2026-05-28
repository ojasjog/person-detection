import cv2
import time
from ultralytics import YOLO

# 1. Initialize the optimized YOLO11 model engine
model = YOLO("yolo11n.pt")

VIDEO_PATH = r"TownCentreXVID.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================================================
# YOUR SPECIFIC CLICKED ROAD COORDINATES
# =========================================================
LINE_P1 = (1798, 747)  
LINE_P2 = (169, 456)   
# =========================================================

# Ensure X1 is always the smaller number to keep our slope math stable
if LINE_P1[0] > LINE_P2[0]:
    START_POINT = LINE_P2
    END_POINT = LINE_P1
else:
    START_POINT = LINE_P1
    END_POINT = LINE_P2

# State track history storage structures
pedestrian_history = {}  # {track_id: previous_cy}
logged_crossings = set()

print("\n🚀 Tracking pipeline started using custom street boundaries.")
print("Press 'q' inside the video window to stop execution.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    
    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            # Track using center-bottom point of the bounding box (pedestrian's feet)
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])  

            # Grab their position from the previous frame
            prev_cy = pedestrian_history.get(track_id, None)
            pedestrian_history[track_id] = cy  # Store current frame position for next time

            if prev_cy is not None and track_id not in logged_crossings:
                
                # Check if the pedestrian is horizontally matching our line bounds
                if START_POINT[0] <= cx <= END_POINT[0]:
                    
                    # Compute the exact expected Y coordinate on the diagonal line at this specific X position
                    # Linear Equation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    dx = END_POINT[0] - START_POINT[0]
                    dy = END_POINT[1] - START_POINT[1]
                    expected_line_y = START_POINT[1] + ((cx - START_POINT[0]) * dy / dx)
                    
                    # Check if their feet passed across the line segment between the last frame and this frame
                    if (prev_cy <= expected_line_y < cy) or (cy <= expected_line_y < prev_cy):
                        timestamp = time.strftime('%H:%M:%S')
                        
                        # Direction validation: Did their Y position increase or decrease?
                        if cy > prev_cy:
                            print(f"[{timestamp}] 📥 LINE CROSS: Pedestrian #{track_id} walked DOWN screen / Exited.")
                        else:
                            print(f"[{timestamp}] 📤 LINE CROSS: Pedestrian #{track_id} walked UP screen / Entered.")
                            
                        logged_crossings.add(track_id)

            # Draw visual tracking boxes around active people
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Render your precise imaginary line on the screen layout
    cv2.line(frame, LINE_P1, LINE_P2, (255, 0, 255), 4)
    cv2.putText(frame, "MY IMAGINARY LINE", (LINE_P2[0] + 20, LINE_P2[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Scale the display layout down dynamically so it fits perfectly on your screen
    resized_view = cv2.resize(frame, (1280, 720))
    cv2.imshow("Custom Road Tripwire Live Log", resized_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Run finished successfully.")