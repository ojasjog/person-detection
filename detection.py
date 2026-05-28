import cv2
import time
import pandas as pd
from ultralytics import YOLO, RTDETR

# 1. Configuration & Custom Path Coordinates
VIDEO_PATH = r"TownCentreXVID.mp4"
MAX_FRAMES = 500  # Evaluates the first 500 frames across all models for parity

# Exact points clicked on the road
LINE_P1 = (1798, 747)  
LINE_P2 = (169, 456)   

# Normalize endpoints so the horizontal slope math stays completely stable (X1 < X2)
if LINE_P1[0] > LINE_P2[0]:
    START_POINT = LINE_P2
    END_POINT = LINE_P1
else:
    START_POINT = LINE_P1
    END_POINT = LINE_P2

# Define the models sitting in different architectural weight classes
models_to_test = {
    "YOLOv8-Nano": "yolov8n.pt",
    "YOLO11-Nano": "yolo11n.pt",
    "RT-DETR-Large": "rtdetr-l.pt"  
}

benchmark_results = {}

# 2. Main Evaluation Pipeline
for model_name, model_weights in models_to_test.items():
    print(f"\n--- Starting Evaluation: {model_name} ---")
    
    # Initialize based on specific framework wrapper class
    if "RT-DETR" in model_name:
        model = RTDETR(model_weights)
    else:
        model = YOLO(model_weights)
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file at '{VIDEO_PATH}'.")
        continue

    frame_count = 0
    total_inference_time = 0
    pedestrian_history = {}
    logged_crossings = set()
    
    start_wall_time = time.time()
    
    while cap.isOpened() and frame_count < MAX_FRAMES:
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Isolate and clock the model's pure internal inference latency
        t_start = time.perf_counter()
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        t_end = time.perf_counter()
        total_inference_time += (t_end - t_start)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                # Calculate current horizontal center and bottom (feet anchoring)
                cx = int((box[0] + box[2]) / 2)
                cy = int(box[3])  
                
                prev_cy = pedestrian_history.get(track_id, None)
                pedestrian_history[track_id] = cy
                
                # Check line logic if we have frame history and haven't logged this person yet
                if prev_cy is not None and track_id not in logged_crossings:
                    # Is the person horizontally positioned within our line boundaries?
                    if START_POINT[0] <= cx <= END_POINT[0]:
                        # Linear Interpolation: Compute exact Y of the line at this specific X position
                        dx = END_POINT[0] - START_POINT[0]
                        dy = END_POINT[1] - START_POINT[1]
                        expected_line_y = START_POINT[1] + ((cx - START_POINT[0]) * dy / dx)
                        
                        # Did their vertical vector cross the calculated threshold on this frame?
                        if (prev_cy <= expected_line_y < cy) or (cy <= expected_line_y < prev_cy):
                            logged_crossings.add(track_id)
                            video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            timestamp = time.strftime("%H:%M:%S", time.gmtime(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))

                            if cy > prev_cy:
                                print(f"[{timestamp}] 📥 LINE CROSS: Pedestrian #{track_id} walked DOWN screen / Exited.")
                            else:
                                print(f"[{timestamp}] 📤 LINE CROSS: Pedestrian #{track_id} walked UP screen / Entered.")
                            
                        

                # --- ADDED: DRAW INDIVIDUAL VISUAL HITBOXES & ID LABELS ---
                # Choose color variant depending on whether they have already crossed the line
                box_color = (255, 0, 0) if track_id in logged_crossings else (0, 255, 0)
                
                # Draw main pedestrian hitbox bounding box
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
                
                # Draw the tracking ID number string right above their head
                cv2.putText(frame, f"ID: {track_id}", (int(box[0]), int(box[1]) - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Draw a tracking tracking dot at their feet anchoring point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            
        # --- SURVEILLANCE OVERLAYS ---
        # Render the custom line on the active tracking feed window
        cv2.line(frame, LINE_P1, LINE_P2, (255, 0, 255), 3)
        cv2.putText(frame, f"Evaluating: {model_name}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        resized_view = cv2.resize(frame, (1280, 720))
        cv2.imshow("Multi-Model Custom Benchmarking", resized_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Evaluation cancelled by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    end_wall_time = time.time()
    
    # 3. Process Empirical Performance Metrics
    if frame_count > 0:
        avg_inference_ms = (total_inference_time / frame_count) * 1000
        fps = frame_count / (end_wall_time - start_wall_time)
    else:
        avg_inference_ms, fps = 0, 0
    
    benchmark_results[model_name] = {
        "Avg Latency (ms)": round(avg_inference_ms, 2),
        "Processing Speed (FPS)": round(fps, 2),
        "Total People Tracked Across Line": len(logged_crossings)
    }

# 4. Print Final Assignment Table Out to Terminal
df = pd.DataFrame(benchmark_results).T
print("\n" + "="*60)
print("             FINAL CUSTOM LINE SURVEILLANCE REPORT          ")
print("="*60)
print(df.to_string())
print("="*60)