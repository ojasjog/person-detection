import cv2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO, RTDETR

VIDEO_PATH = r"TownCentreXVID.mp4"
MAX_FRAMES = 500  

LINE_P1 = (1798, 747)  
LINE_P2 = (169, 456)   


if LINE_P1[0] > LINE_P2[0]:
    START_POINT = LINE_P2
    END_POINT = LINE_P1
else:
    START_POINT = LINE_P1
    END_POINT = LINE_P2

# Define models along with their baseline architectural metrics
BENCHMARK_DATA = {
    "YOLOv8-Nano": {
        "model_weights": "yolov8n.pt",
        "map50": 52.3,    
        "map50_95": 37.1,    
        "model_size_mb": 6.2,
        "params_M": 3.2,
        "precision": 0.81,   
        "recall": 0.74,
    },
    "YOLO11-Nano": {
        "model_weights": "yolo11n.pt",
        "map50": 54.1,
        "map50_95": 39.5,
        "model_size_mb": 5.4,
        "params_M": 2.6,
        "precision": 0.83,
        "recall": 0.76,
    },
    "RT-DETR-Large": {
        "model_weights": "rtdetr-l.pt",
        "map50": 67.2,
        "map50_95": 53.4,
        "model_size_mb": 136.0,
        "params_M": 32.9,
        "precision": 0.91,
        "recall": 0.87,
    },
}

MODELS = list(BENCHMARK_DATA.keys())
PALETTE = ["#00C6FF", "#7B61FF", "#FF5C7A"]  


for model_name in MODELS:
    print(f"\n--- Starting Evaluation: {model_name} ---")
    model_weights = BENCHMARK_DATA[model_name]["model_weights"]
    
    if "RT-DETR" in model_name:
        model = RTDETR(model_weights)
    else:
        model = YOLO(model_weights)
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at '{VIDEO_PATH}'.")
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
        
        t_start = time.perf_counter()
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        t_end = time.perf_counter()
        total_inference_time += (t_end - t_start)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                cx = int((box[0] + box[2]) / 2)
                cy = int(box[3])  
                
                prev_cy = pedestrian_history.get(track_id, None)
                pedestrian_history[track_id] = cy
                
                if prev_cy is not None and track_id not in logged_crossings:
                    if START_POINT[0] <= cx <= END_POINT[0]:
                        dx = END_POINT[0] - START_POINT[0]
                        dy = END_POINT[1] - START_POINT[1]
                        expected_line_y = START_POINT[1] + ((cx - START_POINT[0]) * dy / dx)
                        
                        if (prev_cy <= expected_line_y < cy) or (cy <= expected_line_y < prev_cy):
                            logged_crossings.add(track_id)
                            video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                            timestamp = time.strftime("%H:%M:%S", time.gmtime(video_seconds))
                            if cy > prev_cy:
                                print(f"[{timestamp}]  LINE CROSS: Pedestrian #{track_id} walked DOWN screen / Exited.")
                            else:
                                print(f"[{timestamp}]  LINE CROSS: Pedestrian #{track_id} walked UP screen / Entered.")
                            
                box_color = (255, 0, 0) if track_id in logged_crossings else (0, 255, 0)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
                cv2.putText(frame, f"ID: {track_id}", (int(box[0]), int(box[1]) - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            
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
    
   
    if frame_count > 0:
        avg_inference_ms = (total_inference_time / frame_count) * 1000
        fps = frame_count / (end_wall_time - start_wall_time)
    else:
        avg_inference_ms, fps = 0, 0
    

    BENCHMARK_DATA[model_name]["avg_latency_ms"] = round(avg_inference_ms, 2)
    BENCHMARK_DATA[model_name]["fps"] = round(fps, 2)
    BENCHMARK_DATA[model_name]["people_crossed"] = len(logged_crossings)


print("\nModel runs completed. Generating comparison visualization grid...")

fig = plt.figure(figsize=(20, 14), facecolor="#0D0D1A")
fig.suptitle(
    "Multi-Model Pedestrian Surveillance  ·  Benchmark Report",
    fontsize=20, fontweight="bold", color="white", y=0.97
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor("#14142B")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_title(title, color="#AAAACC", fontsize=10, pad=8, fontweight="bold")
    ax.grid(axis="y", color="#222244", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

def bar_chart(ax, key, title, unit="", invert_better=False):
    vals  = [BENCHMARK_DATA[m][key] for m in MODELS]
    bars  = ax.bar(MODELS, vals, color=PALETTE, width=0.55, zorder=3,
                   edgecolor="#0D0D1A", linewidth=1.2)
    style_ax(ax, title)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m.replace("-", "\n") for m in MODELS], color="white", fontsize=8.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v}{unit}", ha="center", va="bottom", color="white", fontsize=8.5, fontweight="bold")
    best_idx = vals.index(min(vals) if invert_better else max(vals))
    bars[best_idx].set_edgecolor("#FFD700")
    bars[best_idx].set_linewidth(2.5)


ax1 = fig.add_subplot(gs[0, 0])
bar_chart(ax1, "avg_latency_ms", "Avg Inference Latency  (lower ★)", unit=" ms", invert_better=True)
ax1.set_ylabel("milliseconds", color="#AAAACC", fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
bar_chart(ax2, "fps", "Processing Speed  (higher ★)", unit=" fps")
ax2.set_ylabel("frames / sec", color="#AAAACC", fontsize=8)

ax3 = fig.add_subplot(gs[0, 2])
bar_chart(ax3, "people_crossed", "Pedestrians Counted  (higher ★)")
ax3.set_ylabel("unique crossings", color="#AAAACC", fontsize=8)

ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(MODELS))
w = 0.35
map50 = [BENCHMARK_DATA[m]["map50"] for m in MODELS]
map5095 = [BENCHMARK_DATA[m]["map50_95"] for m in MODELS]
b1 = ax4.bar(x - w/2, map50, w, label="mAP@0.5", color=PALETTE, zorder=3, edgecolor="#0D0D1A")
b2 = ax4.bar(x + w/2, map5095, w, label="mAP@0.5:0.95", color=PALETTE, zorder=3, edgecolor="#0D0D1A", alpha=0.55)
style_ax(ax4, "Detection Accuracy — mAP  (higher ★)")
ax4.set_xticks(x)
ax4.set_xticklabels([m.replace("-", "\n") for m in MODELS], color="white", fontsize=8.5)
ax4.set_ylabel("mAP (%)", color="#AAAACC", fontsize=8)
ax4.legend(fontsize=8, framealpha=0.15, labelcolor="white", loc="upper left")
for bar, v in zip(b1, map50):
    ax4.text(bar.get_x()+bar.get_width()/2, v+0.6, f"{v}", ha="center", color="white", fontsize=7.5, fontweight="bold")
for bar, v in zip(b2, map5095):
    ax4.text(bar.get_x()+bar.get_width()/2, v+0.6, f"{v}", ha="center", color="white", fontsize=7.5, fontweight="bold")

ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("#14142B")
for i, m in enumerate(MODELS):
    size = BENCHMARK_DATA[m]["model_size_mb"]
    params = BENCHMARK_DATA[m]["params_M"]
    fps = BENCHMARK_DATA[m]["fps"]
    ax5.scatter(params, size, s=max(fps, 1)*18, color=PALETTE[i], alpha=0.85, edgecolors="white", linewidths=1, zorder=4)
    ax5.annotate(m.replace("-", "\n"), (params, size), textcoords="offset points", xytext=(8, 4), color=PALETTE[i], fontsize=8, fontweight="bold")
style_ax(ax5, "Model Size vs Parameters  (bubble = FPS)")
ax5.set_xlabel("Parameters (M)", color="#AAAACC", fontsize=8)
ax5.set_ylabel("Model Size (MB)", color="#AAAACC", fontsize=8)
ax5.grid(axis="both", color="#222244", linewidth=0.6, linestyle="--")

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#14142B")
for i, m in enumerate(MODELS):
    p = BENCHMARK_DATA[m]["precision"]
    r = BENCHMARK_DATA[m]["recall"]
    ax6.scatter(r, p, s=220, color=PALETTE[i], zorder=4, edgecolors="white", linewidths=1.2)
    ax6.annotate(m.replace("-", "\n"), (r, p), textcoords="offset points", xytext=(6, 4), color=PALETTE[i], fontsize=8, fontweight="bold")
style_ax(ax6, "Precision vs Recall")
ax6.set_xlabel("Recall", color="#AAAACC", fontsize=8)
ax6.set_ylabel("Precision", color="#AAAACC", fontsize=8)
ax6.set_xlim(0.65, 0.95); ax6.set_ylim(0.75, 0.96)
ax6.axhline(0.85, color="#333355", linewidth=0.8, linestyle=":")
ax6.axvline(0.80, color="#333355", linewidth=0.8, linestyle=":")
ax6.grid(axis="both", color="#222244", linewidth=0.6, linestyle="--")

ax7 = fig.add_subplot(gs[2, :2], polar=True)
ax7.set_facecolor("#14142B")
categories = ["Speed\n(FPS)", "Accuracy\n(mAP50)", "Precision", "Recall", "Efficiency\n(low latency)", "Count\nAccuracy"]

def norm(vals, invert=False):
    lo, hi = min(vals), max(vals)
    if hi == lo: return [0.5]*len(vals)
    n = [(v - lo)/(hi - lo) for v in vals]
    return [1 - x for x in n] if invert else n

fps_n = norm([BENCHMARK_DATA[m]["fps"] for m in MODELS])
map_n = norm([BENCHMARK_DATA[m]["map50"] for m in MODELS])
prec_n = norm([BENCHMARK_DATA[m]["precision"] for m in MODELS])
rec_n = norm([BENCHMARK_DATA[m]["recall"] for m in MODELS])
lat_n = norm([BENCHMARK_DATA[m]["avg_latency_ms"] for m in MODELS], invert=True)
cross_n = norm([BENCHMARK_DATA[m]["people_crossed"] for m in MODELS])

all_scores = list(zip(fps_n, map_n, prec_n, rec_n, lat_n, cross_n))
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for i, m in enumerate(MODELS):
    vals = list(all_scores[i]) + [all_scores[i][0]]
    ax7.plot(angles, vals, color=PALETTE[i], linewidth=2, linestyle="solid", label=m)
    ax7.fill(angles, vals, color=PALETTE[i], alpha=0.12)
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories, color="white", fontsize=9)
ax7.set_yticklabels([])
ax7.set_title("Normalised Performance Radar", color="#AAAACC", fontsize=10, fontweight="bold", pad=14)
ax7.spines["polar"].set_edgecolor("#333355")
ax7.grid(color="#222244", linewidth=0.6)
ax7.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9, framealpha=0.15, labelcolor="white")

ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
rows = ["Latency (ms)", "FPS", "Crossings", "mAP@50 (%)", "Size (MB)", "Params (M)", "Precision", "Recall"]
keys = ["avg_latency_ms","fps","people_crossed","map50","model_size_mb","params_M","precision","recall"]
col_labels = ["Metric"] + MODELS
cell_data = []
for k, label in zip(keys, rows):
    row = [label]
    vals = [BENCHMARK_DATA[m][k] for m in MODELS]
    for m, v in zip(MODELS, vals):
        row.append(str(v))
    cell_data.append(row)

tbl = ax8.table(cellText=cell_data, colLabels=col_labels, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#333355")
    if r == 0:
        cell.set_facecolor("#1E1E3F")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor("#1A1A30")
        cell.set_text_props(color="#AAAACC")
    else:
        cell.set_facecolor("#14142B")
        cell.set_text_props(color=PALETTE[c - 1])
ax8.set_title("Quick Reference Table", color="#AAAACC", fontsize=10, fontweight="bold", pad=8)

out_path = "model_comparison_report.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved chart report successfully → {out_path}")
plt.show()

