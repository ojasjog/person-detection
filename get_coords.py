import cv2

VIDEO_PATH = r"TownCentreXVID.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
success, original_frame = cap.read()
cap.release()

if not success:
    print("Error: Could not read video frame.")
    exit()


ORIG_H, ORIG_W = original_frame.shape[:2]


DISPLAY_W, DISPLAY_H = 1280, 720
display_frame = cv2.resize(original_frame, (DISPLAY_W, DISPLAY_H))


clicked_points = []

def click_line_event(event, x, y, flags, params):
    global display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            true_x = int(x * (ORIG_W / DISPLAY_W))
            true_y = int(y * (ORIG_H / DISPLAY_H))
        
            clicked_points.append((true_x, true_y))
            print(f"📍 Point {len(clicked_points)} Logged -> True X: {true_x}, True Y: {true_y}")
            
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{len(clicked_points)} ({true_x},{true_y})", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if len(clicked_points) == 2:
                p1_disp = (int(clicked_points[0][0] * (DISPLAY_W / ORIG_W)), int(clicked_points[0][1] * (DISPLAY_H / ORIG_H)))
                p2_disp = (int(clicked_points[1][0] * (DISPLAY_W / ORIG_W)), int(clicked_points[1][1] * (DISPLAY_H / ORIG_H)))
                cv2.line(display_frame, p1_disp, p2_disp, (0, 255, 0), 3)
                print("\n✅ Imaginary line established! You can close the window now.")
                
            cv2.imshow("Two-Point Line Grabber", display_frame)

cv2.namedWindow("Two-Point Line Grabber")
cv2.setMouseCallback("Two-Point Line Grabber", click_line_event)

print("--- INSTRUCTIONS ---")
print("1. Click once on the left/upper side of the road.")
print("2. Click a second time directly across on the right/lower side of the road to form your line.")
print("3. Press 'q' to close when finished.\n")

while True:
    cv2.imshow("Two-Point Line Grabber", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()