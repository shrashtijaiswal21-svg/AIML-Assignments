
import cv2
from ultralytics import YOLO
from pathlib import Path

# Step 1: Input video
video_path = "cctv_footage.mp4"

if not Path(video_path).exists():
    print("ERROR: cctv_footage.mp4 file not found!")
    print("Place the video inside AIML_Assignment folder.")
    raise SystemExit

# Step 2: Load the pretrained YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Step 3: Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit

# Step 4: Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

# Step 5: Create output video
output_path = "detected_cctv.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    print("ERROR: Could not create output video.")
    raise SystemExit

print("Object detection started...")

# Step 6: Process each frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    # Draw object labels and bounding boxes
    result_frame = results[0].plot()

    # Save the processed frame
    writer.write(result_frame)

    # Show the detection window
    cv2.imshow("CCTV Object Detection", result_frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 7: Close everything properly
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Object detection completed!")
print("Output saved as:", output_path)
