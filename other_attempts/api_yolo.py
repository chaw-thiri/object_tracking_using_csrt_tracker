# CSRT + YOLO

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT.create()

# Open video file
video_path = r"C:\Users\chawt\Desktop\DIP\test_videos\people.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables
tracker_initialized = False
last_bbox = None  # Store the last known bounding box

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    if tracker_initialized:
        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            # Draw the tracking box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            last_bbox = bbox  # Update the last known bounding box
        else:
            # Tracking failure: fallback to YOLO detection
            results = model(frame)
            detections = results[0].boxes

            closest_bbox = None
            min_distance = float('inf')

            # Find the closest detection to the last known location
            for detection in detections:
                x1, y1, x2, y2 = detection.xyxy[0]
                detection_bbox = (x1, y1, x2 - x1, y2 - y1)
                # Calculate the distance to the last known location
                if last_bbox:
                    dx = (last_bbox[0] + last_bbox[2] / 2) - (detection_bbox[0] + detection_bbox[2] / 2)
                    dy = (last_bbox[1] + last_bbox[3] / 2) - (detection_bbox[1] + detection_bbox[3] / 2)
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_bbox = detection_bbox

            # Reinitialize the tracker with the closest detected object
            if closest_bbox:
                tracker = cv2.TrackerCSRT.create()
                tracker.init(frame, tuple(map(int, closest_bbox)))
                tracker_initialized = True
    else:
        # Initial object detection
        results = model(frame)
        detections = results[0].boxes

        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]

            if conf > 0.5:
                roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                tracker = cv2.TrackerCSRT.create()
                tracker.init(frame, roi)
                tracker_initialized = True
                break  # Use the first detection for tracking

    # Display the frame
    cv2.imshow("CSRT Tracker with YOLO Recovery", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
