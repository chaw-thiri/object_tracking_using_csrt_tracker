# default + preprocessing + refresh
import cv2

# Initialize default CSRT tracker
default_tracker = cv2.TrackerCSRT.create()

# Open video file or capture device
video_path = r"C:\Users\chawt\Desktop\DIP\test_videos\rabbit.MOV"  # for video
#video_path = 0 # for camera
cap = cv2.VideoCapture(video_path)

# Check if the video is loaded
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame of the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    exit()

# Get video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
desired_width = 800
desired_height = 600
resized = False

# Resize the video only if it is larger than the desired frame
if (video_width > desired_width) and (video_height > desired_height):
    frame = cv2.resize(frame, (desired_width, desired_height))
    resized = True

# Select the initial region_of_interest
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")  # Close the selection window

# Initialize the default tracker with selected bounding box
default_tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the desired size
    if resized:
        frame = cv2.resize(frame, (desired_width, desired_height))

    # Preprocessing steps
    # Histogram Equalization (Enhance contrast for each channel)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # Equalize the L channel (lightness), A ( red - green ), B ( blue -yellow)
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gaussian Blur (Slight noise reduction)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blurred_frame, 100, 200)

    # Update tracker with preprocessed frame
    ret, bbox = default_tracker.update(frame)  # Use the color image

    if ret:
        # Draw the bounding box
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # If tracking fails, ask the user to select a new ROI
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Wait for user to select a new ROI
        bbox = cv2.selectROI("Select New Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select New Object")
        if bbox != (0, 0, 0, 0):  # Check if user selected a new ROI
            # Create a new tracker instance and reinitialize with the new ROI
            default_tracker = cv2.TrackerCSRT.create()  # Create a new tracker
            default_tracker.init(frame, bbox)  # Reinitialize tracker with the new ROI

    # Display the result
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q' or 'Esc' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  # 27 is the 'Esc' key
        print("Exiting...")
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
