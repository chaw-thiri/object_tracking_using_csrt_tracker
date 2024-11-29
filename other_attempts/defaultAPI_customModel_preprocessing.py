import cv2

def create_custom_csrt_tracker():
    params = cv2.TrackerCSRT_Params()
    params.psr_threshold = 0.03  # Lower threshold for robustness
    params.filter_lr = 0.02  # Smooth filter updates
    return cv2.TrackerCSRT_create(params)

# Initialize CSRT tracker
custom_tracker = create_custom_csrt_tracker()
default_tracker = cv2.TrackerCSRT_create()

# Open video file or capture device
video_path = r"C:\Users\chawt\Desktop\DIP\test_videos\rabbit.MOV"  # for video
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

# Select the region_of_interest
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")  # Close the selection window

# Initialize the default tracker with selected bounding box
default_tracker.init(frame, bbox)
current_tracker = default_tracker
active_custom_tracker = False

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
    l = cv2.equalizeHist(l)  # Equalize the L channel (lightness)
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gaussian Blur (Slight noise reduction)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Edge Detection (Optional: Highlight edges if needed)
    edges = cv2.Canny(blurred_frame, 100, 200)

    # Update tracker with preprocessed frame (use original frame for tracking)
    ret, bbox = current_tracker.update(frame)  # Use the color image

    if ret:
        # Draw the bounding box
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # If tracking fails
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Only re-select the object if it was lost and manually reset
        if not active_custom_tracker:
            print("Custom tracker activated...")
            bbox = cv2.selectROI("Re-select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Re-select Object")
            custom_tracker.init(frame, bbox)
            current_tracker = custom_tracker
            active_custom_tracker = True

    # Handle switching back to default tracker manually
    if active_custom_tracker:
        if cv2.waitKey(1) & 0xFF == ord('r'):  # 'r' key to reset manually
            print("Switching back to default tracker.")
            default_tracker.init(frame, bbox)
            current_tracker = default_tracker
            active_custom_tracker = False

    # Display the result
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
