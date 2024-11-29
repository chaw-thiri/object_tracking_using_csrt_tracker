# CSRT + custom CSRT ( swap to custom when the object is lost )
# problem : not losing the object

import cv2
# TODO : solve the retracking
# TODO : add parameter adjustment



def create_custom_csrt_tracker():
    params = cv2.TrackerCSRT.Params()

    # Minimal adjustments for occlusion handling
    #params.use_segmentation = True  # Helps with re-detecting after occlusion
    params.psr_threshold = 0.03  # Lower threshold for robustness, default = 0.035
    # params.padding = 2.0  # Adds context around the target for stability
    params.filter_lr = 0.02  # Smooth filter updates
    # params.scale_lr = 0.025  # Smooth scale updates

    # Create the tracker with these custom parameters
    return cv2.TrackerCSRT.create(params)





# Initialize CSRT tracker
custom_tracker = create_custom_csrt_tracker()
default_tracker = cv2.TrackerCSRT().create()

# Open video file or capture device
#video_path = 0 # for webcam
video_path = r"C:\Users\chawt\Desktop\DIP\test_videos\people.mp4"  # for video
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
# Set the desired frame size (width and height)
desired_width = 800  # Adjust this to fit your screen
desired_height = 600
resized = False
# resize the video only if it is larger than the desired frame
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




    # Update tracker
    ret, bbox = current_tracker.update(frame)

    if ret:
        # Draw the bounding box
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # If tracking fails
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # bbox = reset_tracker(tracker, frame)
        if not active_custom_tracker:
            print("Custom tracker activated...")
            bbox =cv2.selectROI("Re-select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Re-select Object")
            custom_tracker.init(frame, bbox)
            current_tracker = custom_tracker
            active_custom_tracker = True
    if active_custom_tracker:
        # swap to default tracker manually
        if cv2.waitKey(1) & 0xFF == ord('r'):  # 'r' key to reset manually
            print("Switching back to default tracker.")
            default_tracker.init(frame, bbox)
            current_tracker = default_tracker
            active_custom_tracker = False

    # Display the resized result
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
