# pure CSRT api
import cv2

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT.create()

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
if (video_width > desired_width) or (video_height > desired_height):
    frame = cv2.resize(frame, (desired_width, desired_height))
    resized = True

# Select the object to track manually
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")  # Close the selection window

# Initialize the tracker with the selected bounding box
tracker.init(frame, bbox)



while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the desired size
    if resized:
        frame = cv2.resize(frame, (desired_width, desired_height))




    # Update tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw the bounding box
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # If tracking fails
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # bbox = reset_tracker(tracker, frame)
    # Display the resized result
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()