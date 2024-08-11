import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


def xywh_to_xyxy(box):
    x, y, w, h = box
    return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

def segment_crosswalk(image, lower_thresh, upper_thresh, area_thresh=100):
    # Threshold on specified color ranges
    thresh = cv2.inRange(image, lower_thresh, upper_thresh)

    # Apply morphology to fill interior regions in mask
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((15,15), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # Get contours
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Filter on area
    largest_contour = None
    max_area = 0

    for c in cntrs:
        area = cv2.contourArea(c)
        if area > area_thresh:
            if area > max_area:
                max_area = area
                largest_contour = c

    if largest_contour is not None:
        # Get the convex hull of the largest contour
        hull = cv2.convexHull(largest_contour)

        # Get bounding box of the convex hull
        x, y, w, h = cv2.boundingRect(hull)
        roi = (x, y, x+w, y+h)
        
        return roi, hull
    else:
        return (0, 0, 0, 0), None




####################################################################
# Load the YOLO model
model = YOLO('yolov5su.pt')

# Open the video file
video_path = "TrafficVideo.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = "TrafficCount.mp4"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

# Store the track history and counts
track_history = defaultdict(lambda: [])
vehicle_counts = {'car': 0, 'bike': 0}
person_count = 0
crosswalk_detected = False

# Set to track IDs
counted_persons = set()
crossed_vehicles = set()

# Confidence threshold
confidence_threshold = 0.75

# Define the center line position
center_x = width // 2

# Read the first frame to detect crosswalk ROI
ret, first_frame = cap.read()
if ret:
    crosswalk_roi, crosswalk_contour = segment_crosswalk(first_frame, (120, 120, 120), (160, 160, 160))
    crosswalk_detected = crosswalk_roi != (0, 0, 0, 0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        # Get the boxes, track IDs, and confidences
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        # Filter by confidence
        filtered_boxes = [box for box, conf in zip(boxes, confidences) if conf >= confidence_threshold]
        filtered_ids = [track_id for track_id, conf in zip(track_ids, confidences) if conf >= confidence_threshold]
        filtered_classes = [cls for cls, conf in zip(classes, confidences) if conf >= confidence_threshold]

        annotated_frame = frame.copy()

        # Draw vertical reference line at the center
        cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
        
        # Draw the crosswalk ROI contours
        if crosswalk_detected and crosswalk_contour is not None:
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [crosswalk_contour], (255, 255, 0))
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, annotated_frame, 0.5, 0, annotated_frame)
            cv2.drawContours(annotated_frame, [crosswalk_contour], -1, (255, 0, 0), 2)
        else:
            print("Crosswalk not detected")

        for box, track_id, cls in zip(filtered_boxes, filtered_ids, filtered_classes):

            x1, y1, x2, y2 = xywh_to_xyxy(box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


            class_labels = {0: "Person", 1: "Bike", 2: "Car"} 
            label = class_labels.get(cls, "Unknown")
            # Adjust text size and position as needed
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            # Count vehicles crossing the line
            if track_id not in crossed_vehicles:
                if (x1 < center_x < x2) or (x2 < center_x < x1):
                    if cls == 2:  # for cars
                        vehicle_counts['car'] += 1
                    elif cls == 1:  # for bikes
                        vehicle_counts['bike'] += 1
                    crossed_vehicles.add(track_id)

            # Count people crossing the ROI/crosswalk
            if cls == 0: 
                box_x1, box_y1, box_x2, box_y2 = x1, y1, x2, y2
                if crosswalk_detected:
                    roi_x1, roi_y1, roi_x2, roi_y2 = crosswalk_roi
                    # Calculate the intersection area
                    ix1 = max(box_x1, roi_x1)
                    iy1 = max(box_y1, roi_y1)
                    ix2 = min(box_x2, roi_x2)
                    iy2 = min(box_y2, roi_y2)
                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
                    
                    if (inter_area / box_area) >= 0.5:  # Check for 50% overlap
                        if track_id not in counted_persons:
                            person_count += 1
                            counted_persons.add(track_id)

        # Print counts on frame
        counts = [
            f"Cars: {vehicle_counts['car']}",
            f"Bikes: {vehicle_counts['bike']}",
            f"Persons: {person_count}"]
        y0, dy = 30, 40
        for i, line in enumerate(counts):
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            cv2.rectangle(annotated_frame, (10, y0 + i * dy - text_h - 10), (10 + text_w, y0 + i * dy + 10), (255, 0, 0), -1)
            cv2.putText(annotated_frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize the frame for display
        display_frame = cv2.resize(annotated_frame, (width // 2, height // 2))
        out.write(annotated_frame)

        # Display 
        cv2.imshow("YOLOv8 Tracking", display_frame)
 
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # if the end of the video is reached
        break

cap.release()
out.release()
cv2.destroyAllWindows()
