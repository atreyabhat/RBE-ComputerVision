# Pedestrian and Vehicle Detection on Crosswalks

This project leverages the YOLOv8 model and OpenCV to detect and track pedestrians and vehicles on a crosswalk in a video stream. It counts the number of pedestrians crossing the road and the number of vehicles that pass through a predefined reference line.

## Features:
- **Crosswalk detection**: Segments and highlights the crosswalk area in the video.
- **Vehicle and pedestrian tracking**: Tracks cars, bikes, and pedestrians using YOLOv8.
- **Counts vehicles and pedestrians**: Counts vehicles (cars/bikes) crossing a reference line and pedestrians crossing the crosswalk.
- **Real-time display**: Shows the live video feed with detection and tracking annotations.

## Requirements:
- Python 3.x
- OpenCV (`cv2`)
- Numpy
- `ultralytics` (for YOLOv8)

## How it Works:
1. **Crosswalk Segmentation**: The `segment_crosswalk` function detects and isolates the crosswalk area using color thresholding and contour detection.
2. **Tracking with YOLOv8**: The `model.track()` method from the YOLOv8 package is used to track moving objects (vehicles and pedestrians).
3. **Crossing Detection**: Vehicles and pedestrians crossing the reference line or the crosswalk are counted in real-time.


<img src="https://github.com/user-attachments/assets/3b06801d-28c5-49ae-a147-750c96192dde" width="500" />

<img src="https://github.com/user-attachments/assets/837b88cb-ffdf-497f-928a-3b7f29620e6e" width="500" />
