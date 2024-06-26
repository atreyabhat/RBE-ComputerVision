# Basic Camera Application

A simple camera application to read camera feed and perform various operations on it.

![camera_app_intro](https://github.com/atreyabhat/RBE-ComputerVision/assets/39030188/5863199d-8a22-4ef7-8a12-de8d805521e9)


## Usage Instructions
1. **Launching the Application:**
   - Run the script `cameraApplication.py`.
   - Ensure that a webcam is connected to the system.
   - Press 'Esc' to exit.
   
2. **Keyboard Controls:**
   - Press the following keys to activate corresponding features. The features being used are displayed on the GUI:
     - 'c': Capture an image.
     - 'v': Start/Stop video recording.
     - 'r': Rotate the displayed image by 10 degrees.
     - 'b': Toggle Gaussian blur effect.
     - 't': Toggle threshold effect.
     - 's': Toggle sharpening effect.
     - 'e': Toggle color segmentation.
   
3. **Trackbars:**
   - The application window displays trackbars for adjusting various parameters:
     - 'Threshold': Adjust the threshold value for binary thresholding.
     - 'Blur-Sigma': Adjust the sigma value for Gaussian blur effect.
     - 'Sharpen-Strength': Adjust the strength of the sharpening filter.
     - 'Zoom': Adjust the zoom factor for resizing the frame.
   
4. **Image Capture:**
   - Press 'c' to capture and save an image.
   - The image is also modified with the timestamp RoI overlaid on the top right corner.
   
5. **Video Recording:**
   - Press 'v' to start recording. Press again to stop.

## Requirements
- Python 3.x
- OpenCV library
- Numpy

## Usage
```bash
python camera_application.py
