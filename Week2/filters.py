'''
Author: Atreya Bhat
RBE 549 Computer Vision @ WPI
'''

import cv2
import datetime
import numpy as np

def zoom(frame, zoom_factor):
    """
    Applies zoom to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        zoom_factor (float): The zoom factor.
    Returns:
        numpy.ndarray: The zoomed frame.
    """
    height, width = frame.shape[:2]
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    # Calculate top left corner of the cropping box
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    
    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
    return cv2.resize(cropped_frame, (width, height))

def threshold_frame(frame, threshold_value):
    """
    Applies thresholding to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        threshold_value (int): The threshold value.
    Returns:
        numpy.ndarray: The thresholded frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def border(frame, border_color=(0, 0, 255), border_size=5):
    """
    Adds a border to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        border_color (tuple): The color of the border.
        border_size (int): The size of the border.
    Returns:
        numpy.ndarray: The frame with border.
    """
    return cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

def capture_image(frame):
    """
    Captures and saves an image from the frame.
    Args:
        frame (numpy.ndarray): The input frame.
    """
    img_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, timestamp, (450, frame.shape[0] - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('WPI_CAM', np.ones_like(frame) * 255)  # Display white screen
    cv2.waitKey(200)  # Wait for 0.5 seconds

    date_time_roi = frame[440:frame.shape[1], 440:].copy()
    frame[0:date_time_roi.shape[0], frame.shape[1] - date_time_roi.shape[1]:frame.shape[1]] = date_time_roi

    img_filename = f"{img_timestamp}.jpg"
    cv2.imwrite(img_filename, frame)
    print(f"Image saved: {img_filename}")

def video_record(frame, fourcc,is_recording):
    if not is_recording:
        img_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.putText(frame, "Recording Started", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('WPI_CAM', frame)
        cv2.waitKey(500)
        video_filename = f"{img_timestamp}.avi"
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        is_recording = True
        print("Recording started")
    else:
        out.release()
        is_recording = False
        cv2.putText(frame, "Recording stopped", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('WPI_CAM', frame)
        cv2.waitKey(500)
        print("Recording stopped")

    return is_recording

def rotate_image(frame, angle):
    """
    Rotates the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        angle (float): The angle to rotate.
    Returns:
        numpy.ndarray: The rotated frame.
    """
    rows, cols = frame.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(frame, M, (cols, rows))

def gaussian_blur(frame, sigma):
    """
    Applies Gaussian blur to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        sigma (float): The sigma for Gaussian blur.
    Returns:
        numpy.ndarray: The frame with Gaussian blur applied.
    """
    return cv2.GaussianBlur(frame, (5, 5), sigma)

def sharpen(frame, sharpen_strength):
    """
    Applies sharpening to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        sharpen_strength (float): The strength of the sharpening.
    Returns:
        numpy.ndarray: The frame with sharpening applied.
    """
    return cv2.filter2D(frame, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * sharpen_strength/2)

def extract_color(frame):
    """
    Extracts a specific color range from the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        lower_color (numpy.ndarray): The lower color range.
        upper_color (numpy.ndarray): The upper color range.
    Returns:
        numpy.ndarray: The frame with the extracted color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    
    full_mask = lower_mask + upper_mask

    return cv2.bitwise_and(frame, frame, mask=full_mask)


def sobel_custom(frame,direction='x'):

    if(direction=='x'):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        # return cv2.filter2D(frame, -1, sobel_x)
        return filter2D_custom(frame, sobel_x)
        
    elif(direction=='y'):
        sobel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=np.float32)
        # return cv2.filter2D(frame, -1, sobel_y)
        return filter2D_custom(frame, sobel_y)
    

def laplacian_custom(frame):
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)
    # return cv2.filter2D(frame, -1, laplacian)
    return filter2D_custom(frame, laplacian)

def filter2D_custom(frame, kernel):
    filtered_frame = np.zeros_like(frame, dtype=np.float32)
    for i in range(1, frame.shape[0] - 1):
        for j in range(1, frame.shape[1] - 1):
            filtered_frame[i, j] = np.sum(frame[i-1:i+2, j-1:j+2] * kernel)

    # Normalize the filtered frame
    filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)
    return filtered_frame


def sobel(frame,direction,ksize):
    if(direction=='x'):
        return cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize)
    elif(direction=='y'):
        return cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize)
    
def canny(frame,threshold1=100,threshold2=200):
    return cv2.Canny(frame, threshold1, threshold2)