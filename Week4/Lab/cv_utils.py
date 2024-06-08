'''
Author: Atreya Bhat
RBE 549 Computer Vision @ WPI
'''

import cv2
import datetime
import numpy as np

####################################################
# WEEK 1
####################################################

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



####################################################
# WEEK 2
####################################################


def sobel_custom(frame,direction='x'):

    if(direction=='x'):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        return cv2.filter2D(frame, -1, sobel_x)
        
    elif(direction=='y'):
        sobel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=np.float32)
        return cv2.filter2D(frame, -1, sobel_y)
    

def laplacian_custom(frame):
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(frame, -1, laplacian)


def sobel(frame,direction,ksize):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(direction=='x'):
        return cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize)
    elif(direction=='y'):
        return cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize)
    
    
def canny(frame,threshold1=100,threshold2=200):
    return cv2.Canny(frame, threshold1, threshold2)

def scale_frame(frame, scale):
    """
    Scales the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        scale (float): The scale factor.
    Returns:
        numpy.ndarray: The scaled frame.
    """
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)

def affine_transform(frame, angle):
    """
    Applies affine transformation to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
    Returns:
        numpy.ndarray: The frame with affine transformation applied.
    """
    rows, cols = frame.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(frame, M, (cols, rows))

def perspective_transform(frame,pts1,pts2):
    """
    Applies perspective transformation to the frame.
    Args:
        frame (numpy.ndarray): The input frame.
    Returns:
        numpy.ndarray: The frame with perspective transformation applied.
    """
    rows, cols = frame.shape[:2]
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, M, (cols, rows))


def harris(frame, blockSize, ksize, k,corner_threshold=0.1):
    """
    Performs Harris corner detection on the input frame and refines the corner points.

    Parameters:
    - frame: The input frame
    - blockSize: The size of the neighbourhood considered for corner detection
    - ksize: Aperture parameter of the Sobel derivative used
    - k: Harris detector free parameter in the equation

    Returns:
    - harris_frame: The frame with highlighted Harris corners
    - refined_frame: The frame with refined corner points
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # Apply Harris corner detection
    harris_corners = cv2.cornerHarris(gray_float, blockSize=blockSize, ksize=ksize, k=k)

    # Dilate corner image to enhance corner points
    harris_corners = cv2.dilate(harris_corners, None)

    # Threshold to mark the corners in the original image
    harris_frame = frame.copy()
    harris_frame[harris_corners > corner_threshold * harris_corners.max()] = [0, 0, 255]

    # Refine corner points
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(harris_corners > 0.01 * harris_corners.max()))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Draw refined corners
    refined_frame = frame.copy()
    for corner in corners:
        x, y = corner
        cv2.circle(refined_frame, (int(x), int(y)), 5, (0, 0, 255), 1)

    return harris_frame, refined_frame

def sift(frame, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.2, edgeThreshold=10, sigma=1.6):
    """
    Performs SIFT feature detection on the input frame.

    Parameters:
    - frame: The input frame

    Returns:
    - sift_frame: The frame with SIFT keypoints drawn
    """

    # Create SIFT object with custom parameters
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma)

    sift_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
    kp, _ = sift.detectAndCompute(sift_frame, None)
    sift_frame = cv2.drawKeypoints(sift_frame, kp, sift_frame, color=(0, 255, 0))
    return sift_frame


####################################################
# WEEK 3
####################################################


def custom_draw_matches(img1, kp1, img2, kp2, matches):
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result_img


# Apply RANSAC for outlier rejection
def filter_matches_with_ransac(kp1, kp2, matches):
    if len(matches) < 4:
        return matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
    filtered_matches = []
    for i, m in enumerate(matches):
        if mask[i]:
            filtered_matches.append(m)

    return filtered_matches, M

def sift_feature_matching(img1, img2):
    # Create SIFT feature detector
    sift = cv2.SIFT_create()

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    

    # Detect keypoints and compute descriptors using SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    new_matches, M = filter_matches_with_ransac(keypoints1, keypoints2, good_matches)

    result_img = custom_draw_matches(img1, keypoints1, img2, keypoints2, good_matches)

    return result_img, new_matches, M

def surf_feature_matching(img1, img2, use_ransac=False, hessian_threshold=800):
    # Create SURF feature detector
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

    # Detect keypoints and compute descriptors using SURF
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Apply RANSAC for outlier rejection if specified
    if use_ransac:
        good_matches = filter_matches_with_ransac(keypoints1, keypoints2, good_matches[:50])

    result_img = custom_draw_matches(img1, keypoints1, img2, keypoints2, good_matches[:50])

    return result_img


####################################################
# WEEK 4
####################################################

def image_stitcher(frames):

    # Create stitcher object
    stitcher = cv2.Stitcher_create()
    
    # Stitch frames together
    status, stitched_frame = stitcher.stitch(frames)
    
    if status == cv2.Stitcher_OK:
        return stitched_frame
    else:
        print("Stitching failed!")
        return None