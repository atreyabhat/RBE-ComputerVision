import cv2
import numpy as np

# Function to detect coins in the frame
def detect_coins(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30, param1=100, param2=40, minRadius=30, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    
    return circles, frame

# Function to recognize coins in the detected circles
def recognize_coins(frame, circles, reference_coins):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    coin_data = {}

    for (x, y, r) in circles:
        # Ensure the region of interest is within the image bounds
        y_min = max(y - r, 0)
        y_max = min(y + r, gray.shape[0])
        x_min = max(x - r, 0)
        x_max = min(x + r, gray.shape[1])
        
        coin = gray[y_min:y_max, x_min:x_max]
        kp, des = sift.detectAndCompute(coin, None)
        coin_data[(x, y, r)] = (kp, des, coin)
    
    return coin_data

# Function to calculate the total value of recognized coins
def calculate_total_value(coin_data, reference_coins):
    total_value = 0.0
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    for coin_key, (kp, des, coin) in coin_data.items():
        best_match = None
        max_matches = 0

        for ref_value, (ref_kp, ref_des) in reference_coins.items():
            if des is not None and ref_des is not None:
                matches = flann.knnMatch(des, ref_des, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
                
                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    best_match = ref_value
        
        if best_match is not None:
            total_value += best_match
    
    return total_value

# Function to read and process reference images
def process_reference_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des, image

# Read and process the reference image of a quarter
ref_kp_quarter, ref_des_quarter, ref_img_quarter = process_reference_image('quarter.jpg')

# Reference coins with their values and precomputed keypoints and descriptors
reference_coins = {
    0.25: (ref_kp_quarter, ref_des_quarter)
}

# Capture video from the camera
cap = cv2.VideoCapture(2)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Detect coins in the frame
    circles, frame_with_circles = detect_coins(frame)

    # Recognize and calculate the total value if circles are detected
    if circles is not None:
        coin_data = recognize_coins(frame, circles, reference_coins)
        total_value = calculate_total_value(coin_data, reference_coins)
        cv2.putText(frame_with_circles, f"Total Value: ${total_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("VisionCoin", frame_with_circles)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
