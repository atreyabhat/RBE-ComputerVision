import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

folder_path = '../Data/Train/Set1/'
image_files = os.listdir(folder_path)
images = []

# Load each image and append it to the list
for file_name in image_files:
    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path)

    images.append(image)

# Initialize the panorama with the first image
panorama = images[0]
for i in range(1, len(images)):
    img1 = images[i]
    img2 = panorama

    res_img, _,M = sift_feature_matching(img2, img1)

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    corner_pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    corner_pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    corner_pts2_warped = cv2.perspectiveTransform(corner_pts2, M)

    corners = np.concatenate((corner_pts1, corner_pts2_warped), axis=0)

    [xmin, ymin] = np.int32(corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel())

    M_shift = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    canvas_size = (xmax-xmin, ymax-ymin)

    img1_warped = cv2.warpPerspective(img1, M_shift, canvas_size)
    img2_warped = cv2.warpPerspective(img2, M_shift.dot(M), canvas_size)

    # panorama = cv2.warpPerspective(img2, M_shift.dot(M), canvas_size, flags = cv2.INTER_LINEAR)

    # panorama[(-ymin):h1+(-ymin),(-xmin):w1+(-xmin)] = img1

    panorama = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

# Display the final panorama
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Panorama')
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(res_img)
