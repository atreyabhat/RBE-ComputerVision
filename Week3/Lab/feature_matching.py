import cv2
import numpy as np
from cv_utils import *
import os

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def visualize_save(img1, kp1, img2, kp2, matches, title):
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Draw red blobs around the endpoints of the matching lines
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        x2 += img1.shape[1]  # Shift x2 for display on the combined image
        cv2.circle(result_img, (int(x1), int(y1)), 5, (0, 0, 255), -1)  # Red blob in image 1
        cv2.circle(result_img, (int(x2), int(y2)), 5, (0, 0, 255), -1)  # Red blob in image 2

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 600, 400)
    cv2.imwrite(os.path.join(results_dir, title +'.jpg'),result_img)
    cv2.imshow(title, result_img)

# Apply RANSAC for outlier rejection
def filter_matches_with_ransac(kp1, kp2, matches):
    if len(matches) < 4:
        return matches 
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    filtered_matches = []
    for i, m in enumerate(matches):
        if mask[i]:
            filtered_matches.append(m)

    return matches

# Load the images
book_img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)
table_img = cv2.imread('table.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT and SURF feature detectors
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(800)

# Create Brute-Force matcher with cross-check
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Create FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


# Detect keypoints and compute descriptors using SIFT
book_keypoints_sift, book_descriptors_sift = sift.detectAndCompute(book_img, None)
table_keypoints_sift, table_descriptors_sift = sift.detectAndCompute(table_img, None)

# Detect keypoints and compute descriptors using SURF
book_keypoints_surf, book_descriptors_surf = surf.detectAndCompute(book_img, None)
table_keypoints_surf, table_descriptors_surf = surf.detectAndCompute(table_img, None)


# Match descriptors using Brute-Force for SIFT
matches_sift_bf = bf.match(book_descriptors_sift, table_descriptors_sift)
matches_sift_bf = sorted(matches_sift_bf, key=lambda x: x.distance)

# Match descriptors using Brute-Force for SURF
matches_surf_bf = bf.match(book_descriptors_surf, table_descriptors_surf)
matches_surf_bf = sorted(matches_surf_bf, key=lambda x: x.distance)


matches_sift_flann = flann.knnMatch(book_descriptors_sift, table_descriptors_sift, k=2)
matches_surf_flann = flann.knnMatch(book_descriptors_surf, table_descriptors_surf, k=2)

# Apply ratio test to filter good matches for FLANN
good_matches_sift_flann = []
for m, n in matches_sift_flann:
    if m.distance < 0.7 * n.distance:
        good_matches_sift_flann.append(m)

# Apply ratio test to filter good matches for FLANN
good_matches_surf_flann = []
for m, n in matches_surf_flann:
    if m.distance < 0.7 * n.distance:
        good_matches_surf_flann.append(m)



good_matches_sift_flann_ransac = filter_matches_with_ransac(book_keypoints_sift, table_keypoints_sift, good_matches_sift_flann)
good_matches_surf_flann_ransac = filter_matches_with_ransac(book_keypoints_surf, table_keypoints_surf, good_matches_surf_flann)


visualize_save(book_img, book_keypoints_sift, table_img, table_keypoints_sift, matches_sift_bf[:30], 'SIFT + Brute-Force')
visualize_save(book_img, book_keypoints_sift, table_img, table_keypoints_sift, good_matches_sift_flann[:30], 'SIFT + FLANN')
visualize_save(book_img, book_keypoints_surf, table_img, table_keypoints_surf, matches_surf_bf[:30], 'SURF + Brute-Force')
visualize_save(book_img, book_keypoints_surf, table_img, table_keypoints_surf, good_matches_surf_flann[:30], 'SURF + FLANN')
visualize_save(book_img, book_keypoints_sift, table_img, table_keypoints_sift, good_matches_sift_flann_ransac[:30], 'SIFT + FLANN + RANSAC')
visualize_save(book_img, book_keypoints_surf, table_img, table_keypoints_surf, good_matches_surf_flann_ransac[:30], 'SURF + FLANN + RANSAC')

cv2.waitKey(0)
cv2.destroyAllWindows()



