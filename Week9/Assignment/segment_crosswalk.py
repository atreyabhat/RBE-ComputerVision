import cv2
import numpy as np

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