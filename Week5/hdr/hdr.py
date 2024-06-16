import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def perform_hdr_fusion(folder_path):
    """
    Perform HDR fusion on a set of images in the given folder path.

    Returns:
        numpy.ndarray: The HDR fused image.
    """
    # Read all images from the folder
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            images.append(image)

    # Perform HDR fusion
    merge_mertens = cv2.createMergeMertens()
    hdr = merge_mertens.process(images)

    return hdr

# Example usage
folder_path = "image_set1/"
hdr_image = perform_hdr_fusion(folder_path)

# Display the HDR image
cv2.imshow("HDR Image", cv2.resize(hdr_image, dsize=(0, 0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
