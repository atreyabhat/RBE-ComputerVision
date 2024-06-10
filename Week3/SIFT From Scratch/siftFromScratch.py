#################################
#Author: Atreya Bhat
# agbhat@wpi.edu
#################################

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def convolution2D(image, kernel):
    """
    Perform 2D convolution on the input image with the given kernel.

    Args:
        image (numpy.ndarray): Input image.
        kernel (numpy.ndarray): Convolution kernel.

    Returns:
        numpy.ndarray: Convolved image.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output


def generate_gaussian_pyramid(image, num_octaves, num_scales, initial_sigma=1.6):
    """
    Generate a Gaussian pyramid for the input image.

    Args:
        image (numpy.ndarray): Input image.
        num_octaves (int): Number of octaves in the pyramid.
        num_scales (int): Number of scales in each octave.
        initial_sigma (float): Initial sigma value for the Gaussian kernel.

    Returns:
        list: Gaussian pyramid.
    """
    gaussian_pyramid = []
    for _ in range(num_octaves):
        octave_images = [image]
        sigma = initial_sigma
        for _ in range(num_scales):
            sigma *= 2
            kernel_size = 7
            gaussian = cv2.getGaussianKernel(kernel_size, sigma)
            image = convolution2D(image, gaussian)
            octave_images.append(image)
        gaussian_pyramid.append(octave_images)
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    return gaussian_pyramid


def generate_dog_pyramid(gaussian_pyramid):
    """
    Generate a Difference of Gaussians (DoG) pyramid from the given Gaussian pyramid.

    Args:
        gaussian_pyramid (list): Gaussian pyramid.

    Returns:
        list: DoG pyramid.
    """
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        dog_images = []
        for i in range(1, len(octave_images)):
            dog = octave_images[i] - octave_images[i - 1]
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    return dog_pyramid


def compute_first_derivatives(D, x, y, s):
    """
    Compute the first derivatives of the Difference of Gaussians (DoG) pyramid at the specified location.

    Args:
        D (numpy.ndarray): DoG pyramid.
        x (int): x-coordinate of the location.
        y (int): y-coordinate of the location.
        s (int): Scale index.

    Returns:
        tuple: First derivatives (dx, dy, ds).
    """
    dx = (D[y, x + 1, s] - D[y, x - 1, s]) / 2.0
    dy = (D[y + 1, x, s] - D[y - 1, x, s]) / 2.0
    ds = (D[y, x, s + 1] - D[y, x, s - 1]) / 2.0
    return dx, dy, ds


def compute_second_derivatives(D, x, y, s):
    """
    Compute the second derivatives of the Difference of Gaussians (DoG) pyramid at the specified location.

    Args:
        D (numpy.ndarray): DoG pyramid.
        x (int): x-coordinate of the location.
        y (int): y-coordinate of the location.
        s (int): Scale index.

    Returns:
        tuple: Second derivatives (dxx, dxy, dxs, dyy, dys, dss).
    """
    dxx = D[y, x + 1, s] - 2 * D[y, x, s] + D[y, x - 1, s]
    dyy = D[y + 1, x, s] - 2 * D[y, x, s] + D[y - 1, x, s]
    dss = D[y, x, s + 1] - 2 * D[y, x, s] + D[y, x, s - 1]
    
    dxy = (D[y + 1, x + 1, s] - D[y + 1, x - 1, s] - D[y - 1, x + 1, s] + D[y - 1, x - 1, s]) / 4.0
    dxs = (D[y, x + 1, s + 1] - D[y, x - 1, s + 1] - D[y, x + 1, s - 1] + D[y, x - 1, s - 1]) / 4.0
    dys = (D[y + 1, x, s + 1] - D[y - 1, x, s + 1] - D[y + 1, x, s - 1] + D[y - 1, x, s - 1]) / 4.0
    
    return dxx, dxy, dxs, dyy, dys, dss


def compute_JacobHess(D, x, y, s):
    """
    Compute the Jacobian and Hessian matrices at the specified location in the Difference of Gaussians (DoG) pyramid.

    Args:
        D (numpy.ndarray): DoG pyramid.
        x (int): x-coordinate of the location.
        y (int): y-coordinate of the location.
        s (int): Scale index.

    Returns:
        tuple: Jacobian matrix, Hessian matrix.
    """
    dx, dy, ds = compute_first_derivatives(D, x, y, s)
    dxx, dxy, dxs, dyy, dys, dss = compute_second_derivatives(D, x, y, s)
    
    Jacob = np.array([dx, dy, ds])
    Hess = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    
    return Jacob, Hess


def localize_keypoint(D, x, y, s):
    """
    Localize a keypoint at the specified location in the Difference of Gaussians (DoG) pyramid.

    Args:
        D (numpy.ndarray): DoG pyramid.
        x (int): x-coordinate of the location.
        y (int): y-coordinate of the location.
        s (int): Scale index.

    Returns:
        tuple: Offset, Jacobian matrix, Hessian matrix, x-coordinate, y-coordinate, scale index.
    """
    Jacob, Hess = compute_JacobHess(D, x, y, s)
    offset = -np.linalg.inv(Hess).dot(Jacob)
    return offset, Jacob, Hess[:2, :2], x, y, s


def is_extrema(pixel_value, dog_pyramid, octave, level, i, j):
    """
    Check if the pixel at the specified location is an extrema in the Difference of Gaussians (DoG) pyramid.

    Args:
        pixel_value (float): Pixel value at the location.
        dog_pyramid (list): DoG pyramid.
        octave (int): Octave index.
        level (int): Level index.
        i (int): y-coordinate of the location.
        j (int): x-coordinate of the location.

    Returns:
        bool: True if the pixel is an extrema, False otherwise.
    """
    region_of_interest = (dog_pyramid[octave][level - 1][i - 1:i + 2, j - 1:j + 2].flatten().tolist() +
                          dog_pyramid[octave][level][i - 1:i + 2, j - 1:j + 2].flatten().tolist() +
                          dog_pyramid[octave][level + 1][i - 1:i + 2, j - 1:j + 2].flatten().tolist())
    
    return (pixel_value == max(region_of_interest) or pixel_value == min(region_of_interest)) and region_of_interest.count(pixel_value) == 1


def scale_space_and_localize(dog_pyramid):
    """
    Perform scale space extrema detection and localization on the Difference of Gaussians (DoG) pyramid.

    Args:
        dog_pyramid (list): DoG pyramid.

    Returns:
        list: List of extrema blocks.
    """
    extrema_blocks = []
    threshold = 100
    r_lowe = 10

    for octave, octave_images in enumerate(dog_pyramid):
        extrema_locations = set()
        for level, image in enumerate(octave_images):
            if level != 0 and level != len(octave_images) - 1:
                print(f"Processing octave {octave+1} level {level+1}")
                for i in range(1, image.shape[0] - 2):
                    for j in range(1, image.shape[1] - 2):
                        if is_extrema(image[i, j], dog_pyramid, octave, level, i, j):

                            offset, Jacob, Hess, _, _, _ = localize_keypoint(np.stack(octave_images, axis=-1), j, i, level)
                            # Contrast thresholding
                            contrast = image[i, j] + (1/2)* Jacob.dot(offset)
                            if abs(contrast) < threshold:
                                continue
                            # Edge response elimination
                            [alpha_eig,beta_eig], _ = np.linalg.eig(Hess)
                            r = beta_eig / alpha_eig
                            Ratio = (r + 1)**2 / r
                            if Ratio > (r_lowe + 1)**2 / r_lowe:
                                continue
                            extrema_locations.add((i, j))
        extrema_blocks.append(list(extrema_locations))
    return extrema_blocks


def visualize_keypoints(image, keypoints):
    """
    Visualize the detected keypoints on the input image.

    Args:
        image (numpy.ndarray): Input image.
        keypoints (list): List of keypoints.

    Returns:
        None
    """
    plt.imshow(image, cmap='gray')
    plt.plot([kp[1] for kp in keypoints], [kp[0] for kp in keypoints], marker='+', markersize=6, linestyle='', color='blue')
    plt.title('Detected Keypoints')
    plt.axis('off')
    plt.show()
    cv2.imwrite("output.png", image)
    file_name = f"KeyPoints_SIFT{random.randint(0, 9)}.png"
    cv2.imwrite(file_name, image)




############################################################
# Main code
############################################################

image = cv2.imread("UnityHall.jpg", cv2.IMREAD_GRAYSCALE)
num_octaves = 4
num_scales = 5

gaussian_pyramid = generate_gaussian_pyramid(image, num_octaves, num_scales)
dog_pyramid = generate_dog_pyramid(gaussian_pyramid)

# Show all images in dog_pyramid in a subplot
fig, axs = plt.subplots(len(dog_pyramid), len(dog_pyramid[0]), figsize=(10, 10))
for i in range(len(dog_pyramid)):
    print("Image size: ", dog_pyramid[i][0].shape)
    for j in range(len(dog_pyramid[i])):
        axs[i, j].imshow(dog_pyramid[i][j], cmap='gray')
plt.show()

keypoints = scale_space_and_localize(dog_pyramid)
visualize_keypoints(image, keypoints[0])
print("Found " + int(len(keypoints[0])) + " keypoints")