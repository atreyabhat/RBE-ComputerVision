import numpy as np
import cv2
import matplotlib.pyplot as plt



# Function to generate Gaussian pyramid
def generate_gaussian_pyramid(image, num_octaves, num_scales, initial_sigma=1.6):
    gaussian_pyramid = []
    for _ in range(num_octaves):
        octave_images = [image]
        sigma = initial_sigma
        for _ in range(num_scales):
            sigma *= 2
            kernel_size = 10
            gaussian = cv2.getGaussianKernel(kernel_size, sigma)
            image = cv2.filter2D(image, -1, gaussian)
            octave_images.append(image)
        gaussian_pyramid.append(octave_images)
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    return gaussian_pyramid

# Function to generate Difference of Gaussians (DoG) pyramid
def generate_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        dog_images = []
        for i in range(1, len(octave_images)):
            dog = octave_images[i] - octave_images[i - 1]
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    return dog_pyramid

def compute_first_derivatives(D, x, y, s):
    dx = (D[y, x + 1, s] - D[y, x - 1, s]) / 2.0
    dy = (D[y + 1, x, s] - D[y - 1, x, s]) / 2.0
    ds = (D[y, x, s + 1] - D[y, x, s - 1]) / 2.0
    return dx, dy, ds

def compute_second_derivatives(D, x, y, s):
    dxx = D[y, x + 1, s] - 2 * D[y, x, s] + D[y, x - 1, s]
    dyy = D[y + 1, x, s] - 2 * D[y, x, s] + D[y - 1, x, s]
    dss = D[y, x, s + 1] - 2 * D[y, x, s] + D[y, x, s - 1]
    
    dxy = (D[y + 1, x + 1, s] - D[y + 1, x - 1, s] - D[y - 1, x + 1, s] + D[y - 1, x - 1, s]) / 4.0
    dxs = (D[y, x + 1, s + 1] - D[y, x - 1, s + 1] - D[y, x + 1, s - 1] + D[y, x - 1, s - 1]) / 4.0
    dys = (D[y + 1, x, s + 1] - D[y - 1, x, s + 1] - D[y + 1, x, s - 1] + D[y - 1, x, s - 1]) / 4.0
    
    return dxx, dxy, dxs, dyy, dys, dss

def compute_JacobHess(D, x, y, s):
    dx, dy, ds = compute_first_derivatives(D, x, y, s)
    dxx, dxy, dxs, dyy, dys, dss = compute_second_derivatives(D, x, y, s)
    
    Jacob = np.array([dx, dy, ds])
    Hess = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    
    return Jacob, Hess

def localize_keypoint(D, x, y, s):
    Jacob, Hess = compute_JacobHess(D, x, y, s)
    offset = -np.linalg.inv(Hess).dot(Jacob)
    return offset, Jacob, Hess[:2, :2], x, y, s

def is_extrema(pixel_value, dog_pyramid, octave, level, i, j):
    region_of_interest = (dog_pyramid[octave][level - 1][i - 1:i + 2, j - 1:j + 2].flatten().tolist() +
                          dog_pyramid[octave][level][i - 1:i + 2, j - 1:j + 2].flatten().tolist() +
                          dog_pyramid[octave][level + 1][i - 1:i + 2, j - 1:j + 2].flatten().tolist())
    
    return (pixel_value == max(region_of_interest) or pixel_value == min(region_of_interest)) and region_of_interest.count(pixel_value) == 1

def scale_space_and_localize(dog_pyramid):
    extrema_blocks = []
    threshold = 200
    r_lowe = 10

    for octave, octave_images in enumerate(dog_pyramid):
        extrema_locations = set()
        for level, image in enumerate(octave_images):
            if level != 0 and level != len(octave_images) - 1:
                print(f"Processing octave {octave+1} level {level+1}")
                for i in range(2, image.shape[0] - 2, 2):  # Iterate over even rows
                    for j in range(2, image.shape[1] - 2, 2):  # Iterate over even columns
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
    cv_keypoints = [cv2.KeyPoint(x=kp[1], y=kp[0], size=6) for kp in keypoints]
    image_with_keypoints = cv2.drawKeypoints(image, cv_keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Detected Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    file_name = f"KeyPoints_SIFT{random.randint(0, 9)}.png"
    cv2.imwrite(file_name, image_with_keypoints)


image = cv2.imread("UnityHall.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
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
print("Found " + str(len(keypoints[0])) + " keypoints")