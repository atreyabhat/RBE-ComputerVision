import cv2
import numpy as np
import matplotlib.pyplot as plt

color_image = cv2.imread('texas.png')
image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# apply gaussina blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# canny edge detection and hough transform to detect lines
edges = cv2.Canny(blurred_image, 300, 200)
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

print("Total number of lines: ", len(lines))

# filter lines based on angle
filtered_lines = []
for line in lines:
    rho, theta = line[0]
    if 0.5 < theta < 3 or 3.4 < theta < np.pi:
        filtered_lines.append((rho, theta))

# Visualize the filtered lines
for rho, theta in filtered_lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


plt.imshow(edges, cmap='gray')
plt.show()

plt.imshow(image)
plt.show()


# least squares
A = []
B = []

for rho, theta in filtered_lines:
    A.append([np.cos(theta), np.sin(theta)])
    B.append([rho])

A = np.array(A)
B = np.array(B)

# Solve the least squares problem to find the vanishing point
ATA_inv = np.linalg.inv(A.T @ A)
x = ATA_inv @ A.T @ B


# Print the vanishing point
print("Vanishing Point: ", x)

image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
cv2.circle(image, (int(x[0]), int(x[1])), 20, (255, 0, 0), -1)
plt.imshow(image)
plt.show()

cv2.imwrite('vanishPoint.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
