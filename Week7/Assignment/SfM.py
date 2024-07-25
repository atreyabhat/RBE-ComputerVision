import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os

def featureMatching(img1, img2, method='sift'):
    
    if method == 'sift':
        featureDetector = cv2.SIFT_create()
    elif method == 'surf':
        featureDetector = cv2.xfeatures2d.SURF_create()


    
    # Detect keypoints and compute descriptors
    kp1, des1 = featureDetector.detectAndCompute(img1, None)
    kp2, des2 = featureDetector.detectAndCompute(img2, None)
    
    # Match descriptors using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # # Apply RANSAC to filter out outliers
    # _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]
    
    return pts1, pts2



def computeEssentialMat(K, F):
    E = np.dot(np.dot(K.T, F), K)
    
    U, D, Vt = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vt

    return E


def estimateCameraPose(E):
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    c1 = U[:, 2]
    c2 = -U[:, 2]
    c3 = U[:, 2]
    c4 = -U[:, 2]
    r1 = U @ W @ Vt
    r2 = U @ W @ Vt
    r3 = U @ W.T @ Vt
    r4 = U @ W.T @ Vt

    def check_pose(r, c):
        if np.linalg.det(r) < 0:
            r = -r
            c = -c
        return r, c
    
    r1, c1 = check_pose(r1, c1)
    r2, c2 = check_pose(r2, c2)
    r3, c3 = check_pose(r3, c3)
    r4, c4 = check_pose(r4, c4)
    
    poses = [(c1, r1), (c2, r2), (c3, r3), (c4, r4)]
    
    return poses

def computeFundamentalMat(pts):

    A = np.zeros((pts.shape[0],9))
    
    def getRowOfA(row):
        x1,y1,x2,y2 = row
        return [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]
    
    #create A matrix
    for i in range(pts.shape[0]):
        A[i,:]=getRowOfA(pts[i])

    _,_,V = np.linalg.svd(A)
    F = V[-1,:]
    F_base = F.reshape(3,3)

    #reestimate F
    UF,UD,UV = np.linalg.svd(F_base)
    UD[-1]=0
    F = UF @ np.diag(UD) @ UV
    F /= F[2,2]
    return F

def getInlierRANSAC(pts, num_iterations=100, eps=0.03):
    """
    Estimate inlier correspondences using fundamental matrix based RANSAC.
    """
   
    best_inliers = []
    best_count = 0
    for _ in range(num_iterations):
        # Randomly sample 8 correspondences
        inliers = []
        inlier_count=0
        idx = np.random.randint(0,pts.shape[0],8)
        sample = pts[idx,:]
        # print(sample.shape)
        
        F = computeFundamentalMat(sample)
        
        for i, pt_pair in enumerate(pts):
            x1,y1,x2,y2 = pt_pair
            pt1 = np.array([x1,y1,1])
            pt2 = np.array([x2,y2,1])
            error = pt2.T @ F @ pt1

            if  np.abs(error) < eps:
                inliers.append(i)
                inlier_count+=1
        
        # Update the best inliers if the current set is better
        if inlier_count > best_count:
            best_inliers = inliers
            best_count = inlier_count

    # print(np.array(best_inliers).shape)
    
    #return F computed from the best inliers, and the best inliers
    return computeFundamentalMat(pts[best_inliers,:])



def constructProjectionMatrix(K, R, t):
    R = np.array(R)
    t = np.array(t).reshape(3,1)
    extrinsic = np.hstack((R, t))
    return   K @ extrinsic


def cv2Triangulation(pts1, pts2, C1,R1, C2, R2, K):

    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

    P1 = K @ R1 @ np.hstack((np.eye(3), -C1.reshape((-1,1))))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2.reshape((-1,1))))
    
    points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3D = points4D[:3] / points4D[3]
    
    return points3D,P1,P2


def LSTriangulation(pts1, pts2, C1, R1, C2, R2, K):

    P1 = K @ R1 @ np.hstack((np.eye(3), -C1.reshape((-1, 1))))
    P2 = K @ R2 @ np.hstack((np.eye(3), -C2.reshape((-1, 1))))
        
    pts3D = []
    for i in range(pts1.shape[0]):
        A = np.array([
            pts1[i][0] * P1[2] - P1[0],
            pts1[i][1] * P1[2] - P1[1],
            pts2[i][0] * P2[2] - P2[0],
            pts2[i][1] * P2[2] - P2[1]])
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1,:]
        X /= X[-1]
        pts3D.append(X)
    
    return np.array(pts3D),P1,P2  

def DisambiguateCameraPose(E, pts1, pts2, K):
    poses = estimateCameraPose(E)
    max_inliers = -1
    best_cam_pose = None
    optimal_X = None

    for C, R in poses:
        X,P1,P2 = LSTriangulation(pts1, pts2, np.zeros(3), np.eye(3), C, R, K)

        
        # Check chirality and positive depth
        C = C.reshape(3, 1)  # Reshape to (3, 1) for broadcasting
        X_cam1 = X.T[:3, :]  # Use only (X, Y, Z) coordinates
        X_cam2 = R @ X_cam1 + C

        chirality_condition_check_vector = X_cam2[2, :]  # Z-coordinates in the second camera
        Z_positive_condition = X_cam1[2, :] > 0  # Z-coordinates in the first camera

        inlier_count = np.sum((chirality_condition_check_vector > 0) & Z_positive_condition)

        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_cam_pose = (C, R)
            optimal_X = X  # Ensure optimal_X has shape (N, 3)
            best_P1 = P1
            best_P2 = P2

    return optimal_X, best_cam_pose, best_P1, best_P2


# Open3D visualization and saving point cloud with colors
def SavePCDtoFile(points, colors, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(filename, pcd)


def get_dominant_colors(X, img, P):
    dominant_colors = []
    for pt in X:
        pt = np.append(pt, 1)
        pt_proj = np.dot(P, pt)  # Project to image plane using P
        
        x, y = int(pt_proj[0] / pt_proj[2]), int(pt_proj[1] / pt_proj[2])  # Normalize by dividing by z
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Check if within image bounds
            color = img[y, x]  # Assuming img is in RGB format
            dominant_colors.append(color)
        else:
            dominant_colors.append([0, 0, 0])  # Handle out-of-bounds cases
        
    return np.array(dominant_colors)

def visualizePCD(pcd, point_size=10):

    # Set point size (adjust as needed)
    point_size = 10  # Increase this value to increase point size

    # Create visualizer object
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="SfM reconstructed")

    # Add point cloud to visualizer
    visualizer.add_geometry(pcd)

    # Get render option and set point size
    render_option = visualizer.get_render_option()
    render_option.point_size = point_size

    # Run the visualizer
    visualizer.run()
    visualizer.destroy_window()


def meanReprojError(img1_pts, img2_pts, X, P1, P2):

    mean_error = 0
    for i in range(len(img1_pts)):
        u1_proj = X[i] @ P1.T
        u1_proj /= u1_proj[2]  # Normalize by the homogeneous coordinate
        error1 = np.linalg.norm(img1_pts[i] - u1_proj[:2])

        u2_proj = X[i] @ P2.T
        u2_proj /= u2_proj[2]  # Normalize by the homogeneous coordinate
        error2 = np.linalg.norm(img2_pts[i] - u2_proj[:2])

    mean_error += (error1 + error2) / 2  # Average error for each point
    return mean_error / len(img1_pts)  # Average error for all points


#main
import numpy as np
import cv2
import os


## custom data

# K = np.array([[531.12213,   0.     , 407.19257],
#               [  0.     , 531.54175, 313.30872],
#               [  0.     ,   0.     ,   1.     ]], dtype=np.float32)

# img1 = cv2.imread(os.path.join(os.getcwd(),'Data/1.png'))
# img2 = cv2.imread(os.path.join(os.getcwd(),'Data/2.png'))



#ETH facade
K = np.array([[3414.66, 0, 3113.46],
              [0, 3413.37, 2064.47],
              [0, 0, 1]])

img1 = cv2.imread(os.path.join(os.getcwd(), 'Data/facade_dslr_undistorted/facade/images/dslr_images_undistorted/DSC_0390.JPG'))
img2 = cv2.imread(os.path.join(os.getcwd(), 'Data/facade_dslr_undistorted/facade/images/dslr_images_undistorted/DSC_0391.JPG'))



#########################################################################################################################################

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

print("Computing correspondences...")
correspondences = featureMatching(img1,img2, method='surf')
print("Total correspondences found: ", len(correspondences[0]))

img1_pts = correspondences[0]
img2_pts = correspondences[1]


# Display img1 with correspondences
plt.figure(figsize=(10, 10))
plt.imshow(img1)
plt.scatter(img1_pts[:, 0], img1_pts[:, 1], c='r', marker='x')
plt.title('Image 1 with Correspondences')
plt.show()



F= cv2.findFundamentalMat(img1_pts, img2_pts, cv2.FM_RANSAC, 0.01, 0.99)[0]

# F = getInlierRANSAC(np.hstack((img1_pts,img2_pts)),1000,0.05)

print("F mat ",F)
E = computeEssentialMat(K,F)
print("E mat ",E)


X, pose,P1,P2 = DisambiguateCameraPose(E, img1_pts, img2_pts,K)
print("Reprojection Error ", meanReprojError(img1_pts, img2_pts, X, P1, P2))

# X_new = NonLinearTriangulation(X, P1, P2, img1_pts, img2_pts)
# print("Reprojection Error after Non Linear Triangulation: ", meanReprojError(img1_pts, img2_pts, X_new, P1, P2))

optimal_X = X[:,:3]
optimal_X = optimal_X[(optimal_X[:,2]>-50) & (optimal_X[:,2]<50)]


dominant_colors = get_dominant_colors(optimal_X, img1,P1)

SavePCDtoFile(optimal_X, dominant_colors, "reconstructed.pcd")

#load pcd file
pcd = o3d.io.read_point_cloud("reconstructed.pcd")
visualizePCD(pcd, point_size=5)