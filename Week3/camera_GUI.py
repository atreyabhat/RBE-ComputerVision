'''
Author: Atreya Bhat
RBE 549 Computer Vision @ WPI
'''

import cv2
import datetime
import numpy as np
from cv_utils import *
import matplotlib.pyplot as plt


class CameraApplication:
    def __init__(self, logo_path='OpenCV_Logo.png',train_image_path='train.jpg'):
       
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # Initialize recording variables
        self.is_recording = False
        self.out = None
        
        # Define codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Read and resize logo image
        self.logo = cv2.imread(logo_path)
        self.logo_resized = cv2.resize(self.logo, (100, 100))

        #for feature matching
        self.train_img = cv2.imread(train_image_path)
        
        # Initialize variables
        self.zoom_factor = 1.0
        self.zoom_max = 10
        self.angle = 10
        self.sigma = 0
        self.threshold_value = 127 
        self.sharpen_strength = 2 
        self.canny_thresh_default = 100
        self.sobel_default = 3
        
        # Initialize flags
        self.rotated_flag = False
        self.threshold_flag = False
        self.blur_flag = False
        self.sharpen_flag = False
        self.extract_color_flag = False
        self.canny_flag = False
        self.sobelx_flag = False
        self.sobely_flag = False
        self.sift_flag = False
        self.surf_flag = False

        # Create a window for trackbars
        cv2.namedWindow('WPI_CAM')
        
        # Create trackbars
        cv2.createTrackbar('Threshold', 'WPI_CAM', self.threshold_value, 255, self.update_threshold)
        cv2.createTrackbar('Blur-Sigma', 'WPI_CAM', self.sigma, 30, self.update_sigma)
        cv2.createTrackbar('Sharpen-Strength', 'WPI_CAM', self.sharpen_strength, 10, self.update_sharpen_strength)
        cv2.createTrackbar('Zoom', 'WPI_CAM', int(self.zoom_factor * 10), int(self.zoom_max * 10), self.update_zoom_factor)
        cv2.createTrackbar('Sobel Kernel', 'WPI_CAM', self.sobel_default, 50, self.update_sobel_kernel)
        cv2.createTrackbar('Canny Threshold1', 'WPI_CAM', self.canny_thresh_default, 5000, self.update_canny_threshold_1)
        cv2.createTrackbar('Canny Threshold2', 'WPI_CAM', self.canny_thresh_default, 5000, self.update_canny_threshold_2)

    def update_sigma(self, value):
        """
        Tracker callback function to update the sigma value.
        """
        self.sigma = value
        print("Updated Sigma: ", self.sigma)

    def update_zoom_factor(self, value):
        """
        Tracker callback function to update the zoom value.
        """
        self.zoom_factor = value / 10.0
        print("Zoom Factor: ", self.zoom_factor)

    def update_threshold(self, value):
        """
        Tracker callback function to update the Threshold value.
        """
        self.threshold_value = value
        print("Threshold Value: ", self.threshold_value)
    
    def update_sharpen_strength(self, value):
        """
        Tracker callback function to update the Sharpening intensity value.
        """
        self.sharpen_strength = value
        print("Sharpen Strength: ", self.sharpen_strength)
    
    def update_sobel_kernel(self, value):
        """
        Tracker callback function to update the Sobel kernel size.
        """
        self.sobel_kernel = value
        print("Sobel Kernel: ", self.sobel_kernel)
    
    def update_canny_threshold_1(self, value):
        """
        Tracker callback function to update the Canny Threshold 1 value.
        """
        self.canny_threshold_1 = value
        print("Canny Threshold 1: ", self.canny_threshold_1)
    
    def update_canny_threshold_2(self, value):  
        """
        Tracker callback function to update the Canny Threshold 2 value.
        """
        self.canny_threshold_2 = value
        print("Canny Threshold 2: ", self.canny_threshold_2)


    def process_frame(self, frame):
        """
        Processes the frame by applying various transformations based on flags.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The processed frame.
        """
        # Apply transformations based on flags
        if self.zoom_factor > 1.0:
            frame = zoom(frame, self.zoom_factor)
        if self.rotated_flag:
            frame = rotate_image(frame, self.angle)
        if self.threshold_flag:
            frame = threshold_frame(frame, self.threshold_value)
        if self.blur_flag:
            frame = gaussian_blur(frame, self.sigma)
        if self.sharpen_flag:
            frame = sharpen(frame, self.sharpen_strength)
        if self.extract_color_flag:
            lower = np.array([0, 100, 20])
            upper = np.array([10, 255, 255])
            frame = extract_color(frame, lower, upper)
        if self.canny_flag:
            frame = canny(frame, self.canny_threshold_1, self.canny_threshold_2)
        if self.sobelx_flag:
            frame = sobel(frame, 'x', self.sobel_kernel)
        if self.sobely_flag:
            frame = sobel(frame, 'y', self.sobel_kernel)
        if self.sift_flag:
            frame = sift_feature_matching(frame,self.train_img, use_ransac=True)
        if self.surf_flag:
            frame = surf_feature_matching(frame,self.train_img, use_ransac=True)

        return frame


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                capture_image(frame)
            elif key == ord('v'):
                video_record(frame,self.fourcc,self.is_recording)
            elif key == ord('r'):
                self.rotated_flag = True
                if self.rotated_flag:
                    self.angle += 10
            elif key == ord('b'):
                self.blur_flag = not self.blur_flag
            elif key == ord('t'):
                self.threshold_flag = not self.threshold_flag
                self.extract_color_flag = False
            elif key == ord('q'):
                self.sharpen_flag = not self.sharpen_flag          
            elif key == ord('e'):
                self.threshold_flag = False
                self.extract_color_flag = not self.extract_color_flag
            elif key == ord('d'):
                self.canny_flag = not self.canny_flag
            elif key == ord('s'):
                # Check for combination keypresses
                second_key = cv2.waitKey(0) & 0xFF
                if second_key == ord('x'):
                    self.sobelx_flag = not self.sobelx_flag
                elif second_key == ord('y'):
                    self.sobely_flag = not self.sobely_flag

            #Week 2 - Display 4 pictures with sobel and laplacian
            elif key == ord('4'):
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)               
                laplacian = laplacian_custom(gray)
                sobelx = sobel_custom(gray, 'x')
                sobely = sobel_custom(gray, 'y')
                cv2.imshow('Original', frame)
                cv2.imshow('Laplacian', laplacian)
                cv2.imshow('Sobel X', sobelx)
                cv2.imshow('Sobel Y', sobely)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #Week 3 - Sift and Surf
            elif key == ord('s'):
                # Check for combination keypresses
                second_key = cv2.waitKey(0) & 0xFF
                if second_key == ord('i'):
                    self.sift_flag = not self.sift_flag
                elif second_key == ord('u'):
                    self.surf_flag = not self.surf_flag
            elif key == 27:  # ESC key to exit
                break
            

            text_org = 120
            if self.threshold_flag:
                cv2.putText(frame, "Threshold", (10, text_org), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.blur_flag:
                cv2.putText(frame, "Blur", (10, text_org+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.sharpen_flag:
                cv2.putText(frame, "Sharpen", (10, text_org+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.extract_color_flag:
                cv2.putText(frame, "Color Segmentation", (10, text_org+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            

            # # Blend logo and add border
            # if frame.shape[0] >= 100 and frame.shape[1] >= 100:
            #     logo_resized_uint8 = self.logo_resized.astype(np.uint8)
            #     frame_uint8 = frame[0:100, 0:100].astype(np.uint8)
            #     frame[0:100, 0:100] = cv2.addWeighted(frame_uint8, 0.5, logo_resized_uint8, 0.5, 0)

            frame = border(frame)

            #Show the image
            cv2.imshow('WPI_CAM', frame)
            
            if self.is_recording:
                self.out.write(frame)

        self.cap.release()
        if self.is_recording:
            self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = CameraApplication()
    processor.run()
