import cv2
import datetime
import numpy as np

class CameraApplication:
    def __init__(self, logo_path='OpenCV_Logo.png'):
       
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
        
        # Initialize variables
        self.zoom_factor = 1.0
        self.zoom_max = 10
        self.angle = 10
        self.sigma = 0
        self.threshold_value = 127 
        self.sharpen_strength = 2 
        
        # Initialize flags
        self.rotated_flag = False
        self.threshold_flag = False
        self.blur_flag = False
        self.sharpen_flag = False
        self.extract_color_flag = False

        # Create a window for trackbars
        cv2.namedWindow('WPI_CAM')
        
        # Create trackbars
        cv2.createTrackbar('Threshold', 'WPI_CAM', self.threshold_value, 255, self.update_threshold)
        cv2.createTrackbar('Blur-Sigma', 'WPI_CAM', self.sigma, 30, self.update_sigma)
        cv2.createTrackbar('Sharpen-Strength', 'WPI_CAM', self.sharpen_strength, 10, self.update_sharpen_strength)
        cv2.createTrackbar('Zoom', 'WPI_CAM', int(self.zoom_factor * 10), int(self.zoom_max * 10), self.update_zoom_factor)

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

    def zoom(self, frame):
        """
        Applies zoom to the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The zoomed frame.
        """
        height, width = frame.shape[:2]
        new_width, new_height = int(width / self.zoom_factor), int(height / self.zoom_factor)
        
        # Calculate top left corner of the cropping box
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2
        
        cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
        return cv2.resize(cropped_frame, (width, height))

    def threshold_frame(self, frame):
        """
        Applies thresholding to the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: 
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def border(self, frame):
        """
        Adds a border to the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The frame with border.
        """
        border_color = (0, 0, 255)
        border_size = 5
        return cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

    def capture_image(self, frame):
        """
        Captures and saves an image from the frame.

        Args:
            frame (numpy.ndarray): The input frame.
        """
        img_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imshow('WPI_CAM', np.ones_like(frame) * 255)  # Display white screen
        cv2.waitKey(200)  # Wait for 0.5 seconds

        date_time_roi = frame[440:frame.shape[1], 440:].copy()
        frame[0:date_time_roi.shape[0], frame.shape[1] - date_time_roi.shape[1]:frame.shape[1]] = date_time_roi

        img_filename = f"Week1.jpg"
        cv2.imwrite(img_filename, frame)
        print(f"Image saved: {img_filename}")

    def rotate_image(self, frame):
        """
        Rotates the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The rotated frame.
        """
        rows, cols = frame.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        return cv2.warpAffine(frame, M, (cols, rows))

    def gaussian_blur(self, frame):
        """
        Applies Gaussian blur to the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The frame with Gaussian blur applied.
        """
        return cv2.GaussianBlur(frame, (5, 5), self.sigma)

    def sharpen(self, frame):
        """
        Applies sharpening to the frame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            numpy.ndarray: The frame with sharpening applied.
        """
        return cv2.filter2D(frame, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * self.sharpen_strength/2)

    def extract_color(self, frame, lower_color, upper_color):
        """
        Extracts a specific color range from the frame.

        Args:
            frame (numpy.ndarray): The input frame.
            lower_color (numpy.ndarray): The lower color range.
            upper_color (numpy.ndarray): The upper color range.

        Returns:
            numpy.ndarray: The frame with the extracted color.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        
        full_mask = lower_mask + upper_mask
 
        return cv2.bitwise_and(frame, frame, mask=full_mask)

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
            frame = self.zoom(frame)
        if self.rotated_flag:
            frame = self.rotate_image(frame)
        if self.threshold_flag:
            frame = self.threshold_frame(frame)
        if self.blur_flag:
            frame = self.gaussian_blur(frame)
        if self.sharpen_flag:
            frame = self.sharpen(frame)
        if self.extract_color_flag:
            lower = np.array([0, 100, 20])
            upper = np.array([10, 255, 255])
            frame = self.extract_color(frame, lower, upper)

        # Blend logo and add border
        if frame.shape[0] >= 100 and frame.shape[1] >= 100:
            frame[0:100, 0:100] = cv2.addWeighted(frame[0:100, 0:100], 0.5, self.logo_resized, 0.5, 0)
        frame = self.border(frame)

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                self.capture_image(frame)
            
            elif key == ord('v'):
                if not self.is_recording:
                    img_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.putText(frame, "Recording Started", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('WPI_CAM', frame)
                    cv2.waitKey(500)
                    video_filename = f"Week1.avi"
                    self.out = cv2.VideoWriter(video_filename, self.fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    self.is_recording = True
                    print("Recording started")
                else:
                    self.out.release()
                    self.is_recording = False
                    cv2.putText(frame, "Recording stopped", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow('WPI_CAM', frame)
                    cv2.waitKey(500)
                    print("Recording stopped")
            
            
            elif key == ord('r'):
                self.rotated_flag = not self.rotated_flag
                if self.rotated_flag:
                    self.angle += 10
            
            
            elif key == ord('b'):
            #if self.sigma > 0:
                frame = self.gaussian_blur(frame)
                self.blur_flag = not self.blur_flag

            elif key == ord('t'):
                self.threshold_flag = not self.threshold_flag
                self.extract_color_flag = False


            elif key == ord('s'):
            #if self.sharpen_strength > 0:
                self.sharpen_flag = not self.sharpen_flag
            
            
            elif key == ord('e'):
                self.threshold_flag = False
                self.extract_color_flag = not self.extract_color_flag

            elif key == 27:  # ESC key to exit
                break
            

            timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(frame, timestamp, (450, frame.shape[0] - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 255, 255), 2)
            text_org = 120
            if self.threshold_flag:
                cv2.putText(frame, "Threshold", (10, text_org), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.blur_flag:
                cv2.putText(frame, "Blur", (10, text_org+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.sharpen_flag:
                cv2.putText(frame, "Sharpen", (10, text_org+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.extract_color_flag:
                cv2.putText(frame, "Color Segmentation", (10, text_org+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


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
