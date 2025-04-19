import numpy as np
import cv2

class ImageStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.prev_frames = []
        self.prev_gray = None
        self.prev_points = None
        self.smoothing_factor = 0.8
        
    def stabilize(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame if not exists
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.005,
                                                      minDistance=20, blockSize=5)
            return frame
        
        # Calculate optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)
        
        # Select good points
        good_old = self.prev_points[status == 1]
        good_new = new_points[status == 1]
        
        if len(good_old) < 8 or len(good_new) < 8:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.005,
                                                      minDistance=20, blockSize=5)
            return frame
        
        # Find transformation matrix
        matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        
        if matrix is None:
            return frame
        
        # Apply transformation
        stabilized = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
        
        # Update previous frame and points
        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                                  minDistance=30, blockSize=3)
        
        # Store stabilized frame
        self.prev_frames.append(stabilized)
        if len(self.prev_frames) > self.window_size:
            self.prev_frames.pop(0)
        
        # Average frames for smoother output
        if len(self.prev_frames) > 0:
            # Apply weighted average for smoother transitions
            weights = np.array([self.smoothing_factor ** i for i in range(len(self.prev_frames))])
            weights = weights / np.sum(weights)
            output = np.average(self.prev_frames, axis=0, weights=weights).astype(np.uint8)
            return output
        
        return stabilized