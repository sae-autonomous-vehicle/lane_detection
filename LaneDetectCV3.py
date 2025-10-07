#!/usr/bin/env python3
"""
Jetson-optimized Lane Detection (fully CUDA accelerated where possible)
"""

import time
import cv2
import numpy as np
from collections import deque

# ---------------------------------------------------------
# Helper: capture creation (supports CSI GStreamer on Jetson)
# ---------------------------------------------------------
def create_capture(source=0, width=1280, height=720, use_gst_for_csi=True):
    if isinstance(source, str):
        if source.lower() == 'csi' and use_gst_for_csi:
            gst = (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM),"
                f"width=(int){width},height=(int){height},framerate=(fraction)30/1 ! "
                "nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
            )
            return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        else:
            return cv2.VideoCapture(source)
    else:
        return cv2.VideoCapture(int(source))


# ---------------------------------------------------------
# CUDA-optimized Lane Detection
# ---------------------------------------------------------
class LaneDetect:
    def __init__(self, resize_width=640, smoothing_frames=8, num_samples=200):
        self.RESIZE_WIDTH = resize_width
        self.SMOOTHING_FRAMES = smoothing_frames
        self.NUM_SAMPLES = num_samples
        self.CENTER_DEADZONE = int(0.05 * self.RESIZE_WIDTH)
        self.last_turn = 0
        self.MIN_AREA = 1200
        self.MIN_WIDTH = 170
        self.MIN_LR_CURVE_RATIO = 0.99
        self.SLOPE_THRESH = 1.73

        self.lane_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.center_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.left_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.right_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.curvature_history = deque(maxlen=self.SMOOTHING_FRAMES * 2)

        # CUDA
        self.cuda_available = False
        try:
            self.cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_available = self.cuda_count > 0
            if self.cuda_available:
                self.stream = cv2.cuda.Stream()
        except Exception:
            self.cuda_count = 0
            self.cuda_available = False
            self.stream = None

        # Morphology kernels
        self.small_kernel_gpu = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1,
                                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        self.med_kernel_gpu = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1,
                                                              cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    # ---------------- region of interest (GPU)
    def region_of_interest(self, gmask):
        h, w = gmask.size()
        mask_cpu = np.zeros((h, w), np.uint8)
        polygon = np.array([[(0, int(h*0.9)), (w, int(h*0.9)), (w, int(h*0.7)), (0, int(h*0.7))]], np.int32)
        cv2.fillPoly(mask_cpu, polygon, 255)
        gmask_mask = cv2.cuda_GpuMat()
        gmask_mask.upload(mask_cpu)
        gmask_roi = cv2.cuda.bitwise_and(gmask, gmask_mask)
        return gmask_roi

    # ---------------- perspective transform
    def perspective_matrices(self, shape):
        h, w = shape[:2]
        src = np.float32([[w*0.2,h*0.7],[w*0.8,h*0.7],[w,h*0.9],[0,h*0.9]])
        dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
        return cv2.getPerspectiveTransform(src,dst), cv2.getPerspectiveTransform(dst,src)

    # ---------------- CUDA lane mask
    def combined_lane_mask(self, frame):
        hls_gpu = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame), cv2.COLOR_BGR2HLS, stream=self.stream)
        l_gpu, s_gpu = cv2.cuda.split(hls_gpu)
        
        # S channel mask
        s_bin = cv2.cuda.inRange(s_gpu, 130, 255)
        l_bin = cv2.cuda.inRange(l_gpu, 220, 255)
        color_bin = cv2.cuda.bitwise_or(s_bin, l_bin)

        # Sobel X
        gray_gpu = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame), cv2.COLOR_BGR2GRAY)
        blur_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5,5), 0)
        sobel_gpu = cv2.cuda.Sobel(blur_gpu, cv2.CV_16S, 1, 0, ksize=3)
        abs_sobel = cv2.cuda.abs(sobel_gpu)
        scaled_sobel = cv2.cuda.convertScaleAbs(abs_sobel)
        grad_bin = cv2.cuda.inRange(scaled_sobel, 70, 255)

        # Combine
        combined = cv2.cuda.bitwise_or(color_bin, grad_bin)
        combined = self.region_of_interest(combined)

        # Warp
        matrix, inverse_matrix = self.perspective_matrices(frame.shape)
        warped_gpu = cv2.cuda.warpPerspective(combined, matrix, (combined.size()[1], combined.size()[0]))
        # Morphology
        warped_gpu = self.small_kernel_gpu.apply(warped_gpu)
        warped_gpu = self.med_kernel_gpu.apply(warped_gpu)

        # Download final mask
        warped_mask = warped_gpu.download()
        return warped_mask, inverse_matrix

    # ---------------- lane extraction
    def extract_lane_candidates(self, lane_mask):
        nonzero = lane_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if len(nonzeroy) == 0:
            return None, None

        w = lane_mask.shape[1]
        center_x = w//2
        left_mask = nonzerox < center_x - self.CENTER_DEADZONE
        right_mask = nonzerox >= center_x + self.CENTER_DEADZONE
        left_pts = np.array(list(zip(nonzerox[left_mask], nonzeroy[left_mask])), np.int32) if np.any(left_mask) else None
        right_pts = np.array(list(zip(nonzerox[right_mask], nonzeroy[right_mask])), np.int32) if np.any(right_mask) else None
        return left_pts, right_pts

    # ---------------- resample line
    def resample_line(self, line_pts, num_samples):
        if line_pts is None or len(line_pts)<2: return None
        y_vals = np.linspace(line_pts[:,1].min(), line_pts[:,1].max(), num_samples)
        x_vals = np.interp(y_vals, line_pts[:,1], line_pts[:,0])
        return np.array(list(zip(x_vals,y_vals)), np.int32)

    # ---------------- fill drivable area
    def fill_drivable_area(self, frame, left_pts, right_pts):
        if left_pts is None and right_pts is None: return frame, None, None
        self.lane_history.append((left_pts,right_pts))
        left_avg = np.median([self.resample_line(l,self.NUM_SAMPLES) for l,_ in self.lane_history if l is not None],axis=0).astype(int)
        right_avg = np.median([self.resample_line(r,self.NUM_SAMPLES) for _,r in self.lane_history if r is not None],axis=0).astype(int)
        pts = np.vstack((left_avg,right_avg))
        lane_img = np.zeros_like(frame)
        cv2.fillPoly(lane_img,[pts],(0,255,0))
        return lane_img, left_avg, right_avg

    # ---------------- main driver
    def driver(self, cap):
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: return 1e10,0
        new_h = int(frame.shape[0]*self.RESIZE_WIDTH/frame.shape[1])
        frame = cv2.resize(frame,(self.RESIZE_WIDTH,new_h))

        lane_mask, inverse_matrix = self.combined_lane_mask(frame)
        left_pts, right_pts = self.extract_lane_candidates(lane_mask)
        lane_frame, left_avg, right_avg = self.fill_drivable_area(frame,left_pts,right_pts)

        fps = round(1.0/(time.time()-t0+1e-8),2)
        cv2.putText(lane_frame,f"FPS: {fps}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("Lane Detection CUDA",lane_frame)
        return 0,0  # simplified steering outputs

# -------------------- Main --------------------
if __name__=='__main__':
    VIDEO_PATH = r"D:/Users/Pranil/Github Repos/lane_detection/Lane_Data/WhatsApp Video 2025-10-07 at 07.32.26_b8b4d712.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = LaneDetectCUDA(resize_width=640)
    print("CUDA available:", detector.cuda_available)

    while True:
        detector.driver(cap)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
