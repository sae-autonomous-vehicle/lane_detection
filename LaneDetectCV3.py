# #!/usr/bin/env python3
# """
# Jetson-optimized Lane Detection (fully CUDA accelerated where possible)
# """

# import time
# import cv2
# import numpy as np
# from collections import deque

# # ---------------------------------------------------------
# # Helper: capture creation (supports CSI GStreamer on Jetson)
# # ---------------------------------------------------------
# def create_capture(source=0, width=1280, height=720, use_gst_for_csi=True):
#     if isinstance(source, str):
#         if source.lower() == 'csi' and use_gst_for_csi:
#             gst = (
#                 "nvarguscamerasrc ! video/x-raw(memory:NVMM),"
#                 f"width=(int){width},height=(int){height},framerate=(fraction)30/1 ! "
#                 "nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! "
#                 "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
#             )
#             return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
#         else:
#             return cv2.VideoCapture(source)
#     else:
#         return cv2.VideoCapture(int(source))


# # ---------------------------------------------------------
# # CUDA-optimized Lane Detection
# # ---------------------------------------------------------
# class LaneDetect:
#     def __init__(self, resize_width=640, smoothing_frames=8, num_samples=200):
#         self.RESIZE_WIDTH = resize_width
#         self.SMOOTHING_FRAMES = smoothing_frames
#         self.NUM_SAMPLES = num_samples
#         self.CENTER_DEADZONE = int(0.05 * self.RESIZE_WIDTH)
#         self.last_turn = 0
#         self.MIN_AREA = 1200
#         self.MIN_WIDTH = 170
#         self.MIN_LR_CURVE_RATIO = 0.99
#         self.SLOPE_THRESH = 1.73

#         self.lane_history = deque(maxlen=self.SMOOTHING_FRAMES)
#         self.center_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
#         self.left_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
#         self.right_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
#         self.curvature_history = deque(maxlen=self.SMOOTHING_FRAMES * 2)

#         # CUDA
#         self.cuda_available = False
#         try:
#             self.cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
#             self.cuda_available = self.cuda_count > 0
#             if self.cuda_available:
#                 self.stream = cv2.cuda.Stream()
#         except Exception:
#             self.cuda_count = 0
#             self.cuda_available = False
#             self.stream = None

#         # Morphology kernels
#         self.small_kernel_gpu = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1,
#                                                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
#         self.med_kernel_gpu = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1,
#                                                               cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

#     # ---------------- region of interest (GPU)
#     def region_of_interest(self, gmask):
#         h, w = gmask.size()
#         mask_cpu = np.zeros((h, w), np.uint8)
#         polygon = np.array([[(0, int(h*0.9)), (w, int(h*0.9)), (w, int(h*0.7)), (0, int(h*0.7))]], np.int32)
#         cv2.fillPoly(mask_cpu, polygon, 255)
#         gmask_mask = cv2.cuda_GpuMat()
#         gmask_mask.upload(mask_cpu)
#         gmask_roi = cv2.cuda.bitwise_and(gmask, gmask_mask)
#         return gmask_roi

#     # ---------------- perspective transform
#     def perspective_matrices(self, shape):
#         h, w = shape[:2]
#         src = np.float32([[w*0.2,h*0.7],[w*0.8,h*0.7],[w,h*0.9],[0,h*0.9]])
#         dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
#         return cv2.getPerspectiveTransform(src,dst), cv2.getPerspectiveTransform(dst,src)

#     # ---------------- CUDA lane mask
#     def combined_lane_mask(self, frame):
#         hls_gpu = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame), cv2.COLOR_BGR2HLS, stream=self.stream)
#         l_gpu, s_gpu = cv2.cuda.split(hls_gpu)
        
#         # S channel mask
#         s_bin = cv2.cuda.inRange(s_gpu, 130, 255)
#         l_bin = cv2.cuda.inRange(l_gpu, 220, 255)
#         color_bin = cv2.cuda.bitwise_or(s_bin, l_bin)

#         # Sobel X
#         gray_gpu = cv2.cuda.cvtColor(cv2.cuda_GpuMat().upload(frame), cv2.COLOR_BGR2GRAY)
#         blur_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5,5), 0)
#         sobel_gpu = cv2.cuda.Sobel(blur_gpu, cv2.CV_16S, 1, 0, ksize=3)
#         abs_sobel = cv2.cuda.abs(sobel_gpu)
#         scaled_sobel = cv2.cuda.convertScaleAbs(abs_sobel)
#         grad_bin = cv2.cuda.inRange(scaled_sobel, 70, 255)

#         # Combine
#         combined = cv2.cuda.bitwise_or(color_bin, grad_bin)
#         combined = self.region_of_interest(combined)

#         # Warp
#         matrix, inverse_matrix = self.perspective_matrices(frame.shape)
#         warped_gpu = cv2.cuda.warpPerspective(combined, matrix, (combined.size()[1], combined.size()[0]))
#         # Morphology
#         warped_gpu = self.small_kernel_gpu.apply(warped_gpu)
#         warped_gpu = self.med_kernel_gpu.apply(warped_gpu)

#         # Download final mask
#         warped_mask = warped_gpu.download()
#         return warped_mask, inverse_matrix

#     # ---------------- lane extraction
#     def extract_lane_candidates(self, lane_mask):
#         nonzero = lane_mask.nonzero()
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         if len(nonzeroy) == 0:
#             return None, None

#         w = lane_mask.shape[1]
#         center_x = w//2
#         left_mask = nonzerox < center_x - self.CENTER_DEADZONE
#         right_mask = nonzerox >= center_x + self.CENTER_DEADZONE
#         left_pts = np.array(list(zip(nonzerox[left_mask], nonzeroy[left_mask])), np.int32) if np.any(left_mask) else None
#         right_pts = np.array(list(zip(nonzerox[right_mask], nonzeroy[right_mask])), np.int32) if np.any(right_mask) else None
#         return left_pts, right_pts

#     # ---------------- resample line
#     def resample_line(self, line_pts, num_samples):
#         if line_pts is None or len(line_pts)<2: return None
#         y_vals = np.linspace(line_pts[:,1].min(), line_pts[:,1].max(), num_samples)
#         x_vals = np.interp(y_vals, line_pts[:,1], line_pts[:,0])
#         return np.array(list(zip(x_vals,y_vals)), np.int32)

#     # ---------------- fill drivable area
#     def fill_drivable_area(self, frame, left_pts, right_pts):
#         if left_pts is None and right_pts is None: return frame, None, None
#         self.lane_history.append((left_pts,right_pts))
#         left_avg = np.median([self.resample_line(l,self.NUM_SAMPLES) for l,_ in self.lane_history if l is not None],axis=0).astype(int)
#         right_avg = np.median([self.resample_line(r,self.NUM_SAMPLES) for _,r in self.lane_history if r is not None],axis=0).astype(int)
#         pts = np.vstack((left_avg,right_avg))
#         lane_img = np.zeros_like(frame)
#         cv2.fillPoly(lane_img,[pts],(0,255,0))
#         return lane_img, left_avg, right_avg

#     # ---------------- main driver
#     def driver(self, cap):
#         t0 = time.time()
#         ret, frame = cap.read()
#         if not ret: return 1e10,0
#         new_h = int(frame.shape[0]*self.RESIZE_WIDTH/frame.shape[1])
#         frame = cv2.resize(frame,(self.RESIZE_WIDTH,new_h))

#         lane_mask, inverse_matrix = self.combined_lane_mask(frame)
#         left_pts, right_pts = self.extract_lane_candidates(lane_mask)
#         lane_frame, left_avg, right_avg = self.fill_drivable_area(frame,left_pts,right_pts)

#         fps = round(1.0/(time.time()-t0+1e-8),2)
#         cv2.putText(lane_frame,f"FPS: {fps}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#         cv2.imshow("Lane Detection CUDA",lane_frame)
#         return 0,0  # simplified steering outputs

# # -------------------- Main --------------------
# if __name__=='__main__':
#     VIDEO_PATH = r"D:/Users/Pranil/Github Repos/lane_detection/Lane_Data/WhatsApp Video 2025-10-07 at 07.32.26_b8b4d712.mp4"
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     detector = LaneDetectCUDA(resize_width=640)
#     print("CUDA available:", detector.cuda_available)

#     while True:
#         detector.driver(cap)
#         if cv2.waitKey(1)&0xFF==ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#!/usr/bin/env python3
"""
Jetson-optimized Lane Detection (practical CUDA acceleration)

Notes:
- Designed to run on Jetson with OpenCV built with CUDA (e.g. OpenCV 4.5.4 with CUDA).
- Uses cv2.cuda for color conversion, resize, Gaussian blur, Sobel, and warpPerspective.
- Keeps robust CPU fallbacks for operations that are often unsupported in some builds:
  - color thresholding (inRange), adaptiveThreshold, connectedComponentsWithStats, morphologyEx.
- To maximize utilization: run larger RESIZE_WIDTH (640 or 720), ensure `sudo nvpmodel -m 0` and `sudo jetson_clocks`.
"""

import time
import cv2
import numpy as np
from collections import deque
import sys
import os

# ----------------- User tunables -----------------
FORCE_CPU = False        # If True, forces CPU-only fallbacks
USE_CSI_CAMERA = True   # Set True if using Jetson CSI camera, else False for file/USB
#PATH = "/path/to/your/video.mp4"  # used only if USE_CSI_CAMERA==False
RESIZE_WIDTH_DEFAULT = 640
# -------------------------------------------------

def create_capture(source=0, width=1280, height=720, use_gst_for_csi=True):
    """
    - source: 'csi' for nvarguscamerasrc, string path for files, int for device index
    """
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


class LaneDetectCUDA:
    def __init__(self, resize_width=RESIZE_WIDTH_DEFAULT, smoothing_frames=8, num_samples=200):
        # parameters (kept as in your original)
        self.RESIZE_WIDTH = resize_width
        self.CENTER_DEADZONE = int(0.05 * self.RESIZE_WIDTH)
        self.SMOOTHING_FRAMES = smoothing_frames
        self.MORPH_KERNEL = (5, 5)
        self.MIN_AREA = 1200
        self.MIN_WIDTH = 170
        self.NUM_SAMPLES = num_samples
        self.MIN_LR_CURVE_RATIO = 0.99
        self.SLOPE_THRESH = 1.73
        self.last_turn = 0

        # histories
        self.lane_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.center_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.left_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.right_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.curvature_history = deque(maxlen=self.SMOOTHING_FRAMES * 2)

        # CUDA capability detection
        self.cuda_available = False
        self.cuda_count = 0
        self.stream = None
        if not FORCE_CPU:
            try:
                self.cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
                self.cuda_available = (self.cuda_count > 0)
                if self.cuda_available:
                    # instantiate a stream for async operations
                    self.stream = cv2.cuda.Stream()
            except Exception:
                self.cuda_available = False
                self.cuda_count = 0
                self.stream = None

        # CPU kernels for morphology (used as fallback)
        self.small_kernel_cpu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.med_kernel_cpu = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL)

        # print status
        print(f"[LaneDetectCUDA] CUDA available: {self.cuda_available}, devices: {self.cuda_count}")

    # ---------- ROI ----------
    def region_of_interest_cpu(self, img):
        h, w = img.shape
        polygon = np.array([[(0, int(h * 0.90)), (w, int(h * 0.90)), (w, int(h * 0.7)), (0, int(h * 0.7))]],
                           np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    # ---------- Dynamic THresh ----------
    def detect_lighting_conditions(self, frame):
        """
        Detect lighting conditions to adjust parameters
        (CPU - cheap relative to full pipeline)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        bright_pixels = np.sum(gray > 240) / gray.size
        dark_pixels = np.sum(gray < 10) / gray.size

        if mean_intensity < 70:
            condition = "night"
        elif mean_intensity > 150 or bright_pixels > 0.05:
            condition = "overexposed"
        elif dark_pixels > 0.1:
            condition = "shadows"
        elif std_intensity < 30:
            condition = "low_contrast"
        else:
            condition = "normal"

        return {
            'condition': condition,
            'brightness': float(mean_intensity),
            'contrast': float(std_intensity),
            'bright_ratio': float(bright_pixels),
            'dark_ratio': float(dark_pixels)
        }

    def get_lighting_based_parameters(self, lighting_info):
        condition = lighting_info['condition']
        if condition == "night":
            return {'s_range': (100, 255), 'l_range': (150, 255), 'sobel_range': (40, 255), 'blur_kernel': (7, 7),
                    'morph_iterations': 3}
        elif condition == "shadows":
            return {'s_range': (110, 255), 'l_range': (200, 255), 'sobel_range': (60, 255), 'blur_kernel': (5, 5),
                    'morph_iterations': 2}
        elif condition == "low_contrast":
            return {'s_range': (120, 255), 'l_range': (200, 255), 'sobel_range': (50, 255), 'blur_kernel': (5, 5),
                    'morph_iterations': 2}
        elif condition == "overexposed":
            return {'s_range': (150, 255), 'l_range': (240, 255), 'sobel_range': (80, 255), 'blur_kernel': (3, 3),
                    'morph_iterations': 1}
        else:
            return {'s_range': (130, 255), 'l_range': (220, 255), 'sobel_range': (70, 255), 'blur_kernel': (5, 5),
                    'morph_iterations': 2}

    # ---------- Perspective ----------
    def perspective_transform(self, img):
        height, width = img.shape[:2]
        src_points = np.float32([
            [width * 0.20, height * 0.7],
            [width * 0.80, height * 0.7],
            [width, height * 0.9],
            [0, height * 0.9]
        ])
        dst_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        warped_img = cv2.warpPerspective(img, matrix, (width, height))
        return warped_img, matrix, inverse_matrix

    # ---------------- core combined mask ----------------
    def combined_lane_mask(self, frame):
        """
        High-level strategy:
         - Use CUDA for: resize, cvtColor->GRAY, GaussianBlur, Sobel X, warpPerspective
         - Use CPU for: HLS color thresholds (inRange), morphology (morphologyEx) and connected components
        Reason: inRange + connectedComponents are simple to compute on the small mask and are widely supported.
        """
        lighting = self.detect_lighting_conditions(frame)
        params = self.get_lighting_based_parameters(lighting)

        # ---------- Resize early (CPU) ----------
        # (frame is already resized in driver() but keep in case)
        # frame = cv2.resize(frame, (self.RESIZE_WIDTH, int(frame.shape[0]*self.RESIZE_WIDTH/frame.shape[1])))

        # ---------- HLS color thresholds (CPU) ----------
        # Doing HLS on CPU is cheap compared to whole pipeline and more portable.
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s = hls[:, :, 2]

        s_range = params['s_range']
        l_range = params['l_range']

        s_binary = cv2.inRange(s, s_range[0], s_range[1])
        if lighting['condition'] == "overexposed":
            l_binary = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -40)
        else:
            l_binary = cv2.inRange(l, l_range[0], l_range[1])

        color_binary = cv2.bitwise_or(s_binary, l_binary)  # CPU mask

        # ---------- GPU path for GRADIENT ----------
        # We'll try to compute Sobel on GPU (blur -> sobel -> abs -> convertScaleAbs)
        scaled_sobel = None
        try:
            if self.cuda_available:
                # upload frame once (BGR)
                gframe = cv2.cuda_GpuMat()
                gframe.upload(frame, stream=self.stream)

                # convert to gray on GPU
                ggray = cv2.cuda.cvtColor(gframe, cv2.COLOR_BGR2GRAY, stream=self.stream)

                # Gaussian blur (GPU)
                kx, ky = params['blur_kernel']
                # createGaussianFilter expects odd kernel sizes; ensure that
                if kx % 2 == 0: kx += 1
                if ky % 2 == 0: ky += 1
                try:
                    g_blur_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (kx, ky), 0)
                    g_blur = g_blur_filter.apply(ggray, stream=self.stream)
                except Exception:
                    # fallback to simple upload -> CPU blur if this API missing
                    g_blur = ggray

                # Sobel X (GPU) -> CV_16S
                try:
                    g_sobel = cv2.cuda.Sobel(g_blur, cv2.CV_16S, 1, 0, ksize=3, stream=self.stream)
                    g_abs = cv2.cuda.abs(g_sobel, stream=self.stream)  # absolute value
                    g_scaled = cv2.cuda.convertScaleAbs(g_abs, stream=self.stream)
                    # download scaled sobel to CPU (we need max and inRange)
                    scaled_sobel = g_scaled.download(stream=self.stream)
                    if self.stream: self.stream.waitForCompletion()
                except Exception:
                    # fallback to CPU Sobel if some GPU function missing
                    scaled_sobel = None

            else:
                scaled_sobel = None

        except Exception as e:
            # any GPU errors -> fallback to CPU for gradient
            print("[WARN] GPU gradient failed, falling back to CPU Sobel:", e)
            scaled_sobel = None

        # If GPU path failed or disabled, do CPU Sobel (reliable)
        if scaled_sobel is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, params['blur_kernel'], 0)
            sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobelx = np.absolute(sobelx)
            max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1
            scaled_sobel = np.uint8(255 * abs_sobelx / max_val)

        # ---------- threshold gradient (CPU) ----------
        sobel_range = params['sobel_range']
        grad_binary = cv2.inRange(scaled_sobel, sobel_range[0], sobel_range[1])

        # ---------- Combine masks (CPU) ----------
        combined = cv2.bitwise_or(color_binary, grad_binary)

        # ---------- ROI mask (CPU) ----------
        combined = self.region_of_interest_cpu(combined)

        # ---------- Perspective warp ----------
        # We'll warp on GPU when available to reduce CPU->GPU copies:
        warped_combined = None
        _, matrix, inverse_matrix = None, None, None
        matrix = None
        inverse_matrix = None
        try:
            # compute matrices on CPU (fast)
            _, matrix, inverse_matrix = self.perspective_transform(combined)
            if self.cuda_available:
                # upload combined mask and warp on GPU
                gmask = cv2.cuda_GpuMat()
                gmask.upload(combined, stream=self.stream)
                try:
                    gwarp = cv2.cuda.warpPerspective(gmask, matrix, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_LINEAR, stream=self.stream)
                    warped_combined = gwarp.download(stream=self.stream)
                    if self.stream: self.stream.waitForCompletion()
                except Exception:
                    # if gpu warp fails, fallback to cpu warp
                    warped_combined = cv2.warpPerspective(combined, matrix, (combined.shape[1], combined.shape[0]))
            else:
                warped_combined = cv2.warpPerspective(combined, matrix, (combined.shape[1], combined.shape[0]))
        except Exception as e:
            # fallback: cpu warp (should rarely fail)
            warped_combined = cv2.warpPerspective(combined, matrix, (combined.shape[1], combined.shape[0]))

        # ---------- Morphology (CPU) ----------
        morph_iterations = params['morph_iterations']
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_OPEN, self.small_kernel_cpu,
                                           iterations=morph_iterations)
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_CLOSE, self.med_kernel_cpu,
                                           iterations=morph_iterations)

        # ---------- Remove small components (CPU) ----------
        min_component_area = 50 if lighting['condition'] == "night" else 100
        warped_combined = self.remove_small_components(warped_combined, min_area=min_component_area)

        return warped_combined, lighting, inverse_matrix

    def remove_small_components(self, binary_img, min_area=50):
        # CPU-only: filter connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        output = np.zeros_like(binary_img)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255
        return output

    # ---------- Lane Candidate Extraction ----------
    def extract_lane_candidates(self, lane_mask):
        nonzero = lane_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if len(nonzeroy) == 0:
            return None, None

        w = lane_mask.shape[1]
        center_x = w // 2

        left_candidates = nonzerox[nonzerox < center_x - self.CENTER_DEADZONE], nonzeroy[nonzerox < center_x - self.CENTER_DEADZONE]
        right_candidates = nonzerox[nonzerox >= center_x + self.CENTER_DEADZONE], nonzeroy[nonzerox >= center_x + self.CENTER_DEADZONE]

        def line_confidence(xs, ys, max_width=200):
            if len(xs) < 20:
                return 0
            dx = xs[-1] - xs[0]; dy = ys[-1] - ys[0]
            length = np.sqrt(dx ** 2 + dy ** 2)
            widthx, widthy = np.std(xs), np.std(ys)
            if widthx > max_width and widthy > max_width:
                return 0
            return length

        left_conf = line_confidence(*left_candidates)
        right_conf = line_confidence(*right_candidates)

        left_pts = np.array([[int(np.median(left_candidates[0][left_candidates[1]==y])), y] for y in np.unique(left_candidates[1])], np.int32) if left_conf != 0 else None
        right_pts = np.array([[int(np.median(right_candidates[0][right_candidates[1]==y])), y] for y in np.unique(right_candidates[1])], np.int32) if right_conf != 0 else None

        return left_pts, right_pts

    # ---------- Resample ----------
    def resample_line(self, line_pts, num_samples):
        if line_pts is None or len(line_pts) < 2:
            return None
        y_vals = np.linspace(line_pts[:, 1].min(), line_pts[:, 1].max(), num_samples)
        x_vals = np.interp(y_vals, line_pts[:, 1], line_pts[:, 0])
        return np.array(list(zip(x_vals, y_vals)), dtype=np.int32)

    # ---------- Drivable Area ----------
    def fill_drivable_area(self, frame, left_pts, right_pts):
        if left_pts is None and right_pts is None:
            return frame, None, None

        self.lane_history.append((left_pts, right_pts))

        resampled_left = [self.resample_line(l, self.NUM_SAMPLES) for l, _ in self.lane_history if l is not None]
        resampled_right = [self.resample_line(r, self.NUM_SAMPLES) for _, r in self.lane_history if r is not None]
        if not resampled_left and not resampled_right:
            return frame, None, None

        left_avg = np.median(resampled_left, axis=0).astype(int) if resampled_left else None
        right_avg = np.median(resampled_right, axis=0).astype(int)[::-1] if resampled_right else None

        if left_avg is not None and right_avg is None:
            min_y = np.min(left_avg[:, 1]); max_y = np.max(left_avg[:, 1])
            right_avg = np.array([[self.RESIZE_WIDTH - 1, i] for i in np.linspace(min_y, max_y, 200)], dtype=int)
        elif left_avg is None and right_avg is not None:
            min_y = np.min(right_avg[:, 1]); max_y = np.max(right_avg[:, 1])
            left_avg = np.array([[0, i] for i in np.linspace(min_y, max_y, 200)], dtype=int)
        elif left_avg is None and right_avg is None:
            return frame, None, None

        pts = np.vstack((left_avg, right_avg))
        area = cv2.contourArea(pts)
        avg_width = np.mean(np.abs(right_avg[:, 0] - left_avg[:, 0]))

        min_area_threshold = self.MIN_AREA * 0.3 if (len(resampled_left) == 0 or len(resampled_right) == 0) else self.MIN_AREA
        min_width_threshold = self.MIN_WIDTH * 0.5 if (len(resampled_left) == 0 or len(resampled_right) == 0) else self.MIN_WIDTH

        if area < min_area_threshold or avg_width < min_width_threshold:
            return frame, None, None

        return frame, left_avg, right_avg

    # ---------- Smooth polynomial ----------
    def smooth_with_polynomial(self, points, degree=2):
        if points is None or len(points) < degree + 1:
            return points
        try:
            y_vals = points[:, 1]; x_vals = points[:, 0]
            sort_idx = np.argsort(y_vals); y_sorted = y_vals[sort_idx]; x_sorted = x_vals[sort_idx]
            poly_coeffs = np.polyfit(y_sorted, x_sorted, degree); poly = np.poly1d(poly_coeffs)
            y_smooth = np.linspace(y_sorted.min(), y_sorted.max(), len(points)); x_smooth = poly(y_smooth)
            return np.array(list(zip(x_smooth, y_smooth)), dtype=np.int32)
        except:
            return points

    # ---------- Draw center line polynomial (with look-ahead) ----------
    def draw_center_line_polynomial(self, frame, left_pts, right_pts, inverse_matrix):
        if left_pts is None or right_pts is None:
            return frame, self.left_line_history[-1] if self.left_line_history else None, self.right_line_history[-1] if self.right_line_history else None, -1, 0

        smooth_left = self.smooth_with_polynomial(left_pts)
        smooth_right = self.smooth_with_polynomial(right_pts)
        if smooth_left is None or smooth_right is None or len(smooth_left) < 2 or len(smooth_right) < 2:
            return frame, None, None, -1, 0

        self.left_line_history.append(smooth_left); self.right_line_history.append(smooth_right)
        avg_left = np.mean(self.left_line_history, axis=0).astype(np.int32)
        avg_right = np.mean(self.right_line_history, axis=0).astype(np.int32)

        # look-ahead point (95% down the image)
        h, w = frame.shape[:2]
        look_ahead_y = h * 0.95
        left_y_vals = avg_left[:, 1]; left_x_vals = avg_left[:, 0]
        right_y_vals = avg_right[:, 1]; right_x_vals = avg_right[:, 0]
        left_sort_idx = np.argsort(left_y_vals); right_sort_idx = np.argsort(right_y_vals)
        left_x_at_lookahead = np.interp(look_ahead_y, left_y_vals[left_sort_idx], left_x_vals[left_sort_idx])
        right_x_at_lookahead = np.interp(look_ahead_y, right_y_vals[right_sort_idx], right_x_vals[right_sort_idx])
        center_x_at_lookahead = (left_x_at_lookahead + right_x_at_lookahead) / 2.0
        lane_offset = round((w / 2 - center_x_at_lookahead) / w, 3)

        l_curve, _ = self.calculate_curvature(avg_left, y_eval=look_ahead_y)
        r_curve, _ = self.calculate_curvature(avg_right, y_eval=look_ahead_y)
        self.curvature_history.append(max(min(-(l_curve + r_curve) / 2, 1e8), -1e8))
        curvature = round(np.mean(self.curvature_history), 3)
        self.last_turn = np.sign(curvature)

        center_pts = np.array([[(lx + rx) // 2, y] for (lx, y), (rx, _) in zip(avg_left, avg_right)], np.int32)
        smooth_center = self.smooth_with_polynomial(center_pts)

        lane_img = np.zeros_like(frame)
        drivable_area_pts = np.vstack((avg_left, np.flipud(avg_right)))
        cv2.fillPoly(lane_img, [drivable_area_pts.astype(np.int32)], (0, 255, 0))
        cv2.polylines(lane_img, [smooth_center.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=3)
        cv2.polylines(lane_img, [avg_left.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=3)
        cv2.polylines(lane_img, [avg_right.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=3)

        # Dewarp (CPU)
        dewarped_lanes = cv2.warpPerspective(lane_img, inverse_matrix, (frame.shape[1], frame.shape[0]))
        final_frame = cv2.addWeighted(frame, 1, dewarped_lanes, 0.3, 0)

        return final_frame, avg_left, avg_right, lane_offset, curvature

    # ---------- Curvature ----------
    def calculate_curvature(self, pts, fit=[], y_eval=None):
        if pts is None or len(pts) < 3:
            return 0, []
        ym_per_pix = 0.00064; xm_per_pix = 0.00064
        if y_eval is None:
            y_eval = np.max(pts[:, 1])
        try:
            fit_cr = fit if len(fit) > 0 else np.polyfit(pts[:, 1] * ym_per_pix, pts[:, 0] * xm_per_pix, 2)
            curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
            return curverad, fit_cr
        except np.linalg.LinAlgError:
            return 0, []

    # ---------- driver ----------
    def driver(self, cap):
        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            # return extremely large curvature if read fails
            return 1e10, 0

        # resize for predictable load (do on CPU — cheap)
        new_h = int(frame.shape[0] * self.RESIZE_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (self.RESIZE_WIDTH, new_h))

        lane_mask, lighting, inverse_matrix = self.combined_lane_mask(frame)
        left_pts, right_pts = self.extract_lane_candidates(lane_mask)
        lane_frame, left_avg, right_avg = self.fill_drivable_area(frame, left_pts, right_pts)
        lane_frame, smoothl, smoothr, lane_offset, curve_rad = self.draw_center_line_polynomial(lane_frame, left_avg, right_avg, inverse_matrix)

        t1 = time.time()
        fps = round(1.0 / (t1 - t0 + 1e-8), 2)
        cv2.putText(lane_frame, f"FPS: {fps}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(lane_frame, f"Lighting: {lighting['condition']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(lane_frame, f"RoC: {curve_rad:.2f} m", (lane_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(lane_frame, f"OffSet: {lane_offset}", (lane_frame.shape[1] - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Debug ROI polygon (visual)
        h, w, _ = lane_frame.shape
        roi_polygon_to_draw = np.array([[(0, int(h * 0.90)), (w, int(h * 0.90)), (w, int(h * 0.75)), (0, int(h * 0.75))]], dtype=np.int32)
        cv2.polylines(lane_frame, [roi_polygon_to_draw], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.imshow("Lane Detection", lane_frame)
        cv2.imshow("Lane Mask", cv2.resize(lane_mask, (int(lane_mask.shape[1] * 0.5), int(lane_mask.shape[0] * 0.5))))

        return curve_rad, lane_offset


# -------------------- main --------------------
'''if __name__ == '__main__':
    # pick source
    if USE_CSI_CAMERA:
        SOURCE = 'csi'
    else:
        if not os.path.exists(VIDEO_PATH):
            print("Error: set VIDEO_PATH to an existing video file or enable USE_CSI_CAMERA=True")
            sys.exit(1)
        # Use hardware decode pipeline if you want (uncomment & set path)
        #FILESRC = f"filesrc location={VIDEO_PATH} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        #cap = cv2.VideoCapture(FILESRC, cv2.CAP_GSTREAMER)
        SOURCE = VIDEO_PATH

    cap = create_capture(SOURCE, width=1280, height=720)
    if not cap.isOpened():
        print("Failed to open capture. Exiting.")
        sys.exit(1)

    detector = LaneDetectCUDA(resize_width=RESIZE_WIDTH_DEFAULT)
    print("Starting loop. Press 'q' to quit.")

    while True:
        curvature, offset = detector.driver(cap)
        # your autopilot logic can call steerController(curvature, offset) etc.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()'''
