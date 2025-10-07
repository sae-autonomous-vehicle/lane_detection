# #!/usr/bin/env python3
"""
Jetson-optimized Lane Detection (CUDA-accelerated where possible)

Notes:
- Requires OpenCV built with CUDA (and DNN/CuDNN if using DNN later).
- On Jetson: run `sudo nvpmodel -m 0 && sudo jetson_clocks` for max perf (watch thermals).
- If CUDA not available, falls back to CPU implementation (keeps original logic).
"""

import time
import cv2
import numpy as np
from collections import deque

# ---------------------------------------------------------
# Helper: capture creation (supports CSI GStreamer on Jetson)
# ---------------------------------------------------------
def create_capture(source=0, width=1280, height=720, use_gst_for_csi=True):
    """
    source:
      - int => device index
      - string path => file path
      - 'csi' => Jetson CSI camera (uses nvarguscamerasrc)
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


# ---------------------------------------------------------
# Main class (CUDA-enabled when possible)
# ---------------------------------------------------------
class LaneDetectCUDA:
    def __init__(self, resize_width=480, smoothing_frames=8, num_samples=200):
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

        self.lane_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.center_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.left_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.right_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.curvature_history = deque(maxlen=self.SMOOTHING_FRAMES * 2)

        # CUDA availability & objects
        self.cuda_available = False
        try:
            self.cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_available = self.cuda_count > 0
        except Exception:
            self.cuda_count = 0
            self.cuda_available = False

        if self.cuda_available:
            # create a stream for async operations
            try:
                self.stream = cv2.cuda.Stream()
            except Exception:
                self.stream = None
        else:
            self.stream = None

        # Precreate morph kernels (CPU) and small helpers
        self.small_kernel_cpu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.med_kernel_cpu = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL)

    # ---------------- ROI - keep CPU simple (binary mask small) ----------------
    def region_of_interest(self, img_bin):
        # img_bin is assumed to be single channel (uint8)
        h, w = img_bin.shape
        polygon = np.array([[(0, int(h * 0.90)), (w, int(h * 0.90)), (w, int(h * 0.7)), (0, int(h * 0.7))]],
                           np.int32)
        mask = np.zeros_like(img_bin)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img_bin, mask)

    # --------------- Lighting detection (CPU) ----------------------
    def detect_lighting_conditions(self, frame):
        # frame: BGR CPU numpy array (small - after resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        bright_pixels = float(np.sum(gray > 240)) / gray.size
        dark_pixels = float(np.sum(gray < 10)) / gray.size

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
            'brightness': mean_intensity,
            'contrast': std_intensity,
            'bright_ratio': bright_pixels,
            'dark_ratio': dark_pixels
        }

    def get_lighting_based_parameters(self, lighting_info):
        cond = lighting_info['condition']
        if cond == "night":
            return {'s_range': (100, 255), 'l_range': (150, 255), 'sobel_range': (40, 255), 'blur_kernel': (7, 7),
                    'morph_iterations': 3}
        if cond == "shadows":
            return {'s_range': (110, 255), 'l_range': (200, 255), 'sobel_range': (60, 255), 'blur_kernel': (5, 5),
                    'morph_iterations': 2}
        if cond == "low_contrast":
            return {'s_range': (120, 255), 'l_range': (200, 255), 'sobel_range': (50, 255), 'blur_kernel': (5, 5),
                    'morph_iterations': 2}
        if cond == "overexposed":
            return {'s_range': (150, 255), 'l_range': (240, 255), 'sobel_range': (80, 255), 'blur_kernel': (3, 3),
                    'morph_iterations': 1}
        return {'s_range': (130, 255), 'l_range': (220, 255), 'sobel_range': (70, 255), 'blur_kernel': (5, 5),
                'morph_iterations': 2}

    # ---------------- perspective transform (CPU matrices, GPU warp used) ----------------
    def perspective_transform_matrices(self, img_shape):
        height, width = img_shape[:2]
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
        return matrix, inverse_matrix

    # ---------------- core combined mask (GPU where possible) ----------------
    def combined_lane_mask(self, frame):
        """
        frame: CPU BGR numpy (resized)
        returns: binary mask (CPU uint8), lighting_info, inverse_matrix
        """
        lighting = self.detect_lighting_conditions(frame)
        params = self.get_lighting_based_parameters(lighting)

        hls_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # H, L, S = hls_cpu[:,:,0], hls_cpu[:,:,1], hls_cpu[:,:,2]
        l = hls_cpu[:, :, 1]
        s = hls_cpu[:, :, 2]

        # S and L masks: use CPU (cheap) -> these are small after resize
        s_binary = cv2.inRange(s, params['s_range'][0], params['s_range'][1])

        if lighting['condition'] == "overexposed":
            # adaptive threshold on L: use CPU as CUDA adaptiveThreshold not available
            l_adapt = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -40)
            l_binary = l_adapt
        else:
            l_binary = cv2.inRange(l, params['l_range'][0], params['l_range'][1])

        color_binary = cv2.bitwise_or(s_binary, l_binary)

        # For Sobel & blur, use CUDA path if available
        if self.cuda_available:
            # upload grayscale to GPU
            gray_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gmat = cv2.cuda_GpuMat()
            gmat.upload(gray_cpu, stream=self.stream)

            # Gaussian blur using CUDA filter
            kx, ky = params['blur_kernel']
            # create filter on-the-fly (kernel size must be odd)
            try:
                gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (kx, ky), sigma1=0)
                g_blur = gauss.apply(gmat, stream=self.stream)
            except Exception:
                # fallback to CPU blur if CUDA Gaussian unavailable
                g_blur = gmat

            # Sobel (CUDA supports Sobel for CV_8U -> CV_16S)
            try:
                # Use Sobel in X
                g_sobel = cv2.cuda.Sobel(g_blur, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT,
                                         stream=self.stream)
                # absolute & convert
                g_abs = cv2.cuda.abs(g_sobel)
                # convertScaleAbs equivalent
                g_scaled = cv2.cuda.convertScaleAbs(g_abs, alpha=1.0, beta=0.0, stream=self.stream)
                # normalize to 0-255 might be needed. We'll download and scale on CPU because find max is cheap CPU.
                scaled_sobel = g_scaled.download(stream=self.stream)
                # ensure stream completion before using scaled_sobel
                if self.stream:
                    self.stream.waitForCompletion()
                max_val = np.max(scaled_sobel) if np.max(scaled_sobel) != 0 else 1
                scaled_sobel = np.uint8(255 * (scaled_sobel.astype(np.float32) / float(max_val)))
            except Exception:
                # fallback: CPU Sobel
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, params['blur_kernel'], 0)
                sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
                abs_sobelx = np.absolute(sobelx)
                max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1
                scaled_sobel = np.uint8(255 * abs_sobelx / max_val)

        else:
            # CPU path
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, params['blur_kernel'], 0)
            sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobelx = np.absolute(sobelx)
            max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1
            scaled_sobel = np.uint8(255 * abs_sobelx / max_val)

        # threshold sobel
        sobel_range = params['sobel_range']
        grad_binary = cv2.inRange(scaled_sobel, sobel_range[0], sobel_range[1])

        # Combine color + grad
        combined = cv2.bitwise_or(color_binary, grad_binary)

        # ROI (CPU)
        combined = self.region_of_interest(combined)

        # Perspective transform - we will warp mask (CPU) using GPU if available
        matrix, inverse_matrix = self.perspective_transform_matrices(frame.shape)
        if self.cuda_available:
            # upload combined mask to GPU
            g_mask = cv2.cuda_GpuMat()
            g_mask.upload(combined, stream=self.stream)
            try:
                g_warp = cv2.cuda.warpPerspective(g_mask, matrix, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_LINEAR,
                                                  stream=self.stream)
                warped_combined = g_warp.download(stream=self.stream)
                if self.stream:
                    self.stream.waitForCompletion()
            except Exception:
                warped_combined = cv2.warpPerspective(combined, matrix, (combined.shape[1], combined.shape[0]))
        else:
            warped_combined = cv2.warpPerspective(combined, matrix, (combined.shape[1], combined.shape[0]))

        # Morphology - do on CPU but on the warped mask (small)
        morph_iterations = params['morph_iterations']
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_OPEN, self.small_kernel_cpu,
                                           iterations=morph_iterations)
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_CLOSE, self.med_kernel_cpu,
                                           iterations=morph_iterations)

        # Remove small components (we need connectedComponentsWithStats -> CPU)
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

    # ---------------- lane candidate & resampling (CPU) ----------------
    def extract_lane_candidates(self, lane_mask):
        nonzero = lane_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if len(nonzeroy) == 0:
            return None, None

        w = lane_mask.shape[1]
        center_x = w // 2

        left_mask = nonzerox < center_x - self.CENTER_DEADZONE
        right_mask = nonzerox >= center_x + self.CENTER_DEADZONE

        left_xs = nonzerox[left_mask]
        left_ys = nonzeroy[left_mask]
        right_xs = nonzerox[right_mask]
        right_ys = nonzeroy[right_mask]

        def line_confidence(xs, ys, max_width=200):
            if len(xs) < 20:
                return 0
            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
            slope = abs(dx / (dy + 1e-6))
            length = np.sqrt(dx ** 2 + dy ** 2)
            widthx, widthy = np.std(xs), np.std(ys)
            if widthx > max_width and widthy > max_width:
                return 0
            return length

        left_conf = line_confidence(left_xs, left_ys)
        right_conf = line_confidence(right_xs, right_ys)

        left_pts = None
        right_pts = None
        if left_conf != 0:
            y_unique = np.unique(left_ys)
            pts = []
            for y in y_unique:
                xs_at_y = left_xs[left_ys == y]
                if xs_at_y.size:
                    pts.append([int(np.median(xs_at_y)), int(y)])
            left_pts = np.array(pts, np.int32) if pts else None

        if right_conf != 0:
            y_unique = np.unique(right_ys)
            pts = []
            for y in y_unique:
                xs_at_y = right_xs[right_ys == y]
                if xs_at_y.size:
                    pts.append([int(np.median(xs_at_y)), int(y)])
            right_pts = np.array(pts, np.int32) if pts else None

        return left_pts, right_pts

    def resample_line(self, line_pts, num_samples):
        if line_pts is None or len(line_pts) < 2:
            return None
        y_vals = np.linspace(line_pts[:, 1].min(), line_pts[:, 1].max(), num_samples)
        x_vals = np.interp(y_vals, line_pts[:, 1], line_pts[:, 0])
        return np.array(list(zip(x_vals, y_vals)), dtype=np.int32)

    # ---------------- drivable area (CPU heavy geometry) ----------------
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

        # estimate opposite lane if missing
        if left_avg is not None and right_avg is None:
            min_y = np.min(left_avg[:, 1])
            max_y = np.max(left_avg[:, 1])
            right_avg = np.array([[self.RESIZE_WIDTH - 1, i] for i in np.linspace(min_y, max_y, 200)], dtype=int)
        elif left_avg is None and right_avg is not None:
            min_y = np.min(right_avg[:, 1])
            max_y = np.max(right_avg[:, 1])
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

    # ---------------- polynomial smoothing & drawing (CPU, small geometry) ----------------
    def smooth_with_polynomial(self, points, degree=2):
        if points is None or len(points) < degree + 1:
            return points
        try:
            y_vals = points[:, 1]
            x_vals = points[:, 0]
            sort_idx = np.argsort(y_vals)
            y_sorted = y_vals[sort_idx]
            x_sorted = x_vals[sort_idx]
            poly_coeffs = np.polyfit(y_sorted, x_sorted, degree)
            poly = np.poly1d(poly_coeffs)
            y_smooth = np.linspace(y_sorted.min(), y_sorted.max(), len(points))
            x_smooth = poly(y_smooth)
            return np.array(list(zip(x_smooth, y_smooth)), dtype=np.int32)
        except Exception:
            return points

    def draw_center_line_polynomial(self, frame, left_pts, right_pts, inverse_matrix):
        if left_pts is None or right_pts is None:
            return frame, left_pts, right_pts, -1, 0

        smooth_left = self.smooth_with_polynomial(left_pts)
        smooth_right = self.smooth_with_polynomial(right_pts)
        if smooth_left is None or smooth_right is None:
            return frame, None, None, -1, 0

        bottom_y = max(smooth_left[:, 1].max(), smooth_right[:, 1].max())
        top_y = min(smooth_left[:, 1].min(), smooth_right[:, 1].min())

        y_points = np.linspace(top_y, bottom_y, 10)
        left_x_interp = np.interp(y_points, smooth_left[:, 1], smooth_left[:, 0])
        right_x_interp = np.interp(y_points, smooth_right[:, 1], smooth_right[:, 0])
        lane_widths = right_x_interp - left_x_interp

        self.left_line_history.append(smooth_left)
        self.right_line_history.append(smooth_right)

        avg_left = np.mean(self.left_line_history, axis=0).astype(np.int32)
        avg_right = np.mean(self.right_line_history, axis=0).astype(np.int32)

        l_curve, l_fit = self.calculate_curvature(avg_left)
        r_curve, r_fit = self.calculate_curvature(avg_right)

        # some cleanup if curves disagree (keeps original logic)
        if np.sign(l_curve) != np.sign(r_curve) and (abs(l_curve / r_curve) < self.MIN_LR_CURVE_RATIO or abs(r_curve / l_curve) < self.MIN_LR_CURVE_RATIO):
            if self.last_turn >= 0:
                min_y = 0
                while np.sign(l_curve) != np.sign(r_curve) and min_y < len(right_pts) - 2:
                    r_curve, _ = self.calculate_curvature(right_pts[min_y:], r_fit)
                    min_y += 1
                right_pts = right_pts[min_y:]
            else:
                min_y = 0
                while np.sign(l_curve) != np.sign(r_curve) and min_y < len(left_pts) - 2:
                    l_curve, _ = self.calculate_curvature(left_pts[min_y:], l_fit)
                    min_y += 1
                left_pts = left_pts[min_y:]

        self.curvature_history.append(max(min(-(l_curve + r_curve) / 2, 1e8), -1e8))
        curvature = round(np.mean(self.curvature_history), 3)
        self.last_turn = np.sign(curvature)

        center_pts = np.array([[(lx + rx) // 2, y] for (lx, y), (rx, _) in zip(smooth_left, smooth_right)], np.int32)
        smooth_center = self.smooth_with_polynomial(center_pts)
        lane_offset = round((self.RESIZE_WIDTH / 2 - np.mean(smooth_center[:, 0])) / self.RESIZE_WIDTH, 3)

        lane_img = np.zeros_like(frame)
        drivable_area_pts = np.vstack((smooth_left, np.flipud(smooth_right)))
        cv2.fillPoly(lane_img, [drivable_area_pts.astype(np.int32)], (0, 255, 0))
        cv2.polylines(lane_img, [smooth_center.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=3)
        cv2.polylines(lane_img, [smooth_left.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=3)
        cv2.polylines(lane_img, [smooth_right.reshape((-1, 1, 2)).astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=3)

        # Dewarp with inverse matrix (use GPU warp if available)
        try:
            if self.cuda_available:
                g_lane_img = cv2.cuda_GpuMat()
                g_lane_img.upload(lane_img, stream=self.stream)
                g_dewarp = cv2.cuda.warpPerspective(g_lane_img, inverse_matrix, (frame.shape[1], frame.shape[0]),
                                                    flags=cv2.INTER_LINEAR, stream=self.stream)
                dewarped_lanes = g_dewarp.download(stream=self.stream)
                if self.stream:
                    self.stream.waitForCompletion()
            else:
                dewarped_lanes = cv2.warpPerspective(lane_img, inverse_matrix, (frame.shape[1], frame.shape[0]))
        except Exception:
            dewarped_lanes = cv2.warpPerspective(lane_img, inverse_matrix, (frame.shape[1], frame.shape[0]))

        final_frame = cv2.addWeighted(frame, 1, dewarped_lanes, 0.3, 0)
        l_curve, _ = self.calculate_curvature(smooth_left)
        r_curve, _ = self.calculate_curvature(smooth_right)
        self.curvature_history.append(max(min(-(l_curve + r_curve) / 2, 1e8), -1e8))
        curvature = round(np.mean(self.curvature_history), 3)

        return final_frame, smooth_left, smooth_right, lane_offset, curvature

    def calculate_curvature(self, pts, fit=[]):
        if pts is None or len(pts) < 3:
            return 0, []
        ym_per_pix = 0.00064
        xm_per_pix = 0.00064
        y_eval = np.max(pts[:, 1])
        try:
            fit_cr = fit if len(fit) > 0 else np.polyfit(pts[:, 1] * ym_per_pix, pts[:, 0] * xm_per_pix, 2)
            curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
            return curverad, fit_cr
        except np.linalg.LinAlgError:
            return 0, []

    # ---------------- main per-frame driver ----------------
    def driver(self, cap):
        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame Read Error")
            return 1e10, 0

        # Resize early (CPU) - smaller frame reduces GPU work & transfers
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
        cv2.putText(lane_frame, f"OffSet: {lane_offset}", (lane_frame.shape[1] - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Show results (optional)
        cv2.imshow("Lane Detection (CUDA Optimized)", lane_frame)
        cv2.imshow("Lane Mask (warped)", cv2.resize(lane_mask, (int(lane_mask.shape[1] * 0.5), int(lane_mask.shape[0] * 0.5))))

        return curve_rad, lane_offset


# -------------------- if run as script --------------------
if __name__ == '__main__':
    # Choose source: 'csi', integer device index, or file path
    SOURCE = 0  # or 'csi' or '/path/to/video.mp4'
    cap = create_capture(SOURCE, width=1280, height=720)
    detector = LaneDetectCUDA(resize_width=640)
    print("CUDA available:", detector.cuda_available, "CUDA devices:", detector.cuda_count)
    print("Press q to quit.")

    while True:
        curve, offset = detector.driver(cap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
