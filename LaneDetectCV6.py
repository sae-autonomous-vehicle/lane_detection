import time
import cv2
import numpy as np
from collections import deque


# ---------- Config ----------
class LaneDetect():

    def __init__(self):
        self.RESIZE_WIDTH = 360
        self.CENTER_DEADZONE = int(0.05 * self.RESIZE_WIDTH)
        self.SMOOTHING_FRAMES = 8
        self.MORPH_KERNEL = (3,3)
        self.MIN_AREA = self.RESIZE_WIDTH * 2
        self.MIN_WIDTH = int(self.RESIZE_WIDTH * 0.3)  # self.RESIZE_WIDTH * 0.3
        self.NUM_SAMPLES = int(self.RESIZE_WIDTH * 0.25)  # for resampling lanes
        self.MIN_LR_CURVE_RATIO = 0.99
        self.SLOPE_THRESH = 1.73
        self.last_turn = 0      # 1: left, 0: straight, -1: right
        self.fps=30

        self.lane_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.center_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.left_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.right_line_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.curvature_history = deque(maxlen=self.SMOOTHING_FRAMES*2)

    # ---------- ROI ----------
    def region_of_interest(self, img):
        h, w = img.shape
        #polygon = np.array([[ (0,int(h*0.80)), (w,int(h*0.80)), (w,int(h*0.35)), (0,int(h*0.35)) ]], np.int32)
        polygon = np.array([[ (0,int(h*0.90)), (w,int(h*0.90)), (w,int(h*0.7)), (0,int(h*0.7)) ]], np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)


    # ------- Dynamic THresh -------


    def detect_lighting_conditions(self, frame):
        """
        Detect lighting conditions to adjust parameters
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Check for overexposure
        bright_pixels = np.sum(gray > 240) / gray.size
        
        # Check for underexposure
        dark_pixels = np.sum(gray < 10) / gray.size


        
        # Classify lighting condition
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
        """
        Get parameters based on detected lighting condition
        """
        condition = lighting_info['condition']
        
        if condition == "night":
            return {
                's_range': (100, 255),
                'l_range': (150, 255),
                'sobel_range': (40, 255),
                'blur_kernel': (7, 7),
                'morph_iterations': 3
            }
        elif condition == "shadows":
            return {
                's_range': (110, 255),
                'l_range': (200, 255),
                'sobel_range': (60, 255),
                'blur_kernel': (5, 5),
                'morph_iterations': 2
            }
        elif condition == "low_contrast":
            return {
                's_range': (120, 255),
                'l_range': (200, 255),
                'sobel_range': (50, 255),
                'blur_kernel': (5, 5),
                'morph_iterations': 2
            }
        elif condition == "overexposed":
            return {
                's_range': (150, 255),
                'l_range': (240, 255),
                'sobel_range': (80, 255),
                'blur_kernel': (3, 3),
                'morph_iterations': 1
            }
        else:  # normal
            return {
                's_range': (130, 255),
                'l_range': (220, 255),
                'sobel_range': (70, 255),
                'blur_kernel': (5, 5),
                'morph_iterations': 2
            }


    
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


    # ---------- Lane Mask ----------


    def combined_lane_mask(self, frame):

        lighting = self.detect_lighting_conditions(frame)
        params = self.get_lighting_based_parameters(lighting)


        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        _, l, s = hls[:,:,0], hls[:,:,1], hls[:,:,2]
        
        s_range = params['s_range']
        l_range = params['l_range']
        
        s_binary = cv2.inRange(s, s_range[0], s_range[1])
        
        if lighting['condition'] == "overexposed":
            l_adaptive = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -40)
            l_binary = l_adaptive
        else:
            l_binary = cv2.inRange(l, l_range[0], l_range[1])
        
        color_binary = cv2.bitwise_or(s_binary, l_binary)
        
        # Dynamic Sobel processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_kernel = params['blur_kernel']
        gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)
        
        sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_val = np.max(abs_sobelx) if np.max(abs_sobelx) != 0 else 1
        scaled_sobel = np.uint8(255 * abs_sobelx / max_val)
        
        sobel_range = params['sobel_range']
        grad_binary = cv2.inRange(scaled_sobel, sobel_range[0], sobel_range[1])
        
        # Combine masks
        combined = cv2.bitwise_or(color_binary, grad_binary)
        combined = self.region_of_interest(combined)

        # Apply perspective transform to the combined mask
        warped_combined, _, inverse_matrix = self.perspective_transform(combined)

        morph_iterations = params['morph_iterations']
        
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_OPEN, small_kernel, 
                                       iterations=morph_iterations)
        
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL)
        warped_combined = cv2.morphologyEx(warped_combined, cv2.MORPH_CLOSE, medium_kernel, 
                                       iterations=morph_iterations)
        
        # Dynamic component filtering
        min_component_area = 50 if lighting['condition'] == "night" else 100
        warped_combined = self.remove_small_components(warped_combined, min_area=min_component_area)
        
        return warped_combined, lighting, inverse_matrix


    def remove_small_components(self, binary_img, min_area=50):

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
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


        def line_confidence(xs, ys,max_width=200):
            if len(xs) < 20: return 0
            dx = xs[-1]-xs[0]; dy = ys[-1]-ys[0]
            slope = abs(dx/(dy+1e-6))
            length = np.sqrt(dx**2 + dy**2)


            # --- Width penalty ---
            widthx, widthy = np.std(xs), np.std(ys)  # spread of x-points = line thickness
            if widthx > max_width and widthy > max_width:
                return 0  # discard wide blobs


            return length


        left_conf = line_confidence(*left_candidates)
        right_conf = line_confidence(*right_candidates)


        left_pts = np.array([ [int(np.median(left_candidates[0][left_candidates[1]==y])), y] for y in np.unique(left_candidates[1])], np.int32) if left_conf != 0 else None
        right_pts = np.array([ [int(np.median(right_candidates[0][right_candidates[1]==y])), y] for y in np.unique(right_candidates[1])], np.int32) if right_conf != 0 else None


        return left_pts, right_pts


    # ---------- Resample Line ----------
    def resample_line(self, line_pts, num_samples):


        if line_pts is None or len(line_pts) < 2:
            return None
        y_vals = np.linspace(line_pts[:,1].min(), line_pts[:,1].max(), num_samples)
        x_vals = np.interp(y_vals, line_pts[:,1], line_pts[:,0])
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
        
        # Handle single lane cases - create estimated opposite lane
        if left_avg is not None and right_avg is None:
            print("Using left lane only - estimating right lane")
            min_y = np.min(left_avg[:, 1])
            max_y = np.max(left_avg[:, 1])
            right_avg = np.array([[self.RESIZE_WIDTH-1, i] for i in np.linspace(min_y, max_y, 200)], dtype=int)
            
        elif left_avg is None and right_avg is not None:
            print("Using right lane only - estimating left lane") 
            min_y = np.min(right_avg[:, 1])
            max_y = np.max(right_avg[:, 1])
            left_avg = np.array([[0, i] for i in np.linspace(min_y, max_y, 200)], dtype=int)
            
        elif left_avg is None and right_avg is None:
            print("No lanes detected")
            return frame, None, None


        # polygon for drivable area
        pts = np.vstack((left_avg, right_avg))
        area = cv2.contourArea(pts)
        min_len = min(len(left_avg), len(right_avg))

        if min_len == 0:
            return frame, left_avg, right_avg  # No lanes detected, just return

        # Resample both arrays to have the same number of points
        left_resampled = np.linspace(0, len(left_avg) - 1, min_len).astype(int)
        right_resampled = np.linspace(0, len(right_avg) - 1, min_len).astype(int)

        left_sync = left_avg[left_resampled]
        right_sync = right_avg[right_resampled]

        avg_width = np.mean(np.abs(right_sync[:, 0] - left_sync[:, 0]))
        
        # Relax constraints for single lane detection
        min_area_threshold = self.MIN_AREA * 0.3 if (len(resampled_left) == 0 or len(resampled_right) == 0) else self.MIN_AREA
        min_width_threshold = self.MIN_WIDTH * 0.5 if (len(resampled_left) == 0 or len(resampled_right) == 0) else self.MIN_WIDTH
        
        if area < min_area_threshold or avg_width < min_width_threshold:
            #print(f"Area/width too small: {area:.0f} < {min_area_threshold}, {avg_width:.0f} < {min_width_threshold}")
            return frame, None, None
        
        return frame, left_avg, right_avg



    # ---------- Center Line ----------


    def smooth_with_polynomial(self, points, degree=2):
        """Fit polynomial to points for smoother curve"""
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
        except:
            return points


    def draw_center_line_polynomial(self, frame, left_pts, right_pts, inverse_matrix):
        # Handle missing lanes safely
        if left_pts is None or right_pts is None:
            return frame, (
                self.left_line_history[-1] if self.left_line_history else None
            ), (
                self.right_line_history[-1] if self.right_line_history else None
            ), -1, 0

        # Smooth individual lanes
        smooth_left = self.smooth_with_polynomial(left_pts)
        smooth_right = self.smooth_with_polynomial(right_pts)

        if (
            smooth_left is None
            or smooth_right is None
            or len(smooth_left) < 2
            or len(smooth_right) < 2
        ):
            return frame, None, None, -1, 0

        # --- Ensure consistent lengths before storing in history ---
        target_len = 100  # fixed number of samples for averaging stability
        def resample_points(points, n=target_len):
            if len(points) < 2:
                return points
            idxs = np.linspace(0, len(points) - 1, n).astype(int)
            return points[idxs]

        smooth_left = resample_points(smooth_left)
        smooth_right = resample_points(smooth_right)

        self.left_line_history.append(smooth_left)
        self.right_line_history.append(smooth_right)

        # Limit history size to avoid memory growth
        max_hist = 8
        if len(self.left_line_history) > max_hist:
            self.left_line_history.pop(0)
        if len(self.right_line_history) > max_hist:
            self.right_line_history.pop(0)

        # --- Compute safe averages ---
        avg_left = np.mean(np.array(self.left_line_history), axis=0).astype(np.int32)
        avg_right = np.mean(np.array(self.right_line_history), axis=0).astype(np.int32)

        # --- START OF MODIFICATIONS ---

        h, w = frame.shape[:2]
        look_ahead_y = h * 0.95  # look-ahead control point (95% of frame height)

        # Sort points by y before interpolation
        left_y_vals = avg_left[:, 1]
        left_x_vals = avg_left[:, 0]
        right_y_vals = avg_right[:, 1]
        right_x_vals = avg_right[:, 0]

        left_sort_idx = np.argsort(left_y_vals)
        right_sort_idx = np.argsort(right_y_vals)

        left_x_at_lookahead = np.interp(
            look_ahead_y, left_y_vals[left_sort_idx], left_x_vals[left_sort_idx]
        )
        right_x_at_lookahead = np.interp(
            look_ahead_y, right_y_vals[right_sort_idx], right_x_vals[right_sort_idx]
        )

        # Compute lane center and offset
        center_x_at_lookahead = (left_x_at_lookahead + right_x_at_lookahead) / 2.0
        lane_offset = round((w / 2 - center_x_at_lookahead) / w, 3)

        # --- Safe curvature calculation ---
        def safe_curvature(points, y_eval):
            try:
                curve, _ = self.calculate_curvature(points, y_eval=y_eval)
                if np.isnan(curve) or np.isinf(curve):
                    return 1e6  # treat as straight
                if abs(curve) > 1e8:
                    curve = np.sign(curve) * 1e8
                return curve
            except Exception:
                return 1e6

        l_curve = safe_curvature(avg_left, look_ahead_y)
        r_curve = safe_curvature(avg_right, look_ahead_y)

        # --- END OF MODIFICATIONS ---

        self.curvature_history.append(max(min(-(l_curve + r_curve) / 2, 1e8), -1e8))
        if len(self.curvature_history) > 8:
            self.curvature_history.popleft()
        curvature = round(np.mean(self.curvature_history), 3)

        self.last_turn = np.sign(curvature)

        # --- Visualization section ---
        center_pts = np.array(
            [[(lx + rx) // 2, y] for (lx, y), (rx, _) in zip(avg_left, avg_right)], np.int32
        )
        smooth_center = self.smooth_with_polynomial(center_pts)

        lane_img = np.zeros_like(frame)
        drivable_area_pts = np.vstack((avg_left, np.flipud(avg_right)))
        cv2.fillPoly(lane_img, [drivable_area_pts.astype(np.int32)], (0, 255, 0))

        cv2.polylines(
            lane_img,
            [smooth_center.reshape((-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=(0, 0, 255),
            thickness=3,
        )
        cv2.polylines(
            lane_img,
            [avg_left.reshape((-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=(255, 0, 0),
            thickness=3,
        )
        cv2.polylines(
            lane_img,
            [avg_right.reshape((-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=(255, 0, 0),
            thickness=3,
        )

        # Warp back to original perspective
        dewarped_lanes = cv2.warpPerspective(
            lane_img, inverse_matrix, (frame.shape[1], frame.shape[0])
        )
        final_frame = cv2.addWeighted(frame, 1, dewarped_lanes, 0.3, 0)

        return final_frame, avg_left, avg_right, lane_offset, curvature


    # ---------- Curvature ----------


    def calculate_curvature(self, pts, fit=[], y_eval=None):
        if pts is None or len(pts) < 3:
            return 0, []

        ym_per_pix = 0.00064
        xm_per_pix = 0.00064
        
        # Use the provided y_eval point, or default to the max y-value (furthest point)
        if y_eval is None:
            y_eval = np.max(pts[:,1])
        
        try:
            # Use provided fit if available, otherwise calculate a new one
            fit_cr = fit if len(fit)>0 else np.polyfit(pts[:,1]*ym_per_pix, pts[:,0]*xm_per_pix, 2)
            if np.abs(fit_cr[0]) < 1e-6:
                curverad = 1e6  # assume very large (almost straight line)
            else:
                curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

            return curverad, fit_cr
        except np.linalg.LinAlgError:
            return 0, []
    
    def driver(self, cap):


        t0 = time.time()
        ret, frame = cap.read()


        if frame is None:
            print("Frame Read Error")
            return 1e10, 0


        frame = cv2.resize(frame, (self.RESIZE_WIDTH, int(frame.shape[0]*self.RESIZE_WIDTH/frame.shape[1])))
        
        lane_mask, lighting, inverse_matrix = self.combined_lane_mask(frame)
        
        left_pts, right_pts = self.extract_lane_candidates(lane_mask)
        
        lane_frame, left_avg, right_avg = self.fill_drivable_area(frame, left_pts, right_pts)  

        lane_frame, smoothl, smoothr, lane_offset, curve_rad = self.draw_center_line_polynomial(lane_frame, left_avg, right_avg, inverse_matrix)
        t1 = time.time()
        self.fps = round(1 / (t1 - t0 + 1e-8), 2)
        cv2.putText(lane_frame, f"FPS: {self.fps}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(lane_frame, f"Lighting: {lighting['condition']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(lane_frame, f"RoC: {curve_rad:.2f} m", (lane_frame.shape[1]-0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(lane_frame, f"OffSet: {lane_offset}", (lane_frame.shape[1]-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
 
        # --- ADD THIS DEBUGGING CODE ---
        h, w, _ = lane_frame.shape
        # Use the exact same coordinates as in your region_of_interest function
        roi_polygon_to_draw = np.array([[(0, int(h * 0.90)), (w, int(h * 0.90)), (w, int(h * 0.7)), (0, int(h * 0.7))]], dtype=np.int32)
        # Draw a bright yellow polygon on the final frame
        cv2.polylines(lane_frame, [roi_polygon_to_draw], isClosed=True, color=(0, 255, 255), thickness=2)
        # --- END OF DEBUGGING CODE ---
 
 
        cv2.imshow("Lane Detection", lane_frame)
        cv2.imshow("Lane Mask", cv2.resize(lane_mask, (int(lane_mask.shape[1]*0.5), int(lane_mask.shape[0]*0.5))))
        #cv2.polylines(lane_frame, [np.array([[302, lane_frame.shape[1]//2],[330, lane_frame.shape[1]//2]], dtype=np.int32).reshape((-1, 1, 2))], isClosed=False, color=(255,0,0), thickness=2)


        return curve_rad, lane_offset
