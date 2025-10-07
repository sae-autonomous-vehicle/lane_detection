import math
import time
import cv2
import numpy as np
# from smbus2 import SMBus  # Uncomment if using Arduino I2C link
from LaneDetectCUDA import LaneDetectCUDA  # Use your CUDA lane detection version


class AutoPilot:
    def __init__(self):
        """
        Steering Convention:
            +ve curvature → Left turn, -ve curvature → Right turn
            Steering Angle measured anticlockwise (+ve → left, -ve → right)
            Steering Angle range: -20° to +20°
        """
        self.MAX_SPEED = 150
        self.MAX_STEER_ANG = 20  # degrees
        self.TURNING_RADIUS = 0.635  # meters
        self.WHEELBASE = 0.23  # meters
        self.TIRE_WIDTH = 0.02  # meters

        # PID control parameters
        self.P, self.I, self.D = 50, 1, 0.5
        self.i_term, self.prev_err, self.dt_prev = 0, 0, 0
        self.INTG_WINDUP = 5  # max integral contribution
        self.INTG_MEM = 10
        self.intg_counter = 0

        self.steer = 0
        self.curv = 0

    # ---------------- Steering Controller ----------------
    def steerController(self, curvature, offset):
        self.curv = 0.8 * self.curv + 0.2 * curvature

        if 1.5 > abs(curvature) > 0.05:
            steer = self.turnControl(curvature)
            self.intg_counter = 10
            mode = "Turn"
        else:
            steer = self.laneCentering(offset)
            mode = "Centering"

        steer = int(0.3 * steer + 0.7 * self.steer)
        self.steer = steer
        print(f"{mode} | Curv: {curvature:.2f} | Offset: {offset:.3f} | Steer: {steer}")
        return steer

    def turnControl(self, curvature):
        steer_angle = math.degrees(math.atan(self.WHEELBASE / curvature))
        return int(np.clip(steer_angle, -self.MAX_STEER_ANG, self.MAX_STEER_ANG))

    def laneCentering(self, offset):
        if offset == -1:
            return self.steer  # maintain previous steering if invalid offset

        err = offset
        p = self.P * err

        if self.intg_counter >= self.INTG_MEM:
            self.i_term = 0
            self.intg_counter = 0

        self.i_term += np.clip(self.I * err, -self.INTG_WINDUP, self.INTG_WINDUP)

        t1 = time.time()
        d = self.D * (err - self.prev_err) / (t1 - self.dt_prev + 1e-8)

        self.prev_err = err
        self.dt_prev = t1
        self.intg_counter += 1

        steer = np.clip(p + self.i_term + d, -self.MAX_STEER_ANG, self.MAX_STEER_ANG)
        return int(steer)


# ---------------- Optional I2C Communication ----------------
'''
class DataLink:
    def __init__(self):
        self.I2C_BUS = 1
        self.ARDUINO_ADDRESS = 0x08

    def send_data(self, steer: int, speed: int):
        try:
            bus = SMBus(self.I2C_BUS)
            values = [steer, speed]
            byte_array = [
                (values[0] >> 8) & 0xFF, values[0] & 0xFF,
                (values[1] >> 8) & 0xFF, values[1] & 0xFF
            ]
            bus.write_i2c_block_data(self.ARDUINO_ADDRESS, 0, byte_array)
            print(f"Sent {values} to Arduino at address {self.ARDUINO_ADDRESS}")
        except Exception as e:
            print(f"I2C Error: {e}")
        finally:
            if 'bus' in locals():
                bus.close()
'''


# ---------------- Main Loop ----------------
def get_jetson_gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=640, display_height=360, framerate=30, flip_method=0):
    """Optimized GStreamer pipeline for Jetson hardware acceleration."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


if __name__ == '__main__':
    # Select your source:
    cap = cv2.VideoCapture(get_jetson_gstreamer_pipeline(), cv2.CAP_GSTREAMER)  # Jetson camera
    # cap = cv2.VideoCapture("D:/Users/Pranil/Github Repos/lane_detection/Lane_Data/WhatsApp Video 2025-10-07 at 07.32.26_b8b4d712.mp4")  # Local video

    lane_pred = LaneDetectCUDA()
    pilot = AutoPilot()
    # data = DataLink()  # Uncomment for Arduino I2C communication

    print("=== Starting CUDA AutoPilot System ===")
    print("Press 'q' to quit")

    while cap.isOpened():
        curvature, lane_offset = lane_pred.driver(cap)
        steer = pilot.steerController(curvature, lane_offset)

        # Send data to Arduino (if enabled)
        # data.send_data(steer, 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== System Stopped ===")