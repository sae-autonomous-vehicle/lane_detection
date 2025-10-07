#import smbus2 as smbus
import math
import time
import cv2
from LaneDetectCV import LaneDetectCUDA
import numpy as np

class AutoPilot():
    def __init__(self):
        '''
        Steering Convention: 
            Curvature +ve -> Left turn and -ve -> Right turn
            Steering Angle measured in aclk direction. ie. +ve -> left and -ve -> right
            Steering Angle of car -20 to +20 degree
            Turning Radius of Model car 0.635 m
        '''
        self.MAX_SPEED = 150
        self.MAX_STEER_ANG = 20  # +-20 degree ie. 40deg
        self.TURNING_RADIUS = 0.635 # meter
        self.WHEELBASE = 0.23 # meter
        self.TIRE_WIDTH= 0.02 # meter

        self.P, self.I, self.D = 50, 1, 0.5
        self.i, self.prev_err, self.dt0 = 0, 0, 0
        self.INTG_WINDUP = 5 # Degree max integral affect
        self.INTG_MEM = 10  
        self.intg_cnt = 0

        self.steer, self.curv = 0, 0

    def steerController(self, curvature, offset):

        self.curv = 0.8*self.curv + 0.2*curvature
        if 1.5 > abs(curvature) and abs(curvature) > 0.05:
            steer = self.turnControl(curvature)
            self.intg_cnt = 10
            print('Turn | ',end='')
        else:
            steer = self.laneCentering(offset)
            print('Centering | ',end='')

       
        steer = int(0.3*steer + 0.7*self.steer)
        self.steer = steer
        return steer

    def turnControl(self, curvature):

        steer_angle = math.degrees(math.atan(self.WHEELBASE / curvature))
        steer = int(min(max(steer_angle, -self.MAX_STEER_ANG), self.MAX_STEER_ANG))
        return steer

    def laneCentering(self, offset):
        if offset == -1:
            return self.steer
        err = offset
        p = self.P * err
        if self.intg_cnt >= self.INTG_MEM:
            self.i = 0
            self.intg_cnt = 0
        self.i += max(min(self.I * err, self.INTG_WINDUP), -self.INTG_WINDUP)
        t1 = time.time()
        d = self.D * (err - self.prev_err)/(t1 - self.dt0 + 1e-8)

        self.prev_err = err
        self.dt0 = t1
        self.intg_cnt += 1
        self.steer = int(max(min(p + self.i + d, self.MAX_STEER_ANG), -self.MAX_STEER_ANG))
        return self.steer



'''class DataLink():
    def __init__(self):
        self.I2C_BUS = 1
        self.ARDUINO_ADDRESS = 0x08
    def send_data(self,steer: int, speed: int):
        try:
            bus = smbus.SMBus(self.I2C_BUS)
            # Hardcoded values to send
            values = [steer, speed] # steer, speed
            
            # Convert integers to a 4-byte array (high byte, low byte for each)
            byte_array = [
                (values[0] >> 8) & 0xFF, values[0] & 0xFF,
                (values[1] >> 8) & 0xFF, values[1] & 0xFF
            ]
            
            # Write the block of data
            bus.write_i2c_block_data(self.ARDUINO_ADDRESS, 0, byte_array)
            print(f"Sent {values} to Arduino at address {self.ARDUINO_ADDRESS}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check I2C wiring, address, and that the bus is enabled.")

        finally:
            if 'bus' in locals():
                bus.close()'''


VIDEO_PATH = r"C:/Users/SHLOAK/OneDrive/Pictures/Camera Roll/WIN_20250927_17_33_10_Pro.mp4"
VIDEO_PATH1 = r"D:\Users\Pranil\Github Repos\lane_detection\Lane_Data\WhatsApp Video 2025-10-07 at 07.32.26_b8b4d712.mp4"
VIDEO_PATH2 = "C:/Users/SHLOAK/OneDrive/Pictures/Camera Roll/WIN_20250930_17_20_18_Pro.mp4"
VIDEO_PATH3 = "C:/Users/SHLOAK/Downloads/WhatsApp Video 2025-10-02 at 10.13.59 AM.mp4"

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    lane_pred = LaneDetectCUDA()
    pilot     = AutoPilot()
    #data = DataLink()

    while True:
        curvature, lane_offset = lane_pred.driver(cap)
        steer = pilot.steerController(curvature, lane_offset)

        #data.send_data(steer,100)
        print(f'curv: {curvature}, offset:{lane_offset}, steer:{steer}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
