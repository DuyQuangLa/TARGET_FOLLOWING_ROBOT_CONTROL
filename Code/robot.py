# import the opencv library
import cv2
from cv2 import cuda
import numpy as np
import time
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
from time import sleep
import math
#from numba import jit, cuda

###
pre_timeframe = 0
new_timeframe = 0
###


GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
In1 = 35
In2 = 37
Ena = 33

kit=ServoKit(channels=16)
A=90
kit.servo[0].angle=A

#Define object specific variables
Known_distance = 40.0
Known_width = 7.5
width_in_rf_image = 279.7

# hsv value
hsv_min = (169,93,138)
hsv_max = (255,255,255)

#basic constants for opencv Functs
kernel = np.ones((3,3),'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0,20)
fontScale = 0.6
color = (0, 0, 255)
thickness = 2

# define a video capture object
vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture(0)
vid.set(3, 1280)
vid.set(4, 720)

h = 12
#cudareader = cv2.cudacodec.VideoReader()

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

class Motor():
    def __init__(self,Ena,In1,In2):
        self.Ena = Ena
        self.In1 = In1
        self.In2 = In2
        GPIO.setup(self.Ena, GPIO.OUT)
        GPIO.setup(self.In1, GPIO.OUT)
        GPIO.setup(self.In2, GPIO.OUT)
        self.pwm = GPIO.PWM(self.Ena, 500)
        self.pwm.start(50)

    def moveS(self,z):
        GPIO.output(self.In1, GPIO.LOW)
        GPIO.output(self.In2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(z)

    def moveR(self,z):
        GPIO.output(self.In1, GPIO.LOW)
        GPIO.output(self.In2, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(z)

    def moveN(self, z):
        GPIO.output(self.In1, GPIO.HIGH)
        GPIO.output(self.In2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(z)

# ---------- Blob detecting function: returns keypoints and mask
# -- return keypoints, reversemask

def blob_detect(image,  # -- The frame (cv standard)
                hsv_min,  # -- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,  # -- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,  # -- blur value (default 0)
                blob_params=None,  # -- blob parameters (default None)
                # -- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
                ):
    # - Blur image to remove noise
    if blur > 0:
        image = cv2.blur(image, (blur, blur))
        # - Show result

    # - Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # - Apply HSV threshold
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # - dilate and erode makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # - build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 100

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 500000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

    else:
        params = blob_params

        # - Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255 - mask

    keypoints = detector.detect(reversemask)

    return keypoints

# ---------- Draw detected blobs: returns the image
# -- return(im_with_keypoints)

def draw_keypoints(image,  # -- Input image
                   keypoints,  # -- CV keypoints
                   imshow=False  # -- show the result
                   ):
    # -- Draw detected blobs as red circles.
    # -- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage = np.array([]), color=(0, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return (im_with_keypoints)

#####
# find the distance from then camera

def Focal_Length_Finder(Known_distance, real_width, width_in_rf_image):

    focal_length = (width_in_rf_image * Known_distance) / real_width
    return focal_length

Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, width_in_rf_image)

# find the distance from then camera

def Distance_finder(Focal_Length, Known_width, area, image):
    distance = (Known_width * Focal_Length)/area

    return distance 
###3#
motor1 = Motor(Ena,In1,In2)
targetA = 45
P = 0.5
I = 2
D = 3
pid1 = PID(P, I, D)
pid1.SetPoint = targetA
pid1.setSampleTime(1)
tolerance = 4

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # Display the resulting frame
    keypoints = blob_detect(frame, hsv_min, hsv_max, blur=3,
                            blob_params=None, imshow=True)
    frame = draw_keypoints(frame, keypoints, imshow=True)
    
    
    ###
    new_timeframe = time.time()
    fps = 1/(new_timeframe-pre_timeframe)
    pre_timeframe = new_timeframe
    fps = int(fps)
    text0 = "fps: " + "{:.2f}".format(fps)
    cv2.putText(frame, text0, (5, 75), font, 1, (0, 0, 255), 2)
    ###
    
    #print(len(keypoints))

    # Write Size of first blob
    blobCount = len(keypoints)

    # Write the number of blobs found
    text1 = "Count=" + str(blobCount)
    text4 = "Tien"
    text5 = "Trai"
    text6 = "Phai"
    text7 = "Lui"
    text8 = "Ngung"
    cv2.putText(frame, text1, (5, 25), font, 1, (0, 255, 0), 2)

    if blobCount > 0:
        blob_x = keypoints[0].pt[0]
        text2 = "X=" + "{:.2f}".format(blob_x)
        cv2.putText(frame, text2, (5, 50), font, 1, (0, 0, 255), 2)
        angle_servo = 4*(10**(-19))*(int(blob_x)**2) + (0.1406*int(blob_x)) - 6*(10**(-14))
        #kit.servo[0].angle = angle_servo
#****#
        rows = frame.shape[0]

        y_max_px = int(rows)
        frame = cv2.line(frame, (int(blob_x), 500), (int(blob_x), y_max_px), (0, 0, 255), 3)
#***#    
        # Write Y position of first blob
        blob_y = keypoints[0].pt[1]
        #text3 = "Y=" + "{:.2f}".format(blob_y)
        
    
        # Write Size of first blob
        blob_size = keypoints[0].size
	#print("blob_size = ", blob_size)
        current_distance = Distance_finder(Focal_length_found, Known_width, blob_size, frame)
        ditance_2 = math.sqrt(current_distance**2 - h**2) - 5
        #print(blob_size)
        text3 = "Distance from Camera in CM :" + "{:.2f}".format(ditance_2)
        cv2.putText(frame, text3, (5, 100), font, 1, (0, 0, 255), 2)
        
        pid1.update(ditance_2)
        target_x = pid1.output
        print('pid_output = ', target_x)
        
        """if target_x > 10 and target_x < 20:
           target_x = target_x + 10"""
        if target_x > 100:
           target_x = 100
        elif target_x < -100:
           target_x = -100
           
        if abs(ditance_2 - targetA) <= tolerance:
           target_x = 0
        if target_x == 0:
           motor1.moveS(0)
           time.sleep(1)
        """elif target_x < 10:
           target_x = target_x + 20
        elif target_x < -10 and target_x > -20:
           target_x = target_x - 10"""
        
        """elif target_x > -10:
           target_x = target_x - 20"""
        
        if target_x < 0:
           #pwm = -1*(10**(-17))*(target_x**2) + 0.2*target_x - 29
           pwm = -1*(10**(-17))*(target_x**2) + 0.1*target_x - 29
           """if target_x < -10 and target_x > -20:
              target_x = target_x - 10
           elif target_x > -10:
              target_x = target_x - 20
           motor1.moveR(35-(35+target_x))
           print("v = ",35-(35+target_x))"""
           motor1.moveR(-pwm)
           print('v = ', -pwm)
           kit.servo[0].angle = angle_servo
           cv2.putText(frame, text4, (1100, 25), font, 1, (0, 0, 255), 2)
           """time.sleep(-target_x/100)
           motor1.moveS(0)"""
           #
           if angle_servo > 135:
               """if target_x < -10 and target_x > -20:
                  target_x = target_x - 10
               elif target_x > -10:
                  target_x = target_x - 20
               if target_x > 25:
                  target_x = 25"""
               cv2.putText(frame, text5, (1100, 50), font, 1, (0, 0, 255), 2)
               motor1.moveR(-pwm)
               """motor1.moveR(25-(25-target_x))
               time.sleep(-target_x/100)
               motor1.moveS(0)"""
               
           else:
               """if target_x < -10 and target_x > -20:
                  target_x = target_x - 10
               elif target_x > -10:
                  target_x = target_x - 20
               if target_x > 25:
                  target_x = 25"""
               cv2.putText(frame, text6, (1100, 50), font, 1, (0, 0, 255), 2)
               motor1.moveR(-pwm)
               """motor1.moveR(25-(25-target_x))
               time.sleep(-target_x/100)
               motor1.moveS(0)"""
               
        else:
           #pwm = 3*(10**(-17))*(target_x**2) + 0.2*target_x + 29
           pwm = 1*(10**(-17))*(target_x**2) + 0.1*target_x + 29
           """if target_x > 10 and target_x < 20:
              target_x = target_x + 10
           elif target_x < 10:
              target_x = target_x + 20
           motor1.moveN(35-(35-target_x))
           print("v = ",35-(35-target_x))"""
           motor1.moveN(pwm)
           print('v = ', pwm)
           kit.servo[0].angle = (180 - angle_servo)
           cv2.putText(frame, text7, (1100, 25), font, 1, (0, 0, 255), 2)
           """time.sleep(target_x/100)
           motor1.moveS(0)"""
           #
           if angle_servo > 135:
               """if target_x > 10 and target_x < 20:
                  target_x = target_x + 10
               elif target_x < 10:
                  target_x = target_x + 20
               if target_x < -25:
                  target_x = -25"""
               cv2.putText(frame, text6, (1100, 50), font, 1, (0, 0, 255), 2)
               motor1.moveN(pwm)
               """motor1.moveN(30-(30+target_x))
               time.sleep(target_x/100)
               motor1.moveS(0)
               """
           else:
               """if target_x > 10 and target_x < 20:
                  target_x = target_x + 10
               elif target_x < 10:
                  target_x = target_x + 20
               if target_x < -25:
                  target_x = -25"""
               cv2.putText(frame, text5, (1100, 50), font, 1, (0, 0, 255), 2)
               motor1.moveN(pwm)
               """motor1.moveN(25-(25+target_x))
               time.sleep(target_x/100)
               motor1.moveS(0)"""
               
    elif blobCount == 0:
        cv2.putText(frame, text8, (1100, 25), font, 1, (0, 0, 255), 2)
        motor1.moveS(0)

    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
