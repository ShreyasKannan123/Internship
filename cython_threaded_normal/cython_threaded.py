import time
import socket
import math
import argparse
import threading

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from pymavlink import mavutil

import cv2
import cv2.aruco as aruco
import numpy as np

cimport cython
from libc.math cimport sqrt

cdef int id_to_find = 72
cdef int marker_size = 16         # Marker size in centimeters
cdef int takeoff_height = 5
cdef double velocity = 0.5

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

# ----- Camera Parameters -----
cdef int horizontal_res = 640
cdef int vertical_res = 480

# Field of view in radians (adjust for your camera)
cdef double horizontal_fov = 62.2 * (math.pi / 180)
cdef double vertical_fov = 48.8 * (math.pi / 180)

# Camera calibration files (adjust paths as needed)
calib_path = "/home/pi/video2calibration/calibrationFiles/"
cameraMatrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
cameraDistortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

# ----- Other Globals -----
cdef int found_count = 0
cdef int notfound_count = 0

cdef int first_run = 0        # Used to set initial time for FPS determination
start_time = 0.0
end_time = 0.0

cdef int script_mode = 2      # 1: arm and takeoff; 2: manual LOITER-to-GUIDED landing
cdef int ready_to_land = 0    # Flag: 1 when landing is triggered

# If True, expect the pilot to arm manually; if False, arm via the script.
manualArm = True

# Global vehicle object – will be set in main()
vehicle = None

##########################
# Threaded Camera Capture
##########################

cdef class CameraStream:

    cdef object cap
    cdef object frame
    cdef bint stopped
    cdef object lock
    def __cinit__(self):
        # Initialize the camera capture and related attributes
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, horizontal_res)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vertical_res)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
       
    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self
       
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)
           
    def read(self):
        with self.lock:
            return self.frame
       
    def stop(self):
        self.stopped = True
        self.cap.release()

# Create a global camera stream object.
camera_stream = None

def connectMyCopter():
    parser = argparse.ArgumentParser(description='Drone commands')
    parser.add_argument('--connect')
    args = parser.parse_args()
    connection_string = args.connect
    if not connection_string:
        connection_string = '/dev/ttyACM0'
    veh = connect(connection_string, wait_ready=True)
    return veh

cdef double _get_distance_meters(targetLocation,currentLocation):
    cdef double dLat = targetLocation.lat - currentLocation.lat
    cdef double dLon = targetLocation.lon - currentLocation.lon
    return sqrt(dLon * dLon + dLat * dLat) * 1.113195e5

def get_distance_meters(targetLocation, currentLocation):
    return _get_distance_meters(targetLocation, currentLocation)

def goto(targetLocation):
    global vehicle
    distanceToTargetLocation = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
    vehicle.simple_goto(targetLocation)
    while vehicle.mode.name == "GUIDED":
        currentDistance = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
        if currentDistance < distanceToTargetLocation * 0.02:
            print("Reached target waypoint.")
            time.sleep(2)
            break
        time.sleep(1)

def arm_and_takeoff(targetHeight):
    """
    Arms the vehicle and takes off to the specified altitude.
    """
    global vehicle, manualArm
    while not vehicle.is_armable:
        print("Waiting for vehicle to become armable.")
        time.sleep(1)
    print("Vehicle is now armable")
   
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != 'GUIDED':
        print("Waiting for drone to enter GUIDED flight mode")
        time.sleep(1)
    print("Vehicle now in GUIDED MODE. Have fun!!")
   
    if not manualArm:
        vehicle.armed = True
        while not vehicle.armed:
            print("Waiting for vehicle to be armed")
            time.sleep(1)
    else:
        if not vehicle.armed:
            print("Exiting script. manualArm set to True but vehicle not armed.")
            print("Set manualArm to False if you want the script to arm the drone.")
            return
    print("Propellers are spinning...")
    vehicle.simple_takeoff(targetHeight)
    while True:
        print("Current Altitude: %d" % vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= 0.95 * targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")

def send_distance_message(z):
    global vehicle
    msg = vehicle.message_factory.distance_sensor_encode(
        0,    # time since boot (not used)
        20,   # minimum distance
        7000, # maximum distance
        int(z),  # current distance (integer)
        0,    # type (raw camera, not used)
        0,    # onboard id (not used)
        mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,  # sensor orientation
        0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_local_ned_velocity(vx, vy, vz):
    global vehicle
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_land_message(x, y):
    global vehicle
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,
        0,
        0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

def lander():
    global first_run, notfound_count, found_count, marker_size, start_time
    global vehicle, horizontal_res, vertical_res, horizontal_fov, vertical_fov
    global cameraMatrix, cameraDistortion, aruco_dict, parameters, id_to_find, camera_stream

    cdef double x_sum, y_sum, x_avg, y_avg, x_ang, y_ang

    if first_run == 0:
        print("First run of lander!!")
        first_run = 1
        start_time = time.time()
   
    frame = camera_stream.read()
    if frame is None:
        print("No frame available yet.")
        return

    # Resize and convert to grayscale
    frame = cv2.resize(frame, (horizontal_res, vertical_res))
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detect markers
    ids = None
    corners, ids, rejected = aruco.detectMarkers(image=gray_img,
                                                   dictionary=aruco_dict,
                                                   parameters=parameters)
    # Ensure the drone is in LAND mode.
    if vehicle.mode.name != 'LAND':
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode.name != 'LAND':
            print('WAITING FOR DRONE TO ENTER LAND MODE')
            time.sleep(1)
   
    try:
        # Check if the desired marker was detected.
        if ids is not None and len(ids) > 0 and ids[0] == id_to_find:
            ret_vals = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                       cameraMatrix=cameraMatrix,
                                                       distCoeffs=cameraDistortion)
            # Unpack rotation and translation vectors
            rvec = ret_vals[0][0, 0, :]
            tvec = ret_vals[1][0, 0, :]
           
            # Compute the marker’s yaw angle using the rotation matrix.
            R, _ = cv2.Rodrigues(rvec)
            yaw_rad = np.arctan2(R[1,0], R[0,0])
            yaw_deg = np.degrees(yaw_rad)
            yaw = round((yaw_deg + 360) % 360, 2)
           
            # Compute average pixel coordinates for the marker.
            x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
            y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]
            x_avg = x_sum * 0.25
            y_avg = y_sum * 0.25
           
            # Convert pixel coordinates into angular offsets.
            x_ang = (x_avg - horizontal_res * 0.5) * (horizontal_fov / horizontal_res)
            y_ang = (y_avg - vertical_res * 0.5) * (vertical_fov / vertical_res)
           
            send_land_message(x_ang, y_ang)
            # Optionally send a fake rangefinder message:
            # send_distance_message(tvec[2])
           
            print("X CENTER PIXEL: " + str(x_avg) + " Y CENTER PIXEL: " + str(y_avg))
            print("FOUND COUNT: " + str(found_count) + " NOTFOUND COUNT: " + str(notfound_count))
            print("Marker Heading:" + str(yaw) +
                  " MARKER POSITION: x=" + format(tvec[0], '.2f') +
                  " y=" + format(tvec[1], '.2f') +
                  " z=" + format(tvec[2], '.2f'))
            found_count += 1
        else:
            notfound_count += 1
    except Exception as e:
        print('Target likely not found. Error: ' + str(e))
        notfound_count += 1


def main():
    """
    Main routine:
      - Connects to the vehicle.
      - Sets up precision landing parameters.
      - Initializes the threaded camera stream.
      - Runs the landing detection loop until landing is complete.
    """
    global vehicle, ready_to_land, start_time, end_time, found_count, notfound_count, script_mode, camera_stream

    # Start the threaded camera stream.
    global camera_stream
    camera_stream = CameraStream().start()

    # Connect to the copter.
    vehicle = connectMyCopter()
   
    # SETUP PARAMETERS TO ENABLE PRECISION LANDING
    vehicle.parameters['PLND_ENABLED'] = 1
    vehicle.parameters['PLND_TYPE'] = 1      # 1 for companion computer
    vehicle.parameters['PLND_EST_TYPE'] = 0    # 0 for raw sensor, 1 for Kalman filter estimation
    vehicle.parameters['LAND_SPEED'] = 30      # Descent speed (cm/s)

    # Set static home location from first 3D fix.
    home_lat = vehicle.location.global_relative_frame.lat
    home_lon = vehicle.location.global_relative_frame.lon
    wp_home = LocationGlobalRelative(home_lat, home_lon, takeoff_height)
   
    if script_mode == 1:
        arm_and_takeoff(takeoff_height)
        print(str(time.time()))
        # Optionally, offset the drone from the target with a velocity command.
        # send_local_ned_velocity(velocity, velocity, 0)
        time.sleep(1)
        ready_to_land = 1
    elif script_mode == 2:
        # Wait until a landing point is acquired.
        while True:
            lat_current = vehicle.location.global_relative_frame.lat
            lon_current = vehicle.location.global_relative_frame.lon
            current_altitude = vehicle.location.global_relative_frame.alt
            wp_current = LocationGlobalRelative(lat_current, lon_current, current_altitude)
            distance_to_home = get_distance_meters(wp_current, wp_home)
            altitude = vehicle.rangefinder.distance
           
            if (vehicle.mode.name == 'RTL' and distance_to_home <= 3 and altitude <= 8) or (vehicle.mode.name == 'LAND'):
                print("Landing Point Acquired...")
                ready_to_land = 1
                break
           
            time.sleep(1)
            print("Distance to Home:" + str(distance_to_home) +
                  " Altitude:" + str(altitude) +
                  " Waiting to acquire Landing Point...")
   
    if ready_to_land == 1:
        while vehicle.armed:
            lander()
       
        end_time = time.time()
        total_time = end_time - start_time
        total_count = found_count + notfound_count
        if total_time > 0:
            freq_lander = total_count / total_time
        else:
            freq_lander = 0
        print("Total iterations: " + str(total_count))
        print("Total seconds: " + str(int(total_time)))
        print("------------------")
        print("Lander function had a frequency of: " + str(freq_lander))
        print("------------------")
        print("Precision landing completed...")
       
    # Stop the camera stream when finished.
    camera_stream.stop()

if __name__ == '__main__':
    main()
