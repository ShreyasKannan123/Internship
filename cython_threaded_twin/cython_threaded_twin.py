import time
import socket
import math
import argparse
import threading
import logging

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

import cv2
import cv2.aruco as aruco
import numpy as np

# ---- Configuration using dataclasses ----
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Aruco marker configuration
@dataclass
class ArucoIDs:
    primary: int = 72     # Default marker ID used for landing
    backup: int = 70      # Used if altitude is very low (< 1m)
    size_cm: int = 16     # Size of the ArUco marker in cm

# Camera settings
@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    h_fov_deg: float = 62.2       # Horizontal field of view
    v_fov_deg: float = 48.8       # Vertical field of view
    calibration_path: str = "/home/pi/video2calibration/calibrationFiles/"  # Path to camera calibration files

# Flight behavior configuration
@dataclass
class FlightConfig:
    takeoff_alt: float = 5.0                # Takeoff altitude in meters
    landing_alt_trigger: float = 1.0        # Use backup ID when below this altitude
    land_speed_cmps: int = 30               # Landing speed in cm/s
    distance_threshold_m: float = 3.0       # Trigger landing if within 3m of home
    rangefinder_trigger_alt_m: float = 8.0  # Trigger landing if below this altitude

# ---- Global State and Constants ----
aruco_ids = ArucoIDs()
camera_cfg = CameraConfig()
flight_cfg = FlightConfig()

# Load Aruco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
aruco_params = aruco.DetectorParameters_create()

# Calibration and camera resources
camera_matrix = None
camera_distortion = None
vehicle = None
camera_stream = None

# Status tracking
manual_arm = True  # Set to False for auto arm during takeoff
found_count = 0
notfound_count = 0
first_run = False
start_time = 0.0

# ---- Helper Functions ----

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * (math.pi / 180)

def get_distance_meters(loc1, loc2):
    """Return ground distance in meters between two LocationGlobalRelative points."""
    dLat = loc1.lat - loc2.lat
    dLon = loc1.lon - loc2.lon
    return math.sqrt(dLat ** 2 + dLon ** 2) * 1.113195e5

def send_land_message(x, y):
    """
    Send MAVLink message to guide the drone's landing based on x, y offsets.
    Coordinates are angles in radians relative to camera center.
    """
    msg = vehicle.message_factory.landing_target_encode(
        0, 0,  # time_usec, target_num
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Frame
        x, y,  # X and Y offset angles
        0, 0, 0  # Unused fields
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ---- Video Stream Handler ----

class CameraStream:
    """Threaded camera capture class to continuously read frames."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        """Start the camera thread."""
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Background thread continuously reading camera frames."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        """Return the latest frame (thread-safe)."""
        with self.lock:
            return self.frame

    def stop(self):
        """Stop the thread and release the camera."""
        self.stopped = True
        self.cap.release()

# ---- DroneKit Setup ----

def connect_drone():
    """Connect to the vehicle using the given connection string."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect')
    args = parser.parse_args()
    connection = args.connect or '/dev/ttyACM0'
    return connect(connection, wait_ready=True)

def arm_and_takeoff(alt):
    """Arms vehicle and flies to a target altitude."""
    while not vehicle.is_armable:
        logging.info("Waiting for vehicle to become armable...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        logging.info("Switching to GUIDED mode...")
        time.sleep(1)

    if not manual_arm:
        vehicle.armed = True
        while not vehicle.armed:
            logging.info("Waiting to arm...")
            time.sleep(1)
    elif not vehicle.armed:
        logging.warning("Manual arm required. Exiting.")
        return

    vehicle.simple_takeoff(alt)

    # Wait until reaching target altitude
    while vehicle.location.global_relative_frame.alt < 0.95 * alt:
        logging.info(f"Current Altitude: {vehicle.location.global_relative_frame.alt:.2f}")
        time.sleep(1)

# ---- Landing Detection and Control ----

def detect_and_land():
    """Run Aruco marker detection and send landing control messages."""
    global first_run, found_count, notfound_count, start_time

    if not first_run:
        first_run = True
        start_time = time.time()

    frame = camera_stream.read()
    if frame is None:
        logging.warning("No frame yet.")
        return

    # Preprocess frame
    gray = cv2.cvtColor(cv2.resize(frame, (camera_cfg.width, camera_cfg.height)), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Ensure vehicle is in LAND mode
    if vehicle.mode.name != "LAND":
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode.name != "LAND":
            logging.info("Waiting to switch to LAND mode...")
            time.sleep(1)

    # Choose marker ID based on altitude
    active_id = aruco_ids.primary
    if vehicle.rangefinder.distance <= flight_cfg.landing_alt_trigger:
        active_id = aruco_ids.backup

    try:
        if ids is not None and active_id in ids:
            # Find detected marker index
            index = np.where(ids == active_id)[0][0]

            # Estimate pose
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                [corners[index]], aruco_ids.size_cm,
                camera_matrix, camera_distortion
            )

            # Compute center point
            marker = corners[index][0]
            x_avg = sum(pt[0] for pt in marker) / 4
            y_avg = sum(pt[1] for pt in marker) / 4

            # Convert to angular offset from center of frame
            x_ang = (x_avg - camera_cfg.width / 2) * deg2rad(camera_cfg.h_fov_deg) / camera_cfg.width
            y_ang = (y_avg - camera_cfg.height / 2) * deg2rad(camera_cfg.v_fov_deg) / camera_cfg.height

            send_land_message(x_ang, y_ang)

            logging.info(f"Marker {active_id} found at x={x_avg:.1f}, y={y_avg:.1f}, dist={tvec[0][0][2]:.2f}")
            found_count += 1
        else:
            notfound_count += 1
    except Exception as e:
        logging.error(f"Detection error: {e}")
        notfound_count += 1

# ---- Main Routine ----

def main():
    """Main entry point for setup, takeoff, detection, and landing."""
    global vehicle, camera_matrix, camera_distortion, camera_stream

    # Load camera calibration
    try:
        camera_matrix = np.loadtxt(camera_cfg.calibration_path + "cameraMatrix.txt", delimiter=',')
        camera_distortion = np.loadtxt(camera_cfg.calibration_path + "cameraDistortion.txt", delimiter=',')
    except FileNotFoundError:
        logging.error("Calibration files not found. Exiting.")
        return

    # Start camera feed
    camera_stream = CameraStream().start()

    # Connect to drone
    vehicle = connect_drone()

    # Configure landing parameters
    vehicle.parameters['PLND_ENABLED'] = 1
    vehicle.parameters['PLND_TYPE'] = 1
    vehicle.parameters['PLND_EST_TYPE'] = 0
    vehicle.parameters['LAND_SPEED'] = flight_cfg.land_speed_cmps

    # Save current location as home
    home = LocationGlobalRelative(
        vehicle.location.global_relative_frame.lat,
        vehicle.location.global_relative_frame.lon,
        flight_cfg.takeoff_alt
    )

    # Takeoff if in GUIDED mode
    if vehicle.mode.name == "GUIDED":
        arm_and_takeoff(flight_cfg.takeoff_alt)
        time.sleep(1)

    # Wait until ready to begin landing sequence
    while True:
        altitude = vehicle.rangefinder.distance
        dist_to_home = get_distance_meters(vehicle.location.global_relative_frame, home)

        # Check if conditions are met to begin LAND
        if (vehicle.mode.name == "RTL" and dist_to_home <= flight_cfg.distance_threshold_m and
                altitude <= flight_cfg.rangefinder_trigger_alt_m) or vehicle.mode.name == "LAND":
            logging.info("Landing sequence engaged...")
            break

        logging.info(f"Distance: {dist_to_home:.1f}m | Altitude: {altitude:.1f}m")
        time.sleep(1)

    # Actively run landing detection loop
    while vehicle.armed:
        detect_and_land()

    # Report summary
    duration = time.time() - start_time
    total = found_count + notfound_count
    logging.info(f"Completed in {duration:.1f} sec | Frames: {total} | Rate: {total / duration:.2f} Hz")
    camera_stream.stop()

# Run the main program
if __name__ == '__main__':
    main()
