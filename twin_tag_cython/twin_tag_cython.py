import time
import socket
import math
import argparse

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from pymavlink import mavutil

import cv2
import cv2.aruco as aruco
import numpy as np

cimport cython
from libc.math cimport sqrt, atan2, pi

# Variables
id_to_find = 72
id_to_find_low_alt = 70
marker_size = 16  # cm
takeoff_height = 5
velocity = 0.5

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

horizontal_res = 640
vertical_res = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, horizontal_res)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vertical_res)

horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

calib_path = "/home/pi/video2calibration/calibrationFiles/"
cameraMatrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
cameraDistortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

found_count = 0
notfound_count = 0

first_run = 0
start_time = 0
end_time = 0
script_mode = 2
ready_to_land = 0
manualArm = True


def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()
    connection_string = args.connect

    if not connection_string:
        connection_string = '/dev/ttyACM0'

    vehicle = connect(connection_string, wait_ready=True)
    return vehicle


def get_distance_meters(targetLocation, currentLocation):
    dLat = targetLocation.lat - currentLocation.lat
    dLon = targetLocation.lon - currentLocation.lon
    return sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5


def goto(targetLocation):
    distanceToTargetLocation = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
    vehicle.simple_goto(targetLocation)
    while vehicle.mode.name == "GUIDED":
        currentDistance = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
        if currentDistance < distanceToTargetLocation * .02:
            print("Reached target waypoint.")
            time.sleep(2)
            break
        time.sleep(1)


def arm_and_takeoff(targetHeight):
    while vehicle.is_armable != True:
        print("Waiting for vehicle to become armable.")
        time.sleep(1)
    print("Vehicle is now armable")

    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode != 'GUIDED':
        print("Waiting for drone to enter GUIDED flight mode")
        time.sleep(1)
    print("Vehicle now in GUIDED MODE. Have fun!!")

    if manualArm == False:
        vehicle.armed = True
        while vehicle.armed == False:
            print("Waiting for vehicle to be armed")
            time.sleep(1)
    else:
        if vehicle.armed == False:
            print("Exiting script. manualArm set to True but vehicle not armed.")
            print("Set manualArm to True if desiring script to arm the drone.")
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
    msg = vehicle.message_factory.distance_sensor_encode(
        0, 20, 7000, z, 0, 0, mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_local_ned_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_land_message(x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x, y, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def lander():
    global first_run, notfound_count, found_count, marker_size, start_time
    if first_run == 0:
        print("First run of lander!!")
        first_run = 1
        start_time = time.time()

    ret, frame = cap.read()
    frame = cv2.resize(frame, (horizontal_res, vertical_res))
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ids = ''
    corners, ids, rejected = aruco.detectMarkers(image=gray_img, dictionary=aruco_dict, parameters=parameters)

    if vehicle.mode != 'LAND':
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode != 'LAND':
            print('WAITING FOR DRONE TO ENTER LAND MODE')
            time.sleep(1)

    try:
        current_alt = vehicle.location.global_relative_frame.alt
        active_id = id_to_find_low_alt if current_alt < 2 else id_to_find

        if ids is not None and ids[0] == active_id:
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix=cameraMatrix, distCoeffs=cameraDistortion)
            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

            R, _ = cv2.Rodrigues(rvec)
            yaw_rad = atan2(R[1, 0], R[0, 0])
            yaw = round((math.degrees(yaw_rad) + 360) % 360, 2)

            roll_rad = atan2(R[2, 1], R[2, 2])
            roll = round((math.degrees(roll_rad) + 360) % 360, 2)

            pitch_rad = atan2(-R[2, 0], sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
            pitch = round((math.degrees(pitch_rad) + 360) % 360, 2)

            x = '{:.2f}'.format(tvec[0])
            y = '{:.2f}'.format(tvec[1])
            z = '{:.2f}'.format(tvec[2])

            x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
            y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]

            x_avg = x_sum * .25
            y_avg = y_sum * .25

            x_ang = (x_avg - horizontal_res * .5) * (horizontal_fov / horizontal_res)
            y_ang = (y_avg - vertical_res * .5) * (vertical_fov / vertical_res)

            if vehicle.mode != 'LAND':
                vehicle.mode = VehicleMode('LAND')
                while vehicle.mode != 'LAND':
                    time.sleep(1)
                print("------------------------")
                print("Vehicle now in LAND mode")
                print("------------------------")
                send_land_message(x_ang, y_ang)
            else:
                send_land_message(x_ang, y_ang)

            print("X CENTER PIXEL: " + str(x_avg) + " Y CENTER PIXEL: " + str(y_avg))
            print("FOUND COUNT: " + str(found_count) + " NOTFOUND COUNT: " + str(notfound_count))
            print("Marker Heading:" + str(yaw) + " MARKER POSITION: x=" + x + " y= " + y + " z=" + z)
            found_count += 1
            print("")
        else:
            notfound_count += 1
    except Exception as e:
        print('Target likely not found. Error: ' + str(e))
        notfound_count += 1


def main():
    global vehicle, ready_to_land, found_count, notfound_count, start_time, end_time

    vehicle = connectMyCopter()

    vehicle.parameters['PLND_ENABLED'] = 1
    vehicle.parameters['PLND_TYPE'] = 1
    vehicle.parameters['PLND_EST_TYPE'] = 0
    vehicle.parameters['LAND_SPEED'] = 30

    home_lat = vehicle.location.global_relative_frame.lat
    home_lon = vehicle.location.global_relative_frame.lon
    wp_home = LocationGlobalRelative(home_lat, home_lon, takeoff_height)

    if script_mode == 1:
        arm_and_takeoff(takeoff_height)
        print(str(time.time()))
        time.sleep(1)
        ready_to_land = 1

    elif script_mode == 2:
        while True:
            lat_current = vehicle.location.global_relative_frame.lat
            lon_current = vehicle.location.global_relative_frame.lon
            current_altitude = vehicle.location.global_relative_frame.alt

            wp_current = LocationGlobalRelative(lat_current, lon_current, current_altitude)
            distance_to_home = get_distance_meters(wp_current, wp_home)
            altitude = vehicle.rangefinder.distance

            if (vehicle.mode == 'RTL' and distance_to_home <= 3 and altitude <= 8) or (vehicle.mode == 'LAND'):
                print("Landing Point Acquired...")
                ready_to_land = 1
                break

            time.sleep(1)
            print("Distance to Home:" + str(distance_to_home) + " Altitude:" + str(altitude) + " Waiting to acquire Landing Point...")

    if ready_to_land == 1:
        while vehicle.armed is True:
            lander()

        end_time = time.time()
        total_time = abs(int(end_time - start_time))
        total_count = found_count + notfound_count
        freq_lander = total_count / total_time
        print("Total iterations: " + str(total_count))
        print("Total seconds: " + str(total_time))
        print("------------------")
        print("Lander function had a frequency of: " + str(freq_lander))
        print("------------------")
        print("Precision landing completed...")
      
