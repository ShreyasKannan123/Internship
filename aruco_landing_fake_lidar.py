###########DEPENDENCIES################
import time
import socket
# exceptions
import math
import argparse

from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException
from pymavlink import mavutil

import cv2
import cv2.aruco as aruco
import numpy as np

from imutils.video import WebcamVideoStream
import imutils
#######VARIABLES####################
##Aruco
id_to_find = 72
marker_size = 16 #cm
takeoff_height = 5
velocity = 0.5

#current_altitude = vehicle.location.global_relative_frame.alt

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()
##

##Camera
horizontal_res= 640
vertical_res= 480
#cap = WebcamVideoStream(src=0).start()
cap=cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, horizontal_res)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vertical_res)

horizontal_fov = 62.2 * (math.pi / 180 ) ##Pi cam V1: 53.5 V2: 62.2
vertical_fov = 48.8 * (math.pi / 180)    ##Pi cam V1: 41.41 V2: 48.8

calib_path="/home/pi/video2calibration/calibrationFiles/"
cameraMatrix   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
cameraDistortion   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')
##

##Counters and script triggers
#vertical_position=0
found_count=0
notfound_count=0

first_run=0 #Used to set initial time of function to determine FPS
start_time=0
end_time=0
script_mode = 2 ##1 for arm and takeoff, 2 for manual LOITER to GUIDED land 
ready_to_land=0 ##1 to trigger landing

manualArm=True ##If True, arming from RC controller, If False, arming from this script. 
#########FUNCTIONS#################


def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect

    if not connection_string:
        connection_string = '/dev/serial/by-id/usb-CubePilot_CubeOrange+_330024001751313437363430-if00'

    vehicle = connect(connection_string, wait_ready=True)

    return vehicle


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
            return None

    print("Propellers are spinning...")
    vehicle.simple_takeoff(targetHeight)

    while True:
        print("Current Altitude: %d" % vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= 0.95 * targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")

    return None

# def send_distance_message(z):
#     msg = vehicle.message_factory.distance_sensor_encode(
#         0,
#         1,
#         10000,
#         z,
#         0,
#         0,
#         mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
#         0
#     )
#     vehicle.send_mavlink(msg)
#     vehicle.flush()


def send_local_ned_velocity(vx, vy, vz):
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
    
def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0, #replaced 0 with z for altitude estimation message
        0,
        0,)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def lander():
    global first_run,notfound_count,found_count,marker_size,start_time
    if first_run==0:
        print("First run of lander!!")
        first_run=1
        start_time=time.time()
        
    ret,frame = cap.read()
    #gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray_img=np.array(gray_img).astype('uint8')
    frame = cv2.resize(frame,(horizontal_res,vertical_res))
    frame_np = np.array(frame)   #.astype('uint8')
    gray_img = cv2.cvtColor(frame_np,cv2.COLOR_BGR2GRAY)
    
    ids=''
    corners, ids, rejected = aruco.detectMarkers(image=gray_img,dictionary=aruco_dict,parameters=parameters)
    if vehicle.mode!='RTL': #'LAND':
        vehicle.mode=VehicleMode("RTL") #("LAND")
        while vehicle.mode!='LAND':
            print('WAITING FOR DRONE TO ENTER LAND MODE')
            time.sleep(1)
    
    try:
        if ids is not None and ids[0] == id_to_find:
            ret = aruco.estimatePoseSingleMarkers(corners,marker_size,cameraMatrix=cameraMatrix,distCoeffs=cameraDistortion)
            (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])
            x = '{:.2f}'.format(tvec[0])
            y = '{:.2f}'.format(tvec[1])
            z = '{:.2f}'.format(tvec[2])
            z_f=float(z)
            z_int=int(z_f)
            
            vertical_position = z_int
            
            y_sum = 0
            x_sum = 0
            
            x_sum = corners[0][0][0][0]+ corners[0][0][1][0]+ corners[0][0][2][0]+ corners[0][0][3][0]
            y_sum = corners[0][0][0][1]+ corners[0][0][1][1]+ corners[0][0][2][1]+ corners[0][0][3][1]
    
            x_avg = x_sum*.25
            y_avg = y_sum*.25
            
            x_ang = (x_avg - horizontal_res*.5)*(horizontal_fov/horizontal_res)
            y_ang = (y_avg - vertical_res*.5)*(vertical_fov/vertical_res)
            
            if vehicle.mode!='LAND':
                vehicle.mode = VehicleMode('LAND')
                while vehicle.mode!='LAND':
                    time.sleep(1)
                print("------------------------")
                print("Vehicle now in LAND mode")
                print("------------------------")
                send_land_message(x_ang,y_ang)
                #send_distance_message(vertical_position)     #to be enabled later
            else:
                send_land_message(x_ang,y_ang)
                #send_distance_message(vertical_position)     #to be enabled later
                pass
            print("X CENTER PIXEL: "+str(x_avg)+" Y CENTER PIXEL: "+str(y_avg))
            print("FOUND COUNT: "+str(found_count)+" NOTFOUND COUNT: "+str(notfound_count))
            print("MARKER POSITION: x=" +x+" y= "+y+" z="+z)
            found_count=found_count+1
            print("")
        else:
            notfound_count=notfound_count+1
    except Exception as e:
        print('Target likely not found. Error: '+str(e))
        notfound_count=notfound_count+1
    
     
 
 
######################################################

#######################MAIN###########################

######################################################

vehicle = connectMyCopter()

##
##SETUP PARAMETERS TO ENABLE PRECISION LANDING
##
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1 ##1 for companion computer
vehicle.parameters['PLND_EST_TYPE'] = 0 ##0 for raw sensor, 1 for kalman filter pos estimation
vehicle.parameters['LAND_SPEED'] = 30 ##Descent speed of 30cm/s
#current_altitude = vehicle.location.global_relative_frame.alt
distance = vehicle.rangefinder.distance

####### to be enabled later #######
#vehicle.parameters['RNGFND2_TYPE'] = 10
#vehicle.parameters['RNGFND2_MIN_CM'] = 20 
#vehicle.parameters['RNGFND2_MAX_CM'] = 1000
#vehicle.parameters['RNGFND2_GNDCLEAR'] = 10

if script_mode ==1:
    arm_and_takeoff(takeoff_height)
    print(str(time.time()))
    #send_local_ned_velocity(velocity,velocity,0) ##Offset drone from target
    time.sleep(1)
    ready_to_land=1
    
elif script_mode == 2:
    while vehicle.mode!='RTL':
        time.sleep(1)
        print("Waiting for manual change from mode "+str(vehicle.mode)+" to RTL")
#         while distance < takeoff_height: #vehicle.mode!='RTL' and distance <= takeoff_height:    #modification
#             time.sleep(1)     #modification
#             print("waiting to obtain target altitude") #modification
#         #print("Waiting to reach target altitude")
    ready_to_land=1
    print("Reached Home Location. Initiating Precision Landing...")

if ready_to_land==1:
    while vehicle.armed==True:
        lander()
    end_time=time.time()
    total_time=end_time-start_time
    total_time=abs(int(total_time))

    total_count=found_count+notfound_count
    freq_lander=total_count/total_time
    print("Total iterations: "+str(total_count))
    print("Total seconds: "+str(total_time))
    print("------------------")
    print("lander function had frequency of: "+str(freq_lander))
    print("------------------")
    print("Vehicle has landed")
    print("------------------")
