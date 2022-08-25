#!/usr/bin/env python3
#Basis für Mono Ouptut: https://docs.luxonis.com/projects/api/en/latest/samples/MonoCamera/mono_preview/
import time

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco
from depthai_sdk import FPSHandler


fps = FPSHandler()


def create_pipeline(res=400):

    # Create pipelinqe
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoRight = pipeline.create(dai.node.MonoCamera)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutRight.setStreamName('right')

    # Properties
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    if res == 400:
        monoRight.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
    elif res == 480:
        monoRight.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_480_P)

    # Linking
    monoRight.out.link(xoutRight.input)
    return pipeline


print("OpenCV version :  {0}".format(cv2.__version__))
resolution = 400
#ArUco declarations
# mtx,dist = calibration matrix and distortion coefficients of camera calibration (https://docs.luxonis.com/en/latest/pages/calibration/)
if resolution == 400:
    mtx = np.load(
        'D:\TUBcloud\Bachelorarbeit\Code\99_calibdata\M_right_datacalib_mtx_THE400.pkl', allow_pickle=True)
    dist = np.load(
        'D:\TUBcloud\Bachelorarbeit\Code\99_calibdata\M_right_datacalib_dist_THE400.pkl', allow_pickle=True)
elif resolution == 480:
    mtx = np.load(
        'D:\TUBcloud\Bachelorarbeit\Code\99_calibdata\M_right_datacalib_mtx_THE480.pkl', allow_pickle=True)
    dist = np.load(
        'D:\TUBcloud\Bachelorarbeit\Code\99_calibdata\M_right_datacalib_dist_THE480.pkl', allow_pickle=True)

size_of_marker = 0.052  # 52mm Breite
length_of_axis = 0.05
x_mid = 0
y_mid = 0
# Load ArUco Dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()

# Create pipelinqe
pipeline = create_pipeline(resolution)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    
    # Output queues will be used to get the grayscale frames from the outputs defined above
    qRight = device.getOutputQueue(name="right", maxSize=1, blocking=False)

    while True:
        fps.nextIter()

        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inRight = qRight.tryGet()
        
        if inRight is not None:
            frameRight = inRight.getCvFrame()

            # ArUco processing (ArUco only uses right camera) (on host)
            # ArUco Marker detection
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frameRight, aruco_dict, parameters=parameters)
            # Draw Detected Markers:
            frameRight = aruco.drawDetectedMarkers(frameRight, corners, ids)

            # ArUco PoseEstimation
            rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(
                corners, size_of_marker, mtx, dist)

            # Draw Pose
            frameRight = aruco.drawDetectedMarkers(frameRight, corners, ids)

            # Display
            if tvecs is not None:
                for i in range(len(tvecs)):
                    frameRight = aruco.drawAxis(
                        frameRight, mtx, dist, rvecs[i], tvecs[i], length_of_axis)

                    #calculate euler angles
                    rvec = np.squeeze(rvecs[0], axis=None)
                    tvec = np.squeeze(tvecs[0], axis=None)
                    tvec = np.expand_dims(tvec, axis=1)
                    rvec_matrix = cv2.Rodrigues(rvec)[0]
                    proj_matrix = np.hstack((rvec_matrix, tvec))
                    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    
                    # display angles
                    cv2.putText(frameRight, 'rot X: '+str(
                        int(euler_angles[0])), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                    cv2.putText(frameRight, 'rot Y: '+str(
                        int(euler_angles[1])), (200, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                    cv2.putText(frameRight, 'rot Z: '+str(
                        int(euler_angles[2])), (400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                    
                    # display position
                    cv2.putText(
                        frameRight, 'pos X: %.0fmm' % (1000*tvec[0]), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                    cv2.putText(
                        frameRight, 'pos Y: %.0fmm' % (1000*tvec[1]), (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                    cv2.putText(
                        frameRight, 'pos Z: %.0fmm' % (1000*tvec[2]), (400, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

                    print(1)
                    # tvec = translation vector
                    # rvec = rotation vector
            else: print(0)
            # Display fps
            cv2.putText(frameRight, "Fps: {:.2f}".format(fps.fps()), (2, 396), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow('ArucoA', frameRight)

        if cv2.waitKey(1) == ord('q'):
            break
