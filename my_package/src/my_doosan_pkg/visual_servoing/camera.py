import cv2 as cv
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs
import time
import rospy
import os
from math import *
import sys

class Camera:
    Jp = None
    # O_g_p = None  # coord of goal in pixel coord
    # O_p = None  # current maker position in pixel coord
    Sd = None
    S = None

    def __init__(self):
        self.fps = 30
        self.marker_length = 0.119  # length in meters (The returning translation vectors will be in the same unit
        self.pipe = rs.pipeline()

        self.cfg = rs.config()
        self.cfg.enable_device('244222073634')  # serial no of our camera
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)

        self.profile = self.pipe.start(self.cfg)
    
    def draw_coordinate_frame(self,image, marker_corner=None):
        # Get image dimensions
        height, width = image.shape[:2]

        # Calculate center coordinates
        center_x = width // 2
        center_y = height // 2

        center = (center_x, center_y)

        # Draw coordinate frame
        frame_size = 80  # Adjust frame size as needed
        arrow_size = 10  # Adjust arrow size as needed
        color = (0, 180, 250)  # frame color (B,G,R)
        thickness = 2
        font_scale = 0.5

        # Draw horizontal line (X-axis)
        cv.line(image, (center[0], center[1]), (center[0] + frame_size, center[1]), color, thickness)
        # Draw vertical line (Y-axis)
        cv.line(image, (center[0], center[1]), (center[0], center[1] - frame_size), color, thickness)

        # Draw arrowhead for X-axis
        cv.line(image, (center[0] + frame_size, center[1]), (center[0] + frame_size - arrow_size, center[1] - arrow_size), color, thickness)
        cv.line(image, (center[0] + frame_size, center[1]), (center[0] + frame_size - arrow_size, center[1] + arrow_size), color, thickness)
        # Draw arrowhead for Y-axis
        cv.line(image, (center[0], center[1] - frame_size), (center[0] - arrow_size, center[1] - frame_size + arrow_size), color, thickness)
        cv.line(image, (center[0], center[1] - frame_size), (center[0] + arrow_size, center[1] - frame_size + arrow_size), color, thickness)

        font = cv.FONT_HERSHEY_SIMPLEX
        # Text in horizontal line (X-axis)
        cv.putText(image,'X',(center[0] + frame_size, center[1] + 10), font, font_scale, (0,0,255), thickness, cv.LINE_AA)
        # Text in horizontal line (Y-axis)
        cv.putText(image,'Y',(center[0] + 10, center[1] - frame_size), font, font_scale, (0,255,0), thickness, cv.LINE_AA)

        # Draw goal corners
        cv.circle(image,(240,180), 5, color, 2)
        cv.circle(image,(400,180), 5, color, 2)
        cv.circle(image,(400,300), 5, color, 2)

        # Draw marker corners
        # cv.circless(image,(int(marker_corner[0,0]),int(marker_corner[0,1])), 5, (0,0,255),2)
        # cv.circle(image,(int(marker_corner[1,0]),int(marker_corner[1,1])), 5, (0,0,255), 2)
        # cv.circle(image,(int(marker_corner[2,0]),int(marker_corner[2,1])), 5, (0,0,255), 2)

        return image, center

    def detect_marker(self):
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        u0,v0,fx,fy = 330.24231461, 245.79774661, 603.7866956, 606.00911541
        camera_matrix = np.array([[fx,0,u0],[0,fy,v0],[0,0,1]])
        dist_coeffs = np.array([[-0.01640182,0.92663735,0.00520239,0.00355203,-3.29411829]])

        # dictionary to specify type of the marker
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)

        # detect the marker
        self.param_markers = aruco.DetectorParameters()

        while True:
            # iterate through multiple frames, in a live video feed
            frame = self.pipe.wait_for_frames() # Wait for a coherent pair of frames: depth and color
            aligned_frames = self.align.process(frame) # Align the depth frame to color frame

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue    

            # Defining depth_image
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # turning the frame to grayscale-only (for efficiency)
            gray_frame = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)  # MARKER_CORNER ORDER -> topRight, bottomRight, bottomLeft, and topLeft

            image_with_frame, frame_center = self.draw_coordinate_frame(color_image) # Draw coordinate frame at camera center

            # marker desired center point coord in camera coord frame
            # Zc_g = 0.2
            # Xc_g = 0
            # Yc_g = 0

            # O_g_c = np.array([[Xc_g],[Yc_g],[Zc_g],[1]])

            # K_I = np.array([[fx,0,u0,0],[0,fy,v0,0],[0,0,0,1]]) # K[I|O]
            # O_t_p_homo = K_I @ O_g_c  # (3,1)
            # Camera.O_t_p = np.array([[O_t_p_homo[0,0]/O_t_p_homo[2,0]],[O_t_p_homo[1,0]/O_t_p_homo[2,0]]]) # (2,1)  --> (u_g, v_g)

            Camera.Sd = np.array([[(240-u0)/fx],[(180-v0)/fy],[(400-u0)/fx],[(180-v0)/fy],[(400-u0)/fx],[(300-v0)/fy]])

            if marker_IDs != None:  # allow when marker is within the frame  
                '''
                The returned transformation is the one that transforms points from each marker coordinate system to the camera coordinate system. 
                The marker corrdinate system is centered on the middle of the marker, with the Z axis perpendicular to the marker plane.
                '''
                R_vec, T_vec, _ = aruco.estimatePoseSingleMarkers(marker_corners, self.marker_length, camera_matrix, dist_coeffs) # translation and rotational vector of aruco marker w.r.t camera frame in meters         

                Xc, Yc, Zc = T_vec[0][0][0], T_vec[0][0][1], T_vec[0][0][2]  # marker center coord wrt camera coord frame in meters

                # Camera.S = T_vec[0][0]

                # # Each element in marker_corners corresponds to the corners of a single marker
                # for corners in marker_corners:
                #     # Calculate the mean of x coordinates and y coordinates separately
                #     u1 = np.mean(corners[:,:,0])
                #     v1 = np.mean(corners[:,:,1])
                
                # Camera.O_p = np.array([[u1],[v1]])  # (2,1)  --> (u,v)

                # Image Jacobian matrix
                u1, v1 = marker_corners[0][0][0,0], marker_corners[0][0][0,1]  # coord of P1 marker points wrt pixel coord in image plane
                u2, v2 = marker_corners[0][0][1,0], marker_corners[0][0][1,1]  # coord of P2 marker points wrt pixel coord in image plane
                u3, v3 = marker_corners[0][0][2,0], marker_corners[0][0][2,1]  # coord of P3 marker points wrt pixel coord in image plane

                Camera.S = np.array([[(u1-u0)/fx],[(v1-v0)/fy],[(u2-u0)/fx],[(v2-v0)/fy],[(u3-u0)/fx],[(v3-v0)/fy]])

                # Image Jacobian matrix 
                Camera.Jp = np.array([[-1/Zc,   0   , (u1-u0)/(Zc*fx), (u1-u0)*(v1-v0)/(fy*fx)     ,  -((fx**2)+(u1-u0)**2)/(fx**2),  (v1-v0)/fy],
                                      [   0  , -1/Zc, (v1-v0)/(Zc*fy), ((fy**2)+(v1-v0)**2)/(fy**2),    -(u1-u0)*(v1-v0)/(fx*fy)   , -(u1-u0)/fx],
                                      [-1/Zc,   0   , (u2-u0)/(Zc*fx), (u2-u0)*(v2-v0)/(fy*fx)     ,  -((fx**2)+(u2-u0)**2)/(fx**2),  (v2-v0)/fy],
                                      [   0  , -1/Zc, (v2-v0)/(Zc*fy), ((fy**2)+(v2-v0)**2)/(fy**2),    -(u2-u0)*(v1-v0)/(fx*fy)   , -(u2-u0)/fx],
                                      [-1/Zc,   0   , (u3-u0)/(Zc*fx), (u3-u0)*(v3-v0)/(fy*fx)     ,  -((fx**2)+(u3-u0)**2)/(fx**2),  (v3-v0)/fy],
                                      [   0  , -1/Zc, (v3-v0)/(Zc*fy), ((fy**2)+(v3-v0)**2)/(fy**2),    -(u3-u0)*(v1-v0)/(fx*fy)   , -(u3-u0)/fx]])


                # Camera.Jp = np.array([[-fx/Zc,   0   , (u1-u0)/Zc, (u1-u0)*(v1-v0)/fy,  -fx-(((u1-u0)**2)/fx),  fx*(v1-v0)/fy],
                #                       [   0  , -fy/Zc, (v1-v0)/Zc, fy+((v1-v0)**2)/fy,    -(u1-u0)*(v1-v0)/fx, -fy*(u1-u0)/fx],
                #                       [-fx/Zc,   0   , (u2-u0)/Zc, (u2-u0)*(v2-v0)/fy,  -fx-(((u2-u0)**2)/fx),  fx*(v2-v0)/fy],
                #                       [   0  , -fy/Zc, (v2-v0)/Zc, fy+((v2-v0)**2)/fy,    -(u2-u0)*(v1-v0)/fx, -fy*(u2-u0)/fx],
                #                       [-fx/Zc,   0   , (u3-u0)/Zc, (u3-u0)*(v3-v0)/fy,  -fx-(((u3-u0)**2)/fx),  fx*(v3-v0)/fy],
                #                       [   0  , -fy/Zc, (v3-v0)/Zc, fy+((v3-v0)**2)/fy,    -(u3-u0)*(v1-v0)/fx, -fy*(u3-u0)/fx]])

                for j in range(len(marker_IDs)):
                    cv.drawFrameAxes(color_image, camera_matrix, dist_coeffs, R_vec[j], T_vec[j], 0.1)  # draw frame around marker

            # Show stream
            cv.imshow('rgb', image_with_frame)
            if cv.waitKey(1) == ord('q'):
                break

        self.pipe.stop()

if __name__ == "__main__":
    camera_view = Camera()
    camera_view.detect_marker()