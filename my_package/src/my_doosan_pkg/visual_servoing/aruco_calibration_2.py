import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *

from handeye_calibration import HandEyeCalibration

# 1. Setup RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 2. Setup robot connection
# For example, with a UR robot:
# robot = urx.Robot("192.168.1.100")  # Replace with your robot's IP

# 3. Setup ArUco marker detection
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera matrix and distortion coefficients (from previous calibration)
camera_matrix = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]])
dist_coeffs = np.zeros((5, 1))

# Lists to store poses
robot_rvecs = []
robot_tvecs = []
target_rvecs = []
target_tvecs = []

# 4. Collect calibration data
num_poses = 10  # Number of different poses to collect
for i in range(num_poses):
    # Move robot to a new pose
    # robot.movej([joint1, joint2, joint3, joint4, joint5, joint6], acc=0.1, vel=0.1)
    
    # Get current robot pose
    # robot_pose = robot.getl()  # [x, y, z, rx, ry, rz]
    # Simulate robot pose for this example
    robot_pose = [0.5 + i*0.01, 0.2, 0.5, 0.1, 0.2, 0.3]
    
    # Extract rotation (axis-angle) and translation
    robot_trans = np.array(robot_pose[:3])
    robot_rot = np.array(robot_pose[3:])
    
    # Wait for robot to settle
    # time.sleep(0.5)
    
    # Capture frame from RealSense
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(color_image)
    
    if ids is not None and len(ids) > 0:
        # Estimate pose of the marker
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        
        # Store the poses
        robot_rvecs.append(robot_rot)
        robot_tvecs.append(robot_trans)
        target_rvecs.append(rvecs[0][0])
        target_tvecs.append(tvecs[0][0])
        
        print(f"Collected pose {i+1}/{num_poses}")
    else:
        print(f"No marker detected in pose {i+1}, skipping")

# 5. Perform hand-eye calibration
HandEyeCalibration.set_verbose(True)

# For Eye-to-Hand calibration (camera observes robot):
H_robot_to_camera = HandEyeCalibration.estimate_hand_eye_screw(
    robot_rvecs, robot_tvecs, target_rvecs, target_tvecs)

print("Robot to Camera Transformation:")
print(H_robot_to_camera)

# Clean up
pipeline.stop()
# robot.close()

# 6. Save calibration result
np.save('robot_to_camera_transform.npy', H_robot_to_camera)