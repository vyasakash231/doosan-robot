import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *

from scipy.spatial.transform import Rotation 

class Camera:
    def __init__(self):
        # Initiate/Creates a node with name 'my_turtlebot'
        rospy.init_node('Extrinsic_Calibration',anonymous = False)

        self.fps = 30
        self.marker_length = 0.119  # length in meters (The returning translation vectors will be in the same unit
        self.pipe = rs.pipeline()

        self.cfg = rs.config()
        self.cfg.enable_device('244222073634')  # serial no of our camera
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)

        self.profile = self.pipe.start(self.cfg)
        
        rospy.wait_for_service('/dsr01a0509/motion/move_joint')  # Wait until the service becomes available
        self.move_joint = rospy.ServiceProxy('/dsr01a0509/motion/move_joint', MoveJoint)

        self.my_subscriber = rospy.Subscriber('/tf', TFMessage, self.call_back_func)  # In degrees

        self.T_e_0 = np.eye(4)

        self.rate = rospy.Rate(10)

    def call_back_func(self,msg):
        T_matrix = np.zeros((6,4,4))
        # Process the TF messages
        i = 0
        T_e_0 = np.eye(4)
        Homo_matrix = np.eye(4)
        for transform in msg.transforms:
            # Extract relevant information from the message
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Print the information
            t = np.array([translation.x, translation.y, translation.z])
            r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            Homo_matrix[:3,:3] = np.round(r.as_matrix(),3)
            Homo_matrix[:3,-1] = t.T
            T_e_0 = np.dot(T_e_0,Homo_matrix)

            T_matrix[i,:,:] = Homo_matrix
            i += 1
        self.T_e_0 = T_e_0

    def detect_marker(self):
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        camera_matrix = np.array([[603.7866956,      0      , 330.24231461],
                                  [     0     , 606.00911541, 245.79774661],
                                  [     0     ,      0      ,     1       ]])
        
        dist_coeffs = np.array([[-0.01640182, 0.92663735, 0.00520239, 0.00355203, -3.29411829]])

        # dictionary to specify type of the marker
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)

        # detect the marker
        self.param_markers = aruco.DetectorParameters()

        # move to home
        movej([0,0,0,0,0,0], vel=30, acc=10)
        self.rate.sleep()   

        Joint_angles = np.array([[0,-10,-30,0,-130,0],[-45,20,-120,45,-70,0],[15,-70,-10,-20,-120,0],[20,-20,-65,-10,-100,-10],[40,-30,-90,-50,-75,-20],
                                 [15,10,-120,-20,-40,40],[0,-30,-30,10,-120,-50],[30,10,-80,-30,-90,-10],[0,-45,-60,10,-80,0],[-45,-30,-110,70,-70,20],
                                 [70,-40,-80,-50,-110,0],[0,20,-120,0,-50,0],[30,-40,-45,-20,-110,10],[-28,-60,-40,40,-110,0],[-10,-30,-45,0,-110,0], 
                                 [35,-35,-105,-70,-65,10],[-30,45,-130,30,-70,0],[-35,-50,-70,50,-90,0],[30,-70,-30,-40,-110,0],[-20,-30,-60,20,-100,0]])

        Row,Column = Joint_angles.shape

        R_target2cam =[]
        t_target2cam = []
        R_gripper2base = []
        t_gripper2base = []

        # iterate through multiple frames, in a live video feed
        for i in range(Row):
            self.move_joint(Joint_angles[i,:].tolist(), 20, 10, 0, 0, 0, 0, 0)
            time.sleep(1) # sleep for 1 sec
            
            self.rate.sleep() # dynamically choose the correct time to sleep

            # iterate through multiple frames, in a live video feed
            frame = self.pipe.wait_for_frames() # Wait for a coherent pair of frames: depth and color
            aligned_frames = self.align.process(frame) # Align the depth frame to color frame

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue    

            # Defining depth_image
            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # turning the frame to grayscale-only (for efficiency)
            gray_frame = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)  # MARKER_CORNER ORDER -> topRight, bottomRight, bottomLeft, and topLeft

            cv.aruco.drawDetectedMarkers(color_image, marker_corners, marker_IDs)
       
            R_vec, T_vec, _ = cv.aruco.estimatePoseSingleMarkers(marker_corners, self.marker_length, camera_matrix, dist_coeffs)
        
            t_vec, r_matrix = self.T_e_0[:3,-1], self.T_e_0[:3,:3]   # t is in meters and r is 3x3 matrix

            if marker_IDs != None:
                r = Rotation.from_matrix(r_matrix)
                R_target2cam.append(R_vec[0][0]) # In radians
                t_target2cam.append(T_vec[0][0]) # In meters
                R_gripper2base.append(r.as_rotvec()) # In radians
                t_gripper2base.append(t_vec) # In meters
                
                for j in range(len(marker_IDs)):
                    cv.drawFrameAxes(color_image, camera_matrix, dist_coeffs, R_vec[j], T_vec[j], 0.1)
            
            cv.imshow("frame", color_image) 
            cv.waitKey(500) # sleep for 0.5 sec
        
        cv.destroyAllWindows()
        
        # Finding Transfrmation btw camera and end-effector
        R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam) 
        
        self.pipe.stop()

        return R_cam2gripper, t_cam2gripper

if __name__ == "__main__":
    camera_view = Camera()
    R_cam2gripper, t_cam2gripper = camera_view.detect_marker()
    print(R_cam2gripper)
    print(t_cam2gripper)
