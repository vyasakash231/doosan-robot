#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from impedance_control import CartesianImpedanceControl


# move to initial position first
p1= posj(0,25,110,0,45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
movej(p1, vel=40, acc=20)

time.sleep(1.0)

try:
    # Initialize ROS node first
    rospy.init_node('My_service_node')
    
    # Create control object
    task = CartesianImpedanceControl()
    rospy.sleep(2.0)  # Give time for initialization

    # Start controller in a separate thread
    controller_thread = Thread(target=task.run_controller, args=(1400.0, 200.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 
    controller_thread.daemon = True
    controller_thread.start()
    
    # Keep the main thread running for the plot
    while not rospy.is_shutdown():
        rospy.sleep(0.01)

except rospy.ROSInterruptException:
    pass

finally:
    task.save()  # save data for plotting