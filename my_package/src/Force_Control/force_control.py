#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import *
from scipy.spatial.transform import Rotation

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

def main(args=None):
    rospy.init(args=args)
    node = rospy.create_node("force_control", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    pos = posx([496.06, 93.46, 96.92, 20.75, 179.00, 19.09])

    # 초기 위치
    JReady = [0, 0, 90, 0, 90, 0]
    set_tool("Tool Weight_2FG")
    set_tcp("2FG_TCP")

    while rospy.ok():
        # 초기 위치로 이동
        movej(JReady, vel=VELOCITY, acc=ACC)
        movel(pos, vel=VELOCITY, acc=ACC, ref=DR_BASE)
        
        set_desired_force(fd=[0, 0, -10, 0, 0, 0], dir=[0, 0, 1, 0, 0, 0], mod=DR_FC_MOD_REL)
        while not check_force_condition(DR_AXIS_Z, max=5):
            pass

        release_compliance_ctrl()

    rospy.shutdown()


if __name__ == "__main__":
    main()