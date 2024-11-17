#!/usr/bin/env python
import rospy
from dsr_msgs.srv import StopRtControl, DisconnectRtControl
import threading

class RtShutdownNode:
    def __init__(self):
        rospy.init_node('rt_shutdown_node')
        
        # Initialize service clients
        self.client1_ = rospy.ServiceProxy('/dsr01/realtime/stop_rt_control', StopRtControl)
        self.client2_ = rospy.ServiceProxy('/dsr01/realtime/disconnect_rt_control', DisconnectRtControl)
        
        # Initialize flags
        self.rt_started = True  # Initially True since we're shutting down
        self.rt_connected = True
        
        # Start service client threads
        self.client1_thread_ = threading.Thread(target=self.stop_rt_control_client)
        self.client2_thread_ = threading.Thread(target=self.disconnect_rt_control_client)
        
        self.client1_thread_.start()
        self.client2_thread_.start()

    def stop_rt_control_client(self):
        rate = rospy.Rate(0.5)  # 0.5 Hz
        while not rospy.is_shutdown() and self.rt_started:
            rate.sleep()
            try:
                rospy.wait_for_service('/dsr01/realtime/stop_rt_control', timeout=1.0)
                
                request = StopRtControlRequest()
                response = self.client1_(request)
                if response.success:
                    self.rt_started = False
                    rospy.loginfo("RT stopped")
            except rospy.ROSException as e:
                rospy.logwarn("thread1: Waiting for the server to be up...")
            except Exception as e:
                rospy.logerr("Service call failed: %s", str(e))

    def disconnect_rt_control_client(self):
        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown() and self.rt_connected:
            rate.sleep()
            if not self.rt_started:
                try:
                    rospy.wait_for_service('/dsr01/realtime/disconnect_rt_control', timeout=1.0)
                    
                    request = DisconnectRtControlRequest()
                    response = self.client2_(request)
                    if response.success:
                        self.rt_connected = False
                        rospy.loginfo("RT disconnected")
                except rospy.ROSException as e:
                    rospy.logwarn("thread1: Waiting for the server to be up...")
                except Exception as e:
                    rospy.logerr("Service call failed: %s", str(e))

    def __del__(self):
        # Join threads on deletion
        if hasattr(self, 'client1_thread_') and self.client1_thread_.is_alive():
            self.client1_thread_.join()
            rospy.loginfo("client1_thread_ joined")
        if hasattr(self, 'client2_thread_') and self.client2_thread_.is_alive():
            self.client2_thread_.join()
            rospy.loginfo("client2_thread_ joined")

def main():
    try:
        node = RtShutdownNode()
        while not rospy.is_shutdown() and node.rt_connected:
            rospy.sleep(0.1)  # Small sleep to prevent CPU hogging
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
