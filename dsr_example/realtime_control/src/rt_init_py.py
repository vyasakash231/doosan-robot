#!/usr/bin/env python
import rospy
from dsr_msgs.srv import ConnectRtControl, SetRtControlOutput, StartRtControl
import threading

class RtInitNode:
    def __init__(self):
        rospy.init_node('rt_init_node')
        
        # Initialize service clients
        self.client1_ = rospy.ServiceProxy('/dsr01/realtime/connect_rt_control', ConnectRtControl)
        self.client2_ = rospy.ServiceProxy('/dsr01/realtime/set_rt_control_output', SetRtControlOutput)
        self.client3_ = rospy.ServiceProxy('/dsr01/realtime/start_rt_control', StartRtControl)
        
        # Initialize flags
        self.rt_connected = False
        self.rt_output_set = False
        self.rt_started = False
        
        # Start service client threads
        self.client1_thread_ = threading.Thread(target=self.connect_rt_control_client)
        self.client2_thread_ = threading.Thread(target=self.set_rt_control_output_client)
        self.client3_thread_ = threading.Thread(target=self.start_rt_control_client)
        
        self.client1_thread_.start()
        self.client2_thread_.start()
        self.client3_thread_.start()

    def connect_rt_control_client(self):
        rate = rospy.Rate(0.5)  # 0.5 Hz
        while not rospy.is_shutdown() and not self.rt_connected:
            rate.sleep()
            try:
                # Wait for service
                rospy.wait_for_service('/dsr01/realtime/connect_rt_control', timeout=1.0)
                
                # Create request
                request = ConnectRtControlRequest()
                request.ip_address = "192.168.137.100"
                request.port = 12347
                
                # Call service
                response = self.client1_(request)
                if response.success:
                    self.rt_connected = True
                    rospy.loginfo("RT connected")
            except rospy.ROSException as e:
                rospy.logwarn("thread1: Waiting for the server to be up...")
            except Exception as e:
                rospy.logerr("Service call failed: %s", str(e))

    def set_rt_control_output_client(self):
        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown() and not self.rt_output_set:
            rate.sleep()
            if self.rt_connected:
                try:
                    rospy.wait_for_service('/dsr01/realtime/set_rt_control_output', timeout=1.0)
                    
                    request = SetRtControlOutputRequest()
                    request.version = "v1.0"
                    request.period = 0.001
                    request.loss = 4
                    
                    response = self.client2_(request)
                    if response.success:
                        self.rt_output_set = True
                        rospy.loginfo("RT control output set")
                except rospy.ROSException as e:
                    rospy.logwarn("thread2: Waiting for the server to be up...")
                except Exception as e:
                    rospy.logerr("Service call failed: %s", str(e))

    def set_rt_control_client(self):
        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown() and not self.rt_started:
            rate.sleep()
            if self.rt_connected and self.rt_output_set:
                try:
                    rospy.wait_for_service('/dsr01/realtime/start_rt_control', timeout=1.0)
                    
                    request = StartRtControlRequest()
                    response = self.client3_(request)
                    if response.success:
                        self.rt_started = True
                        rospy.loginfo("RT started")
                except rospy.ROSException as e:
                    rospy.logwarn("thread3: Waiting for the server to be up...")
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
        if hasattr(self, 'client3_thread_') and self.client3_thread_.is_alive():
            self.client3_thread_.join()
            rospy.loginfo("client3_thread_ joined")

def main():
    try:
        node = RtInitNode()
        while not rospy.is_shutdown() and not node.rt_started:
            rospy.sleep(0.1)  # Small sleep to prevent CPU hogging
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
