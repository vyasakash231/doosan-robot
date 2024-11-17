#!/usr/bin/env python
import os
import ctypes
import errno
import threading
import ctypes.util

# Load the POSIX threads library
librt = ctypes.CDLL('librt.so.1')
libc = ctypes.CDLL(ctypes.util.find_library('c'))

# Define schedule policies
SCHED_OTHER = 0
SCHED_FIFO = 1
SCHED_RR = 2

class SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]

def set_thread_scheduling(thread, policy, sched_priority):
    """
    Set thread scheduling policy and priority.
    
    Args:
        thread: Thread native handle (thread.ident)
        policy: Scheduling policy (SCHED_FIFO, SCHED_RR, or SCHED_OTHER)
        sched_priority: Priority level
        
    Raises:
        RuntimeError: If setting scheduling parameters fails
    """
    param = SchedParam()
    param.sched_priority = sched_priority
    
    # Get thread ID for the current thread
    if thread is None:
        thread = threading.current_thread().ident
        
    # Attempt to set scheduling parameters
    result = libc.pthread_setschedparam(
        thread,
        policy,
        ctypes.byref(param)
    )
    
    if result > 0:
        error_msg = os.strerror(result)
        raise RuntimeError(f"Couldn't set scheduling priority and policy. Error code: {error_msg}")

def get_thread_scheduling(thread=None):
    """
    Get current thread scheduling policy and priority.
    
    Args:
        thread: Thread native handle (optional, defaults to current thread)
        
    Returns:
        tuple: (policy, priority)
        
    Raises:
        RuntimeError: If getting scheduling parameters fails
    """
    policy = ctypes.c_int()
    param = SchedParam()
    
    if thread is None:
        thread = threading.current_thread().ident
        
    result = libc.pthread_getschedparam(
        thread,
        ctypes.byref(policy),
        ctypes.byref(param)
    )
    
    if result > 0:
        error_msg = os.strerror(result)
        raise RuntimeError(f"Couldn't get scheduling parameters. Error code: {error_msg}")
        
    return policy.value, param.sched_priority
