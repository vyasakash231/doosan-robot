#!/usr/bin/env python
import resource
import threading
import ctypes
import os

# Define constants
RUSAGE_SELF = 0
RUSAGE_CHILDREN = -1
RUSAGE_THREAD = -2  # This might not be available on all systems

def get_involuntary_context_switches(who=RUSAGE_THREAD):
    """
    Get the number of involuntary context switches.
    
    Args:
        who: Process to measure (RUSAGE_SELF, RUSAGE_CHILDREN, or RUSAGE_THREAD)
        
    Returns:
        int: Number of involuntary context switches
    """
    try:
        usage = resource.getrusage(who)
        return usage.ru_nivcsw
    except ValueError:
        # Fall back to /proc filesystem if RUSAGE_THREAD is not supported
        if who == RUSAGE_THREAD:
            try:
                tid = ctypes.CDLL('libc.so.6').syscall(186)  # gettid()
                with open(f'/proc/{os.getpid()}/task/{tid}/status', 'r') as f:
                    for line in f:
                        if line.startswith('nonvoluntary_ctxt_switches'):
                            return int(line.split(':')[1].strip())
            except:
                return 0
        return 0

class ContextSwitchesCounter:
    """
    Counter for tracking context switches between measurements.
    """
    def __init__(self, who=RUSAGE_THREAD):
        self.who = who
        self.involuntary_context_switches_previous = 0
        self._lock = threading.Lock()
        self._initialized = False
    
    def init(self):
        """Initialize the counter with the current number of context switches."""
        with self._lock:
            self.involuntary_context_switches_previous = get_involuntary_context_switches(self.who)
            self._initialized = True
    
    def get(self):
        """
        Get the number of context switches since the last measurement.
        
        Returns:
            int: Number of context switches since last call
        """
        if not self._initialized:
            self.init()
            
        with self._lock:
            current = get_involuntary_context_switches(self.who)
            diff = current - self.involuntary_context_switches_previous
            self.involuntary_context_switches_previous = current
            return diff
