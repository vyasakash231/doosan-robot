#!/usr/bin/env python
import time
import ctypes
import os
import threading
import random
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Load C library for accessing thread-specific CPU time
libc = ctypes.CDLL('libc.so.6', use_errno=True)

class TimeSpec(ctypes.Structure):
    """Structure to hold time specification"""
    _fields_ = [
        ('tv_sec', ctypes.c_long),
        ('tv_nsec', ctypes.c_long)
    ]

def get_thread_clock_id(thread_id: int) -> int:
    """
    Get the clock ID for a specific thread.
    
    Args:
        thread_id: Thread ID to get clock for
        
    Returns:
        Clock ID for the thread
        
    Raises:
        OSError: If clock ID cannot be retrieved
    """
    clock_id = ctypes.c_int()
    ret = libc.pthread_getcpuclockid(thread_id, ctypes.byref(clock_id))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"Failed to get thread clock ID: {os.strerror(errno)}")
    return clock_id.value

def get_native_thread_time(thread_handle: Optional[int] = None) -> int:
    """
    Get the CPU time for a specific thread in nanoseconds.
    
    Args:
        thread_handle: Thread handle to get time for. Uses current thread if None.
        
    Returns:
        Thread CPU time in nanoseconds
        
    Raises:
        OSError: If time cannot be retrieved
    """
    if thread_handle is None:
        thread_handle = threading.get_ident()
        
    clock_id = get_thread_clock_id(thread_handle)
    timespec = TimeSpec()
    
    ret = libc.clock_gettime(clock_id, ctypes.byref(timespec))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"Failed to get thread time: {os.strerror(errno)}")
        
    return timespec.tv_sec * 1_000_000_000 + timespec.tv_nsec

def get_current_thread_time() -> int:
    """
    Get the CPU time for the current thread in nanoseconds.
    
    Returns:
        Current thread CPU time in nanoseconds
    """
    return get_native_thread_time(threading.get_ident())

def burn_cpu_cycles(duration_ns: int) -> None:
    """
    Burn CPU cycles for a specified duration.
    
    This function will keep the CPU busy for the specified duration by performing
    meaningless calculations. This is useful for testing CPU scheduling and
    real-time performance.
    
    Args:
        duration_ns: Duration to burn CPU cycles in nanoseconds
        
    Example:
        >>> # Burn CPU for 1 millisecond
        >>> burn_cpu_cycles(1_000_000)
    """
    if duration_ns <= 0:
        return
        
    end_time = get_current_thread_time() + duration_ns
    x = 0
    
    while True:
        # Perform meaningless calculations to burn CPU cycles
        while x != random.randint(0, 1000000) and x % 1000 != 0:
            x += 1
            
        # Check if we've reached the target duration
        if get_current_thread_time() >= end_time:
            break

class CpuBurner:
    """Utility class for burning CPU cycles with context manager support"""
    
    def __init__(self, duration_ns: int):
        """
        Initialize CPU burner.
        
        Args:
            duration_ns: Duration to burn CPU cycles in nanoseconds
        """
        self.duration_ns = duration_ns
        self.start_time = 0
        
    def __enter__(self):
        """Start burning CPU cycles"""
        self.start_time = get_current_thread_time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure we burn CPU for the full duration"""
        elapsed = get_current_thread_time() - self.start_time
        remaining = max(0, self.duration_ns - elapsed)
        if remaining > 0:
            burn_cpu_cycles(remaining)

def example_usage():
    """Example usage of CPU burning utilities"""
    # Example 1: Simple CPU burning
    print("Burning CPU for 100ms...")
    burn_cpu_cycles(100_000_000)  # 100ms in nanoseconds
    
    # Example 2: Using context manager
    print("Burning CPU with context manager...")
    with CpuBurner(100_000_000):  # 100ms in nanoseconds
        # Do some work here
        print("Working while ensuring minimum CPU time...")
    
    # Example 3: Get thread times
    print(f"Current thread CPU time: {get_current_thread_time()} ns")
    
    # Example 4: Measure execution time
    start = get_current_thread_time()
    burn_cpu_cycles(50_000_000)  # 50ms
    elapsed = get_current_thread_time() - start
    print(f"Actual time burned: {elapsed/1_000_000:.2f} ms")

if __name__ == "__main__":
    example_usage()
