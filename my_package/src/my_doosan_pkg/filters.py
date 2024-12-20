import numpy as np

class Filters:
    def __init__(self, dt):
        # Initialize torque filtering variables
        self.prev_torque = None
        self.alpha = 0.5  # Filter coefficient (0.25 means 25% new, 75% previous)

        # Initialize moving average filter
        self.window_size = 10.0
        self.torque_history = []  # Changed from fixed-size list to empty list

        # Initialize torque smoothing parameters
        self.max_torque_rate = 25.0  # Maximum change in torque per second
        self.dt = dt  # Control period

    def low_pass_filter_torque(self, new_torque):
        """ Butterworth-like low pass filter for torque smoothing """
        if self.prev_torque is None:
            self.prev_torque = new_torque
            return new_torque
            
        filtered_torque = (self.alpha * new_torque + (1 - self.alpha) * self.prev_torque)
        self.prev_torque = filtered_torque
        return filtered_torque
    
    def moving_average_filter(self, new_torque):
        """ Moving average filter with exponential weighting """
        # Add new torque to history
        self.torque_history.append(new_torque)
        
        # Keep only window_size most recent values
        if len(self.torque_history) > self.window_size:
            self.torque_history.pop(0)
            
        # If history isn't full yet, return current torque
        if len(self.torque_history) < self.window_size:
            return new_torque
            
        # Calculate exponentially weighted moving average
        weights = np.exp(np.linspace(-1, 0, len(self.torque_history)))
        weights = weights / np.sum(weights)
        
        filtered_torque = np.zeros_like(new_torque)
        for i in range(len(self.torque_history)):
            filtered_torque += weights[i] * self.torque_history[i]
        return filtered_torque
    
    def smooth_torque(self, new_torque):
        """ Safe torque smoothing using exponential filter and rate limiting """
        if self.prev_torque is None:
            self.prev_torque = new_torque
            return new_torque
            
        # Calculate maximum allowed change in this timestep
        max_change = self.max_torque_rate * self.dt
        
        # Rate limiting
        torque_diff = new_torque - self.prev_torque
        limited_diff = np.clip(torque_diff, -max_change, max_change)
        rate_limited_torque = self.prev_torque + limited_diff
        
        # Exponential smoothing
        smoothed_torque = (self.alpha * rate_limited_torque + (1 - self.alpha) * self.prev_torque)
        
        # Update previous torque
        self.prev_torque = smoothed_torque
        return smoothed_torque