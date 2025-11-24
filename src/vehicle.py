# File: vehicle.py

class Vehicle:
    def __init__(self, vehicle_id, approach, maneuver, priority_status, velocity, delay=0.0):
        """
        Initializes a Vehicle object.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            approach: Direction of approach (N, S, E, W)
            maneuver: Type of maneuver (S, L, R for straight, left, right)
            priority_status: Boolean for emergency vehicle status
            velocity: Tuple (min, max) velocity range or initial velocity
            delay: Optional initial delay value (default 0.0)
        """
        self.vehicle_id = vehicle_id
        self.id = vehicle_id
        self.approach = approach
        self.maneuver = maneuver
        self.priority_status = priority_status
        self.velocity = velocity  # Keep for compatibility (the range tuple)
        self.delay = delay
        self.path = []
        
        # NEW: Segment-wise speeds (5 segments)
        # Initialize with None, will be set by optimizer
        self.segment_speeds = [None] * 5
    
    def set_segment_speeds(self, speeds_list):
        """Set the 5 segment speeds for this vehicle."""
        if len(speeds_list) != 5:
            raise ValueError(f"Expected 5 speeds, got {len(speeds_list)}")
        self.segment_speeds = list(speeds_list)
    
    def get_segment_speeds(self):
        """Returns the 5 segment speeds."""
        return self.segment_speeds
    
    def get_average_speed(self):
        """Returns average of the 5 segment speeds."""
        if None in self.segment_speeds:
            return 0.0
        return sum(self.segment_speeds) / 5
    
    def get_vehicle_info(self):
        """Returns formatted string with vehicle details."""
        avg_speed = self.get_average_speed()
        return (f"ID: {self.vehicle_id}, Approach: {self.approach}, "
                f"Maneuver: {self.maneuver}, Priority: {self.priority_status}, "
                f"Avg Speed: {avg_speed:.2f} m/s")
    
    def set_delay(self, delay):
        """Sets the delay for the vehicle."""
        self.delay = delay