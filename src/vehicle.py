# File: vehicle.py

class Vehicle:
    def __init__(self, vehicle_id, approach, maneuver, priority_status, velocity, delay):
        """
        Initializes a Vehicle object.
        """
        self.vehicle_id = vehicle_id
        self.approach = approach      # "N", "E", "S", or "W"
        self.maneuver = maneuver      # "S", "L", or "R"
        self.priority_status = priority_status
        self.velocity = velocity
        self.delay = delay
        # This will be set by the Geometry class
        self.path = []         

    def get_vehicle_info(self):
        """
        Returns a formatted string with the vehicle's details.
        """
        return (f"ID: {self.vehicle_id}, Approach: {self.approach}, "
                f"Maneuver: {self.maneuver}, Priority: {self.priority_status}, "
                f"Vel_Range: {self.velocity}")