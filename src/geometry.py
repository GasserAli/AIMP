# File: geometry.py

import config  # Import the config.py file

class Geometry:
    def __init__(self):
        """
        Initializes the Geometry class.
        It automatically reads 'safety_distance' from the imported config file.
        """
        self.safety_distance = config.safety_distance 
        
        # This is where you will store the ordered queues (v1, v2, v3...)
        # for each lane, just as you described.
        self.entry_queues = {"N": [], "E": [], "S": [], "W": []}

        # This map defines the *path* (conflict points) and the
        # *exit* (merge point) for every possible move.
        self.path_map = {
            # North Approach
            "N": {
                "S": {'conflicts': ['C1', 'C5', 'C10', 'C13'], 'merge': 'M_S'},
                "L": {'conflicts': ['C2', 'C6', 'C9', 'C12'], 'merge': 'M_E'},
                "R": {'conflicts': [], 'merge': 'M_W'},
            },
            # East Approach
            "E": {
                # Your example: E-S vehicle goes through C4, C3, C2, C1...
                "S": {'conflicts': ['C4', 'C3', 'C2', 'C1'], 'merge': 'M_W'}, # ...and exits at M_W
                "L": {'conflicts': ['C7', 'C9', 'C11', 'C14'], 'merge': 'M_S'},
                "R": {'conflicts': [], 'merge': 'M_N'},
            },
            # South Approach
            "S": {
                "S": {'conflicts': ['C16', 'C12', 'C7', 'C4'], 'merge': 'M_N'},
                "L": {'conflicts': ['C15', 'C11', 'C8', 'C5'], 'merge': 'M_W'},
                "R": {'conflicts': [], 'merge': 'M_E'},
            },
            # West Approach
            "W": {
                "S": {'conflicts': ['C13', 'C14', 'C15', 'C16'], 'merge': 'M_E'},
                "L": {'conflicts': ['C10', 'C8', 'C6', 'C3'], 'merge': 'M_N'},
                "R": {'conflicts': [], 'merge': 'M_S'},
            }
        }

    def set_trajectory(self, vehicle):
        """
        Sets the trajectory (path as a list of string names) for a vehicle
        based on its approach and maneuver using the path_map.
        """
        approach = vehicle.approach
        maneuver = vehicle.maneuver

        # 1. This is the unique "entry point" for the lane.
        # e.g., 'S_N' for North, 'S_E' for East.
        # A vehicle must acquire this point to enter the intersection.
        path_info = self.path_map.get(approach, {}).get(maneuver)

        if path_info:
            conflict_points = path_info['conflicts']
            
            # 2. This is the "merge point" (exit) for that specific path.
            merge_point = path_info['merge']
            
            # 3. The vehicle's full path is set: [Entry Point, Conflicts..., Merge Point]
            vehicle.path = conflict_points + [merge_point]
        else:
            print(f"Warning: No path defined for approach {approach}, maneuver {maneuver}")
            vehicle.path = []

    def create_entry_queue(self, vehicles):
        """
        Creates a queue for each entry lane based on the vehicles' approach.

        """
        # Clear queues in case this is run multiple times
        self.entry_queues = {"N": [], "E": [], "S": [], "W": []}
        
        for vehicle in vehicles:
            if vehicle.approach in self.entry_queues:
                # Appending vehicles creates the First-In, First-Out queue
                self.entry_queues[vehicle.approach].append(vehicle)
            else:
                print(f"Warning: Vehicle {vehicle.vehicle_id} has invalid approach: {vehicle.approach}")



# Example usage:
# [Running] python -u "c:\Users\Dell\Desktop\GUC\Semester 9\Optimization\AIMP\src\main.py"
# --- Running Test with Config Permutation ---
# Loaded 4 vehicles from config.pi.
# Geometry initialized (Safety Distance: 2.0m).

# ---Assigned Vehicle Paths ---
#   Vehicle 1 (N, S): ['C1', 'C5', 'C10', 'C13', 'M_S']
#   Vehicle 2 (N, L): ['C2', 'C6', 'C9', 'C12', 'M_E']
#   Vehicle 3 (N, R): ['M_W']
#   Vehicle 4 (W, S): ['C13', 'C14', 'C15', 'C16', 'M_E']

# ---Output Lane Queues ---
#   N Queue: [1, 2, 3]
#   E Queue: []
#   S Queue: []
#   W Queue: [4]

# ---Test Complete ---
