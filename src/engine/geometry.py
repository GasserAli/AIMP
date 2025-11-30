# File: geometry.py

import config # Import the config.py file

class Geometry:
    def __init__(self):
        """
        Initializes the Geometry class.
        It automatically reads 'safety_distance' from the imported config file.
        """
        self.safety_distance = config.safety_distance 
        
        # This is where you will store the ordered queues (v1, v2, v3...)
        self.entry_queues = {"N": [], "E": [], "S": [], "W": []}

        # This map defines the *path* (conflict points) and the *exit* (merge point).
        self.path_map = {
            # North Approach
            "N": {
                "S": {'conflicts': ['C1', 'C5', 'C11', 'C13'], 'merge': 'M_S'},
                "L": {'conflicts': ['C2', 'C6', 'C9', 'C12'], 'merge': 'M_E'},
                "R": {'conflicts': [], 'merge': 'M_W'},
            },
            # East Approach
            "E": {
                "S": {'conflicts': ['C4', 'C3', 'C2', 'C1'], 'merge': 'M_W'}, 
                "L": {'conflicts': ['C7', 'C9', 'C10', 'C14'], 'merge': 'M_S'},
                "R": {'conflicts': [], 'merge': 'M_N'},
            },
            # South Approach
            "S": {
                "S": {'conflicts': ['C16', 'C12', 'C7', 'C4'], 'merge': 'M_N'},
                "L": {'conflicts': ['C15', 'C10', 'C8', 'C5'], 'merge': 'M_W'},
                "R": {'conflicts': [], 'merge': 'M_E'},
            },
            # West Approach
            "W": {
                "S": {'conflicts': ['C13', 'C14', 'C15', 'C16'], 'merge': 'M_E'},
                "L": {'conflicts': ['C11', 'C8', 'C6', 'C3'], 'merge': 'M_N'},
                "R": {'conflicts': [], 'merge': 'M_S'},
            }
        }

    def set_trajectory(self, vehicle):
        """
        Sets the trajectory (path as a list of string names) for a vehicle.
        All paths start with the single, common base entry point.
        """
        approach = vehicle.approach
        maneuver = vehicle.maneuver

        path_info = self.path_map.get(approach, {}).get(maneuver)

        if path_info:
            conflict_points = path_info['conflicts']
            merge_point = path_info['merge']

            # --- IMPLEMENTED FIX ---
            # Define the mandatory single common entry point (e.g., S_N at (0, 80))
            base_entry_point = f"S_{approach}"
            
            # The vehicle's full path is set: [Base Entry Point] + [Conflicts] + [Merge Point]
            # This is correct for the logic required in visualization.py.
            vehicle.path = [base_entry_point] + conflict_points + [merge_point]

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