# File: config.py

from vehicle import Vehicle  # Import the Vehicle class from vehicle.py
# --- Tunable Parameters ---
# These variables can be imported by any other file.

velocity_range = (10,12)  # Minimum and maximum velocity for all vehicles
tau = 1.0                # Headway time (in seconds)
alpha = 10.0
beta = 1.0
safety_distance = 4    # Safety distance between vehicles (in meters)
inter_conflict_distance = 4  # Distance between conflict points (in meters)
# --- Permutation of vehicles (pi) ---
# This list can also be imported by other files.
pi = [
    Vehicle(vehicle_id=1, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=2, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=3, approach="N", maneuver="S", priority_status=True, velocity=velocity_range),
    Vehicle(vehicle_id=4, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=5, approach="E", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=6, approach="E", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=7, approach="S", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=8, approach="W", maneuver="L", priority_status=True, velocity=velocity_range),
    Vehicle(vehicle_id=9, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=10, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=11, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=12, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=13, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=14, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=15, approach="N", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=16, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=17, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=18, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=19, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=20, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=21, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=22, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=23, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=24, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=25, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=26, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=27, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=28, approach="W", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=29, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=30, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
]

# --- Function to display all vehicle configurations ---
def display_vehicle_configurations():
    for vehicle in pi:
        print(vehicle.get_vehicle_info())

# This 'if' block only runs when you execute 'python config.py' directly.
# It will NOT run when another file imports 'config'.
if __name__ == "__main__":
    print("--- Running config.py directly to test ---")
    display_vehicle_configurations()
    print("------------------------------------------")
    print(f"Safety distance is: {safety_distance}")
    print(f"Alpha is: {alpha}")