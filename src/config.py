# File: config.py
import random
from vehicle import Vehicle  # Import the Vehicle class from vehicle.py

# --- Tunable Parameters ---
velocity_range = (10,15)  # Minimum and maximum velocity for all vehicles
tau = 1                # Headway time (in seconds)
alpha = 1              # Weight for emergency vehicle delay
beta = 1               # Weight for all vehicle delay
gamma = 3   # <---- NEW: speed reward weight (tune 0.1 to 2.0)
safety_distance = 3    # Safety distance between vehicles (in meters)
inter_conflict_distance = 6  # Distance between conflict points (in meters)

# --- MODIFICATION: New randomly generated static list of 60 vehicles ---
pi = [
    Vehicle(vehicle_id=1, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=2, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=3, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=4, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=5, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=6, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=7, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=8, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=9, approach="W", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=10, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=11, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=12, approach="E", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=13, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=14, approach="N", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=15, approach="S", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=16, approach="N", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=17, approach="S", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=18, approach="W", maneuver="L", priority_status=True, velocity=velocity_range),
    Vehicle(vehicle_id=19, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=20, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=21, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=22, approach="S", maneuver="L", priority_status=True, velocity=velocity_range),
    Vehicle(vehicle_id=23, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=24, approach="E", maneuver="S", priority_status=True, velocity=velocity_range),
    Vehicle(vehicle_id=25, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=26, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=27, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=28, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=29, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    Vehicle(vehicle_id=30, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=31, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=32, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=33, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=34, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=35, approach="E", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=36, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=37, approach="E", maneuver="R", priority_status=True, velocity=velocity_range),
    # Vehicle(vehicle_id=38, approach="S", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=39, approach="S", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=40, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=41, approach="E", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=42, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=43, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=44, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=45, approach="S", maneuver="R", priority_status=True, velocity=velocity_range),
    # Vehicle(vehicle_id=46, approach="N", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=47, approach="E", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=48, approach="S", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=49, approach="W", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=50, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    # # Vehicle(vehicle_id=51, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=52, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=53, approach="W", maneuver="S", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=54, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=55, approach="E", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=56, approach="W", maneuver="L", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=57, approach="W", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=58, approach="N", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=59, approach="S", maneuver="R", priority_status=False, velocity=velocity_range),
    # Vehicle(vehicle_id=60, approach="N", maneuver="S", priority_status=False, velocity=velocity_range),
]
# --- END MODIFICATION ---


# --- Function to display all vehicle configurations ---
def display_vehicle_configurations():
    print("--- Static Vehicle List ---")
    for vehicle in pi:
         print(f"  ID {vehicle.id}: {vehicle.approach} -> {vehicle.maneuver}, Priority: {vehicle.priority_status}")
    print("---------------------------")


if __name__ == "__main__":
    print("--- Running config.py directly to test ---")
    display_vehicle_configurations()
    print("------------------------------------------")
    print(f"Total Vehicles: {len(pi)}")
    print(f"Safety distance is: {safety_distance}")
    print(f"Alpha is: {alpha}")