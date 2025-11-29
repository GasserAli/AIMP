# File: config.py
import random
from engine.vehicle import Vehicle  # Import the Vehicle class from vehicle.py

# --- Tunable Config Parameters, this is the enviroment setup ---
velocity_range = (6,18)  # Minimum and maximum velocity for all vehicles
avg_velocity = sum(velocity_range) / 2  # Average velocity for reference
avg_vehicle_length = 4.5  # Average vehicle length (in meters)
tau = avg_vehicle_length/avg_velocity   # Headway time (in seconds)
alpha = 1.0              # Weight for emergency vehicle delay
beta = 1.0               # Weight for all vehicle delay
safety_distance = 5    # Safety distance between vehicles (in meters)
inter_conflict_distance = 5  # Distance between conflict points (in meters)

# --- Constraint Allowance ---
speed_penalty_coeff = 0.0   # penalize solutions with speeds far from v_max
# conflict_weight = 0.0  # penalize conflicts in objective function
follow_slack= 0.05  # Additional slack time for following vehicles (in seconds), this allows more flexibilty for constraint C0


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
    # Vehicle(vehicle_id=51, approach="S", maneuver="L", priority_status=False, velocity=velocity_range),
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