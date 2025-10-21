# File: main.py

# Import the config module to get the permutation (pi)
import config

# Import the Geometry class
from geometry import Geometry
from decoder import is_permutation_valid
from objective import objective_from_queues

def main():
    """
    This function runs the test you described:
    1. Takes permutation from config
    2. Applies geometry
    3. Shows the resulting queues and paths
    """
    print("--- Running Test with Config Permutation ---")

    # 1. Get the permutation (pi) from config
    all_vehicles = config.pi
    print(f"Loaded {len(all_vehicles)} vehicles from config.pi.")

    # 2. Apply Geometry
    # Create an instance of the Geometry class
    intersection_geom = Geometry()
    print(f"Geometry initialized (Safety Distance: {intersection_geom.safety_distance}m).")

    # A) Set the path for each vehicle
    for vehicle in all_vehicles:
        intersection_geom.set_trajectory(vehicle)

    print(is_permutation_valid(all_vehicles,distance_to_first_conflict=10.0,inter_conflict_distance=10.0,safety_time=config.tau))

    for vehicle in all_vehicles:
        print(f"Vehicle {vehicle.vehicle_id} delay: {vehicle.delay}")
        
    # B) Create the entry queues
    intersection_geom.create_entry_queue(all_vehicles)

    # 3. Show Assigned Paths (Output 2)
    print("\n---Assigned Vehicle Paths ---")
    for v in all_vehicles:
        # We access v.path, which was set by intersection_geom
        print(f"  Vehicle {v.vehicle_id} ({v.approach}, {v.maneuver}): {v.path}")

    # 4. Show Lane Queues (Output 1)
    print("\n---Output Lane Queues ---")
    for approach, queue in intersection_geom.entry_queues.items():
        # Get just the IDs for a cleaner print
        v_ids = [v.vehicle_id for v in queue]
        print(f"  {approach} Queue: {v_ids}")

    print("\n---Test Complete ---")

    result = objective_from_queues(all_vehicles, alpha=config.alpha, beta=config.beta)
    print("\nObjective Results:")
    print(f"  Delays: {result['delays']}")  
    print(f"  Emergency Delay (fem): {result['fem']}")
    print(f"  Total Delay (fall): {result['fall']}")
    print(f"  Weighted Objective (f): {result['f']}")


# This standard Python line calls the main() function when you run the script
if __name__ == "__main__":
    main()