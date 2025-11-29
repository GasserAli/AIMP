# File: main.py
from ast import List
import random
# Import the config module to get the permutation (pi) and parameters
import config

# Import the necessary classes and functions
from engine.geometry import Geometry
# --- Use the final, correct decoder and objective functions ---
from engine.decoder import run_decoder # Make sure this is the final version
from engine.objective import calculate_objective
# -----------------------------------------------------------
# Import Vehicle class for type hinting if needed, though config handles creation
from engine.vehicle import Vehicle

def main():
    """
    This function tests the final decoder and objective function:
    1. Takes permutation from config
    2. Applies geometry to get paths
    3. Generates example speeds
    4. Runs the decoder
    5. Calculates the objective
    6. Shows paths, queues, and objective results
    """
    print("--- Running Test with Final Decoder ---")

    # --- 1. Initialization ---
    all_vehicles: list[Vehicle] = config.pi # Type hint for clarity
    print(f"Loaded {len(all_vehicles)} vehicles from config.pi.")

    # Apply Geometry to get paths and initial queues
    intersection_geom = Geometry()
    # Create queues based on the order defined in config.pi for t_ear calculation
    intersection_geom.create_entry_queue(all_vehicles)
    for vehicle in all_vehicles:
        intersection_geom.set_trajectory(vehicle)

    print(f"Geometry initialized (Safety Distance: {config.safety_distance}m).")
    # Display all relevant config parameters being used
    print(f"Config Parameters: Velocity=({config.velocity_range[0]}-{config.velocity_range[1]}) m/s, Tau={config.tau}s, "
          f"SafetyDist={config.safety_distance}m, InterConflictDist={config.inter_conflict_distance}m")

    # --- 2. Generate Example Speeds ---
    # Use the minimum speed from the range, respecting C0
    speeds_dict = {}
    v_min, v_max = config.velocity_range
    print("\nGenerating Example Speeds (using min speed, respecting C0):")
    for approach, queue in intersection_geom.entry_queues.items():
        last_speed_in_queue = v_max # Start constraint check with max speed
        if not queue: continue
        print(f"  Queue {approach}:")
        for v in queue:
            # Assign min speed, but ensure it's not faster than the car ahead
            assigned_speed = min(v_min, last_speed_in_queue)
            speeds_dict[v.id] = assigned_speed
            last_speed_in_queue = assigned_speed # Update constraint for next car
            print(f"    V{v.id}: {assigned_speed:.2f} m/s")

    # Create the speeds list in the same order as the permutation (config.pi)
    try:
        example_speeds = [speeds_dict[v.id] for v in all_vehicles]
    except KeyError as e:
         print(f"\nError: Could not find speed for vehicle ID {e}. "
               "Check if all vehicles in config.pi are correctly processed.")
         return


    # --- 3. Create Tau Dictionary ---
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    if not all_points:
        print("\nError: No conflict points found in assigned vehicle paths. Check geometry.py.")
        return
    # Use the global tau from config for all points
    tau_p_dict = {p: config.tau for p in all_points}

    # --- 4. Run the Decoder ---
    print("\nRunning Decoder...")
    try:
        # Pass the geom object (which has the initial queues needed for t_ear)
        decoder_results = run_decoder(
            permutation=all_vehicles, # Using config.pi order as the permutation for this test
            speeds=example_speeds,
            geom=intersection_geom,
            tau_p_dict=tau_p_dict,
            return_full_schedule=False # Don't need full schedule details here
        )
        print("Decoder finished.")
    except Exception as e:
        print(f"\nError during decoder execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return


    # --- 5. Calculate the Objective ---
    print("\nCalculating Objective...")
    try:
        result = calculate_objective(decoder_results)
        print("Objective calculation finished.")
    except Exception as e:
        print(f"\nError during objective calculation: {e}")
        return

    # --- 6. Show Results ---
    print("\n--- Assigned Vehicle Paths ---")
    for v in all_vehicles:
        # Use getattr for safety in case 'id' attribute is missing
        print(f"  Vehicle {getattr(v, 'id', 'N/A')} ({getattr(v, 'approach', 'N/A')}, {getattr(v, 'maneuver', 'N/A')}): {getattr(v, 'path', 'N/A')}")

    print("\n--- Initial Lane Queues (from config order) ---")
    for approach, queue in intersection_geom.entry_queues.items():
        v_ids = [getattr(v, 'id', 'N/A') for v in queue]
        print(f"  {approach} Queue: {v_ids}")

    print("\n--- Objective Function Results ---")
    try:
        # Sort delays by vehicle ID for clarity
        sorted_delays = dict(sorted(result['delays'].items()))
        # Format delays to 2 decimal places
        formatted_delays = {k: f"{v:.2f}" for k, v in sorted_delays.items()}
        print(f"  Calculated Delays (s): {formatted_delays}")
        print(f"  Emergency Delay (fem): {result['fem']:.2f}")
        print(f"  Total Delay (fall): {result['fall']:.2f}")
        print(f"  Weighted Objective (f): {result['f']:.2f}")
        print(f"  (Based on config: alpha={config.alpha}, beta={config.beta})")
    except KeyError:
         print("  Error: Objective result dictionary is missing expected keys ('delays', 'fem', 'fall', 'f').")
    except Exception as e:
         print(f"  Error formatting or printing objective results: {e}")


    print("\n--- Test Complete ---")


# Run the main function
if __name__ == "__main__":
    main()