# File: main.py
from ast import List
import random
import csv
import sys
# Import the config module to get the permutation (pi) and parameters
import config

# Import the necessary classes and functions
from geometry import Geometry
# --- Use the final, correct decoder and objective functions ---
from decoder import run_decoder # Make sure this is the final version
from objective import calculate_objective
# -----------------------------------------------------------
# Import Vehicle class for type hinting if needed, though config handles creation
from vehicle import Vehicle
# Import GA and SA runner functions for comparison tests
from ga import run_ga
from sa import run_sa
import sa as sa
import ga as ga
import matplotlib.pyplot as plt


def run_experiment(num_runs: int = 100):
    """Run GA and SA experiments and save results to CSV."""
    csv_filename = 'ga_sa_comparison_results.csv'

    print(f"Starting GA vs SA experiment ({num_runs} runs each)...")
    print(f"Results will be saved to: {csv_filename}\n")

    results = []

    # Run GA 100 times
    print("=" * 60)
    print("Running GA experiments...")
    print("=" * 60)
    for run_num in range(1, num_runs + 1):
        seed = run_num
        try:
            ga_best_perm, ga_best_speeds, ga_best_obj, _ = run_ga(
                pop_size=ga.POP_SIZE,
                generations=ga.GENERATIONS,
                random_seed=seed,
                return_history=False
            )

            results.append({
                'run': run_num,
                'algorithm': 'GA',
                'seed': seed,
                'best_f': ga_best_obj['f'],
                'total_delay': ga_best_obj['fall'],
                'emergency_delay': ga_best_obj['fem']
            })

            if run_num % 10 == 0:
                print(f"GA Run {run_num}/{num_runs} completed - f: {ga_best_obj['f']:.2f}")
        except Exception as e:
            print(f"GA Run {run_num} failed: {e}")

    print(f"\nGA experiments completed ({num_runs} runs)\n")

    # Run SA 100 times
    print("=" * 60)
    print("Running SA experiments...")
    print("=" * 60)
    for run_num in range(1, num_runs + 1):
        seed = run_num
        random.seed(seed)
        try:
            sa_best_perm, sa_best_speeds, sa_best_obj, sa_history = run_sa(
                T_init=sa.T_INITIAL,
                T_min=sa.T_MIN,
                cool_rate=sa.COOLING_RATE,
                iter_per_temp=sa.MAX_ITER_PER_TEMP,
                max_iter=sa.MAX_TOTAL_ITERATIONS,
                animation_enabled=False,
                return_history=True
            )

            # Extract total delay and emergency delay from history
            sa_total_delay = 0.0
            sa_emergency_delay = 0.0
            if sa_history and len(sa_history.get('total_delays', [])) > 0:
                sa_total_delay = sa_history['total_delays'][-1]
                sa_emergency_delay = sa_history['emergency_delays'][-1]

            # sa_best_obj is now an objective dict with keys 'f', 'fall', 'fem'
            sa_total = sa_best_obj.get('fall', sa_total_delay)
            sa_em = sa_best_obj.get('fem', sa_emergency_delay)
            results.append({
                'run': run_num,
                'algorithm': 'SA',
                'seed': seed,
                'best_f': sa_best_obj.get('f', float(sa_best_obj)),
                'total_delay': sa_total,
                'emergency_delay': sa_em
            })

            if run_num % 10 == 0:
                print(f"SA Run {run_num}/{num_runs} completed - f: {sa_best_obj.get('f', float(sa_best_obj)):.2f}")
        except Exception as e:
            print(f"SA Run {run_num} failed: {e}")

    print(f"\nSA experiments completed ({num_runs} runs)\n")

    # Save results to CSV
    print("=" * 60)
    print(f"Saving {len(results)} results to {csv_filename}...")
    print("=" * 60)

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['run', 'algorithm', 'seed', 'best_f', 'total_delay', 'emergency_delay']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to: {csv_filename}\n")

    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    ga_results = [r for r in results if r['algorithm'] == 'GA']
    sa_results = [r for r in results if r['algorithm'] == 'SA']

    if ga_results:
        ga_f_values = [r['best_f'] for r in ga_results]
        ga_delay_values = [r['total_delay'] for r in ga_results]
        ga_emergency_values = [r['emergency_delay'] for r in ga_results]

        print(f"\nGA Results (n={len(ga_results)}):")
        print(f"  Best f:")
        print(f"    Mean: {sum(ga_f_values) / len(ga_f_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_f_values)/len(ga_f_values))**2 for x in ga_f_values) / len(ga_f_values))**0.5:.2f}")
        print(f"    Min:  {min(ga_f_values):.2f}")
        print(f"    Max:  {max(ga_f_values):.2f}")
        print(f"  Total Delay:")
        print(f"    Mean: {sum(ga_delay_values) / len(ga_delay_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_delay_values)/len(ga_delay_values))**2 for x in ga_delay_values) / len(ga_delay_values))**0.5:.2f}")
        print(f"  Emergency Delay:")
        print(f"    Mean: {sum(ga_emergency_values) / len(ga_emergency_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_emergency_values)/len(ga_emergency_values))**2 for x in ga_emergency_values) / len(ga_emergency_values))**0.5:.2f}")

    if sa_results:
        sa_f_values = [r['best_f'] for r in sa_results]
        sa_delay_values = [r['total_delay'] for r in sa_results]
        sa_emergency_values = [r['emergency_delay'] for r in sa_results]

        print(f"\nSA Results (n={len(sa_results)}):")
        print(f"  Best f:")
        print(f"    Mean: {sum(sa_f_values) / len(sa_f_values):.2f}")
        print(f"    Std:  {(sum((x - sum(sa_f_values)/len(sa_f_values))**2 for x in sa_f_values) / len(sa_f_values))**0.5:.2f}")
        print(f"    Min:  {min(sa_f_values):.2f}")
        print(f"    Max:  {max(sa_f_values):.2f}")
        print(f"  Total Delay:")
        print(f"    Mean: {sum(sa_delay_values) / len(sa_delay_values):.2f}")
        print(f"    Std:  {(sum((x - sum(sa_delay_values)/len(sa_delay_values))**2 for x in sa_delay_values) / len(sa_delay_values))**0.5:.2f}")
        print(f"  Emergency Delay:")
        print(f"    Mean: {sum(sa_emergency_values) / len(sa_emergency_values):.2f}")
        print(f"    Std:  {(sum((x - sum(sa_emergency_values)/len(sa_emergency_values))**2 for x in sa_emergency_values) / len(sa_emergency_values))**0.5:.2f}")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)


def compare_ga_sa(ga_history: dict, sa_history: dict):
    """Compare GA and SA by plotting cost and total delay vs evaluations.

    ga_history: produced by run_ga with return_history=True
    sa_history: produced by run_sa with return_history=True
    """
    # Prepare SA x (iterations)
    sa_costs = sa_history.get('costs', [])
    sa_totals = sa_history.get('total_delays', [])
    sa_x = list(range(1, len(sa_costs) + 1))

    # Prepare GA x as cumulative evaluations (generations * pop_size)
    ga_costs = ga_history.get('costs', [])
    ga_totals = ga_history.get('total_delays', [])
    pop_size = ga_history.get('pop_size', 1)
    ga_x = [(g + 1) * pop_size for g in ga_history.get('generations', list(range(len(ga_costs))))]

    # Plot costs
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(sa_x, sa_costs, label='SA Cost', color='tab:blue')
    ax[0].plot(ga_x, ga_costs, label='GA Cost', color='tab:orange')
    ax[0].set_xlabel('Objective Evaluations')
    ax[0].set_ylabel('Weighted Objective (f)')
    ax[0].set_title('Cost vs Evaluations')
    ax[0].legend()
    ax[0].grid(True)

    # Plot total delay
    ax[1].plot(sa_x, sa_totals, label='SA Total Delay', color='tab:green')
    ax[1].plot(ga_x, ga_totals, label='GA Total Delay', color='tab:red')
    ax[1].set_xlabel('Objective Evaluations')
    ax[1].set_ylabel('Total Delay (s)')
    ax[1].set_title('Total Delay vs Evaluations')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

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


# Run the main function and then do a quick GA vs SA comparison
if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--experiment":
        # Run experiment mode (100 iterations)
        num_runs = 100
        if len(sys.argv) > 2:
            try:
                num_runs = int(sys.argv[2])
            except ValueError:
                pass
        run_experiment(num_runs=num_runs)
    else:
        # Run normal mode (single quick test)
        main()

    print("\n=== Running Quick GA vs SA Comparison ===")

    # Run GA with small settings for a quick test (request history)
    try:
        ga_best_perm, ga_best_speeds, ga_best_obj, ga_history = run_ga(pop_size=ga.POP_SIZE, generations=ga.GENERATIONS, random_seed=2, return_history=True)
    except Exception as e:
        print(f"GA run failed: {e}")
        ga_best_perm, ga_best_speeds, ga_best_obj, ga_history = None, None, None, None

    # Run SA with conservative settings for a quick test (request history)
    try:
        sa_best_perm, sa_best_speeds, sa_best_obj, sa_history = run_sa(T_init=sa.T_INITIAL, T_min=sa.T_MIN, cool_rate=sa.COOLING_RATE, iter_per_temp=sa.MAX_ITER_PER_TEMP, max_iter=sa.MAX_TOTAL_ITERATIONS, return_history=True)
    except Exception as e:
        print(f"SA run failed: {e}")
        sa_best_perm, sa_best_speeds, sa_best_obj, sa_history = None, None, None, None

    print("\n--- Comparison Summary ---")
    if ga_best_obj:
        print(f"GA Best f: {ga_best_obj['f']:.2f}")
        print(f"  Total Delay (fall): {ga_best_obj['fall']:.2f} s")
        print(f"  Emergency Delay (fem): {ga_best_obj['fem']:.2f} s")
    else:
        print("GA result not available.")

    if sa_best_obj:
        # sa_best_obj is an objective dict
        try:
            print(f"SA Best f: {sa_best_obj['f']:.2f}")
            print(f"  Total Delay (fall): {sa_best_obj.get('fall', 0):.2f} s")
            print(f"  Emergency Delay (fem): {sa_best_obj.get('fem', 0):.2f} s")
        except Exception:
            # Fallback if older return format
            print(f"SA Best f: {float(sa_best_obj):.2f}")
    else:
        print("SA result not available.")

    print("=== Quick Comparison Complete ===")

    # If we have both histories, show a combined comparison plot (after printing summaries)
    if ga_history and sa_history:
        try:
            compare_ga_sa(ga_history, sa_history)
        except Exception as e:
            print(f"Comparison plot failed: {e}")