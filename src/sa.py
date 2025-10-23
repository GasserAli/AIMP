 # File: sa.py
import math
import random
import copy
import matplotlib.pyplot as plt
import os
import traceback # For detailed error printing

# --- Import Project Files ---
import config
import objective
from geometry import Geometry
from decoder import run_decoder
# --- Import visualization with check ---
try:
    from visualization import IntersectionVisualization
    # *** RE-ENABLE ANIMATION ***
    animation_enabled = True
    print("Successfully imported visualization module.")
except ImportError:
    print("Warning: visualization.py not found or class missing. Animation disabled.")
    animation_enabled = False
# --- End Import ---

# --- SA Parameters ---
T_INITIAL = 100.0        # Initial temperature
T_MIN = 1.0               # Final temperature
COOLING_RATE = 0.99       # Cooling rate (e.g., 0.99)
MAX_ITER_PER_TEMP = 1    # Iterations at each temperature step
MAX_TOTAL_ITERATIONS = 100000 # Total iteration limit

def create_initial_solution(geom):
    """
    Generates a valid initial solution (permutation, speeds).
    Respects C0 constraint.
    """
    initial_perm = config.pi
    initial_speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range

    for approach, queue in geom.entry_queues.items():
        if not queue: continue
        v_leader = queue[0]
        # Ensure leader is in the solution
        if v_leader.id not in [v.id for v in initial_perm]: continue
        last_speed = random.uniform(v_min_global, v_max_global)
        initial_speeds_dict[v_leader.id] = last_speed
        for v_follower in queue[1:]:
             if v_follower.id not in [v.id for v in initial_perm]: continue
             current_max = min(v_max_global, last_speed)
             current_min = min(v_min_global, current_max)
             if current_min > current_max: current_min = current_max
             # Add epsilon for uniform range if min == max
             new_speed = random.uniform(current_min, current_max + 1e-9)
             initial_speeds_dict[v_follower.id] = new_speed
             last_speed = new_speed

    # Ensure all vehicles in initial_perm get a speed
    initial_speeds_list = []
    for v in initial_perm:
        if v.id not in initial_speeds_dict:
             initial_speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
             # print(f"Warning: V {v.id} missing from queues during initial speed gen.") # Reduced noise
        initial_speeds_list.append(initial_speeds_dict[v.id])

    print(f"Initial Permutation (IDs): {[v.id for v in initial_perm]}")
    return initial_perm, initial_speeds_list


def validate_speeds(permutation, speeds, geom):
    """
    Enforces the C0 (no-catch-up) constraint.
    """
    v_new = list(speeds)
    speed_dict = {p.id: s for p, s in zip(permutation, v_new)}
    # Use geom passed in (should have original config queues)
    for queue in geom.entry_queues.values():
        if not queue: continue
        if queue[0].id not in speed_dict: continue # Leader might not be in current permutation subset
        last_speed = speed_dict[queue[0].id]
        for v_follower in queue[1:]:
            if v_follower.id not in speed_dict: continue # Follower might not be in current permutation subset
            follower_speed = speed_dict[v_follower.id]
            if follower_speed > last_speed:
                speed_dict[v_follower.id] = last_speed
            last_speed = speed_dict[v_follower.id] # Update for the next car

    # Rebuild list based on the *input permutation* order
    validated_speeds_list = [speed_dict[v.id] for v in permutation]
    return validated_speeds_list


def generate_neighbor(perm_current, speeds_current, geom):
    """
    Generates a new "neighbor" solution (perm or speed change).
    Ensures C0 constraint is re-validated.
    """
    perm_new = copy.deepcopy(perm_current)
    speeds_new = copy.deepcopy(speeds_current)
    v_min_global, v_max_global = config.velocity_range

    # Choose move type
    if random.random() < 0.5 and len(perm_new) > 1:
        # Swap Permutation
        idx1, idx2 = random.sample(range(len(perm_new)), 2)
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]
    elif len(speeds_new) > 0:
        # Change Speed
        idx = random.randrange(len(speeds_new))
        change = random.uniform(-2.0, 2.0) # Speed adjustment range
        speeds_new[idx] += change
        # Clamp speed to global min/max bounds
        speeds_new[idx] = max(v_min_global, min(speeds_new[idx], v_max_global))

    # Re-validate C0 constraint *after* the change, using the *new* permutation
    speeds_new = validate_speeds(perm_new, speeds_new, geom)
    return perm_new, speeds_new


# --- evaluate_solution WITH return_full_schedule flag ---
def evaluate_solution(permutation, speeds, geom, tau_p_dict,
                      return_full_schedule=False):
    """
    Evaluates a solution (Π, v) by running the decoder and objective function.
    Handles the return_full_schedule flag.
    """
    try:
        # 1. Run the Decoder
        decoder_output = run_decoder(
            permutation=permutation,
            speeds=speeds,
            geom=geom,
            tau_p_dict=tau_p_dict,
            return_full_schedule=return_full_schedule # Pass the flag
        )

        # 2. Unpack decoder output
        if return_full_schedule:
            if isinstance(decoder_output, tuple) and len(decoder_output) == 3:
                 decoder_results, scheduled_times, t_ear = decoder_output
            else: # Handle unexpected decoder return
                 print("Error: Decoder didn't return expected tuple for full schedule.")
                 decoder_results = [{"id": v.id, "delay": 9999.0, "is_emergency": v.priority_status} for v in permutation]
                 scheduled_times, t_ear = {}, {}
        else:
            decoder_results = decoder_output
            scheduled_times, t_ear = {}, {} # Placeholders when not needed

        # 3. Calculate the Objective
        obj_dict = objective.calculate_objective(decoder_results)

        # 4. Return correct data structure
        if return_full_schedule:
            return obj_dict, scheduled_times, t_ear
        else:
            return obj_dict

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        # Return high penalty values on error
        penalized_results = [{"id": v.id, "delay": 99999.0, "is_emergency": v.priority_status} for v in permutation]
        penalized_obj = objective.calculate_objective(penalized_results)
        if return_full_schedule:
             return penalized_obj, {}, {}
        else:
             return penalized_obj
# --- END evaluate_solution modification ---


def plot_results(history_data):
    """Create a 2x2 grid of SA performance plots, each with temperature overlay."""
    costs = history_data['costs']
    avg_delays = history_data['avg_delays']
    total_delays = history_data['total_delays']
    emergency_delays = history_data['emergency_delays']
    temps = history_data['temps']

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Simulated Annealing Performance Metrics with Temperature Overlay",
                 fontsize=14, fontweight='bold')

    # --- Common function for each subplot ---
    def plot_with_temp(ax, data, color, title, ylabel):
        ax2 = ax.twinx()
        ax.plot(data, '-', color=color, label=ylabel)
        ax2.plot(temps, '--', color='purple', alpha=0.4, label='Temperature (T)')
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel, color=color)
        ax2.set_ylabel('Temperature (T)', color='purple')
        ax.grid(True)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    # --- 1. Objective Cost ---
    plot_with_temp(axs[0, 0], costs, 'blue', 'Weighted Objective Cost (f)', 'Cost (f)')

    # --- 2. Average Delay ---
    plot_with_temp(axs[0, 1], avg_delays, 'orange', 'Average Delay per Vehicle', 'Avg Delay (s)')

    # --- 3. Total Delay ---
    plot_with_temp(axs[1, 0], total_delays, 'green', 'Total Delay (f_all)', 'Total Delay (s)')

    # --- 4. Emergency Delay ---
    plot_with_temp(axs[1, 1], emergency_delays, 'red', 'Emergency Delay (f_em)', 'Emergency Delay (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS):
    """Main Simulated Annealing (SA) algorithm."""
    print("--- Starting Simulated Annealing ---")

    # --- 1. Initialization ---
    print("Initializing geometry and parameters...")
    geom_for_validation = Geometry()
    all_vehicles = config.pi
    geom_for_validation.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom_for_validation.set_trajectory(v)
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    if not all_points:
        print("Error: No vehicles or no paths found. Exiting.")
        return [], [], 0.0
    tau_p_dict = {p: config.tau for p in all_points}

    (perm_current, speeds_current) = create_initial_solution(geom_for_validation)

    # Evaluate initial solution (flag is implicitly False)
    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom_for_validation, tau_p_dict)
    obj_current = obj_dict_current['f']

    perm_best = perm_current
    speeds_best = speeds_current
    obj_best = obj_current

    T = T_init
    iter_count = 0
    history = {'costs': [], 'avg_delays': [], 'total_delays': [], 'emergency_delays': [], 'temps': []}

    print(f"Initial Solution Cost (f): {obj_best:.2f}")

    # --- 2. SA Loop ---
    while T > T_min and iter_count < max_iter:
        for i in range(iter_per_temp):
            (perm_new, speeds_new) = generate_neighbor(perm_current, speeds_current, geom_for_validation)

            # Evaluate neighbor (explicitly False for flag)
            obj_dict_new = evaluate_solution(
                permutation=perm_new,
                speeds=speeds_new,
                geom=geom_for_validation,
                tau_p_dict=tau_p_dict,
                return_full_schedule=False # Important: False during search
            )
            obj_new = obj_dict_new['f']

            ΔE = obj_new - obj_current

            # Acceptance Decision
            if ΔE < 0 or (T > 1e-9 and math.exp(-ΔE / T) > random.random()):
                perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                obj_dict_current = obj_dict_new
                if obj_current < obj_best:
                    perm_best, speeds_best, obj_best = perm_current, speeds_current, obj_current
                    print(f"  Iter {iter_count}: * New Best Solution: {obj_best:.2f}")

            iter_count += 1

            # Store History
            history['costs'].append(obj_current)
            history['temps'].append(T)
            history['total_delays'].append(obj_dict_current.get('fall', 0))
            history['emergency_delays'].append(obj_dict_current.get('fem', 0))
            current_delays = obj_dict_current.get('delays', {})
            avg_delay = sum(current_delays.values()) / len(current_delays) if current_delays else 0.0
            history['avg_delays'].append(avg_delay)

            if iter_count >= max_iter: break

        T = T * cool_rate # Geometric cooling

    # --- 3. Termination ---
    print("\n--- SA Finished ---")
    if iter_count >= max_iter: print(f"Termination: Reached max iteration limit ({max_iter}).")
    if T <= T_min: print(f"Termination: Reached minimum temperature ({T_min}). Final T={T:.2f}")
    print(f"Total iterations: {iter_count}")
    print(f"Best Objective (f): {obj_best:.2f}")
    print(f"Best Permutation (IDs): {[v.id for v in perm_best]}")
    print(f"Best Speeds: {[round(s, 2) for s in speeds_best]}")

    # --- 4. Plot History ---
    plot_results(history)

    # --- 5. Animate the BEST solution (conditional) ---
    if animation_enabled: # Check the flag set during import
        print("\n--- Animating Best Solution ---")
        print("Re-running decoder to get full schedule for animation...")
        try:
            # Correctly call evaluate_solution with the flag
            obj_dict, final_schedule, final_tear = evaluate_solution(
                permutation=perm_best,
                speeds=speeds_best,
                geom=geom_for_validation, # Use the geom object with original queues
                tau_p_dict=tau_p_dict,
                return_full_schedule=True # Request full schedule
            )

            # --- Create speeds dictionary needed by animator ---
            final_speeds_dict = {v.id: s for v, s in zip(perm_best, speeds_best)}
            # --- End modification ---

            animator = IntersectionVisualization()
            # --- Pass speeds_dict to load_schedule ---
            animator.load_schedule(perm_best, final_schedule, final_tear, final_speeds_dict, tau_p_dict)
            # --- End modification ---

            print("Starting animation window...")
            animator.start_animation() # This will block until the window is closed
        except NameError as ne:
            print(f"Animation skipped: Required name not found - {ne}")
        except Exception as e:
             print(f"An error occurred during animation setup or execution: {e}")
             traceback.print_exc()
    else:
        print("\nAnimation disabled (visualization module not found or failed to import).")

    return perm_best, speeds_best, obj_best


if __name__ == "__main__":
    run_sa()