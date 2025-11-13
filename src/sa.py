# File: sa.py
import math
import random
import copy
import matplotlib.pyplot as plt
import os
import traceback 

# --- Import Project Files ---
import config
import objective
from geometry import Geometry
from decoder import run_decoder
try:
    from visualization import IntersectionVisualization
    animation_enabled = True
    print("Successfully imported visualization module.")
except ImportError:
    print("Warning: visualization.py not found or class missing. Animation disabled.")
    animation_enabled = False
# --- End Import ---

# --- SA Parameters ---
T_INITIAL = 100.0
T_MIN = 1.0
COOLING_RATE = 0.99
MAX_ITER_PER_TEMP = 1
MAX_TOTAL_ITERATIONS = 100000 

def create_initial_solution(geom):
    """
    Generates a valid initial solution (permutation, speeds).
    Respects C0 constraint.
    """
    # --- FIX: This now creates a random permutation ---
    initial_perm = copy.deepcopy(config.pi)
    random.shuffle(initial_perm)
    
    initial_speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range

    for approach, queue in geom.entry_queues.items():
        if not queue: continue
        
        leader_in_queue = None
        for v in queue:
            if v.id in [p.id for p in initial_perm]:
                leader_in_queue = v
                break
        
        if leader_in_queue is None:
            continue # No vehicles from this queue

        last_speed = random.uniform(v_min_global, v_max_global)
        initial_speeds_dict[leader_in_queue.id] = last_speed
        
        # Find followers
        followers_in_queue = [v for v in queue if v.id != leader_in_queue.id and v.id in [p.id for p in initial_perm]]
        
        for v_follower in followers_in_queue:
             current_max = min(v_max_global, last_speed)
             current_min = min(v_min_global, current_max)
             if current_min > current_max: current_min = current_max
             
             new_speed = random.uniform(current_min, current_max + 1e-9) # Add epsilon
             initial_speeds_dict[v_follower.id] = new_speed
             last_speed = new_speed

    # Ensure all vehicles in initial_perm get a speed
    initial_speeds_list = []
    for v in initial_perm:
        if v.id not in initial_speeds_dict:
             initial_speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
        initial_speeds_list.append(initial_speeds_dict[v.id])

    print(f"  Initial SA Perm (IDs): {[v.id for v in initial_perm]}")
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
        
        # Find leader in this permutation
        leader_in_queue = None
        for v in queue:
            if v.id in speed_dict:
                leader_in_queue = v
                break
        if not leader_in_queue: continue
        
        last_speed = speed_dict[leader_in_queue.id]
        
        # Find followers
        followers_in_queue = [v for v in queue if v.id != leader_in_queue.id and v.id in speed_dict]
        
        for v_follower in followers_in_queue:
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
    """
    perm_new = copy.deepcopy(perm_current)
    speeds_new = copy.deepcopy(speeds_current)
    v_min_global, v_max_global = config.velocity_range

    if random.random() < 0.5 and len(perm_new) > 1:
        # Swap Permutation
        idx1, idx2 = random.sample(range(len(perm_new)), 2)
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]
    elif len(speeds_new) > 0:
        # Change Speed
        idx = random.randrange(len(speeds_new))
        change = random.uniform(-2.0, 2.0)
        speeds_new[idx] += change
        speeds_new[idx] = max(v_min_global, min(speeds_new[idx], v_max_global))

    speeds_new = validate_speeds(perm_new, speeds_new, geom)
    return perm_new, speeds_new


def evaluate_solution(permutation, speeds, geom, tau_p_dict,
                      return_full_schedule=False):
    """
    Evaluates a solution (Π, v) by running the decoder and objective function.
    """
    try:
        decoder_output = run_decoder(
            permutation=permutation,
            speeds=speeds,
            geom=geom,
            tau_p_dict=tau_p_dict,
            return_full_schedule=return_full_schedule
        )

        if return_full_schedule:
            if isinstance(decoder_output, tuple) and len(decoder_output) == 3:
                 decoder_results, scheduled_times, t_ear = decoder_output
            else: 
                 print("Error: Decoder didn't return expected tuple for full schedule.")
                 decoder_results = [{"id": v.id, "delay": 9999.0, "is_emergency": v.priority_status} for v in permutation]
                 scheduled_times, t_ear = {}, {}
        else:
            decoder_results = decoder_output
            scheduled_times, t_ear = {}, {} 

        obj_dict = objective.calculate_objective(decoder_results)

        if return_full_schedule:
            return obj_dict, scheduled_times, t_ear
        else:
            return obj_dict

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        penalized_results = [{"id": v.id, "delay": 99999.0, "is_emergency": v.priority_status} for v in permutation]
        penalized_obj = objective.calculate_objective(penalized_results)
        if return_full_schedule:
             return penalized_obj, {}, {}
        else:
             return penalized_obj


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

    plot_with_temp(axs[0, 0], costs, 'blue', 'Weighted Objective Cost (f)', 'Cost (f)')
    plot_with_temp(axs[0, 1], avg_delays, 'orange', 'Average Delay per Vehicle', 'Avg Delay (s)')
    plot_with_temp(axs[1, 0], total_delays, 'green', 'Total Delay (f_all)', 'Total Delay (s)')
    plot_with_temp(axs[1, 1], emergency_delays, 'red', 'Emergency Delay (f_em)', 'Emergency Delay (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ---
# --- THIS IS THE NEW, CORRECTED FUNCTION DEFINITION ---
# ---
def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS,
           initial_solution=None, verbose=True):
    """Main Simulated Annealing (SA) algorithm."""
    
    if verbose:
        print("--- Starting Simulated Annealing ---")
        print("Initializing geometry and parameters...")
        
    geom_for_validation = Geometry()
    all_vehicles = config.pi
    geom_for_validation.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom_for_validation.set_trajectory(v)
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    if not all_points:
        print("Error: No vehicles or no paths found. Exiting.")
        return [], [], 0.0, {}, None, None, 0

    tau_p_dict = {p: config.tau for p in all_points}

    # --- Use provided initial solution if available ---
    if initial_solution:
        if verbose:
            print("Using provided initial solution.")
        (perm_current, speeds_current) = (copy.deepcopy(initial_solution[0]), copy.deepcopy(initial_solution[1]))
    else:
        if verbose:
            print("Creating new initial solution...")
        (perm_current, speeds_current) = create_initial_solution(geom_for_validation)

    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom_for_validation, tau_p_dict)
    obj_current = obj_dict_current['f']

    perm_best = perm_current
    speeds_best = speeds_current
    obj_best = obj_current

    T = T_init
    iter_count = 1 
    history = {'costs': [], 'avg_delays': [], 'total_delays': [], 'emergency_delays': [], 'temps': []}
    
    history['costs'].append(obj_current)
    history['temps'].append(T)
    history['total_delays'].append(obj_dict_current.get('fall', 0))
    history['emergency_delays'].append(obj_dict_current.get('fem', 0))
    current_delays = obj_dict_current.get('delays', {})
    avg_delay = sum(current_delays.values()) / len(current_delays) if current_delays else 0.0
    history['avg_delays'].append(avg_delay)

    if verbose:
        print(f"Initial Solution Cost (f): {obj_best:.2f}")

    while T > T_min and iter_count < max_iter:
        for i in range(iter_per_temp):
            (perm_new, speeds_new) = generate_neighbor(perm_current, speeds_current, geom_for_validation)

            obj_dict_new = evaluate_solution(
                permutation=perm_new,
                speeds=speeds_new,
                geom=geom_for_validation,
                tau_p_dict=tau_p_dict,
                return_full_schedule=False
            )
            obj_new = obj_dict_new['f']
            
            iter_count += 1 

            ΔE = obj_new - obj_current

            if ΔE < 0 or (T > 1e-9 and math.exp(-ΔE / T) > random.random()):
                perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                obj_dict_current = obj_dict_new
                if obj_current < obj_best:
                    perm_best, speeds_best, obj_best = perm_current, speeds_current, obj_current
                    if verbose:
                        print(f"  Iter {iter_count}: * New Best Solution: {obj_best:.2f}")

            history['costs'].append(obj_current)
            history['temps'].append(T)
            history['total_delays'].append(obj_dict_current.get('fall', 0))
            history['emergency_delays'].append(obj_dict_current.get('fem', 0))
            current_delays = obj_dict_current.get('delays', {})
            avg_delay = sum(current_delays.values()) / len(current_delays) if current_delays else 0.0
            history['avg_delays'].append(avg_delay)

            if iter_count >= max_iter: break
        
        if iter_count >= max_iter: break 

        T = T * cool_rate 

    if verbose:
        print("\n--- SA Finished ---")
        if iter_count >= max_iter: print(f"Termination: Reached max iteration/evaluation limit ({max_iter}).")
        if T <= T_min: print(f"Termination: Reached minimum temperature ({T_min}). Final T={T:.2f}")
        print(f"Total evaluations: {iter_count}")
        print(f"Best Objective (f): {obj_best:.2f}")
        # print(f"Best Permutation (IDs): {[v.id for v in perm_best]}")
        # print(f"Best Speeds: {[round(s, 2) for s in speeds_best]}")
    
    return perm_best, speeds_best, obj_best, history, geom_for_validation, tau_p_dict, iter_count


if __name__ == "__main__":
    try:
        from visualization import IntersectionVisualization
        animation_enabled_standalone = True
    except ImportError:
        animation_enabled_standalone = False
        print("Warning: visualization.py not found. Animation disabled.")

    
    (perm_best, speeds_best, obj_best, history, 
     geom, tau_p_dict, evals) = run_sa()
    
    plot_results(history)
    
    if animation_enabled_standalone:
        print("\n--- Animating Best Solution (Standalone Run) ---")
        try:
            obj_dict, final_schedule, final_tear = evaluate_solution(
                permutation=perm_best,
                speeds=speeds_best,
                geom=geom,
                tau_p_dict=tau_p_dict,
                return_full_schedule=True
            )
            final_speeds_dict = {v.id: s for v, s in zip(perm_best, speeds_best)}
            animator = IntersectionVisualization()
            animator.load_schedule(perm_best, final_schedule, final_tear, final_speeds_dict, tau_p_dict)
            print("Starting animation window...")
            animator.start_animation()
        except Exception as e:
             print(f"An error occurred during animation setup: {e}")
             traceback.print_exc()