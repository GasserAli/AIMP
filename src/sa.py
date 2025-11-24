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
T_INITIAL = 100.0  # Increased from 50.0 to allow more exploration
T_MIN = 0.5        # Lowered from 1.0 to continue longer
COOLING_RATE = 0.98  # Slowed from 0.99 to cool more gradually
MAX_ITER_PER_TEMP = 3  # Increased from 2 for more iterations per temp
MAX_TOTAL_ITERATIONS = 100000 

def create_initial_solution(geom):
    """
    Generates a valid initial solution (permutation, segment_speeds_matrix).
    Each vehicle has 5 segment speeds.
    """
    initial_perm = copy.deepcopy(config.pi)
    random.shuffle(initial_perm)
    
    v_min, v_max = config.velocity_range
    num_vehicles = len(initial_perm)
    num_segments = 5
    
    # Initialize: random speed for each vehicle-segment pair
    segment_speeds_matrix = [
        [random.uniform(v_min, v_max) for _ in range(num_segments)]
        for _ in range(num_vehicles)
    ]
    
    # Apply C0 constraint PER SEGMENT
    segment_speeds_matrix = validate_segment_speeds(initial_perm, segment_speeds_matrix, geom)
    
    print(f"  Initial Permutation (IDs): {[v.id for v in initial_perm]}")
    return initial_perm, segment_speeds_matrix


def validate_segment_speeds(permutation, segment_speeds_matrix, geom):
    """
    Enforces C0 constraint PER SEGMENT: followers cannot exceed leader speed in same segment.
    """
    v_min, v_max = config.velocity_range
    validated = [row[:] for row in segment_speeds_matrix]
    
    for segment_idx in range(5):  # 5 segments
        for approach, queue in geom.entry_queues.items():
            if not queue:
                continue
            
            # Find leader in this queue
            leader_idx = None
            for v in queue:
                perm_indices = [i for i, p in enumerate(permutation) if p.id == v.id]
                if perm_indices:
                    leader_idx = perm_indices[0]
                    break
            
            if leader_idx is None:
                continue
            
            last_speed = validated[leader_idx][segment_idx]
            
            # Apply to followers
            for v in queue:
                follower_indices = [i for i, p in enumerate(permutation) if p.id == v.id]
                if not follower_indices or follower_indices[0] == leader_idx:
                    continue
                
                follower_idx = follower_indices[0]
                if validated[follower_idx][segment_idx] > last_speed:
                    validated[follower_idx][segment_idx] = last_speed
                last_speed = validated[follower_idx][segment_idx]
    
    return validated


def generate_neighbor(perm_current, segment_speeds_current, geom):
    """
    Generates a neighbor by mutating segment speeds (80%) or permutation (20%).
    Each vehicle has 5 segment speeds.
    """
    perm_new = copy.deepcopy(perm_current)
    speeds_new = [row[:] for row in segment_speeds_current]  # Deep copy 2D list
    v_min, v_max = config.velocity_range
    
    if random.random() < 0.2 and len(perm_new) > 1:
        # 20%: Permutation swap
        idx1, idx2 = random.sample(range(len(perm_new)), 2)
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]
    else:
        # 80%: Speed mutation
        vehicle_idx = random.randrange(len(speeds_new))
        segment_idx = random.randrange(5)  # 5 segments
        
        mutation_strategy = random.random()
        if mutation_strategy < 0.3:  # 30%: Large increase
            change = random.uniform(0.5, 3.0)
        elif mutation_strategy < 0.5:  # 20%: Small increase
            change = random.uniform(0.1, 1.0)
        elif mutation_strategy < 0.8:  # 30%: Small decrease
            change = random.uniform(-1.0, -0.1)
        else:  # 20%: Large decrease
            change = random.uniform(-3.0, -0.5)
        
        speeds_new[vehicle_idx][segment_idx] += change
        speeds_new[vehicle_idx][segment_idx] = max(v_min, min(speeds_new[vehicle_idx][segment_idx], v_max))
    
    speeds_new = validate_segment_speeds(perm_new, speeds_new, geom)
    return perm_new, speeds_new


def evaluate_solution(permutation, segment_speeds_matrix, geom, tau_p_dict, return_full_schedule=False):
    """
    Evaluates a solution (Π, segment_speeds) by running decoder and objective.
    NOW passes speeds_matrix to objective for speed incentive calculation.
    """
    try:
        decoder_output = run_decoder(
            permutation=permutation,
            segment_speeds_matrix=segment_speeds_matrix,
            geom=geom,
            tau_p_dict=tau_p_dict,
            return_full_schedule=return_full_schedule
        )

        if return_full_schedule:
            if isinstance(decoder_output, tuple) and len(decoder_output) >= 3:
                decoder_results, scheduled_times, t_ear = decoder_output[:3]
            else:
                print("Error: Decoder didn't return expected tuple.")
                decoder_results = [{"id": v.id, "delay": 9999.0, "is_emergency": v.priority_status} for v in permutation]
                scheduled_times, t_ear = {}, {}
        else:
            decoder_results = decoder_output
            scheduled_times, t_ear = {}, {}

        # NEW: Pass segment_speeds_matrix to objective for speed incentive
        obj_dict = objective.calculate_objective(decoder_results, speeds_matrix=segment_speeds_matrix)
        
        if return_full_schedule:
            return obj_dict, scheduled_times, t_ear
        else:
            return obj_dict

    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        penalized_results = [{"id": v.id, "delay": 99999.0, "is_emergency": v.priority_status} for v in permutation]
        # NEW: Still pass speeds to penalized objective
        penalized_obj = objective.calculate_objective(penalized_results, speeds_matrix=segment_speeds_matrix)
        if return_full_schedule:
            return penalized_obj, {}, {}
        else:
            return penalized_obj


def plot_results(history_data):
    """Create a 3x2 grid of SA performance plots, each with temperature overlay."""
    costs = history_data['costs']
    avg_delays = history_data['avg_delays']
    total_delays = history_data['total_delays']
    emergency_delays = history_data['emergency_delays']
    conflicts = history_data.get('conflicts', [0] * len(costs))  # Handle missing conflicts
    temps = history_data['temps']

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Simulated Annealing Performance Metrics with Temperature Overlay",
                 fontsize=14, fontweight='bold')

    def plot_with_temp(ax, data, color, title, ylabel):
        ax2 = ax.twinx()
        ax.plot(data, '-', color=color, label=ylabel, linewidth=1.5)
        ax2.plot(temps, '--', color='purple', alpha=0.4, label='Temperature (T)')
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel, color=color)
        ax2.set_ylabel('Temperature (T)', color='purple')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

    plot_with_temp(axs[0, 0], costs, 'blue', 'Weighted Objective Cost (f)', 'Cost (f)')
    plot_with_temp(axs[0, 1], avg_delays, 'orange', 'Average Delay per Vehicle', 'Avg Delay (s)')
    plot_with_temp(axs[1, 0], total_delays, 'green', 'Total Delay (f_all)', 'Total Delay (s)')
    plot_with_temp(axs[1, 1], emergency_delays, 'red', 'Emergency Delay (f_em)', 'Emergency Delay (s)')
    plot_with_temp(axs[2, 0], conflicts, 'brown', 'Number of Conflicts', 'Conflicts')
    
    # Empty plot on bottom right for symmetry
    axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS,
           initial_solution=None, verbose=True):
    """Main Simulated Annealing with segment-wise speeds."""
    
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
        print("Error: No vehicles or paths found.")
        return [], [], 0.0, {}, None, None, 0

    tau_p_dict = {p: config.tau for p in all_points}

    if initial_solution:
        if verbose:
            print("Using provided initial solution.")
        perm_current = copy.deepcopy(initial_solution[0])
        speeds_current = [row[:] for row in initial_solution[1]]
    else:
        if verbose:
            print("Creating new initial solution...")
        perm_current, speeds_current = create_initial_solution(geom_for_validation)

    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom_for_validation, tau_p_dict)
    obj_current = obj_dict_current['f']

    perm_best = perm_current
    speeds_best = [row[:] for row in speeds_current]
    obj_best = obj_current
    obj_dict_best = obj_dict_current

    T = T_init
    iter_count = 1
    history = {'costs': [], 'avg_delays': [], 'total_delays': [], 'emergency_delays': [], 'temps': [], 'conflicts': []}

    history['costs'].append(obj_current)
    history['temps'].append(T)
    history['total_delays'].append(obj_dict_current.get('fall', 0))
    history['emergency_delays'].append(obj_dict_current.get('fem', 0))
    history['conflicts'].append(obj_dict_current.get('conflicts', 0))

    if verbose:
        print(f"Initial Solution Cost (f): {obj_best:.2f}")

    while T > T_min and iter_count < max_iter:
        for i in range(iter_per_temp):
            if iter_count >= max_iter:
                break

            perm_new, speeds_new = generate_neighbor(perm_current, speeds_current, geom_for_validation)

            obj_dict_new = evaluate_solution(perm_new, speeds_new, geom_for_validation, tau_p_dict)
            obj_new = obj_dict_new['f']

            iter_count += 1
            delta_E = obj_new - obj_current

            if delta_E < 0 or (T > 1e-9 and math.exp(-delta_E / T) > random.random()):
                perm_current = perm_new
                speeds_current = [row[:] for row in speeds_new]
                obj_current = obj_new
                obj_dict_current = obj_dict_new

                if obj_current < obj_best:
                    perm_best = perm_current
                    speeds_best = [row[:] for row in speeds_current]
                    obj_best = obj_current
                    obj_dict_best = obj_dict_current
                    if verbose:
                        print(f"  Iter {iter_count}: * New Best Solution: {obj_best:.2f}")

            history['costs'].append(obj_current)
            history['temps'].append(T)
            history['total_delays'].append(obj_dict_current.get('fall', 0))
            history['emergency_delays'].append(obj_dict_current.get('fem', 0))
            history['conflicts'].append(obj_dict_current.get('conflicts', 0))

        if iter_count >= max_iter:
            break

        T *= cool_rate

    if verbose:
        print("\n--- SA Finished ---")
        print(f"Termination: {'Reached minimum temperature' if T <= T_min else 'Reached max iterations'}. Final T={T:.2f}")
        print(f"Total evaluations: {iter_count}")
        print(f"Best Objective (f): {obj_best:.2f}")
        print(f"Best Conflicts: {obj_dict_best.get('conflicts', 0)}")
        print(f"Best Total Delay: {obj_dict_best.get('fall', 0):.2f} s")
        print(f"Best Emergency Delay: {obj_dict_best.get('fem', 0):.2f} s")
        
        # NEW: Print speed incentive component
        print(f"Best Speed Incentive: {obj_dict_best.get('speed_incentive', 0):.2f}")
        print(f"Best Conflict Penalty: {obj_dict_best.get('conflict_penalty', 0):.2f}")
        
        # Print best permutation with segment speeds
        print(f"\nBest Solution Details:")
        print(f"Best Permutation (IDs): {[v.id for v in perm_best]}")
        print(f"\nVehicle Segment Speeds:")
        for i, v in enumerate(perm_best):
            segment_speeds = speeds_best[i]
            avg_speed = sum(segment_speeds) / 5
            print(f"  Vehicle {v.id}: Avg={avg_speed:.2f} m/s, Segments=[{', '.join(f'{s:.2f}' for s in segment_speeds)}]")

    return perm_best, speeds_best, obj_best, history, geom_for_validation, tau_p_dict, iter_count

# def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
#            iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS,
#            initial_solution=None, verbose=True):
#     """Main Simulated Annealing (SA) algorithm."""
    
#     if verbose:
#         print("--- Starting Simulated Annealing ---")
#         print("Initializing geometry and parameters...")
        
#     geom_for_validation = Geometry()
#     all_vehicles = config.pi
#     geom_for_validation.create_entry_queue(all_vehicles)
#     for v in all_vehicles:
#         geom_for_validation.set_trajectory(v)
#     all_points = set().union(*(v.path for v in all_vehicles if v.path))
#     if not all_points:
#         print("Error: No vehicles or no paths found. Exiting.")
#         return [], [], 0.0, {}, None, None, 0

#     tau_p_dict = {p: config.tau for p in all_points}

#     if initial_solution:
#         if verbose:
#             print("Using provided initial solution.")
#         (perm_current, speeds_current) = (copy.deepcopy(initial_solution[0]), copy.deepcopy(initial_solution[1]))
#     else:
#         if verbose:
#             print("Creating new initial solution...")
#         (perm_current, speeds_current) = create_initial_solution(geom_for_validation)

#     obj_dict_current = evaluate_solution(perm_current, speeds_current, geom_for_validation, tau_p_dict)
#     obj_current = obj_dict_current['f']

#     perm_best = perm_current
#     speeds_best = speeds_current
#     obj_best = obj_current

#     T = T_init
#     iter_count = 1 
#     history = {'costs': [], 'avg_delays': [], 'total_delays': [], 'emergency_delays': [], 'temps': []}
    
#     # Record initial state
#     history['costs'].append(obj_current)
#     history['temps'].append(T)
#     history['total_delays'].append(obj_dict_current.get('fall', 0))
#     history['emergency_delays'].append(obj_dict_current.get('fem', 0))
#     current_delays = obj_dict_current.get('delays', {})
#     avg_delay = sum(current_delays.values()) / len(current_delays) if current_delays else 0.0
#     history['avg_delays'].append(avg_delay)

#     if verbose:
#         print(f"Initial Solution Cost (f): {obj_best:.2f}")

#     # FIX: Change primary loop to be driven by iteration count (max_iter)
#     # The temperature T will continue to cool until T_min is reached, 
#     # but the loop itself runs up to max_iter.
#     while iter_count < max_iter:
#         for i in range(iter_per_temp):
#             # Check for termination inside the inner loop as well
#             if iter_count >= max_iter: break 
            
#             (perm_new, speeds_new) = generate_neighbor(perm_current, speeds_current, geom_for_validation)

#             obj_dict_new = evaluate_solution(
#                 permutation=perm_new,
#                 speeds=speeds_new,
#                 geom=geom_for_validation,
#                 tau_p_dict=tau_p_dict,
#                 return_full_schedule=False
#             )
#             obj_new = obj_dict_new['f']
            
#             iter_count += 1 

#             ΔE = obj_new - obj_current

#             # T is used for acceptance, but the loop continues even if T < T_min
#             if ΔE < 0 or (T > 1e-9 and math.exp(-ΔE / T) > random.random()):
#                 perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
#                 obj_dict_current = obj_dict_new
#                 if obj_current < obj_best:
#                     perm_best, speeds_best, obj_best = perm_current, speeds_current, obj_current
#                     if verbose:
#                         print(f"  Iter {iter_count}: * New Best Solution: {obj_best:.2f}")

#             # Always record the current state
#             history['costs'].append(obj_current)
#             history['temps'].append(T)
#             history['total_delays'].append(obj_dict_current.get('fall', 0))
#             history['emergency_delays'].append(obj_dict_current.get('fem', 0))
#             current_delays = obj_dict_current.get('delays', {})
#             avg_delay = sum(current_delays.values()) / len(current_delays) if current_delays else 0.0
#             history['avg_delays'].append(avg_delay)

#             # NOTE: iter_count check moved to the start of the inner loop, but 
#             # the break condition here is still needed if max_iter is hit on the last evaluation

#         if iter_count >= max_iter: break 
        
#         # Only cool the temperature if it hasn't hit T_min yet
#         if T > T_min: 
#             T = T * cool_rate 
#         # Optional: If T has hit T_min, keep it constant at T_min for remaining iterations
#         # else:
#         #     T = T_min 
#         # (The original code's exponential acceptance uses T > 1e-9, so setting T=T_min is safe)

#     if verbose:
#         print("\n--- SA Finished ---")
#         if iter_count >= max_iter: print(f"Termination: Reached max iteration/evaluation limit ({max_iter}).")
#         # Check if T has reached T_min at termination
#         if T <= T_min: print(f"Termination: Reached minimum temperature ({T_min}). Final T={T:.2f}")
#         print(f"Total evaluations: {iter_count}")
#         print(f"Best Objective (f): {obj_best:.2f}")
    
    return perm_best, speeds_best, obj_best, history, geom_for_validation, tau_p_dict, iter_count

if __name__ == "__main__":
    
    # --- MODIFICATION: Must use the new smooth visualizer ---
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
            # --- MODIFICATION: Call the new, simpler load_schedule ---
            final_speeds_dict = {v.id: s for v, s in zip(perm_best, speeds_best)}
            animator = IntersectionVisualization()
            animator.load_schedule(perm_best, final_speeds_dict) # <-- No longer needs schedule
            
            print("Starting animation window...")
            animator.start_animation()
        except Exception as e:
            print(f"Error during animation: {e}")
            traceback.print_exc()