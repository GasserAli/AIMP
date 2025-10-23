# File: sa.py
import math
import random
import copy
import matplotlib.pyplot as plt  # <-- Added for plotting
import os

# --- Import Project Files ---
import config
import objective
from geometry import Geometry
from decoder import run_decoder  # <-- Imports the "Algorithm 1" decoder
from visualization_utils import IntersectionVisualizer 

# --- SA Parameters ---
T_INITIAL = 1000.0        # Initial temperature
T_MIN = 1.0               # Final temperature
COOLING_RATE = 0.99       # Cooling rate (e.g., 0.99)
MAX_ITER_PER_TEMP = 20    # Iterations at each temperature step
MAX_TOTAL_ITERATIONS = 100000 # <-- Total iteration limit

def create_initial_solution(geom):
    """
    Generates a valid initial solution (permutation, speeds) for the SA algorithm
    by reading the problem instance from config.py.
    
    This respects the C0 (no-catch-up) constraint.
    """
    
    initial_perm = config.pi
    initial_speeds_dict = {} 
    v_min_global, v_max_global = config.velocity_range
    
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue
            
        v_leader = queue[0]
        last_speed = random.uniform(v_min_global, v_max_global)
        initial_speeds_dict[v_leader.id] = last_speed

        for v_follower in queue[1:]:
            current_max = min(v_max_global, last_speed)
            current_min = min(v_min_global, current_max)
            new_speed = random.uniform(current_min, current_max)
            initial_speeds_dict[v_follower.id] = new_speed
            last_speed = new_speed

    initial_speeds_list = [initial_speeds_dict[v.id] for v in initial_perm]
    
    return initial_perm, initial_speeds_list


def validate_speeds(permutation, speeds, geom):
    """
    Enforces the C0 (no-catch-up) constraint on a given speed list.
    If v_k > v_k-1 in a lane, v_k is capped to v_k-1.
    """
    
    v_new = list(speeds)
    speed_dict = {p.id: s for p, s in zip(permutation, v_new)}

    for queue in geom.entry_queues.values():
        if not queue:
            continue
        
        last_speed = speed_dict[queue[0].id]
        
        for v_follower in queue[1:]:
            follower_speed = speed_dict[v_follower.id]
            if follower_speed > last_speed:
                speed_dict[v_follower.id] = last_speed
            last_speed = speed_dict[v_follower.id]
            
    validated_speeds_list = [speed_dict[v.id] for v in permutation]
    return validated_speeds_list


def generate_neighbor(perm_current, speeds_current, geom):
    """
    Generates a new "neighbor" solution by jiggling the current one.
    """
    
    perm_new = copy.deepcopy(perm_current)
    speeds_new = copy.deepcopy(speeds_current)
    
    v_min_global, v_max_global = config.velocity_range

    if random.random() < 0.5:
        # --- Move 1: Swap Permutation (Π) ---
        idx1, idx2 = random.sample(range(len(perm_new)), 2)
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]
        
    else:
        # --- Move 2: Change Speed (v) ---
        idx = random.randrange(len(speeds_new))
        change = random.uniform(-2.0, 2.0)
        speeds_new[idx] += change
        speeds_new[idx] = max(v_min_global, min(speeds_new[idx], v_max_global))

    # --- CRITICAL: Re-validate C0 Constraint ---
    speeds_new = validate_speeds(perm_new, speeds_new, geom)
    
    return perm_new, speeds_new


def evaluate_solution(permutation, speeds, geom, tau_p_dict):
    """
    Evaluates a solution (Π, v) by running it through the
    decoder and objective function.
    
    Returns the full objective dictionary: {"delays", "fem", "fall", "f"}
    """
    
    # 1. Run the Decoder from decoder.py
    decoder_results = run_decoder(permutation, speeds, geom, tau_p_dict)
    
    # 2. Calculate the Objective Function from objective.py
    obj_dict = objective.calculate_objective(decoder_results)
    
    # 3. Return the full dictionary
    return obj_dict


def plot_results(history_data):
    """
    Create and display plots for cost and delays in interactive windows.
    
    Parameters
    ----------
    history_data : dict
        A dictionary containing the lists: 'costs', 'avg_delays', 
        'total_delays', 'emergency_delays', 'temps'.
    """
    
    costs = history_data['costs']
    avg_delays = history_data['avg_delays']
    total_delays = history_data['total_delays']
    emergency_delays = history_data['emergency_delays']
    temps = history_data['temps']
    
    # Plot 1: Cost (f)
    plt.figure(figsize=(10, 4))
    plt.plot(costs, '-')
    plt.title('SA: Weighted Objective Cost (f) per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (f)')
    plt.grid(True)
    plt.tight_layout()
    
    # Plot 2: Average Delay
    plt.figure(figsize=(10, 4))
    plt.plot(avg_delays, '-', color='orange')
    plt.title('SA: Average Delay per Vehicle per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Delay (s)')
    plt.grid(True)
    plt.tight_layout()

    # Plot 3: Total Delay (f_all)
    plt.figure(figsize=(10, 4))
    plt.plot(total_delays, '-', color='green')
    plt.title('SA: Total Delay (f_all) per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Total Delay (s)')
    plt.grid(True)
    plt.tight_layout()
    
    # Plot 4: Emergency Delay (f_em)
    plt.figure(figsize=(10, 4))
    plt.plot(emergency_delays, '-', color='red')
    plt.title('SA: Emergency Delay (f_em) per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Emergency Delay (s)')
    plt.grid(True)
    plt.tight_layout()
    
    # Plot 5: Temperature (Cooling Schedule)
    plt.figure(figsize=(10, 4))
    plt.plot(temps, '-', color='purple')
    plt.title('SA: Temperature vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature (T)')
    plt.grid(True)
    plt.tight_layout()

    # Show all plot windows
    print("Displaying 5 plot windows...")
    plt.show()


def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE, 
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS):
    """
    Main Simulated Annealing (SA) algorithm.
    """
    print("---Starting Simulated Annealing ---")

    # --- 1. Initialization (Phase 1) ---
    print("Initializing geometry and parameters...")
    geom = Geometry()
    all_vehicles = config.pi
    
    geom.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom.set_trajectory(v)
        
    all_points = set()
    for v in all_vehicles:
        all_points.update(v.path)
    
    if not all_points:
        print("Error: No vehicles or no paths found. Exiting.")
        return [], [], 0.0

    tau_p_dict = {p: config.tau for p in all_points}
    
    # Generate the initial valid solution
    (perm_current, speeds_current) = create_initial_solution(geom)
    
    # Evaluate the initial solution
    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom, tau_p_dict)
    obj_current = obj_dict_current['f']
    
    # Set initial best
    perm_best = perm_current
    speeds_best = speeds_current
    obj_best = obj_current
    
    T = T_init
    iter_count = 0
    
    # --- History Lists for Plotting ---
    history = {
        'costs': [],
        'avg_delays': [],
        'total_delays': [],
        'emergency_delays': [],
        'temps': []
    }

    print(f"Initial Solution Cost (f): {obj_best:.2f}")

    # --- 2. SA Loop (Phase 2) ---
    while T > T_min and iter_count < max_iter:
        
        for i in range(iter_per_temp):
            
            # A. Generate a new neighbor solution
            (perm_new, speeds_new) = generate_neighbor(perm_current, speeds_current, geom)
            
            # B. Evaluate the new solution
            obj_dict_new = evaluate_solution(perm_new, speeds_new, geom, tau_p_dict)
            obj_new = obj_dict_new['f']
            
            # C. Make Acceptance Decision
            ΔE = obj_new - obj_current
            
            if ΔE < 0:
                # Better solution: Always accept
                perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                obj_dict_current = obj_dict_new
                
                if obj_new < obj_best:
                    perm_best, speeds_best, obj_best = perm_new, speeds_new, obj_new
                    print(f"  Iter {iter_count}: * New Best Solution: {obj_best:.2f}")
            else:
                # Worse solution: Accept with probability P = exp(-ΔE / T)
                if math.exp(-ΔE / T) > random.random():
                    perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                    obj_dict_current = obj_dict_new
            
            iter_count += 1
            
            # --- Store History for Plots ---
            history['costs'].append(obj_current)
            history['temps'].append(T)
            history['total_delays'].append(obj_dict_current['fall'])
            history['emergency_delays'].append(obj_dict_current['fem'])
            
            current_delays = obj_dict_current['delays']
            if current_delays:
                avg_delay = sum(current_delays.values()) / len(current_delays)
            else:
                avg_delay = 0.0
            history['avg_delays'].append(avg_delay)
            
            # Check for total iteration limit *inside* the inner loop
            if iter_count >= max_iter:
                break
            
        # D. Cool down
        T = T * cool_rate

    # --- 3. Termination (Phase 3) ---
    print("\n---SA Finished ---")
    if iter_count >= max_iter:
        print(f"Termination: Reached max iteration limit ({max_iter}).")
    if T <= T_min:
        print(f"Termination: Reached minimum temperature ({T_min}).")
        
    print(f"Total iterations: {iter_count}")
    print(f"Best Objective (f): {obj_best:.2f}")
    print(f"Best Permutation (IDs): {[v.id for v in perm_best]}")
    print(f"Best Speeds: {[round(s, 2) for s in speeds_best]}")
    
    # Call the plotting function
    # plot_results(history)
    
    return perm_best, speeds_best, obj_best


if __name__ == "__main__":
    perm_best, speeds_best, _ = run_sa()
    visualizer = IntersectionVisualizer()
    visualizer.start()  # Start the visualization server

    # Test the visualization with a single update of the vehicles
    visualizer.update_vehicles(vehicles=perm_best, permutation=[v.id for v in perm_best])
    visualizer.start_simulation()