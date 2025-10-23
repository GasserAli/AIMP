# File: main_with_viz.py
"""
Main entry point for running the intersection optimization with selectable visualization.

This module allows you to:
1. Run the Simulated Annealing optimization algorithm
2. Choose between two visualization methods:
   - 'matplotlib': Static matplotlib animation (visualization.py)
   - 'web': Interactive web-based D3.js visualization (visualization_server.py + visualization_utils.py)
"""

# =============================================================================
# AUTOMATIC DEPENDENCY INSTALLATION
# =============================================================================
import subprocess
import sys

def install_dependencies():
    """Automatically install required packages if not present."""
    required_packages = {
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'flask': 'flask',
        'requests': 'requests'
    }
    
    print("\n" + "="*70)
    print("CHECKING AND INSTALLING DEPENDENCIES")
    print("="*70)
    
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"{package_name} already installed")
        except ImportError:
            print(f"{package_name} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"{package_name} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name}: {e}")
                print(f" You may need to run: pip install {pip_name}")
    
    print("="*70 + "\n")

# Run dependency installation
install_dependencies()

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import traceback
import math
import random
import copy
import time

# --- Import Project Files ---
import config
import objective
from geometry import Geometry
from decoder import run_decoder
from vehicle import Vehicle

# =============================================================================
# CONFIGURATION: Choose Visualization Method
# =============================================================================
VISUALIZATION_METHOD = 'matplotlib'  # Options: 'matplotlib', 'web', or 'none'
# =============================================================================

# --- Conditional Imports Based on Visualization Method ---
animation_enabled = False
web_viz_enabled = False

if VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization import IntersectionVisualization
        import matplotlib.pyplot as plt
        animation_enabled = True
        print("Matplotlib visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import matplotlib visualization: {e}")
        print("  Continuing without matplotlib animation.")

elif VISUALIZATION_METHOD == 'web':
    try:
        from visualization_utils import IntersectionVisualizer
        web_viz_enabled = True
        print("Web-based visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import web visualization: {e}")
        print("  Continuing without web visualization.")

elif VISUALIZATION_METHOD == 'none':
    print("Visualization disabled (running in headless mode)")

else:
    print(f"Warning: Unknown VISUALIZATION_METHOD '{VISUALIZATION_METHOD}'")
    print("  Valid options: 'matplotlib', 'web', 'none'")
    print("  Continuing without visualization.")


# =============================================================================
# SA PARAMETERS
# =============================================================================
T_INITIAL = 1000.0
T_MIN = 1.0
COOLING_RATE = 0.99
MAX_ITER_PER_TEMP = 20
MAX_TOTAL_ITERATIONS = 100000


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================
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


# =============================================================================
# SA ALGORITHM FUNCTIONS
# =============================================================================
def create_initial_solution(geom):
    """Generate a valid initial solution respecting the C0 constraint."""
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
    """Enforce the C0 (no-catch-up) constraint."""
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
    """Generate a neighbor solution."""
    perm_new = copy.deepcopy(perm_current)
    speeds_new = copy.deepcopy(speeds_current)
    v_min_global, v_max_global = config.velocity_range

    if random.random() < 0.5:
        idx1, idx2 = random.sample(range(len(perm_new)), 2)
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]
    else:
        idx = random.randrange(len(speeds_new))
        change = random.uniform(-2.0, 2.0)
        speeds_new[idx] += change
        speeds_new[idx] = max(v_min_global, min(speeds_new[idx], v_max_global))

    speeds_new = validate_speeds(perm_new, speeds_new, geom)
    return perm_new, speeds_new


def evaluate_solution(permutation, speeds, geom, tau_p_dict, return_full_schedule=False):
    """Evaluate a solution using decoder and objective function."""
    if return_full_schedule:
        # Decoder returns (decoder_results, scheduled_times, t_ear)
        decoder_results, scheduled_times, t_ear = run_decoder(
            permutation, speeds, geom, tau_p_dict, return_full_schedule=True
        )
        obj_dict = objective.calculate_objective(decoder_results)
        return obj_dict, scheduled_times, t_ear
    else:
        # Decoder returns just decoder_results (list)
        decoder_results = run_decoder(
            permutation, speeds, geom, tau_p_dict, return_full_schedule=False
        )
        obj_dict = objective.calculate_objective(decoder_results)
        return obj_dict


def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS):
    """Main Simulated Annealing algorithm."""
    print("\n" + "="*70)
    print("STARTING SIMULATED ANNEALING OPTIMIZATION")
    print("="*70)

    # --- 1. Initialization ---
    print("\n[1/3] Initializing geometry and parameters...")
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
        return [], [], 0.0, None, None, {}

    tau_p_dict = {p: config.tau for p in all_points}
    
    (perm_current, speeds_current) = create_initial_solution(geom)
    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom, tau_p_dict)
    obj_current = obj_dict_current['f']
    
    perm_best = perm_current
    speeds_best = speeds_current
    obj_best = obj_current
    
    T = T_init
    iter_count = 0
    
    history = {
        'costs': [],
        'avg_delays': [],
        'total_delays': [],
        'emergency_delays': [],
        'temps': []
    }

    print(f"Initial Solution Cost (f): {obj_best:.2f}")
    print(f"Number of vehicles: {len(all_vehicles)}")

    # --- 2. SA Loop ---
    print("\n[2/3] Running optimization...")
    while T > T_min and iter_count < max_iter:
        for i in range(iter_per_temp):
            (perm_new, speeds_new) = generate_neighbor(perm_current, speeds_current, geom)
            obj_dict_new = evaluate_solution(perm_new, speeds_new, geom, tau_p_dict)
            obj_new = obj_dict_new['f']
            
            ΔE = obj_new - obj_current
            
            if ΔE < 0:
                perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                obj_dict_current = obj_dict_new
                
                if obj_new < obj_best:
                    perm_best, speeds_best, obj_best = perm_new, speeds_new, obj_new
                    print(f"  Iter {iter_count}: New Best Solution: {obj_best:.2f}")
            else:
                if math.exp(-ΔE / T) > random.random():
                    perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                    obj_dict_current = obj_dict_new
            
            iter_count += 1
            
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
            
            if iter_count >= max_iter:
                break
            
        T = T * cool_rate

    # --- 3. Termination ---
    print("\n[3/3] Optimization complete!")
    print("="*70)
    if iter_count >= max_iter:
        print(f"Termination: Reached max iteration limit ({max_iter})")
    if T <= T_min:
        print(f"Termination: Reached minimum temperature ({T_min})")
        
    print(f"\nRESULTS:")
    print(f"  Total iterations: {iter_count}")
    print(f"  Best Objective (f): {obj_best:.2f}")
    print(f"  Best Permutation (IDs): {[v.id for v in perm_best]}")
    print(f"  Best Speeds: {[round(s, 2) for s in speeds_best]}")
    print("="*70)
    
    return perm_best, speeds_best, obj_best, geom, tau_p_dict, history


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict):
    """Run matplotlib-based visualization."""
    print("\n" + "="*70)
    print("STARTING MATPLOTLIB VISUALIZATION")
    print("="*70)
    
    try:
        print("Re-running decoder to get full schedule...")
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
        
        print("Opening animation window...")
        print("  (Close the window to continue)")
        animator.start_animation()
        print("Animation window closed.")
        
    except Exception as e:
        print(f"Error during matplotlib visualization: {e}")
        traceback.print_exc()


def visualize_web(perm_best, speeds_best):
    """Run web-based visualization."""
    print("\n" + "="*70)
    print("STARTING WEB-BASED VISUALIZATION")
    print("="*70)
    
    try:
        visualizer = IntersectionVisualizer()
        
        print("Starting web server...")
        visualizer.start()
        
        # Give server time to start
        time.sleep(2)
        
        # Update vehicle speeds
        for vehicle, speed in zip(perm_best, speeds_best):
            vehicle.velocity = round(speed, 2)
        
        print("Sending vehicle data to visualization...")
        visualizer.update_vehicles(vehicles=perm_best, permutation=[v.id for v in perm_best])
        
        print("Starting simulation...")
        visualizer.start_simulation()
        
        print("\n" + "="*70)
        print("Web visualization server running!")
        print("  Open your browser to: http://localhost:5000")
        print("  Press Ctrl+C to stop the server")
        print("="*70 + "\n")
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down visualization server...")
            
    except Exception as e:
        print(f"Error during web visualization: {e}")
        traceback.print_exc()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main function to run optimization and visualization."""
    print("\n" + "="*70)
    print("INTERSECTION TRAFFIC OPTIMIZATION")
    print("="*70)
    print(f"Visualization Method: {VISUALIZATION_METHOD.upper()}")
    print("="*70 + "\n")
    
    # Run SA optimization
    perm_best, speeds_best, obj_best, geom, tau_p_dict, history = run_sa()
    
    # Run visualization based on selected method
    if VISUALIZATION_METHOD == 'matplotlib' and animation_enabled:
        # Show performance plots first
        print("\nDisplaying SA performance plots...")
        print("  (Close the plots window to continue to animation)")
        plot_results(history)
        
        # Then show animation
        visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        
    elif VISUALIZATION_METHOD == 'web' and web_viz_enabled:
        visualize_web(perm_best, speeds_best)
        
    elif VISUALIZATION_METHOD == 'none':
        print("\nOptimization complete (visualization disabled)")
        
    else:
        print("\nNo visualization available with current settings")
        print("  Results have been computed but not visualized")
    
    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
