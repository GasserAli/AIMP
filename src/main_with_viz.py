# File: main_with_viz.py
"""
Main entry point for running the intersection optimization with selectable visualization
and algorithm comparison.
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
            print(f"  + {package_name} already installed")
        except ImportError:
            print(f"  ! {package_name} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  + {package_name} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"  X FAILED to install {package_name}: {e}")
                print(f"    Please run manually: pip install {pip_name}")
    
    print("="*70 + "\n")

install_dependencies()

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import traceback
import math
import random
import copy
import time
import numpy as np
import matplotlib.pyplot as plt # <-- Make sure this is imported

# --- Import Project Files ---
import config
import objective
from geometry import Geometry
from decoder import run_decoder
from vehicle import Vehicle
import sa

# --- Import Algorithms ---
from sa import (
    run_sa, 
    plot_results as plot_sa_results, 
    evaluate_solution,
    validate_speeds
)
try:
    # --- MODIFICATION: Renamed plot function ---
    from ga import run_ga, plot_ga_performance_dashboard
    GA_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import ga.py: {e}")
    print("  'GA' and 'BOTH' modes will not be available.")
    GA_IMPORTED = False


# =============================================================================
# CONFIGURATION
# =============================================================================
# --- 1. CHOOSE ALGORITHM ---
OPTIMIZATION_ALGORITHM = 'GA'  # 'SA', 'GA', or 'BOTH' for comparison

# --- 2. CHOOSE VISUALIZATION ---
# 'matplotlib' or 'web' to see animation, 'none' for just terminal/plots.
VISUALIZATION_METHOD = 'none'

# --- 3. ALGORITHM PARAMETERS ---
COMPARISON_EVALUATION_BUDGET = 5000 
T_INITIAL = sa.T_INITIAL
T_MIN = sa.T_MIN 
COOLING_RATE = sa.COOLING_RATE
MAX_ITER_PER_TEMP = sa.MAX_ITER_PER_TEMP

if GA_IMPORTED:
    from ga import (
        POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, 
        TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
    )
# =============================================================================


# --- Conditional Imports Based on Visualization Method ---
animation_enabled = False
web_viz_enabled = False

if OPTIMIZATION_ALGORITHM != 'BOTH' and VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization import IntersectionVisualization
        animation_enabled = True
        print("Matplotlib visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import matplotlib visualization: {e}")

elif OPTIMIZATION_ALGORITHM != 'BOTH' and VISUALIZATION_METHOD == 'web':
    try:
        from visualization_utils import IntersectionVisualizer
        web_viz_enabled = True
        print("Web-based visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import web visualization: {e}")

elif VISUALIZATION_METHOD == 'none':
    print("Visualization disabled (running in headless mode)")

else:
    if OPTIMIZATION_ALGORITHM == 'BOTH':
        print("Visualization disabled (running in 'BOTH' comparison mode)")
    else:
        print(f"Warning: Unknown VISUALIZATION_METHOD '{VISUALIZATION_METHOD}'")
        print("  Continuing without visualization.")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_final_solution_comparison(obj_dict, algorithm_name):
    """
    --- NEW BAR CHART PLOT ---
    Plots a bar chart of the final solution's cost components.
    """
    if 'f' not in obj_dict:
        print("Cannot plot final solution: Invalid obj_dict")
        return

    print(f"\nPlotting Final Solution Metrics for {algorithm_name}...")
    
    labels = ['Weighted Objective (f)', 'Total Delay (f_all)', 'Emergency Delay (f_em)']
    values = [
        obj_dict.get('f', 0),
        obj_dict.get('fall', 0),
        obj_dict.get('fem', 0)
    ]
    
    colors = ['blue', 'green', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors)
    
    ax.set_ylabel('Cost / Delay (s)')
    ax.set_title(f'Final Best Solution Metrics: {algorithm_name}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values), 
                f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals):
    """
    Plots the convergence of SA and GA on the same graph vs. evaluations.
    """
    if not GA_IMPORTED:
        print("Cannot plot comparison, GA module not loaded.")
        return
    if 'costs' not in sa_history or 'best_f' not in ga_history:
        print("Error: Invalid history data for comparison plot.")
        return
        
    print("\nPlotting SA vs. GA Convergence...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Process SA Data ---
    sa_costs = sa_history['costs']
    sa_best_so_far = np.minimum.accumulate(sa_costs)
    sa_eval_points = range(1, len(sa_best_so_far) + 1)
    if len(sa_best_so_far) > sa_evals:
        sa_best_so_far = sa_best_so_far[:sa_evals]
        sa_eval_points = sa_eval_points[:sa_evals]

    # --- Process GA Data ---
    ga_best_per_gen = ga_history['best_f'] # <-- Use 'best_f'
    ga_evals_per_gen = (POPULATION_SIZE - int(POPULATION_SIZE * ELITISM_RATE))
    if ga_evals_per_gen <= 0: ga_evals_per_gen = 1
    
    ga_eval_points = [POPULATION_SIZE] + [POPULATION_SIZE + (i * ga_evals_per_gen) for i in range(1, len(ga_best_per_gen))]
    
    ga_plot_indices = [i for i, evals in enumerate(ga_eval_points) if evals <= ga_evals]
    if not ga_plot_indices: ga_plot_indices = [0]
    
    if ga_evals < ga_eval_points[-1]:
         ga_eval_points = [ga_eval_points[i] for i in ga_plot_indices]
         ga_best_per_gen = [ga_best_per_gen[i] for i in ga_plot_indices]
         if ga_evals not in ga_eval_points:
             ga_eval_points.append(ga_evals)
             ga_best_per_gen.append(ga_history['best_f'][-1])
    
    # --- Plotting ---
    ax.plot(sa_eval_points, sa_best_so_far, 'b-', label=f'SA (Best Found)', linewidth=2)
    ax.step(ga_eval_points, ga_best_per_gen, 'r-', where='post', label=f'GA (Best Found)', linewidth=2)
    
    ax.set_title(f'SA vs. GA Convergence (Budget: {COMPARISON_EVALUATION_BUDGET} Evals)', fontsize=16)
    ax.set_xlabel('Number of Fitness Evaluations', fontsize=12)
    ax.set_ylabel('Best Objective Cost (f) - Lower is Better', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, COMPARISON_EVALUATION_BUDGET)
    
    sa_final = sa_best_so_far[-1]
    ga_final = ga_best_per_gen[-1]
    
    winner_text = f"SA Final: {sa_final:.2f}\nGA Final: {ga_final:.2f}"
    ax.text(0.6, 0.6, winner_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# =============================================================================
# VISUALIZATION LAUNCHERS
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
        time.sleep(2)
        
        for vehicle, speed in zip(perm_best, speeds_best):
            vehicle.velocity = round(speed, 2)
        
        print("Sending vehicle data to visualization...")
        visualizer.update_vehicles(vehicles=perm_best, permutation=[v.id for v in perm_best])
        
        print("Starting simulation...")
        visualizer.start_simulation()
        
        print("\n" + "="*70)
        print("  Web visualization server running!")
        print("    Open your browser to: http://localhost:5000")
        print("    Press Ctrl+C to stop the server")
        print("="*70 + "\n")
        
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
    print(f"Selected Algorithm:   {OPTIMIZATION_ALGORITHM.upper()}")
    print(f"Visualization Method: {VISUALIZATION_METHOD.upper()}")
    print("="*70 + "\n")
    
    # --- Run SA ---
    if OPTIMIZATION_ALGORITHM == 'SA':
        perm_best, speeds_best, obj_best, sa_history, geom, tau_p_dict, evals = run_sa(
            max_iter=COMPARISON_EVALUATION_BUDGET
        )
        
        print("\nDisplaying SA performance plots...")
        plot_sa_results(sa_history)
        
        # --- NEW: Re-eval to get obj_dict and show bar plot ---
        best_sa_obj_dict = evaluate_solution(perm_best, speeds_best, geom, tau_p_dict)
        plot_final_solution_comparison(best_sa_obj_dict, 'SA')
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run GA ---
    elif OPTIMIZATION_ALGORITHM == 'GA':
        if not GA_IMPORTED:
            print("ERROR: GA algorithm selected but ga.py could not be imported.")
            return
            
        # --- MODIFICATION: Receive new obj_dict ---
        (perm_best, speeds_best, obj_best, ga_history, geom, 
         tau_p_dict, best_ga_obj_dict, evals) = run_ga(
            max_evaluations=COMPARISON_EVALUATION_BUDGET
        )
        
        print("\nDisplaying GA performance plots...")
        plot_ga_performance_dashboard(ga_history) # <-- Use new 2x2 plot
        
        # --- NEW: Show the bar plot ---
        plot_final_solution_comparison(best_ga_obj_dict, 'GA')
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run BOTH for Comparison ---
    elif OPTIMIZATION_ALGORITHM == 'BOTH':
        if not GA_IMPORTED:
            print("ERROR: 'BOTH' mode selected but ga.py could not be imported.")
            return

        print("\n--- RUNNING SIMULATED ANNEALING ---")
        (sa_perm, sa_speeds, sa_obj, sa_history, 
         sa_geom, sa_tau, sa_evals) = run_sa(
            max_iter=COMPARISON_EVALUATION_BUDGET
        )
        
        print("\n--- RUNNING GENETIC ALGORITHM ---")
        (ga_perm, ga_speeds, ga_obj, ga_history, 
         ga_geom, ga_tau, best_ga_obj_dict, ga_evals) = run_ga(
            max_evaluations=COMPARISON_EVALUATION_BUDGET
        )
        
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"Budget: {COMPARISON_EVALUATION_BUDGET} evaluations")
        print(f"  SA Final Best: {sa_obj:.2f} (in {sa_evals} evals)")
        print(f"  GA Final Best: {ga_obj:.2f} (in {ga_evals} evals)")
        
        if sa_obj < ga_obj:
            print("\n  SA found a better solution.")
        else:
            print("\n  GA found a better or equal solution.")
        print("="*70)

        # Show convergence plot
        plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals)
        
        # --- NEW: Show bar plot comparison ---
        sa_obj_dict = evaluate_solution(sa_perm, sa_speeds, sa_geom, sa_tau)
        plot_final_solution_comparison(sa_obj_dict, 'SA (Final)')
        plot_final_solution_comparison(best_ga_obj_dict, 'GA (Final)')
        print("  (Close all plot windows to finish)")


    else:
        print(f"Error: Unknown OPTIMIZATION_ALGORITHM: '{OPTIMIZATION_ALGORITHM}'")

    
    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()