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
        'requests': 'requests',
        'scipy': 'scipy' 
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
import matplotlib.pyplot as plt
import csv
import datetime
import os 
import scipy.stats as stats 

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
    validate_speeds,
    create_initial_solution as sa_create_initial_solution
)
try:
    from ga import run_ga, plot_ga_performance_dashboard, create_initial_population
    GA_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import ga.py: {e}")
    print("  'GA', 'BOTH', and 'EXPERIMENT' modes will not be available.")
    GA_IMPORTED = False


# =============================================================================
# CONFIGURATION
# =============================================================================
# --- 1. CHOOSE ALGORITHM ---
# Options: 'SA', 'GA', 'BOTH', 'EXPERIMENT'
OPTIMIZATION_ALGORITHM = 'EXPERIMENT' 

# --- 2. CHOOSE VISUALIZATION ---
# 'matplotlib', 'web', 'none'
# (Ignored for 'BOTH' and 'EXPERIMENT' modes)
VISUALIZATION_METHOD = 'web' 

# --- 3. ALGORITHM PARAMETERS ---
COMPARISON_EVALUATION_BUDGET = 10000 
NUM_EXPERIMENT_RUNS = 100  
RANDOM_SEED = 42 

# SA Parameters
T_INITIAL = sa.T_INITIAL
T_MIN = sa.T_MIN 
COOLING_RATE = sa.COOLING_RATE
MAX_ITER_PER_TEMP = sa.MAX_ITER_PER_TEMP

# GA Parameters
if GA_IMPORTED:
    from ga import (
        POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, 
        TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
    )
else:
    # Default values if GA not imported, to prevent errors
    POPULATION_SIZE = "N/A"
    NUM_GENERATIONS = "N/A"
    ELITISM_RATE = "N/A"
    TOURNAMENT_SIZE = "N/A"
    MUTATION_RATE_PERM = "N/A"
    MUTATION_RATE_SPEED = "N/A"

# =============================================================================


# --- Conditional Imports Based on Visualization Method ---
animation_enabled = False
web_viz_enabled = False
is_experiment_mode = OPTIMIZATION_ALGORITHM in ('BOTH', 'EXPERIMENT')

if not is_experiment_mode and VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization import IntersectionVisualization
        animation_enabled = True
        print("Matplotlib visualization enabled (SMOOTH animation)")
    except ImportError as e:
        print(f"Warning: Could not import visualization.py: {e}")

elif not is_experiment_mode and VISUALIZATION_METHOD == 'web':
    try:
        from visualization_utils import IntersectionVisualizer
        web_viz_enabled = True
        print("Web-based visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import web visualization: {e}")

elif VISUALIZATION_METHOD == 'none':
    print("Visualization disabled (running in headless mode)")
else:
    if is_experiment_mode:
        print(f"Visualization disabled (running in '{OPTIMIZATION_ALGORITHM}' mode)")
    else:
        print(f"Warning: Unknown VISUALIZATION_METHOD '{VISUALIZATION_METHOD}'")
        print("  Continuing without visualization.")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals):
    if not GA_IMPORTED:
        print("Cannot plot comparison, GA module not loaded.")
        return
    if 'costs' not in sa_history or 'best_f' not in ga_history:
        print("Error: Invalid history data for comparison plot.")
        return
    print("\nPlotting SA vs. GA Convergence...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sa_costs = sa_history['costs']
    sa_best_so_far = np.minimum.accumulate(sa_costs)
    sa_eval_points = range(1, len(sa_best_so_far) + 1)
    if len(sa_best_so_far) > sa_evals:
        sa_best_so_far = sa_best_so_far[:sa_evals]
        sa_eval_points = sa_eval_points[:sa_evals]
        
    ga_best_per_gen = ga_history['best_f']
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

def plot_experiment_results(sa_results, ga_results):
    print("\nPlotting Experiment Statistical Results (Box Plot)...")
    data = [sa_results, ga_results]
    labels = ['Simulated Annealing (SA)', 'Genetic Algorithm (GA)']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'Algorithm Performance Comparison ({NUM_EXPERIMENT_RUNS} Runs Each)')
    ax.set_ylabel('Final Objective Cost (f) - Lower is Better')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_experiment_distribution(sa_results, ga_results):
    print("\nPlotting Experiment Distribution (Histograms + Gaussian KDE)...")
    all_results = sa_results + ga_results
    min_bin = min(all_results)
    max_bin = max(all_results)
    bins = np.linspace(min_bin, max_bin, 30) 
    
    # --- Plot 1: SA Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(sa_results, bins=bins, alpha=0.7, label='SA Histogram', color='blue', density=True)
    try:
        sa_kde = stats.gaussian_kde(sa_results)
        kde_x = np.linspace(min_bin, max_bin, 200)
        plt.plot(kde_x, sa_kde(kde_x), 'blue', linewidth=2, label='SA Distribution (KDE)')
    except Exception as e:
        print(f"  Warning: Could not plot SA Gaussian KDE. {e}")
    plt.title(f'SA Result Distribution ({NUM_EXPERIMENT_RUNS} Runs)')
    plt.xlabel('Final Objective Cost (f)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    sa_mean = np.mean(sa_results)
    sa_std = np.std(sa_results)
    text_str_sa = (f'SA: Mean={sa_mean:.2f}\nSA: Std Dev={sa_std:.2f}')
    ax_sa = plt.gca()
    plt.text(0.95, 0.95, text_str_sa, transform=ax_sa.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    plt.show()

    # --- Plot 2: GA Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(ga_results, bins=bins, alpha=0.7, label='GA Histogram', color='red', density=True)
    try:
        ga_kde = stats.gaussian_kde(ga_results)
        kde_x = np.linspace(min_bin, max_bin, 200)
        plt.plot(kde_x, ga_kde(kde_x), 'red', linewidth=2, label='GA Distribution (KDE)')
    except Exception as e:
        print(f"  Warning: Could not plot GA Gaussian KDE. {e}")
        
    plt.title(f'GA Result Distribution ({NUM_EXPERIMENT_RUNS} Runs)')
    plt.xlabel('Final Objective Cost (f)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    ga_mean = np.mean(ga_results)
    ga_std = np.std(ga_results)
    text_str_ga = (f'GA: Mean={ga_mean:.2f}\nGA: Std Dev={ga_std:.2f}')
    ax_ga = plt.gca()
    plt.text(0.95, 0.95, text_str_ga, transform=ax_ga.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    plt.show()

# =============================================================================
# VISUALIZATION LAUNCHERS
# =============================================================================
def visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict):
    """Run matplotlib-based visualization."""
    print("\n" + "="*70)
    print("STARTING MATPLOTLIB VISUALIZATION (SMOOTH)")
    print("="*70)
    
    try:
        final_speeds_dict = {v.id: s for v, s in zip(perm_best, speeds_best)}
        
        animator = IntersectionVisualization()
        animator.load_schedule(perm_best, final_speeds_dict)
        
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
# CSV Saving Functions
# =============================================================================

SUMMARY_FILENAME = 'experiment_summary_log.csv'
RAW_FILENAME = 'experiment_raw_data_log.csv'

def save_experiment_summary(timestamp, sa_stats, ga_stats):
    """
    Saves the high-level summary statistics to a SINGLE master CSV file.
    Appends the new run; creates header if file doesn't exist.
    Now includes algorithm parameters.
    """
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    try:
        with open(SUMMARY_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Algorithm', 'Mean Cost (f)', 'Std Deviation (f)', 
                    'Avg Run Time (s)', 'Best Cost Overall', 'Num Runs', 'Evaluation Budget',
                    'SA_T_Init', 'SA_T_Min', 'SA_Cool_Rate', 'SA_Iter_Per_Temp',
                    'GA_Pop_Size', 'GA_Elitism', 'GA_Tourn_Size', 'GA_Mut_Perm', 'GA_Mut_Speed'
                ])
            
            # Write SA Data
            writer.writerow([
                timestamp, 'SA', sa_stats['mean'], sa_stats['std'], sa_stats['time'], sa_stats['best'],
                NUM_EXPERIMENT_RUNS, COMPARISON_EVALUATION_BUDGET,
                T_INITIAL, T_MIN, COOLING_RATE, MAX_ITER_PER_TEMP,
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A' 
            ])
            # Write GA Data
            writer.writerow([
                timestamp, 'GA', ga_stats['mean'], ga_stats['std'], ga_stats['time'], ga_stats['best'],
                NUM_EXPERIMENT_RUNS, COMPARISON_EVALUATION_BUDGET,
                'N/A', 'N/A', 'N/A', 'N/A',
                POPULATION_SIZE, ELITISM_RATE, TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
            ])
        print(f"  Successfully appended summary to: {SUMMARY_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save summary CSV. {e}")

def save_experiment_raw_data(timestamp, sa_results, ga_results):
    file_exists = os.path.isfile(RAW_FILENAME)
    try:
        with open(RAW_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Experiment_Timestamp', 'Run', 'SA_Final_Cost', 'GA_Final_Cost'])
            for i in range(NUM_EXPERIMENT_RUNS):
                writer.writerow([timestamp, i+1, sa_results[i], ga_results[i]])
        print(f"  Successfully appended raw data to: {RAW_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save raw data CSV. {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main function to run optimization and visualization."""
    
    print(f"Setting global random seed to: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*70)
    print("INTERSECTION TRAFFIC OPTIMIZATION")
    print("="*70)
    print(f"Selected Algorithm:   {OPTIMIZATION_ALGORITHM.upper()}")
    print(f"Visualization Method: {VISUALIZATION_METHOD.upper()}")
    if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
        print(f"Experiment Runs:      {NUM_EXPERIMENT_RUNS}")
    print("="*70 + "\n")
    
    # --- Use static config.pi (from config.py) ---
    
    # --- Run SA (Single Run) ---
    if OPTIMIZATION_ALGORITHM == 'SA':
        perm_best, speeds_best, obj_best, sa_history, geom, tau_p_dict, evals = run_sa(
            max_iter=COMPARISON_EVALUATION_BUDGET
        )
        print("\nDisplaying SA performance plots...")
        plot_sa_results(sa_history)
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run GA (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'GA':
        if not GA_IMPORTED:
            print("ERROR: GA algorithm selected but ga.py could not be imported.")
            return
            
        (perm_best, speeds_best, obj_best, ga_history, geom, 
         tau_p_dict, best_ga_obj_dict, evals) = run_ga(
            max_evaluations=COMPARISON_EVALUATION_BUDGET,
            visualize_realtime=True # <-- Enable real-time plot for single run
        )
        
        print("\nDisplaying GA performance plots...")
        plot_ga_performance_dashboard(ga_history)
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run BOTH (Single Run Comparison) ---
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
            max_evaluations=COMPARISON_EVALUATION_BUDGET,
            visualize_realtime=False 
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

        plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals)
        print("  (Close all plot windows to finish)")

    # --- Run Experiment (N-Run Statistical Comparison) ---
    elif OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
        if not GA_IMPORTED:
            print("ERROR: 'EXPERIMENT' mode selected but ga.py could not be imported.")
            return
            
        print("\n--- PREPARING EXPERIMENT ---")
        print(f"Running {NUM_EXPERIMENT_RUNS} times for each algorithm.")
        
        geom = Geometry()
        all_vehicles = config.pi 
        geom.create_entry_queue(all_vehicles)
        for v in all_vehicles:
            geom.set_trajectory(v)
        all_points = set().union(*(v.path for v in all_vehicles if v.path))
        if not all_points:
            print("Error: No vehicles or no paths found. Exiting experiment.")
            return
        tau_p_dict = {p: config.tau for p in all_points}

        print("\nGenerating common initial population for GA...")
        common_ga_population = create_initial_population(POPULATION_SIZE, geom)
        print("\nGenerating common initial solution for SA...")
        common_sa_solution = sa_create_initial_solution(geom)
        
        sa_results = []
        ga_results = []
        
        sa_best_obj_overall = math.inf
        sa_best_history_overall = {}
        ga_best_obj_overall = math.inf
        ga_best_history_overall = {}

        # 3. Run SA Experiment
        print("\n" + "="*70)
        print(f"RUNNING SA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
        print("="*70)
        start_time_sa = time.time()
        for i in range(NUM_EXPERIMENT_RUNS):
            print(f"  SA Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
            _, _, sa_obj, sa_history, _, _, _ = run_sa(
                max_iter=COMPARISON_EVALUATION_BUDGET,
                initial_solution=common_sa_solution,
                verbose=False 
            )
            print(f"    ...Run {i+1} Best: {sa_obj:.2f}")
            sa_results.append(sa_obj)
            
            if sa_obj < sa_best_obj_overall:
                sa_best_obj_overall = sa_obj
                sa_best_history_overall = sa_history
            
        end_time_sa = time.time()

        # 4. Run GA Experiment
        print("\n" + "="*70)
        print(f"RUNNING GA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
        print("="*70)
        start_time_ga = time.time()
        for i in range(NUM_EXPERIMENT_RUNS):
            print(f"  GA Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
            _, _, ga_obj, ga_history, _, _, _, _ = run_ga(
                max_evaluations=COMPARISON_EVALUATION_BUDGET,
                initial_population=common_ga_population,
                verbose=False,
                visualize_realtime=False 
            )
            print(f"    ...Run {i+1} Best: {ga_obj:.2f}")
            ga_results.append(ga_obj)
            
            if ga_obj < ga_best_obj_overall:
                ga_best_obj_overall = ga_obj
                ga_best_history_overall = ga_history
            
        end_time_ga = time.time()

        # 5. Calculate and Print Statistics
        sa_stats = {
            'mean': np.mean(sa_results),
            'std': np.std(sa_results),
            'time': (end_time_sa - start_time_sa) / NUM_EXPERIMENT_RUNS,
            'best': sa_best_obj_overall
        }
        ga_stats = {
            'mean': np.mean(ga_results),
            'std': np.std(ga_results),
            'time': (end_time_ga - start_time_ga) / NUM_EXPERIMENT_RUNS,
            'best': ga_best_obj_overall
        }
        
        print("\n" + "="*70)
        print("EXPERIMENT STATISTICAL RESULTS")
        print(f"(Based on {NUM_EXPERIMENT_RUNS} runs each with same initial state)")
        print("="*70)
        print(f"Metric\t\t\tSimulated Annealing (SA)\tGenetic Algorithm (GA)")
        print(f"-------------------------------------------------------------------------")
        print(f"Mean (Avg. Best Cost)\t{sa_stats['mean']:<26.2f}\t{ga_stats['mean']:<24.2f}")
        print(f"Std. Deviation (Cost)\t{sa_stats['std']:<26.2f}\t{ga_stats['std']:<24.2f}")
        print(f"Avg. Run Time (sec)\t{sa_stats['time']:<26.3f}\t{ga_stats['time']:<24.3f}")
        print(f"Best Cost Found (Overall)\t{sa_stats['best']:<26.2f}\t{ga_stats['best']:<24.2f}")
        print("="*70)
        if ga_stats['mean'] < sa_stats['mean']:
            print("\n  GA achieved a better (lower) average final cost.")
        else:
            print("\n  SA achieved a better (lower) average final cost.")
            
        if ga_stats['std'] < sa_stats['std']:
            print("  GA was more consistent (lower standard deviation).")
        else:
            print("  SA was more consistent (lower standard deviation).")
        
        # 6. Save results to CSVs
        print("\n" + "="*70)
        print("SAVING EXPERIMENT RESULTS...")
        
        experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        save_experiment_summary(experiment_timestamp, sa_stats, ga_stats)
        save_experiment_raw_data(experiment_timestamp, sa_results, ga_results)
        
        # 7. Show All Plots
        print("\n" + "="*70)
        print("GENERATING PLOTS (Close each plot to see the next)...")
        
        plot_experiment_results(sa_results, ga_results)
        
        plot_experiment_distribution(sa_results, ga_results)
        
        print("  Displaying performance dashboard for the BEST SA run...")
        plot_sa_results(sa_best_history_overall)
        
        print("  Displaying performance dashboard for the BEST GA run...")
        plot_ga_performance_dashboard(ga_best_history_overall)
        
        print("  (All plots displayed)")

    else:
        print(f"Error: Unknown OPTIMIZATION_ALGORITHM: '{OPTIMIZATION_ALGORITHM}'")

    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()