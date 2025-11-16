# File: main_with_viz.py
"""
Main entry point for running the intersection optimization with selectable visualization
and algorithm comparison.
"""

# =============================================================================
# ❗ CRITICAL NOTE FOR "NATURAL RUN" COMPARISON (COMPARISON B) ❗
# =============================================================================
# This file is now set up to run a "Natural Run" comparison.
# This means:
# 1. GA runs until it hits 'NUM_GENERATIONS' or 'CONVERGENCE_PATIENCE'.
# 2. SA should run until it hits 'T_min'.
#
# FOR THIS TO WORK:
# You MUST ensure your 'sa.py' file is using the "textbook" 'run_sa'
# function (the one that stops with 'while T > T_min...').
# The 'sa.MAX_TOTAL_ITERATIONS' is now only a safety cap, not the main
# stopping condition for SA.
# =============================================================================


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
    print("  'GA', 'BOTH', 'GA_ANALYSIS', and 'EXPERIMENT' modes will not be available.")
    GA_IMPORTED = False


# =============================================================================
# CONFIGURATION
# =============================================================================
# --- 1. CHOOSE ALGORITHM ---
# 'SA':          Single run of SA (uses sa.MAX_TOTAL_ITERATIONS).
# 'GA':          Single run of GA (uses ga.NUM_GENERATIONS).
# 'SA_ANALYSIS': N-run statistical analysis of SA (natural stop).
# 'GA_ANALYSIS': N-run statistical analysis of GA (natural stop).
# 'BOTH':        Single run SA vs. GA (uses COMPARISON_EVALUATION_BUDGET).
# 'EXPERIMENT':  N-run statistical comparison of SA vs. GA (natural stops).
OPTIMIZATION_ALGORITHM = 'GA_ANALYSIS' 

# --- 2. CHOOSE VISUALIZATION ---
# 'matplotlib', 'web', 'none'
# (Ignored for 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS' modes)
VISUALIZATION_METHOD = 'none' 

# --- 3. ALGORITHM PARAMETERS ---
# Budget for *direct comparison modes only* ('BOTH')
COMPARISON_EVALUATION_BUDGET = 5000 
# Number of runs for 'SA_ANALYSIS', 'GA_ANALYSIS', 'EXPERIMENT'
NUM_EXPERIMENT_RUNS = 5  # 
RANDOM_SEED = 42 

# SA Parameters (used for 'SA' and 'SA_ANALYSIS')
T_INITIAL = sa.T_INITIAL
T_MIN = sa.T_MIN 
COOLING_RATE = sa.COOLING_RATE
MAX_ITER_PER_TEMP = sa.MAX_ITER_PER_TEMP

# GA Parameters (used for 'GA' and 'GA_ANALYSIS')
if GA_IMPORTED:
    from ga import (
        POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, 
        TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
    )
else:
    # Default values if GA not imported, to prevent errors
    POPULATION_SIZE = 50
    NUM_GENERATIONS = 100
    ELITISM_RATE = 0.1
    TOURNAMENT_SIZE = 3
    MUTATION_RATE_PERM = 0.1
    MUTATION_RATE_SPEED = 0.1
# =============================================================================


# --- Conditional Imports Based on Visualization Method ---
animation_enabled = False
web_viz_enabled = False
is_analysis_mode = OPTIMIZATION_ALGORITHM in ('BOTH', 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS')

if not is_analysis_mode and VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization import IntersectionVisualization
        animation_enabled = True
        print("Matplotlib visualization enabled (SMOOTH animation)")
    except ImportError as e:
        print(f"Warning: Could not import visualization.py: {e}")

elif not is_analysis_mode and VISUALIZATION_METHOD == 'web':
    try:
        from visualization_utils import IntersectionVisualizer
        web_viz_enabled = True
        print("Web-based visualization enabled")
    except ImportError as e:
        print(f"Warning: Could not import web visualization: {e}")

elif VISUALIZATION_METHOD == 'none':
    print("Visualization disabled (running in headless mode)")
else:
    if is_analysis_mode:
        print(f"Visualization disabled (running in '{OPTIMIZATION_ALGORITHM}' mode)")
    else:
        print(f"Warning: Unknown VISUALIZATION_METHOD '{VISUALIZATION_METHOD}'")
        print("  Continuing without visualization.")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals):
    """
    Plots a direct, fixed-budget comparison. 
    (This is primarily for 'BOTH' mode).
    """
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
    
    if ga_eval_points and ga_evals < ga_eval_points[-1]:
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

def plot_compare(history_sa, history_ga, labels=('SA', 'GA'), sa_evals=None, ga_evals=None):
    """Overlay SA and GA histories on a single 2x2 comparison grid.

    Accepts history dicts produced by `run_sa` and `run_ga` (formats used in this project).
    The function will look for common keys and plot matching series; missing series are skipped.
    """
    def first(h, options):
        for k in options:
            if isinstance(h, dict) and k in h:
                return h[k]
        return []

    # Helper function to get SA x-axis and clipped series
    def sa_align_and_clip(series, budget):
        if not series:
            return np.array([]), []
        
        x = np.arange(1, len(series) + 1)
        
        # If a budget is provided, clip both X and the series itself
        if budget is not None:
            clip_idx = np.searchsorted(x, budget, side='right')
            if clip_idx > len(x):
                clip_idx = len(x)
            
            x = x[:clip_idx]
            series = series[:clip_idx]

        return np.array(x), series
    
    # Helper function to align and clip GA series
    def ga_align_and_clip(series, eval_points):
        if not series or not eval_points:
            return np.array([]), []
        
        # The number of generations is min(len(series), len(eval_points))
        num_generations = min(len(series), len(eval_points))
        
        x_array = np.array(eval_points[:num_generations])
        clipped_series = series[:num_generations]
        
        return x_array, clipped_series

    costs_sa = first(history_sa, ['costs', 'best', 'best_f'])
    avg_sa = first(history_sa, ['avg', 'avg_f'])
    costs_ga = first(history_ga, ['best_f', 'costs', 'best'])
    avg_ga = first(history_ga, ['avg_f', 'avg'])

    avg_delays_sa = first(history_sa, ['avg_delays', 'avg_delays'])
    avg_delays_ga = first(history_ga, ['best_avg_delay', 'avg_delays'])

    total_delays_sa = first(history_sa, ['total_delays', 'total_delays'])
    total_delays_ga = first(history_ga, ['best_fall', 'total_delays'])

    emergency_delays_sa = first(history_sa, ['emergency_delays', 'emergency_delays'])
    emergency_delays_ga = first(history_ga, ['best_fem', 'emergency_delays'])

    # Prepare x-axis mapping to NUMBER OF FITNESS EVALUATIONS
    # GA: history entries are per-generation. Map generation index to cumulative evaluation counts.
    # Determine number of GA generations present in any series
    ga_series_lengths = [
        len(x) for x in (costs_ga, avg_ga, avg_delays_ga, total_delays_ga, emergency_delays_ga) if x
    ]
    ga_len = max(ga_series_lengths) if ga_series_lengths else 0
    ga_eval_points = []
    if ga_len > 0:
        # Assuming POPULATION_SIZE and ELITISM_RATE are accessible globally
        ga_evals_per_gen = (POPULATION_SIZE - int(POPULATION_SIZE * ELITISM_RATE))
        if ga_evals_per_gen <= 0:
            ga_evals_per_gen = 1
        # cumulative eval counts at generation boundaries
        ga_eval_points = [POPULATION_SIZE] + [POPULATION_SIZE + (i * ga_evals_per_gen) for i in range(1, ga_len)]
        
        # If a ga_evals budget is provided, clip to that budget
        if ga_evals is not None and ga_eval_points:
            # find index where points exceed budget
            clip_idx = np.searchsorted(ga_eval_points, ga_evals, side='right')
            pts = ga_eval_points[:clip_idx].copy()
            
            # if the budget is after the last point, add the budget point itself
            if pts and ga_evals > pts[-1]:
                pts.append(ga_evals)
            
            ga_eval_points = pts

    # SA x arrays and clipped series
    x_costs_sa, costs_sa = sa_align_and_clip(costs_sa, sa_evals)
    x_avg_sa, avg_sa = sa_align_and_clip(avg_sa, sa_evals)
    x_avg_delays_sa, avg_delays_sa = sa_align_and_clip(avg_delays_sa, sa_evals)
    x_total_delays_sa, total_delays_sa = sa_align_and_clip(total_delays_sa, sa_evals)
    x_emergency_delays_sa, emergency_delays_sa = sa_align_and_clip(emergency_delays_sa, sa_evals)

    # GA x arrays and clipped series
    x_costs_ga, costs_ga = ga_align_and_clip(costs_ga, ga_eval_points)
    x_avg_ga, avg_ga = ga_align_and_clip(avg_ga, ga_eval_points)
    x_avg_delays_ga, avg_delays_ga = ga_align_and_clip(avg_delays_ga, ga_eval_points)
    x_total_delays_ga, total_delays_ga = ga_align_and_clip(total_delays_ga, ga_eval_points)
    x_emergency_delays_ga, emergency_delays_ga = ga_align_and_clip(emergency_delays_ga, ga_eval_points)
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Overlay: {labels[0]} vs {labels[1]}")

    # Top-left: Cost (mapped to fitness evaluation counts)
    ax = axs[0, 0]
    if x_costs_sa.size:
        ax.plot(x_costs_sa, costs_sa, label=f"{labels[0]} best", color='tab:blue')
    if x_costs_ga.size:
        ax.plot(x_costs_ga, costs_ga, label=f"{labels[1]} best", color='tab:orange')
    if x_avg_sa.size:
        ax.plot(x_avg_sa, avg_sa, label=f"{labels[0]} avg", color='tab:cyan', linestyle=':')
    if x_avg_ga.size:
        ax.plot(x_avg_ga, avg_ga, label=f"{labels[1]} avg", color='tab:gray', linestyle='-.')
    ax.set_title('Objective Cost vs Evaluations')
    ax.set_xlabel('Number of Fitness Evaluations')
    ax.set_ylabel('Cost (f)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Top-right: Average Delay
    ax = axs[0, 1]
    if x_avg_delays_sa.size:
        ax.plot(x_avg_delays_sa, avg_delays_sa, label=f"{labels[0]}", color='tab:green')
    if x_avg_delays_ga.size:
        ax.plot(x_avg_delays_ga, avg_delays_ga, label=f"{labels[1]}", color='tab:red')
    ax.set_title('Average Delay per Vehicle')
    ax.set_xlabel('Number of Fitness Evaluations')
    ax.set_ylabel('Avg Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Bottom-left: Total Delay
    ax = axs[1, 0]
    if x_total_delays_sa.size:
        ax.plot(x_total_delays_sa, total_delays_sa, label=f"{labels[0]}", color='tab:cyan')
    if x_total_delays_ga.size:
        ax.plot(x_total_delays_ga, total_delays_ga, label=f"{labels[1]}", color='tab:olive')
    ax.set_title('Total Delay (f_all)')
    ax.set_xlabel('Number of Fitness Evaluations')
    ax.set_ylabel('Total Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Bottom-right: Emergency Delay
    ax = axs[1, 1]
    if x_emergency_delays_sa.size:
        ax.plot(x_emergency_delays_sa, emergency_delays_sa, label=f"{labels[0]}", color='tab:blue')
    if x_emergency_delays_ga.size:
        ax.plot(x_emergency_delays_ga, emergency_delays_ga, label=f"{labels[1]}", color='tab:orange')
    ax.set_title('Emergency Delay (f_em)')
    ax.set_xlabel('Number of Fitness Evaluations')
    ax.set_ylabel('Emergency Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Set common x-limits based on provided evaluation budgets or global comparison budget
    x_max = None
    if sa_evals is not None and ga_evals is not None:
        x_max = max(sa_evals, ga_evals) # For Natural Run, just show longest run
    elif sa_evals is not None:
        x_max = sa_evals
    elif ga_evals is not None:
        x_max = ga_evals
    else:
        try:
            # Fallback for 'BOTH' mode
            x_max = COMPARISON_EVALUATION_BUDGET
        except NameError:
            x_max = None

    if x_max is not None:
        for ax in axs.flatten():
            ax.set_xlim(0, x_max)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_experiment_results(results_data, labels):
    """
    Creates a box plot comparing the final distributions of
    one or more algorithms.
    """
    print("\nPlotting Experiment Statistical Results (Box Plot)...")
    
    data = results_data
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    
    ax.set_title(f'Algorithm Performance Comparison ({NUM_EXPERIMENT_RUNS} Runs Each)')
    ax.set_ylabel('Final Objective Cost (f) - Lower is Better')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_experiment_distribution(results_data, labels, colors):
    """
    Creates separate histogram/KDE plots for one or more algorithms.
    """
    print("\nPlotting Experiment Distribution (Histograms + Gaussian KDE)...")
    
    all_results = []
    for res in results_data:
        all_results.extend(res)
    
    if not all_results:
        print("  No data to plot for distribution.")
        return
        
    min_bin = min(all_results)
    max_bin = max(all_results)
    bins = np.linspace(min_bin, max_bin, 30) 

    for i, res in enumerate(results_data):
        if not res: # Skip if no results for this algo
            continue
            
        label = labels[i]
        color = colors[i]
        
        plt.figure(figsize=(10, 6))
        plt.hist(res, bins=bins, alpha=0.7, label=f'{label} Histogram', color=color, density=True)
        
        try:
            # Check for sufficient data
            if len(res) > 1:
                kde = stats.gaussian_kde(res)
                kde_x = np.linspace(min_bin, max_bin, 200)
                plt.plot(kde_x, kde(kde_x), color, linewidth=2, label=f'{label} Distribution (KDE)')
            else:
                print(f"  Warning: Not enough data points to plot Gaussian KDE for {label}.")
        except Exception as e:
            print(f"  Warning: Could not plot {label} Gaussian KDE. {e}")

        plt.title(f'{label} Result Distribution ({NUM_EXPERIMENT_RUNS} Runs)')
        plt.xlabel('Final Objective Cost (f)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        mean = np.mean(res)
        std = np.std(res)
        text_str = (f'{label}: Mean={mean:.2f}\n{label}: Std Dev={std:.2f}')
        ax = plt.gca()
        plt.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10,
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

def save_experiment_summary(timestamp, all_stats):
    """
    Saves the high-level summary statistics to a SINGLE master CSV file.
    Appends the new run; creates header if file doesn't exist.
    """
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    try:
        with open(SUMMARY_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # --- MODIFIED: Changed CSV Header ---
                writer.writerow([
                    'Timestamp', 'Algorithm', 'Mean Cost (f)', 'Std Deviation (f)', 
                    'Avg Run Time (s)', 'Best Cost Overall', 'Num Runs', 
                    'Avg Evals Used', # <-- This is the important change
                    'SA_T_Init', 'SA_T_Min', 'SA_Cool_Rate', 'SA_Iter_Per_Temp',
                    'GA_Pop_Size', 'GA_Generations', 'GA_Elitism', 'GA_Tourn_Size', 'GA_Mut_Perm', 'GA_Mut_Speed'
                ])
            
            if 'SA' in all_stats:
                stats = all_stats['SA']
                writer.writerow([
                    timestamp, 'SA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    T_INITIAL, T_MIN, COOLING_RATE, MAX_ITER_PER_TEMP,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A' 
                ])
            if 'GA' in all_stats:
                stats = all_stats['GA']
                writer.writerow([
                    timestamp, 'GA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    'N/A', 'N/A', 'N/A', 'N/A',
                    POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
                ])
        print(f"  Successfully appended summary to: {SUMMARY_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save summary CSV. {e}")


def save_experiment_raw_data(timestamp, sa_results=None, ga_results=None, sa_evals=None, ga_evals=None):
    """
    Saves the raw final cost AND evals from every single run to a SINGLE master CSV file.
    Appends the new runs; creates header if file doesn't exist.
    """
    file_exists = os.path.isfile(RAW_FILENAME)
    try:
        with open(RAW_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # --- MODIFIED: Added evals columns ---
                writer.writerow(['Experiment_Timestamp', 'Run', 'SA_Final_Cost', 'SA_Evals', 'GA_Final_Cost', 'GA_Evals'])
            
            # Determine max number of runs
            num_runs = 0
            if sa_results: num_runs = len(sa_results)
            if ga_results: num_runs = max(num_runs, len(ga_results))
            
            for i in range(num_runs):
                sa_val = sa_results[i] if sa_results and i < len(sa_results) else 'N/A'
                sa_ev = sa_evals[i] if sa_evals and i < len(sa_evals) else 'N/A'
                ga_val = ga_results[i] if ga_results and i < len(ga_results) else 'N/A'
                ga_ev = ga_evals[i] if ga_evals and i < len(ga_evals) else 'N/A'
                # --- MODIFIED: Write all data points ---
                writer.writerow([timestamp, i+1, sa_val, sa_ev, ga_val, ga_ev])
                    
        print(f"  Successfully appended raw data to: {RAW_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save raw data CSV. {e}")
# --- END MODIFICATION ---

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
    if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS'):
        print(f"Experiment Runs:      {NUM_EXPERIMENT_RUNS}")
    print("="*70 + "\n")
    
    print("Using the STATIC problem instance (from config.pi)")
    
    
    # --- Run SA (Single Run) ---
    if OPTIMIZATION_ALGORITHM == 'SA':
        perm_best, speeds_best, obj_best, sa_history, geom, tau_p_dict, evals = run_sa(
            max_iter=sa.MAX_TOTAL_ITERATIONS # Use default safety cap
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
            max_evaluations=None, # <-- This tells GA to use NUM_GENERATIONS or converge
            visualize_realtime=True 
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
            max_iter=COMPARISON_EVALUATION_BUDGET # Uses fixed budget
        )
        
        print("\n--- RUNNING GENETIC ALGORITHM ---")
        (ga_perm, ga_speeds, ga_obj, ga_history, 
         ga_geom, ga_tau, best_ga_obj_dict, ga_evals) = run_ga(
            max_evaluations=COMPARISON_EVALUATION_BUDGET, # Uses fixed budget
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
        try:
            print("Generating overlay comparison (SA vs GA)...")
            plot_compare(sa_history, ga_history, labels=('SA', 'GA'), sa_evals=sa_evals, ga_evals=ga_evals)
        except Exception as e:
            print(f"Could not generate overlay comparison plot: {e}")
            traceback.print_exc()

    # --- MODIFICATION: Split Experiment logic for "Natural Run" ---
    elif OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS'):
        if not GA_IMPORTED:
            print("ERROR: 'GA' dependent modes selected but ga.py could not be imported.")
            return
            
        print("\n--- PREPARING EXPERIMENT (NATURAL RUN) ---")
        print(f"Running {NUM_EXPERIMENT_RUNS} times for selected algorithm(s).")
        
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

        sa_results = []
        ga_results = []
        sa_evals_list = [] # <-- MODIFIED: Track evals per run
        ga_evals_list = [] # <-- MODIFIED: Track evals per run
        
        sa_best_obj_overall = math.inf
        sa_best_history_overall = {}
        ga_best_obj_overall = math.inf
        ga_best_history_overall = {}
        all_stats = {}
        
        # --- FIX: Defined experiment_timestamp here ---
        experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. Run SA Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS'):
            print("\nGenerating common initial solution for SA...")
            common_sa_solution = sa_create_initial_solution(geom)
            
            # --- MODIFICATION: Use default iter cap for NATURAL run ---
            # The 'run_sa' function itself should stop on T < T_min
            sa_iter_limit = sa.MAX_TOTAL_ITERATIONS
            print(f"SA using NATURAL STOP (T<T_min), safety cap: {sa_iter_limit} iterations")
            # ---
            
            print("\n" + "="*70)
            print(f"RUNNING SA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
            print("="*70)
            start_time_sa = time.time()
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"  SA Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                (sa_perm, sa_speeds, sa_obj, sa_history, 
                 sa_geom, sa_tau, sa_evals)= run_sa(
                    max_iter=sa_iter_limit,
                    initial_solution=common_sa_solution,
                    verbose=False 
                )
                print(f"    ...Run {i+1} Best: {sa_obj:.2f} (in {sa_evals} evals)")
                sa_results.append(sa_obj)
                sa_evals_list.append(sa_evals) # <-- MODIFIED
                if sa_obj < sa_best_obj_overall:
                    sa_best_obj_overall = sa_obj
                    sa_best_history_overall = sa_history
            end_time_sa = time.time()
            
            all_stats['SA'] = {
                'mean': np.mean(sa_results),
                'std': np.std(sa_results),
                'time': (end_time_sa - start_time_sa) / NUM_EXPERIMENT_RUNS,
                'best': sa_best_obj_overall,
                'avg_evals': np.mean(sa_evals_list) # <-- MODIFIED
            }

        # 4. Run GA Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'GA_ANALYSIS'):
            print("\nGenerating common initial population for GA...")
            common_ga_population = create_initial_population(POPULATION_SIZE, geom)
            
            # --- MODIFICATION: Use default generations for NATURAL run ---
            # 'None' tells ga.py to use NUM_GENERATIONS or CONVERGENCE_PATIENCE
            ga_eval_limit = None 
            print(f"GA using NATURAL STOP (Num_Gens={NUM_GENERATIONS} or Convergence)")
            # ---

            print("\n" + "="*70)
            print(f"RUNNING GA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
            print("="*70)
            start_time_ga = time.time()
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"  GA Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                _, _, ga_obj, ga_history, _, _, _, ga_evals = run_ga(
                    max_evaluations=ga_eval_limit, # <-- This is None
                    initial_population=common_ga_population,
                    verbose=False,
                    visualize_realtime=False 
                )
                print(f"    ...Run {i+1} Best: {ga_obj:.2f} (in {ga_evals} evals)")
                ga_results.append(ga_obj)
                ga_evals_list.append(ga_evals) # <-- MODIFIED
                if ga_obj < ga_best_obj_overall:
                    ga_best_obj_overall = ga_obj
                    ga_best_history_overall = ga_history
            end_time_ga = time.time()
            
            all_stats['GA'] = {
                'mean': np.mean(ga_results),
                'std': np.std(ga_results),
                'time': (end_time_ga - start_time_ga) / NUM_EXPERIMENT_RUNS,
                'best': ga_best_obj_overall,
                'avg_evals': np.mean(ga_evals_list) # <-- MODIFIED
            }
        
        # 5. Calculate and Print Statistics
        print("\n" + "="*70)
        print("EXPERIMENT STATISTICAL RESULTS")
        print(f"(Based on {NUM_EXPERIMENT_RUNS} runs each with same initial state)")
        print("="*70)
        
        if 'SA' in all_stats:
            stats = all_stats['SA']
            print("--- Simulated Annealing (SA) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")

        if 'GA' in all_stats:
            stats = all_stats['GA']
            print("--- Genetic Algorithm (GA) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")
        
        print("="*70)
        
        if 'SA' in all_stats and 'GA' in all_stats:
            if all_stats['GA']['mean'] < all_stats['SA']['mean']:
                print("\n  GA achieved a better (lower) average final cost.")
            else:
                print("\n  SA achieved a better (lower) average final cost.")
            if all_stats['GA']['std'] < all_stats['SA']['std']:
                print("  GA was more consistent (lower standard deviation).")
            else:
                print("  SA was more consistent (lower standard deviation).")
        
        # 6. Save results to CSVs
        print("\n" + "="*70)
        print("SAVING EXPERIMENT RESULTS...")
        
        # --- FIX: Use correct variable name ---
        save_experiment_summary(experiment_timestamp, all_stats)
        
        if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
            save_experiment_raw_data(experiment_timestamp, sa_results, ga_results, sa_evals_list, ga_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'SA_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, sa_results=sa_results, sa_evals=sa_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'GA_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, ga_results=ga_results, ga_evals=ga_evals_list)
        # --- END FIX ---
        
        # 7. Show All Plots
        print("\n" + "="*70)
        print("GENERATING PLOTS (Close each plot to see the next)...")
        
        if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
            plot_experiment_results([sa_results, ga_results], ['SA', 'GA'])
            plot_experiment_distribution([sa_results, ga_results], ['SA', 'GA'], ['blue', 'red'])
            print("  Displaying performance dashboard for the BEST SA run...")
            plot_sa_results(sa_best_history_overall)
            print("  Displaying performance dashboard for the BEST GA run...")
            plot_ga_performance_dashboard(ga_best_history_overall)
            
            # Plotting the overlay plot is now more complex, as the x-axes (evals)
            # are different for every run. We plot the "best" run for each.
            try:
                print("Generating overlay comparison (SA vs GA - Best Runs)...")
                # We need to find the evals for the *best* run, not the avg
                best_sa_evals = sa_evals_list[np.argmin(sa_results)]
                best_ga_evals = ga_evals_list[np.argmin(ga_results)]
                plot_compare(sa_best_history_overall, ga_best_history_overall, labels=('SA (Best)', 'GA (Best)'), 
                             sa_evals=best_sa_evals, ga_evals=best_ga_evals)
            except Exception as e:
                print(f"Could not generate overlay comparison plot: {e}")
                traceback.print_exc()
            
        elif OPTIMIZATION_ALGORITHM == 'SA_ANALYSIS':
            plot_experiment_results([sa_results], ['SA'])
            plot_experiment_distribution([sa_results], ['SA'], ['blue'])
            print("  Displaying performance dashboard for the BEST SA run...")
            plot_sa_results(sa_best_history_overall)

        elif OPTIMIZATION_ALGORITHM == 'GA_ANALYSIS':
            plot_experiment_results([ga_results], ['GA'])
            plot_experiment_distribution([ga_results], ['GA'], ['red'])
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