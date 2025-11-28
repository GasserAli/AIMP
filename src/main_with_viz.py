# File: main_with_viz.py
"""
Main entry point for running the intersection optimization with selectable visualization
and algorithm comparison. Now includes support for MMAS (Max-Min Ant System).
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
    create_initial_solution as sa_create_initial_solution,
    T_INITIAL, T_MIN, COOLING_RATE, MAX_ITER_PER_TEMP
)

try:
    from ga import run_ga, plot_ga_performance_dashboard, create_initial_population
    GA_IMPORTED = True
except Exception as e:
    print(f"WARNING: Could not import ga.py: {e}")
    GA_IMPORTED = False

# --- Import MMAS (new) ---
try:
    from mmas import (
        run_mmas,
        MAX_ITER,
        plot_mmas_results,
        plot_mmas_performance_dashboard
    )
    MMAS_IMPORTED = True
except Exception as e:
    print(f"WARNING: Could not import mmas.py: {e}")
    MMAS_IMPORTED = False


# =============================================================================
# CONFIGURATION
# =============================================================================
# Choose algorithm:
# 'SA', 'GA', 'MMAS', 'SA_ANALYSIS', 'GA_ANALYSIS', 'MMAS_ANALYSIS', 'BOTH', 'EXPERIMENT'
OPTIMIZATION_ALGORITHM = 'MMAS'

# Visualization options: 'matplotlib', 'web', 'none'
VISUALIZATION_METHOD = 'matplotlib'

# Algorithm parameters
COMPARISON_EVALUATION_BUDGET = 50000
NUM_EXPERIMENT_RUNS = 50
RANDOM_SEED = 42

# GA params fallback (if ga.py not imported)
if GA_IMPORTED:
    from ga import (POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED)
else:
    POPULATION_SIZE = 150
    NUM_GENERATIONS = 150
    ELITISM_RATE = 0.1
    TOURNAMENT_SIZE = 3
    MUTATION_RATE_PERM = 0.1
    MUTATION_RATE_SPEED = 0.1

# Visualization conditional imports
animation_enabled = False
web_viz_enabled = False
is_analysis_mode = OPTIMIZATION_ALGORITHM in ('BOTH', 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'MMAS_ANALYSIS')

if not is_analysis_mode and VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization import IntersectionVisualization
        animation_enabled = True
        print("Matplotlib visualization enabled (SMOOTH animation)")
    except Exception as e:
        print(f"Warning: Could not import visualization.py: {e}")

elif not is_analysis_mode and VISUALIZATION_METHOD == 'web':
    try:
        from visualization_utils import IntersectionVisualizer
        web_viz_enabled = True
        print("Web-based visualization enabled")
    except Exception as e:
        print(f"Warning: Could not import web visualization: {e}")

elif VISUALIZATION_METHOD == 'none':
    print("Visualization disabled (running in headless mode)")
else:
    if is_analysis_mode:
        print(f"Visualization disabled (running in '{OPTIMIZATION_ALGORITHM}' mode)")
    else:
        print(f"Warning: Unknown VISUALIZATION_METHOD '{VISUALIZATION_METHOD}'")

# =============================================================================
# PLOTTING & COMPARISON HELPERS
# =============================================================================

def plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals):
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
    def first(h, options):
        for k in options:
            if isinstance(h, dict) and k in h:
                return h[k]
        return []

    def sa_align_and_clip(series, budget):
        if not series:
            return np.array([]), []
        x = np.arange(1, len(series) + 1)
        if budget is not None:
            clip_idx = np.searchsorted(x, budget, side='right')
            if clip_idx > len(x):
                clip_idx = len(x)
            x = x[:clip_idx]
            series = series[:clip_idx]
        return np.array(x), series

    def ga_align_and_clip(series, eval_points):
        if not series or not eval_points:
            return np.array([]), []
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

    ga_series_lengths = [
        len(x) for x in (costs_ga, avg_ga, avg_delays_ga, total_delays_ga, emergency_delays_ga) if x
    ]
    ga_len = max(ga_series_lengths) if ga_series_lengths else 0
    ga_eval_points = []
    if ga_len > 0:
        ga_evals_per_gen = (POPULATION_SIZE - int(POPULATION_SIZE * ELITISM_RATE))
        if ga_evals_per_gen <= 0:
            ga_evals_per_gen = 1
        ga_eval_points = [POPULATION_SIZE] + [POPULATION_SIZE + (i * ga_evals_per_gen) for i in range(1, ga_len)]
        if ga_evals is not None and ga_eval_points:
            clip_idx = np.searchsorted(ga_eval_points, ga_evals, side='right')
            pts = ga_eval_points[:clip_idx].copy()
            if pts and ga_evals > pts[-1]:
                pts.append(ga_evals)
            ga_eval_points = pts

    x_costs_sa, costs_sa = sa_align_and_clip(costs_sa, sa_evals)
    x_avg_sa, avg_sa = sa_align_and_clip(avg_sa, sa_evals)
    x_avg_delays_sa, avg_delays_sa = sa_align_and_clip(avg_delays_sa, sa_evals)
    x_total_delays_sa, total_delays_sa = sa_align_and_clip(total_delays_sa, sa_evals)
    x_emergency_delays_sa, emergency_delays_sa = sa_align_and_clip(emergency_delays_sa, sa_evals)

    x_costs_ga, costs_ga = ga_align_and_clip(costs_ga, ga_eval_points)
    x_avg_ga, avg_ga = ga_align_and_clip(avg_ga, ga_eval_points)
    x_avg_delays_ga, avg_delays_ga = ga_align_and_clip(avg_delays_ga, ga_eval_points)
    x_total_delays_ga, total_delays_ga = ga_align_and_clip(total_delays_ga, ga_eval_points)
    x_emergency_delays_ga, emergency_delays_ga = ga_align_and_clip(emergency_delays_ga, ga_eval_points)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Overlay: {labels[0]} vs {labels[1]}")

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

    x_max = None
    if sa_evals is not None and ga_evals is not None:
        x_max = max(sa_evals, ga_evals)
    elif sa_evals is not None:
        x_max = sa_evals
    elif ga_evals is not None:
        x_max = ga_evals
    else:
        try:
            x_max = COMPARISON_EVALUATION_BUDGET
        except NameError:
            x_max = None

    if x_max is not None:
        for ax in axs.flatten():
            ax.set_xlim(0, x_max)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_experiment_results(results_data, labels):
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
        if not res:
            continue
        label = labels[i]
        color = colors[i]
        plt.figure(figsize=(10, 6))
        plt.hist(res, bins=bins, alpha=0.7, label=f'{label} Histogram', color=color, density=True)
        try:
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
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    try:
        with open(SUMMARY_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Algorithm', 'Mean Cost (f)', 'Std Deviation (f)',
                    'Avg Run Time (s)', 'Best Cost Overall', 'Num Runs', 'Avg Evals Used',
                    'SA_T_Init', 'SA_T_Min', 'SA_Cool_Rate', 'SA_Iter_Per_Temp',
                    'GA_Pop_Size', 'GA_Generations', 'GA_Elitism', 'GA_Tourn_Size', 'GA_Mut_Perm', 'GA_Mut_Speed'
                ])

            if 'SA' in all_stats:
                stats = all_stats['SA']
                writer.writerow([
                    timestamp, 'SA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'],
                    T_INITIAL, T_MIN, COOLING_RATE, MAX_ITER_PER_TEMP,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])
            if 'GA' in all_stats:
                stats = all_stats['GA']
                writer.writerow([
                    timestamp, 'GA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'],
                    'N/A', 'N/A', 'N/A', 'N/A',
                    POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED
                ])
            if 'MMAS' in all_stats:
                stats = all_stats['MMAS']
                writer.writerow([
                    timestamp, 'MMAS', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'],
                    'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])
        print(f"  Successfully appended summary to: {SUMMARY_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save summary CSV. {e}")

def save_experiment_raw_data(timestamp, sa_results=None, ga_results=None, mmas_results=None, sa_evals=None, ga_evals=None, mmas_evals=None):
    file_exists = os.path.isfile(RAW_FILENAME)
    try:
        with open(RAW_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Experiment_Timestamp', 'Run', 'SA_Final_Cost', 'SA_Evals', 'GA_Final_Cost', 'GA_Evals', 'MMAS_Final_Cost', 'MMAS_Evals'])
            num_runs = 0
            if sa_results: num_runs = len(sa_results)
            if ga_results: num_runs = max(num_runs, len(ga_results))
            if mmas_results: num_runs = max(num_runs, len(mmas_results))
            for i in range(num_runs):
                sa_val = sa_results[i] if sa_results and i < len(sa_results) else 'N/A'
                sa_ev = sa_evals[i] if sa_evals and i < len(sa_evals) else 'N/A'
                ga_val = ga_results[i] if ga_results and i < len(ga_results) else 'N/A'
                ga_ev = ga_evals[i] if ga_evals and i < len(ga_evals) else 'N/A'
                mmas_val = mmas_results[i] if mmas_results and i < len(mmas_results) else 'N/A'
                mmas_ev = mmas_evals[i] if mmas_evals and i < len(mmas_evals) else 'N/A'
                writer.writerow([timestamp, i+1, sa_val, sa_ev, ga_val, ga_ev, mmas_val, mmas_ev])
        print(f"  Successfully appended raw data to: {RAW_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save raw data CSV. {e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    print(f"Setting global random seed to: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*70)
    print("INTERSECTION TRAFFIC OPTIMIZATION")
    print("="*70)
    print(f"Selected Algorithm:   {OPTIMIZATION_ALGORITHM.upper()}")
    print(f"Visualization Method: {VISUALIZATION_METHOD.upper()}")
    if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'MMAS_ANALYSIS'):
        print(f"Experiment Runs:      {NUM_EXPERIMENT_RUNS}")
    print("="*70 + "\n")

    print("Using the STATIC problem instance (from config.pi)")

    # --- Run SA (Single Run) ---
    if OPTIMIZATION_ALGORITHM == 'SA':
        perm_best, speeds_best, obj_best, sa_history, geom, tau_p_dict, evals = run_sa(
            max_iter=sa.MAX_TOTAL_ITERATIONS
        )
        print("\nDisplaying SA performance plots...")
        plot_sa_results(sa_history)
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
            max_evaluations=None,
            visualize_realtime=True
        )
        print("\nDisplaying GA performance plots...")
        plot_ga_performance_dashboard(ga_history)
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run MMAS (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'MMAS':
        if not MMAS_IMPORTED:
            print("ERROR: MMAS selected but mmas.py could not be imported.")
            return
        perm_best, speeds_best, obj_best, mmas_history, geom, tau_p_dict, evals = run_mmas(max_iter=MAX_ITER)
        print("\nDisplaying MMAS performance plots...")
        try:
            # Full MMAS Dashboard (GA-style)
            plot_mmas_performance_dashboard(mmas_history)

            # SA-style progression plot
            plot_mmas_results(mmas_history)
        except Exception as e:
            print("MMAS plotting failed:", e)

        except Exception:
            # Fallback: simple print if plotting fails
            print("MMAS history plotting failed or not in expected format.")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run BOTH (Single Run Comparison) ---
    elif OPTIMIZATION_ALGORITHM == 'BOTH':
        if not GA_IMPORTED or not MMAS_IMPORTED:
            print("ERROR: 'BOTH' mode selected but GA or MMAS module could not be imported.")
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

        print("\n--- RUNNING MMAS ---")
        (mmas_perm, mmas_speeds, mmas_obj, mmas_history,
         mmas_geom, mmas_tau, mmas_evals) = run_mmas(max_iter=MAX_ITER)

        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"Budget: {COMPARISON_EVALUATION_BUDGET} evaluations")
        print(f"  SA Final Best: {sa_obj:.2f} (in {sa_evals} evals)")
        print(f"  GA Final Best: {ga_obj:.2f} (in {ga_evals} evals)")
        print(f"  MMAS Final Best: {mmas_obj:.2f} (in {mmas_evals} evals)")

        # simple winner print
        best_overall = min((('SA', sa_obj), ('GA', ga_obj), ('MMAS', mmas_obj)), key=lambda x: x[1])
        print(f"\n  Best algorithm: {best_overall[0]} with cost {best_overall[1]:.2f}")
        print("="*70)

        # Plot comparison: SA vs GA (as before)
        try:
            plot_sa_vs_ga_comparison(sa_history, ga_history, sa_evals, ga_evals)
        except Exception as e:
            print(f"Could not plot SA vs GA: {e}")

        # Also show MMAS progression plot if available
        print("Displaying MMAS progression plot...")
        try:
            plot_mmas_performance_dashboard(mmas_history)
        except:
            pass


    # --- Experiment / Analysis Modes (natural runs) ---
    elif OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'MMAS_ANALYSIS'):
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
        mmas_results = []
        sa_evals_list = []
        ga_evals_list = []
        mmas_evals_list = []

        sa_best_obj_overall = math.inf
        sa_best_history_overall = {}
        ga_best_obj_overall = math.inf
        ga_best_history_overall = {}
        mmas_best_obj_overall = math.inf
        mmas_best_history_overall = {}

        all_stats = {}
        experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # SA Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS'):
            print("\nGenerating common initial solution for SA...")
            common_sa_solution = sa_create_initial_solution(geom)
            sa_iter_limit = sa.MAX_TOTAL_ITERATIONS
            print(f"SA using NATURAL STOP (T<T_min), safety cap: {sa_iter_limit} iterations")
            print(f"\nRUNNING SA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
            start_time_sa = time.time()
            for i in range(NUM_EXPERIMENT_RUNS):
                (sa_perm, sa_speeds, sa_obj, sa_history,
                 sa_geom, sa_tau, sa_evals)= run_sa(
                    max_iter=sa_iter_limit,
                    initial_solution=common_sa_solution,
                    verbose=False
                )
                print(f"  SA Run {i+1}/{NUM_EXPERIMENT_RUNS} Best: {sa_obj:.2f} (in {sa_evals} evals)")
                sa_results.append(sa_obj)
                sa_evals_list.append(sa_evals)
                if sa_obj < sa_best_obj_overall:
                    sa_best_obj_overall = sa_obj
                    sa_best_history_overall = sa_history
            end_time_sa = time.time()
            all_stats['SA'] = {
                'mean': np.mean(sa_results),
                'std': np.std(sa_results),
                'time': (end_time_sa - start_time_sa) / NUM_EXPERIMENT_RUNS,
                'best': sa_best_obj_overall,
                'avg_evals': np.mean(sa_evals_list)
            }

        # GA Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'GA_ANALYSIS'):
            if not GA_IMPORTED:
                print("ERROR: GA analysis requested but ga.py unavailable.")
            else:
                print("\nGenerating common initial population for GA...")
                common_ga_population = create_initial_population(POPULATION_SIZE, geom)
                print(f"GA using NATURAL STOP (Num_Gens={NUM_GENERATIONS} or Convergence)")
                print(f"\nRUNNING GA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                start_time_ga = time.time()
                for i in range(NUM_EXPERIMENT_RUNS):
                    _, _, ga_obj, ga_history, _, _, _, ga_evals = run_ga(
                        max_evaluations=None,
                        initial_population=common_ga_population,
                        verbose=False,
                        visualize_realtime=False
                    )
                    print(f"  GA Run {i+1}/{NUM_EXPERIMENT_RUNS} Best: {ga_obj:.2f} (in {ga_evals} evals)")
                    ga_results.append(ga_obj)
                    ga_evals_list.append(ga_evals)
                    if ga_obj < ga_best_obj_overall:
                        ga_best_obj_overall = ga_obj
                        ga_best_history_overall = ga_history
                end_time_ga = time.time()
                all_stats['GA'] = {
                    'mean': np.mean(ga_results),
                    'std': np.std(ga_results),
                    'time': (end_time_ga - start_time_ga) / NUM_EXPERIMENT_RUNS,
                    'best': ga_best_obj_overall,
                    'avg_evals': np.mean(ga_evals_list)
                }

        # MMAS Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'MMAS_ANALYSIS'):
            if not MMAS_IMPORTED:
                print("ERROR: MMAS analysis requested but mmas.py unavailable.")
            else:
                print(f"\nRUNNING MMAS EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                start_time_mmas = time.time()
                for i in range(NUM_EXPERIMENT_RUNS):
                    perm, speeds, obj, hist, _, _, evals = run_mmas(max_iter=MAX_ITER, initial_solution=None, verbose=False)
                    print(f"  MMAS Run {i+1}/{NUM_EXPERIMENT_RUNS} Best: {obj:.2f} (in {evals} evals)")
                    mmas_results.append(obj)
                    mmas_evals_list.append(evals)
                    if obj < mmas_best_obj_overall:
                        mmas_best_obj_overall = obj
                        mmas_best_history_overall = hist
                end_time_mmas = time.time()
                all_stats['MMAS'] = {
                    'mean': np.mean(mmas_results),
                    'std': np.std(mmas_results),
                    'time': (end_time_mmas - start_time_mmas) / NUM_EXPERIMENT_RUNS,
                    'best': mmas_best_obj_overall,
                    'avg_evals': np.mean(mmas_evals_list)
                }

        # Print experiment stats
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
        if 'MMAS' in all_stats:
            stats = all_stats['MMAS']
            print("--- MMAS (Max-Min Ant System) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")

        print("="*70)

        # Save results
        save_experiment_summary(experiment_timestamp, all_stats)
        save_experiment_raw_data(experiment_timestamp,
                                 sa_results=sa_results, ga_results=ga_results, mmas_results=mmas_results,
                                 sa_evals=sa_evals_list, ga_evals=ga_evals_list, mmas_evals=mmas_evals_list)

        # Show plots (if experiment asked)
        if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
            labels = []
            results_to_plot = []
            colors = []
            if 'SA' in all_stats:
                labels.append('SA'); results_to_plot.append(sa_results); colors.append('blue')
            if 'GA' in all_stats:
                labels.append('GA'); results_to_plot.append(ga_results); colors.append('red')
            if 'MMAS' in all_stats:
                labels.append('MMAS'); results_to_plot.append(mmas_results); colors.append('green')
            if results_to_plot:
                plot_experiment_results(results_to_plot, labels)
                plot_experiment_distribution(results_to_plot, labels, colors)

            if 'SA' in all_stats:
                print("  Displaying performance dashboard for the BEST SA run...")
                plot_sa_results(sa_best_history_overall)
            if 'GA' in all_stats:
                print("  Displaying performance dashboard for the BEST GA run...")
                plot_ga_performance_dashboard(ga_best_history_overall)
            if 'MMAS' in all_stats:
                print("  Displaying performance dashboard for the BEST MMAS run...")

                try:
                    plot_mmas_performance_dashboard(mmas_best_history_overall)
                    plot_mmas_results(mmas_best_history_overall)
                except Exception as e:
                    print("MMAS plotting failed:", e)


    else:
        print(f"Error: Unknown OPTIMIZATION_ALGORITHM: '{OPTIMIZATION_ALGORITHM}'")

    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
