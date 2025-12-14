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
import engine.objective as objective
from engine.geometry import Geometry
from engine.decoder import run_decoder
from engine.vehicle import Vehicle
import metahueristics.sa as sa

# --- Import Algorithms ---
from metahueristics.sa import (
    run_sa, 
    plot_results as plot_sa_results, 
    evaluate_solution,
    validate_speeds,
    create_initial_solution as sa_create_initial_solution
)
try:
    from metahueristics.ga import run_ga, plot_ga_performance_dashboard, create_initial_population
    GA_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import ga.py: {e}")
    print("  'GA', 'BOTH', 'GA_ANALYSIS', and 'EXPERIMENT' modes will not be available.")
    GA_IMPORTED = False

try:
    from metahueristics.aco import run_aco, plot_aco_performance_dashboard, create_initial_population as aco_create_initial_population
    ACO_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import aco.py: {e}")
    print("  'ACO', 'ACO_ANALYSIS', and ACO comparison modes will not be available.")
    ACO_IMPORTED = False

try:
    from metahueristics.pso import run_pso, plot_pso_performance_dashboard
    PSO_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import pso.py: {e}")
    PSO_IMPORTED = False

try:
    from metahueristics.sequential_hybrid import run_sequential_hybrid, plot_sequential_hybrid_dashboard
    HYBRID_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import sequential_hybrid.py: {e}")
    print("  'HYBRID', 'HYBRID_ANALYSIS' modes will not be available.")
    HYBRID_IMPORTED = False

try:
    from metahueristics.dragonfly import TwoStageDragonflyOptimizer
    DA_IMPORTED = True
except ImportError as e:
    print(f"WARNING: Could not import dragonfly.py: {e}")
    print("  'DA', 'DA_ANALYSIS' modes will not be available.")
    DA_IMPORTED = False


# =============================================================================
# CONFIGURATION
# =============================================================================
# --- 1. CHOOSE ALGORITHM ---
# 'SA':          Single run of SA (uses sa.MAX_TOTAL_ITERATIONS).
# 'GA':          Single run of GA (uses ga.NUM_GENERATIONS).
# 'ACO':         Single run of ACO (uses aco.NUM_ITERATIONS).
# 'PSO':         Single run of PSO (uses pso.NUM_ITERATIONS).
# 'HYBRID':      Single run of Hybrid ACO+PSO (uses ACO and PSO iterations).
# 'DA':          Single run of Dragonfly Algorithm (two-stage: permutation + speeds).
# 'SA_ANALYSIS': N-run statistical analysis of SA (natural stop).
# 'GA_ANALYSIS': N-run statistical analysis of GA (natural stop).
# 'ACO_ANALYSIS': N-run statistical analysis of ACO (natural stop).
# 'PSO_ANALYSIS': N-run statistical analysis of PSO (natural stop).
# 'DA_ANALYSIS': N-run statistical analysis of DA (natural stop).
# 'BOTH':        Single run SA vs. GA (uses COMPARISON_EVALUATION_BUDGET).
# 'EXPERIMENT':  N-run statistical comparison of SA vs. GA vs. Hybrid (natural stops).
# 'COMPARE_ALL': N-run statistical comparison of GA, SA, HYBRID, and DA (natural stops).
OPTIMIZATION_ALGORITHM = 'HYBRID'  # 'SA', 'GA', 'ACO', 'PSO', 'HYBRID', 'DA', 'SA_ANALYSIS', 'GA_ANALYSIS', 'ACO_ANALYSIS', 'PSO_ANALYSIS', 'DA_ANALYSIS', 'BOTH', 'EXPERIMENT', 'COMPARE_ALL'

# --- 2. CHOOSE VISUALIZATION ---
# 'matplotlib', 'web', 'none'
# (Ignored for 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS' modes)
VISUALIZATION_METHOD = 'matplotlib'  # 'matplotlib', 'web', 'none'

# --- 3. ALGORITHM PARAMETERS ---
# Budget for *direct comparison modes only* ('BOTH')
COMPARISON_EVALUATION_BUDGET = 5000 
# Number of runs for 'SA_ANALYSIS', 'GA_ANALYSIS', 'EXPERIMENT'
NUM_EXPERIMENT_RUNS = 10  # 
RANDOM_SEED = 42 

# SA Parameters (used for 'SA' and 'SA_ANALYSIS')
T_INITIAL = sa.T_INITIAL
T_MIN = sa.T_MIN 
COOLING_RATE = sa.COOLING_RATE
MAX_ITER_PER_TEMP = sa.MAX_ITER_PER_TEMP

# GA Parameters (used for 'GA' and 'GA_ANALYSIS')
if GA_IMPORTED:
    from metahueristics.ga import (
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

# ACO Parameters (used for 'ACO' and 'ACO_ANALYSIS')
if ACO_IMPORTED:
    from metahueristics.aco import (
        NUM_ANTS, NUM_ITERATIONS, ALPHA, RHO, Q, 
        TAU_INITIAL, ELITIST_WEIGHT, CONVERGENCE_PATIENCE as ACO_CONVERGENCE_PATIENCE
    )
else:
    # Default values if ACO not imported, to prevent errors
    NUM_ANTS = 50
    NUM_ITERATIONS = 100
    ALPHA = 1.0
    RHO = 0.3
    Q = 100.0
    TAU_INITIAL = 0.1
    ELITIST_WEIGHT = 2.0
    ACO_CONVERGENCE_PATIENCE = 20

# PSO Parameters (used for 'PSO' and 'PSO_ANALYSIS')
if PSO_IMPORTED:
    from metahueristics.pso import (
        SWARM_SIZE as PSO_SWARM_SIZE, 
        NUM_ITERATIONS as PSO_NUM_ITERATIONS,
        W as PSO_W, 
        C1 as PSO_C1, 
        C2 as PSO_C2,
        CONVERGENCE_PATIENCE as PSO_CONVERGENCE_PATIENCE
    )
else:
    # Default values if PSO not imported, to prevent errors
    PSO_SWARM_SIZE = 30
    PSO_NUM_ITERATIONS = 100
    PSO_W = 0.6
    PSO_C1 = 1.4
    PSO_C2 = 1.4
    PSO_CONVERGENCE_PATIENCE = 20

# DA Parameters (used for 'DA' and 'DA_ANALYSIS')
if DA_IMPORTED:
    from metahueristics.dragonfly import (
        DISCRETE_SWARM_SIZE, DISCRETE_MAX_ITERATIONS,
        CONTINUOUS_SWARM_SIZE, CONTINUOUS_MAX_ITERATIONS,
        MAX_VELOCITY_LENGTH
    )
else:
    # Default values if DA not imported, to prevent errors
    DISCRETE_SWARM_SIZE = 10
    DISCRETE_MAX_ITERATIONS = 150
    CONTINUOUS_SWARM_SIZE = 10
    CONTINUOUS_MAX_ITERATIONS = 150
    MAX_VELOCITY_LENGTH = 100
# =============================================================================


# --- Conditional Imports Based on Visualization Method ---
animation_enabled = False
web_viz_enabled = False
is_analysis_mode = OPTIMIZATION_ALGORITHM in ('BOTH', 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'ACO_ANALYSIS', 'PSO_ANALYSIS', 'HYBRID_ANALYSIS', 'DA_ANALYSIS', 'COMPARE_ALL')

if not is_analysis_mode and VISUALIZATION_METHOD == 'matplotlib':
    try:
        from visualization.visualization import IntersectionVisualization
        animation_enabled = True
        print("Matplotlib visualization enabled (SMOOTH animation)")
    except ImportError as e:
        print(f"Warning: Could not import visualization.py: {e}")

elif not is_analysis_mode and VISUALIZATION_METHOD == 'web':
    try:
        from visualization.visualization_utils import IntersectionVisualizer
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

def plot_compare(history_sa, history_ga, labels=('SA', 'GA'), sa_evals=None, ga_evals=None, history_hybrid=None, hybrid_evals=None, history_da=None, da_evals=None):
    """Overlay SA, GA, and optionally HYBRID and DA histories on a single 2x2 comparison grid.

    Accepts history dicts produced by `run_sa`, `run_ga`, `run_sequential_hybrid`, and DA optimizer (formats used in this project).
    The function will look for common keys and plot matching series; missing series are skipped.
    Note: DA history has a two-stage structure with 'stage1' and 'stage2' keys.
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

    # HYBRID data extraction
    costs_hybrid = []
    avg_delays_hybrid = []
    total_delays_hybrid = []
    emergency_delays_hybrid = []
    if history_hybrid:
        costs_hybrid = first(history_hybrid, ['best_f', 'costs', 'best'])
        avg_delays_hybrid = first(history_hybrid, ['best_avg_delay', 'avg_delays'])
        total_delays_hybrid = first(history_hybrid, ['best_fall', 'total_delays'])
        emergency_delays_hybrid = first(history_hybrid, ['best_fem', 'emergency_delays'])
    
    # DA data extraction (two-stage structure)
    costs_da = []
    if history_da:
        # DA has stage1 and stage2 histories, we'll combine them for visualization
        # Note: DA doesn't track avg_delays, total_delays, emergency_delays in the same way
        # We'll only plot the cost/fitness data
        stage1_history = history_da.get('stage1', {})
        stage2_history = history_da.get('stage2', {})
        
        # Extract best fitness from both stages if available
        stage1_best = first(stage1_history, ['best_fitness', 'best_f', 'costs'])
        stage2_best = first(stage2_history, ['best_fitness', 'best_f', 'costs'])
        
        # Combine both stages
        if stage1_best and stage2_best:
            costs_da = list(stage1_best) + list(stage2_best)
        elif stage1_best:
            costs_da = list(stage1_best)
        elif stage2_best:
            costs_da = list(stage2_best)

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
    
    # HYBRID x arrays and clipped series (HYBRID uses combined ACO+PSO evaluation)
    x_costs_hybrid = np.array([])
    x_avg_delays_hybrid = np.array([])
    x_total_delays_hybrid = np.array([])
    x_emergency_delays_hybrid = np.array([])
    if history_hybrid and costs_hybrid:
        # HYBRID: combined history from ACO and PSO phases
        hybrid_len = len(costs_hybrid)
        # Simply map iterations to evaluation indices
        hybrid_eval_points = list(range(1, hybrid_len + 1))
        
        # Clip to budget if provided
        if hybrid_evals is not None and hybrid_eval_points:
            clip_idx = min(hybrid_evals, len(hybrid_eval_points))
            hybrid_eval_points = hybrid_eval_points[:clip_idx]
            costs_hybrid = costs_hybrid[:clip_idx]
            avg_delays_hybrid = avg_delays_hybrid[:clip_idx] if avg_delays_hybrid else []
            total_delays_hybrid = total_delays_hybrid[:clip_idx] if total_delays_hybrid else []
            emergency_delays_hybrid = emergency_delays_hybrid[:clip_idx] if emergency_delays_hybrid else []
        
        x_costs_hybrid = np.array(hybrid_eval_points)
        x_avg_delays_hybrid = np.array(hybrid_eval_points[:len(avg_delays_hybrid)])
        x_total_delays_hybrid = np.array(hybrid_eval_points[:len(total_delays_hybrid)])
        x_emergency_delays_hybrid = np.array(hybrid_eval_points[:len(emergency_delays_hybrid)])
    
    # DA x arrays and clipped series
    x_costs_da = np.array([])
    if history_da and costs_da:
        # DA: simple iteration-based mapping
        da_len = len(costs_da)
        da_eval_points = list(range(1, da_len + 1))
        
        # Clip to budget if provided
        if da_evals is not None and da_eval_points:
            clip_idx = min(da_evals, len(da_eval_points))
            da_eval_points = da_eval_points[:clip_idx]
            costs_da = costs_da[:clip_idx]
        
        x_costs_da = np.array(da_eval_points)
    
    # Determine title based on what algorithms are present
    title_parts = []
    if history_sa: title_parts.append(labels[0] if len(labels) > 0 else 'SA')
    if history_ga: title_parts.append(labels[1] if len(labels) > 1 else 'GA')
    if history_hybrid: title_parts.append(labels[2] if len(labels) > 2 else 'HYBRID')
    if history_da: title_parts.append(labels[3] if len(labels) > 3 else 'DA')
    title = f"Overlay: {' vs '.join(title_parts)}"
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title)

    # Top-left: Cost (mapped to fitness evaluation counts)
    ax = axs[0, 0]
    if x_costs_sa.size:
        ax.plot(x_costs_sa, costs_sa, label=f"{labels[0]} best", color='tab:blue')
    if x_costs_ga.size:
        ax.plot(x_costs_ga, costs_ga, label=f"{labels[1]} best", color='tab:orange')
    if x_costs_hybrid.size:
        hybrid_label = labels[2] if len(labels) > 2 else 'HYBRID'
        ax.plot(x_costs_hybrid, costs_hybrid, label=f"{hybrid_label} best", color='tab:green')
    if x_costs_da.size:
        da_label = labels[3] if len(labels) > 3 else 'DA'
        ax.plot(x_costs_da, costs_da, label=f"{da_label} best", color='tab:purple')
    if x_avg_sa.size:
        ax.plot(x_avg_sa, avg_sa, label=f"{labels[0]} avg", color='tab:cyan', linestyle=':')
    if x_avg_ga.size:
        ax.plot(x_avg_ga, avg_ga, label=f"{labels[1]} avg", color='tab:gray', linestyle='-.')
    ax.set_title('Objective Cost vs Iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (f)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Top-right: Average Delay
    ax = axs[0, 1]
    if x_avg_delays_sa.size:
        ax.plot(x_avg_delays_sa, avg_delays_sa, label=f"{labels[0]}", color='tab:green')
    if x_avg_delays_ga.size:
        ax.plot(x_avg_delays_ga, avg_delays_ga, label=f"{labels[1]}", color='tab:red')
    if x_avg_delays_hybrid.size and len(avg_delays_hybrid) > 0:
        hybrid_label = labels[2] if len(labels) > 2 else 'HYBRID'
        ax.plot(x_avg_delays_hybrid, avg_delays_hybrid, label=f"{hybrid_label}", color='tab:purple')
    ax.set_title('Average Delay per Vehicle')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Avg Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Bottom-left: Total Delay
    ax = axs[1, 0]
    if x_total_delays_sa.size:
        ax.plot(x_total_delays_sa, total_delays_sa, label=f"{labels[0]}", color='tab:cyan')
    if x_total_delays_ga.size:
        ax.plot(x_total_delays_ga, total_delays_ga, label=f"{labels[1]}", color='tab:olive')
    if x_total_delays_hybrid.size and len(total_delays_hybrid) > 0:
        hybrid_label = labels[2] if len(labels) > 2 else 'HYBRID'
        ax.plot(x_total_delays_hybrid, total_delays_hybrid, label=f"{hybrid_label}", color='tab:brown')
    ax.set_title('Total Delay (f_all)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Bottom-right: Emergency Delay
    ax = axs[1, 1]
    if x_emergency_delays_sa.size:
        ax.plot(x_emergency_delays_sa, emergency_delays_sa, label=f"{labels[0]}", color='tab:blue')
    if x_emergency_delays_ga.size:
        ax.plot(x_emergency_delays_ga, emergency_delays_ga, label=f"{labels[1]}", color='tab:orange')
    if x_emergency_delays_hybrid.size and len(emergency_delays_hybrid) > 0:
        hybrid_label = labels[2] if len(labels) > 2 else 'HYBRID'
        ax.plot(x_emergency_delays_hybrid, emergency_delays_hybrid, label=f"{hybrid_label}", color='tab:pink')
    ax.set_title('Emergency Delay (f_em)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Emergency Delay (s)')
    ax.grid(True)
    ax.legend(loc='upper left')

    # Set common x-limits based on provided evaluation budgets or global comparison budget
    x_max = None
    eval_budgets = [e for e in [sa_evals, ga_evals, hybrid_evals, da_evals] if e is not None]
    if eval_budgets:
        x_max = max(eval_budgets)  # Show the longest run
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
    ax.boxplot(data, tick_labels=labels, patch_artist=True,
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
                    'GA_Pop_Size', 'GA_Generations', 'GA_Elitism', 'GA_Tourn_Size', 'GA_Mut_Perm', 'GA_Mut_Speed',
                    'ACO_Num_Ants', 'ACO_Num_Iter', 'ACO_Alpha', 'ACO_Rho', 'ACO_Q', 'ACO_Tau_Init', 'ACO_Elitist_Weight',
                    'PSO_Num_Iter', 'PSO_Swarm_Size', 'PSO_W', 'PSO_C1', 'PSO_C2', 'PSO_Convergence_Patience'
                ])
            
            if 'SA' in all_stats:
                stats = all_stats['SA']
                writer.writerow([
                    timestamp, 'SA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    T_INITIAL, T_MIN, COOLING_RATE, MAX_ITER_PER_TEMP,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A' ,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])
            if 'GA' in all_stats:
                stats = all_stats['GA']
                writer.writerow([
                    timestamp, 'GA', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    'N/A', 'N/A', 'N/A', 'N/A',
                    POPULATION_SIZE, NUM_GENERATIONS, ELITISM_RATE, TOURNAMENT_SIZE, MUTATION_RATE_PERM, MUTATION_RATE_SPEED,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])
            if 'ACO' in all_stats:
                stats = all_stats['ACO']
                writer.writerow([
                    timestamp, 'ACO', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    NUM_ANTS, NUM_ITERATIONS, ALPHA, RHO, Q, TAU_INITIAL, ELITIST_WEIGHT,
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
                ])
            if 'PSO' in all_stats:
                stats = all_stats['PSO']
                writer.writerow([
                    timestamp, 'PSO', stats['mean'], stats['std'], stats['time'], stats['best'],
                    NUM_EXPERIMENT_RUNS, stats['avg_evals'], # <-- MODIFIED
                    'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    PSO_NUM_ITERATIONS, PSO_SWARM_SIZE, PSO_W, PSO_C1, PSO_C2, PSO_CONVERGENCE_PATIENCE
                ])
        print(f"  Successfully appended summary to: {SUMMARY_FILENAME}")
    except IOError as e:
        print(f"  ERROR: Could not save summary CSV. {e}")


def save_experiment_raw_data(timestamp, sa_results=None, ga_results=None, sa_evals=None, ga_evals=None, aco_results=None, aco_evals=None, pso_results=None, pso_evals=None, hybrid_results=None, hybrid_evals=None, da_results=None, da_evals=None):
    """
    Saves the raw final cost AND evals from every single run to a SINGLE master CSV file.
    Appends the new runs; creates header if file doesn't exist.
    """
    file_exists = os.path.isfile(RAW_FILENAME)
    try:
        with open(RAW_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # --- MODIFIED: Added ACO, PSO, HYBRID, and DA columns ---
                writer.writerow(['Experiment_Timestamp', 'Run', 'SA_Final_Cost', 'SA_Evals', 'GA_Final_Cost', 'GA_Evals', 'ACO_Final_Cost', 'ACO_Evals', 'PSO_Final_Cost', 'PSO_Evals', 'HYBRID_Final_Cost', 'HYBRID_Evals', 'DA_Final_Cost', 'DA_Evals'])
            
            # Determine max number of runs
            num_runs = 0
            if sa_results: num_runs = len(sa_results)
            if ga_results: num_runs = max(num_runs, len(ga_results))
            if aco_results: num_runs = max(num_runs, len(aco_results))
            if pso_results: num_runs = max(num_runs, len(pso_results))
            if hybrid_results: num_runs = max(num_runs, len(hybrid_results))
            if da_results: num_runs = max(num_runs, len(da_results))
            
            for i in range(num_runs):
                sa_val = sa_results[i] if sa_results and i < len(sa_results) else 'N/A'
                sa_ev = sa_evals[i] if sa_evals and i < len(sa_evals) else 'N/A'
                ga_val = ga_results[i] if ga_results and i < len(ga_results) else 'N/A'
                ga_ev = ga_evals[i] if ga_evals and i < len(ga_evals) else 'N/A'
                aco_val = aco_results[i] if aco_results and i < len(aco_results) else 'N/A'
                aco_ev = aco_evals[i] if aco_evals and i < len(aco_evals) else 'N/A'
                pso_val = pso_results[i] if pso_results and i < len(pso_results) else 'N/A'
                pso_ev = pso_evals[i] if pso_evals and i < len(pso_evals) else 'N/A'
                hybrid_val = hybrid_results[i] if hybrid_results and i < len(hybrid_results) else 'N/A'
                hybrid_ev = hybrid_evals[i] if hybrid_evals and i < len(hybrid_evals) else 'N/A'
                da_val = da_results[i] if da_results and i < len(da_results) else 'N/A'
                da_ev = da_evals[i] if da_evals and i < len(da_evals) else 'N/A'
                # --- MODIFIED: Write all data points including ACO, PSO, HYBRID, and DA ---
                writer.writerow([timestamp, i+1, sa_val, sa_ev, ga_val, ga_ev, aco_val, aco_ev, pso_val, pso_ev, hybrid_val, hybrid_ev, da_val, da_ev])
                    
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
    if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'ACO_ANALYSIS', 'PSO_ANALYSIS'):
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

    # --- Run ACO (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'ACO':
        if not ACO_IMPORTED:
            print("ERROR: ACO algorithm selected but aco.py could not be imported.")
            return
            
        (perm_best, speeds_best, obj_best, aco_history, geom, 
         tau_p_dict, best_aco_obj_dict, evals) = run_aco(
            max_iterations=None, # <-- This tells ACO to use NUM_ITERATIONS or converge
            visualize_realtime=True,
            verbose=True
        )
        
        print("\nDisplaying ACO performance plots...")
        plot_aco_performance_dashboard(aco_history)
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run PSO (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'PSO':
        if not PSO_IMPORTED:
            print("ERROR: PSO algorithm selected but pso.py could not be imported.")
            return
            
        (perm_best, speeds_best, obj_best, pso_history, geom, 
         tau_p_dict, best_pso_obj_dict, evals) = run_pso(
            max_iterations=100,
            visualize_realtime=True,
            verbose=True
        )
        
        print("\nDisplaying PSO performance plots...")
        plot_pso_performance_dashboard(pso_history)
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run Hybrid ACO+PSO (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'HYBRID':
        if not HYBRID_IMPORTED:
            print("ERROR: HYBRID mode requires sequential_hybrid module.")
            return
            
        (perm_best, speeds_best, obj_best, hybrid_history, geom, 
         tau_p_dict, best_hybrid_obj_dict, evals, aco_iters, pso_iters) = run_sequential_hybrid(
            aco_iterations=100,
            pso_iterations=50,
            verbose=True
        )
        
        print("\nDisplaying Sequential Hybrid performance plots...")
        plot_sequential_hybrid_dashboard(hybrid_history, aco_iters, pso_iters)
        
        print("  (Close all plot windows to continue to animation)")
        if animation_enabled:
            visualize_matplotlib(perm_best, speeds_best, geom, tau_p_dict)
        elif web_viz_enabled:
            visualize_web(perm_best, speeds_best)

    # --- Run Dragonfly Algorithm (Single Run) ---
    elif OPTIMIZATION_ALGORITHM == 'DA':
        if not DA_IMPORTED:
            print("ERROR: DA algorithm selected but dragonfly.py could not be imported.")
            return
            
        print("\n" + "="*70)
        print("RUNNING DRAGONFLY ALGORITHM (Two-Stage)")
        print("="*70)
        
        # Initialize geometry
        geom = Geometry()
        all_vehicles = config.pi
        geom.create_entry_queue(all_vehicles)
        for v in all_vehicles:
            geom.set_trajectory(v)
        all_points = set().union(*(v.path for v in all_vehicles if v.path))
        tau_p_dict = {p: config.tau for p in all_points}
        
        # Run DA optimizer
        optimizer = TwoStageDragonflyOptimizer(verbose=True, visualize=False, log_to_csv=True, csv_prefix="da_run")
        perm_best, speeds_best, obj_best, results = optimizer.optimize()
        
        print("\n" + "="*70)
        print("DRAGONFLY ALGORITHM RESULTS")
        print("="*70)
        print(f"  Best Objective:     {obj_best:.2f}")
        print(f"  Best Permutation:   {[v.id for v in perm_best]}")
        print(f"  Best Speeds:        {[f'{s:.2f}' for s in speeds_best]}")
        print("="*70)
        
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
    elif OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS', 'ACO_ANALYSIS', 'PSO_ANALYSIS', 'HYBRID_ANALYSIS', 'DA_ANALYSIS', 'COMPARE_ALL'):
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
        aco_results = []
        pso_results = []
        hybrid_results = []
        sa_evals_list = []
        ga_evals_list = []
        aco_evals_list = []
        pso_evals_list = []
        hybrid_evals_list = []
        
        sa_best_obj_overall = math.inf
        sa_best_history_overall = {}
        ga_best_obj_overall = math.inf
        ga_best_history_overall = {}
        aco_best_obj_overall = math.inf
        aco_best_history_overall = {}
        pso_best_obj_overall = math.inf
        pso_best_history_overall = {}
        hybrid_best_obj_overall = math.inf
        hybrid_best_history_overall = {}
        hybrid_best_aco_iters = 100
        hybrid_best_pso_iters = 50
        all_stats = {}
        
        # --- FIX: Defined experiment_timestamp here ---
        experiment_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. Run SA Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'SA_ANALYSIS', 'COMPARE_ALL'):
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
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'GA_ANALYSIS', 'COMPARE_ALL'):
            if not GA_IMPORTED:
                print("WARNING: GA selected for experiment but ga.py could not be imported. Skipping GA.")
            else:
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

        # 5. Run ACO Experiment (DISABLED FOR EXPERIMENT MODE)
        if OPTIMIZATION_ALGORITHM == 'ACO_ANALYSIS':
            if not ACO_IMPORTED:
                print("WARNING: ACO selected for experiment but aco.py could not be imported. Skipping ACO.")
            else:
                print("\nGenerating common initial ant colony for ACO...")
                common_aco_population = aco_create_initial_population(NUM_ANTS, geom)
                
                aco_iter_limit = None
                print(f"ACO using NATURAL STOP (Num_Iters={NUM_ITERATIONS} or Convergence)")

                print("\n" + "="*70)
                print(f"RUNNING ACO EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                print("="*70)
                start_time_aco = time.time()
                for i in range(NUM_EXPERIMENT_RUNS):
                    print(f"  ACO Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                    (aco_perm, aco_speeds, aco_obj, aco_history, 
                     aco_geom, aco_tau, aco_obj_dict, aco_evals) = run_aco(
                        max_iterations=aco_iter_limit,
                        visualize_realtime=False,
                        verbose=False
                    )
                    print(f"    ...Run {i+1} Best: {aco_obj:.2f} (in {aco_evals} evals)")
                    aco_results.append(aco_obj)
                    aco_evals_list.append(aco_evals)
                    if aco_obj < aco_best_obj_overall:
                        aco_best_obj_overall = aco_obj
                        aco_best_history_overall = aco_history
                end_time_aco = time.time()
                
                all_stats['ACO'] = {
                    'mean': np.mean(aco_results),
                    'std': np.std(aco_results),
                    'time': (end_time_aco - start_time_aco) / NUM_EXPERIMENT_RUNS,
                    'best': aco_best_obj_overall,
                    'avg_evals': np.mean(aco_evals_list)
                }
        
        # 6. Run PSO Experiment (DISABLED FOR EXPERIMENT MODE)
        if OPTIMIZATION_ALGORITHM == 'PSO_ANALYSIS':
            if not PSO_IMPORTED:
                print("WARNING: PSO selected but pso.py not imported. Skipping PSO.")
            else:
                print("\n" + "="*70)
                print(f"RUNNING PSO EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                print("="*70)
                start_time_pso = time.time()
                for i in range(NUM_EXPERIMENT_RUNS):
                    print(f"  PSO Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                    (pso_perm, pso_speeds, pso_obj, pso_history, 
                     pso_geom, pso_tau, pso_obj_dict, pso_evals) = run_pso(
                        max_iterations=100,
                        visualize_realtime=False,
                        verbose=False
                    )
                    print(f"    ...Run {i+1} Best: {pso_obj:.2f} (in {pso_evals} evals)")
                    pso_results.append(pso_obj)
                    pso_evals_list.append(pso_evals)
                    if pso_obj < pso_best_obj_overall:
                        pso_best_obj_overall = pso_obj
                        pso_best_history_overall = pso_history
                end_time_pso = time.time()
                
                all_stats['PSO'] = {
                    'mean': np.mean(pso_results),
                    'std': np.std(pso_results),
                    'time': (end_time_pso - start_time_pso) / NUM_EXPERIMENT_RUNS,
                    'best': pso_best_obj_overall,
                    'avg_evals': np.mean(pso_evals_list)
                }
        
        # 7. Run HYBRID Experiment
        if OPTIMIZATION_ALGORITHM in ('EXPERIMENT', 'HYBRID_ANALYSIS', 'COMPARE_ALL'):
            if not HYBRID_IMPORTED:
                print("WARNING: HYBRID requires sequential_hybrid module. Skipping HYBRID.")
            else:
                print("\n" + "="*70)
                print(f"RUNNING SEQUENTIAL HYBRID EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                print("="*70)
                start_time_hybrid = time.time()
                for i in range(NUM_EXPERIMENT_RUNS):
                    print(f"  HYBRID Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                    (hybrid_perm, hybrid_speeds, hybrid_obj, hybrid_history,
                     hybrid_geom, hybrid_tau, hybrid_obj_dict, hybrid_evals, aco_iters, pso_iters) = run_sequential_hybrid(
                        aco_iterations=100,
                        pso_iterations=50,
                        verbose=False
                    )
                    print(f"    ...Run {i+1} Best: {hybrid_obj:.2f} (in {hybrid_evals} evals)")
                    hybrid_results.append(hybrid_obj)
                    hybrid_evals_list.append(hybrid_evals)
                    if hybrid_obj < hybrid_best_obj_overall:
                        hybrid_best_obj_overall = hybrid_obj
                        hybrid_best_history_overall = hybrid_history
                        hybrid_best_aco_iters = aco_iters
                        hybrid_best_pso_iters = pso_iters
                end_time_hybrid = time.time()
                
                all_stats['HYBRID'] = {
                    'mean': np.mean(hybrid_results),
                    'std': np.std(hybrid_results),
                    'time': (end_time_hybrid - start_time_hybrid) / NUM_EXPERIMENT_RUNS,
                    'best': hybrid_best_obj_overall,
                    'avg_evals': np.mean(hybrid_evals_list)
                }
        if OPTIMIZATION_ALGORITHM in ('DA_ANALYSIS', 'COMPARE_ALL'):
            if not DA_IMPORTED:
                print("WARNING: DA selected but da.py not imported. Skipping DA.")
            else:
                # Initialize DA result tracking variables
                da_results = []
                da_evals_list = []
                da_best_obj_overall = math.inf
                da_best_history_overall = {}
                
                print("\n" + "="*70)
                print(f"RUNNING DA EXPERIMENT ({NUM_EXPERIMENT_RUNS} runs)...")
                print("="*70)
                start_time_da = time.time()
                
                for i in range(NUM_EXPERIMENT_RUNS):
                    print(f"  DA Run {i+1}/{NUM_EXPERIMENT_RUNS}...")
                    
                    # Create a new optimizer instance for each run
                    optimizer = TwoStageDragonflyOptimizer(verbose=False, visualize=False, log_to_csv=False)
                    
                    # optimize() returns: (best_permutation, best_speeds, best_fitness, results_dict)
                    da_perm, da_speeds, da_obj, da_results_dict = optimizer.optimize()
                    
                    # Extract total evaluations from results_dict or calculate from parameters
                    da_evals = (DISCRETE_SWARM_SIZE * DISCRETE_MAX_ITERATIONS + 
                               CONTINUOUS_SWARM_SIZE * CONTINUOUS_MAX_ITERATIONS)
                    
                    print(f"    ...Run {i+1} Best: {da_obj:.2f} (in {da_evals} evals)")
                    da_results.append(da_obj)
                    da_evals_list.append(da_evals)
                    
                    if da_obj < da_best_obj_overall:
                        da_best_obj_overall = da_obj
                        # Store both stage histories from results_dict
                        da_best_history_overall = {
                            'stage1': da_results_dict.get('stage1_history', {}),
                            'stage2': da_results_dict.get('stage2_history', {})
                        }
                
                end_time_da = time.time()
                
                all_stats['DA'] = {
                    'mean': np.mean(da_results),
                    'std': np.std(da_results),
                    'time': (end_time_da - start_time_da) / NUM_EXPERIMENT_RUNS,
                    'best': da_best_obj_overall,
                    'avg_evals': np.mean(da_evals_list)
                }
        
        # 8. Calculate and Print Statistics
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
        
        if 'ACO' in all_stats:
            stats = all_stats['ACO']
            print("--- Ant Colony Optimization (ACO) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")
        
        if 'PSO' in all_stats:
            stats = all_stats['PSO']
            print("--- Particle Swarm Optimization (PSO) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")
        
        if 'HYBRID' in all_stats:
            stats = all_stats['HYBRID']
            print("--- Hybrid ACO+PSO ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")
        
        if 'DA' in all_stats:
            stats = all_stats['DA']
            print("--- Dragonfly Algorithm (DA) ---")
            print(f"  Avg. Evals Used:     {stats['avg_evals']:.1f}")
            print(f"  Mean (Avg. Best Cost): {stats['mean']:.2f}")
            print(f"  Std. Deviation (Cost): {stats['std']:.2f}")
            print(f"  Avg. Run Time (sec):   {stats['time']:.3f}")
            print(f"  Best Cost (Overall):   {stats['best']:.2f}")
        
        print("="*70)
        
        # Determine winner based on mean performance
        if len(all_stats) >= 2:
            best_algo = min(all_stats.items(), key=lambda x: x[1]['mean'])[0]
            print(f"\n  {best_algo} achieved the best (lowest) average final cost.")
            
            most_consistent = min(all_stats.items(), key=lambda x: x[1]['std'])[0]
            print(f"  {most_consistent} was the most consistent (lowest standard deviation).")
        
        # 7. Save results to CSVs
        print("\n" + "="*70)
        print("SAVING EXPERIMENT RESULTS...")
        
        # --- FIX: Use correct variable name ---
        save_experiment_summary(experiment_timestamp, all_stats)
        
        if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
            save_experiment_raw_data(experiment_timestamp, sa_results, ga_results, sa_evals_list, ga_evals_list, aco_results, aco_evals_list, pso_results, pso_evals_list, hybrid_results, hybrid_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'COMPARE_ALL':
            save_experiment_raw_data(experiment_timestamp, sa_results, ga_results, sa_evals_list, ga_evals_list, 
                                   aco_results=None, aco_evals=None, pso_results=None, pso_evals=None, 
                                   hybrid_results=hybrid_results, hybrid_evals=hybrid_evals_list, 
                                   da_results=da_results, da_evals=da_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'SA_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, sa_results=sa_results, sa_evals=sa_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'GA_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, ga_results=ga_results, ga_evals=ga_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'ACO_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, aco_results=aco_results, aco_evals=aco_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'PSO_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, pso_results=pso_results, pso_evals=pso_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'HYBRID_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, hybrid_results=hybrid_results, hybrid_evals=hybrid_evals_list)
        elif OPTIMIZATION_ALGORITHM == 'DA_ANALYSIS':
            save_experiment_raw_data(experiment_timestamp, da_results=da_results, da_evals=da_evals_list)
        # --- END FIX ---
        
        # 8. Show All Plots
        print("\n" + "="*70)
        print("GENERATING PLOTS (Close each plot to see the next)...")
        
        if OPTIMIZATION_ALGORITHM == 'EXPERIMENT':
            # Determine which algorithms were run
            results_list = []
            labels_list = []
            colors_list = []
            
            if sa_results:
                results_list.append(sa_results)
                labels_list.append('SA')
                colors_list.append('blue')
            if ga_results:
                results_list.append(ga_results)
                labels_list.append('GA')
                colors_list.append('red')
            if hybrid_results:
                results_list.append(hybrid_results)
                labels_list.append('HYBRID')
                colors_list.append('orange')
            
            if results_list:
                plot_experiment_results(results_list, labels_list)
                plot_experiment_distribution(results_list, labels_list, colors_list)
            
            if sa_results and sa_best_history_overall:
                print("  Displaying performance dashboard for the BEST SA run...")
                plot_sa_results(sa_best_history_overall)
            
            if ga_results and ga_best_history_overall:
                print("  Displaying performance dashboard for the BEST GA run...")
                plot_ga_performance_dashboard(ga_best_history_overall)
            
            if hybrid_results and hybrid_best_history_overall:
                print("  Displaying performance dashboard for the BEST HYBRID run...")
                plot_sequential_hybrid_dashboard(hybrid_best_history_overall, hybrid_best_aco_iters, hybrid_best_pso_iters)
            
            # Plotting overlay comparison for best runs
            try:
                if sa_results and ga_results and hybrid_results:
                    print("Generating overlay comparison (SA vs GA vs HYBRID - Best Runs)...")
                    best_sa_evals = sa_evals_list[np.argmin(sa_results)]
                    best_ga_evals = ga_evals_list[np.argmin(ga_results)]
                    best_hybrid_evals = hybrid_evals_list[np.argmin(hybrid_results)]
                    plot_compare(sa_best_history_overall, ga_best_history_overall, 
                                labels=('SA (Best)', 'GA (Best)', 'HYBRID (Best)'), 
                                sa_evals=best_sa_evals, ga_evals=best_ga_evals,
                                history_hybrid=hybrid_best_history_overall, hybrid_evals=best_hybrid_evals)
                elif sa_results and ga_results:
                    print("Generating overlay comparison (SA vs GA - Best Runs)...")
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
        
        elif OPTIMIZATION_ALGORITHM == 'ACO_ANALYSIS':
            plot_experiment_results([aco_results], ['ACO'])
            plot_experiment_distribution([aco_results], ['ACO'], ['green'])
            print("  Displaying performance dashboard for the BEST ACO run...")
            plot_aco_performance_dashboard(aco_best_history_overall)
        
        elif OPTIMIZATION_ALGORITHM == 'PSO_ANALYSIS':
            plot_experiment_results([pso_results], ['PSO'])
            plot_experiment_distribution([pso_results], ['PSO'], ['purple'])
            print("  Displaying performance dashboard for the BEST PSO run...")
            plot_pso_performance_dashboard(pso_best_history_overall)
        
        elif OPTIMIZATION_ALGORITHM == 'DA_ANALYSIS':
            plot_experiment_results([da_results], ['DA'])
            plot_experiment_distribution([da_results], ['DA'], ['orange'])
            print("  Note: DA uses a two-stage approach. Detailed stage-by-stage plots are available in individual runs.")
        
        elif OPTIMIZATION_ALGORITHM == 'COMPARE_ALL':
            # Comprehensive comparison of GA, SA, HYBRID, and DA
            results_list = []
            labels_list = []
            colors_list = []
            
            if sa_results:
                results_list.append(sa_results)
                labels_list.append('SA')
                colors_list.append('blue')
            if ga_results:
                results_list.append(ga_results)
                labels_list.append('GA')
                colors_list.append('red')
            if hybrid_results:
                results_list.append(hybrid_results)
                labels_list.append('HYBRID')
                colors_list.append('green')
            if da_results:
                results_list.append(da_results)
                labels_list.append('DA')
                colors_list.append('orange')
            
            if results_list:
                plot_experiment_results(results_list, labels_list)
                plot_experiment_distribution(results_list, labels_list, colors_list)
            
            # Display individual performance dashboards for best runs
            if sa_results and sa_best_history_overall:
                print("  Displaying performance dashboard for the BEST SA run...")
                plot_sa_results(sa_best_history_overall)
            
            if ga_results and ga_best_history_overall:
                print("  Displaying performance dashboard for the BEST GA run...")
                plot_ga_performance_dashboard(ga_best_history_overall)
            
            if hybrid_results and hybrid_best_history_overall:
                print("  Displaying performance dashboard for the BEST HYBRID run...")
                plot_sequential_hybrid_dashboard(hybrid_best_history_overall, hybrid_best_aco_iters, hybrid_best_pso_iters)
            
            if da_results:
                print("  Note: DA uses a two-stage approach. Detailed stage-by-stage plots are available in individual runs.")
            
            # Plotting overlay comparison for best runs (SA vs GA vs HYBRID vs DA)
            try:
                if sa_results and ga_results and hybrid_results and da_results:
                    print("Generating overlay comparison (SA vs GA vs HYBRID vs DA - Best Runs)...")
                    best_sa_evals = sa_evals_list[np.argmin(sa_results)]
                    best_ga_evals = ga_evals_list[np.argmin(ga_results)]
                    best_hybrid_evals = hybrid_evals_list[np.argmin(hybrid_results)]
                    best_da_evals = da_evals_list[np.argmin(da_results)]
                    plot_compare(sa_best_history_overall, ga_best_history_overall, 
                                labels=('SA (Best)', 'GA (Best)', 'HYBRID (Best)', 'DA (Best)'), 
                                sa_evals=best_sa_evals, ga_evals=best_ga_evals,
                                history_hybrid=hybrid_best_history_overall, hybrid_evals=best_hybrid_evals,
                                history_da=da_best_history_overall, da_evals=best_da_evals)
                elif sa_results and ga_results and hybrid_results:
                    print("Generating overlay comparison (SA vs GA vs HYBRID - Best Runs)...")
                    best_sa_evals = sa_evals_list[np.argmin(sa_results)]
                    best_ga_evals = ga_evals_list[np.argmin(ga_results)]
                    best_hybrid_evals = hybrid_evals_list[np.argmin(hybrid_results)]
                    plot_compare(sa_best_history_overall, ga_best_history_overall, 
                                labels=('SA (Best)', 'GA (Best)', 'HYBRID (Best)'), 
                                sa_evals=best_sa_evals, ga_evals=best_ga_evals,
                                history_hybrid=hybrid_best_history_overall, hybrid_evals=best_hybrid_evals)
                elif sa_results and ga_results:
                    print("Generating overlay comparison (SA vs GA - Best Runs)...")
                    best_sa_evals = sa_evals_list[np.argmin(sa_results)]
                    best_ga_evals = ga_evals_list[np.argmin(ga_results)]
                    plot_compare(sa_best_history_overall, ga_best_history_overall, labels=('SA (Best)', 'GA (Best)'), 
                                 sa_evals=best_sa_evals, ga_evals=best_ga_evals)
            except Exception as e:
                print(f"Could not generate overlay comparison plot: {e}")
                traceback.print_exc()
        
        print("  (All plots displayed)")

    else:
        print(f"Error: Unknown OPTIMIZATION_ALGORITHM: '{OPTIMIZATION_ALGORITHM}'")

    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()