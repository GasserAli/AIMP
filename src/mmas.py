# File: mmas.py
"""
Max–Min Ant System (MMAS) implementation for intersection optimization.
Matches SA return format:
    return perm_best, speeds_best, best_f, history, geom, tau_p_dict, evals
"""

import random
import math
import copy
import numpy as np

import config
from geometry import Geometry
from vehicle import Vehicle
from decoder import run_decoder
from sa import evaluate_solution, validate_speeds
import matplotlib.pyplot as plt
import numpy as np


# =============================================================
# MMAS PARAMETERS
# =============================================================
NUM_ANTS = 90
MAX_ITER = 500
PHEROMONE_MAX = 5.0
PHEROMONE_MIN = 0.001
EVAP_RATE = 0.35

ALPHA = 1.0    # pheromone influence
BETA = 4.0     # heuristic influence


# =============================================================
# INITIAL SOLUTION (reuse SA helper)
# =============================================================
def create_initial_solution(geom):
    from sa import create_initial_solution as sa_create
    return sa_create(geom)


# =============================================================
# ANT CONSTRUCTION
# =============================================================
def construct_ant_solution(vehicles, pheromone, geom):
    """
    Builds a solution permutation using pheromone & heuristic
    and assigns random valid speeds.
    """
    unvisited = vehicles.copy()
    permutation = []

    # --- Rank-based heuristic: lower ID = better (arbitrary consistent heuristic)
    heuristic = {v.id: 1.0 / (i + 1) for i, v in enumerate(unvisited)}

    while unvisited:
        weights = []
        for v in unvisited:
            tau = pheromone[v.id]
            eta = heuristic[v.id] ** BETA
            weights.append((tau ** ALPHA) * eta)

        total = sum(weights)
        probs = [w / total for w in weights]

        # Roulette wheel selection
        selected_idx = np.random.choice(len(unvisited), p=probs)
        permutation.append(unvisited[selected_idx])
        del unvisited[selected_idx]

    # --- Construct speeds using same initialize logic as SA
    speeds = []
    v_min, v_max = config.velocity_range
    for v in permutation:
        speeds.append(random.uniform(v_min, v_max))

    # Validate leader–follower ordering constraints
    speeds = validate_speeds(permutation, speeds, geom)
    return permutation, speeds


# =============================================================
# UPDATE PHEROMONES
# =============================================================
def update_pheromones(pheromone, ants_solutions, best_perm, best_cost):
    # Evaporate
    for key in pheromone:
        pheromone[key] *= (1 - EVAP_RATE)

    # Deposit (only best ant: MMAS rule)
    for v in best_perm:
        pheromone[v.id] += PHEROMONE_MAX / (1 + best_cost)

    # Clamp pheromone
    for key in pheromone:
        pheromone[key] = max(PHEROMONE_MIN, min(PHEROMONE_MAX, pheromone[key]))

    return pheromone


# =============================================================
# MAIN MMAS RUN
# =============================================================
def run_mmas(max_iter=MAX_ITER, initial_solution=None, verbose=True):
    """
    Returns SA-style structure:
        perm_best, speeds_best, best_f, history, geom, tau_p_dict, evals
    """
    # -----------------------
    # Problem Setup
    # -----------------------
    geom = Geometry()
    vehicles = config.pi
    geom.create_entry_queue(vehicles)
    for v in vehicles:
        geom.set_trajectory(v)

    all_points = set().union(*(v.path for v in vehicles if v.path))
    tau_p_dict = {p: config.tau for p in all_points}

    # -----------------------
    # PHEROMONE INITIALIZATION
    # -----------------------
    pheromone = {v.id: PHEROMONE_MAX for v in vehicles}

    # -----------------------
    # BEST TRACKING
    # -----------------------
    best_perm = None
    best_speeds = None
    best_cost = math.inf

    history = {
        "best_f": [],
        "avg_f": [],
        "costs": []
    }

    evals = 0

    if verbose:
        print("\n=== Starting MMAS Optimization ===")

    # =========================================================
    # MAIN LOOP
    # =========================================================
    for iteration in range(max_iter):

        ants_solutions = []
        ants_costs = []

        for k in range(NUM_ANTS):
            perm, speeds = construct_ant_solution(vehicles, pheromone, geom)
            result = evaluate_solution(perm, speeds, geom, tau_p_dict)
            cost = result['f']
            evals += 1

            ants_solutions.append((perm, speeds))
            ants_costs.append(cost)

            # Global best
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
                best_speeds = speeds

        # Update pheromone
        pheromone = update_pheromones(pheromone, ants_solutions, best_perm, best_cost)

        # Save history
        history["best_f"].append(best_cost)
        history["avg_f"].append(sum(ants_costs) / len(ants_costs))
        history["costs"].append(best_cost)

        if verbose and iteration % 10 == 0:
            print(f"  Iter {iteration}: Best f = {best_cost:.2f}")

    if verbose:
        print("\n=== MMAS Finished ===")
        print(f"Best Objective f = {best_cost:.2f}")
        print(f"Evaluations used: {evals}")

    return best_perm, best_speeds, best_cost, history, geom, tau_p_dict, evals
# =============================================================
# MMAS PLOTTING FUNCTIONS (MATCH SA + GA STYLE)
# =============================================================



def plot_mmas_results(history):
    """
    Simple SA-style plot (best, avg, convergence curve).
    """
    best_f = history.get("best_f", [])
    avg_f = history.get("avg_f", [])
    costs = history.get("costs", [])

    iters = range(1, len(best_f) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(iters, best_f, label="Best f", linewidth=2)
    plt.plot(iters, avg_f, label="Avg f", linestyle="--")
    plt.title("MMAS Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Objective f")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Convergence only
    plt.figure(figsize=(12, 4))
    plt.plot(iters, best_f, label="Convergence Curve", color="purple")
    plt.title("MMAS Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far f")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_mmas_performance_dashboard(history):
    """
    GA-style 2×2 performance dashboard.
    """
    best_f = history.get("best_f", [])
    avg_f = history.get("avg_f", [])
    costs = history.get("costs", [])

    iterations = np.arange(1, len(best_f) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MMAS Performance Dashboard", fontsize=16)

    # Panel 1 – Best f
    ax = axs[0, 0]
    ax.plot(iterations, best_f, label="Best f", color='blue')
    ax.set_title("Best Objective f over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best f")
    ax.grid(True)

    # Panel 2 – Avg f
    ax = axs[0, 1]
    ax.plot(iterations, avg_f, label="Avg f", color='orange')
    ax.set_title("Average Objective f per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg f")
    ax.grid(True)

    # Panel 3 – Histogram of Best f
    ax = axs[1, 0]
    ax.hist(best_f, bins=20, color='green', alpha=0.7)
    ax.set_title("Distribution of Best f Values")
    ax.set_xlabel("Best f")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    # Panel 4 – Convergence curve
    ax = axs[1, 1]
    ax.plot(iterations, best_f, color='purple')
    ax.set_title("Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far f")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
