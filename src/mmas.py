# File: mmas.py
"""
Max–Min Ant System (MMAS) implementation for intersection optimization.
Heuristic: PRIORITY-BASED (emergency vehicles favored early positions)
Interface/return signature matches SA & ACO modules:
    return perm_best, speeds_best, best_f, history, geom, tau_p_dict, evals
"""

import math
import random
import copy
from datetime import datetime
from typing import List, Dict, Tuple
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

# Project imports (assumes these modules exist in your project)
import config
from geometry import Geometry
from vehicle import Vehicle
from sa import evaluate_solution, validate_speeds

# =============================================================
# MMAS PARAMETERS
# =============================================================
NUM_ANTS = 90
MAX_ITER = 300
PHEROMONE_MAX = 5.0
PHEROMONE_MIN = 0.001
EVAP_RATE = 0.6
ALPHA = 1.0
BETA = 2.0
ELITIST_WEIGHT = 1.5
TAU_INITIAL = 1.0

ITER_LOG_FILENAME = "mmas_iterations.csv"
RUN_SUMMARY_FILENAME = "mmas_run_summary.csv"

# =============================================================
# REAL-TIME VISUALIZER (same style as ACO)
# =============================================================
class MMASVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_conv, self.ax_pher) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle("Real-Time MMAS Optimization", fontsize=14, fontweight='bold')

        # --- Convergence Plot ---
        self.ax_conv.set_title("Convergence: Best vs Iteration Best")
        self.ax_conv.set_xlabel("Iteration")
        self.ax_conv.set_ylabel("Cost (f)")
        self.ax_conv.grid(True)
        self.line_best, = self.ax_conv.plot([], [], 'b-', linewidth=2, label="Best-so-far f")
        self.line_iter_best, = self.ax_conv.plot([], [], 'g--', linewidth=1.5, label="Iteration Best f")
        self.ax_conv.legend()

        # --- Pheromone Plot ---
        self.ax_pher.set_title("Pheromone Statistics")
        self.ax_pher.set_xlabel("Iteration")
        self.ax_pher.set_ylabel("Pheromone Level")
        self.ax_pher.grid(True)
        self.line_pher_max, = self.ax_pher.plot([], [], 'r-', linewidth=1.8, label="Max τ")
        self.line_pher_avg, = self.ax_pher.plot([], [], 'orange', linewidth=1.8, label="Avg τ")
        self.line_pher_min, = self.ax_pher.plot([], [], 'brown', linewidth=1.8, label="Min τ")
        self.ax_pher.legend()

        plt.tight_layout()

    def update(self, history):
        if not plt.fignum_exists(self.fig.number):
            return

        iters = range(len(history["best_f"]))

        # Convergence
        self.line_best.set_data(iters, history["best_f"])
        self.line_iter_best.set_data(iters, history["iter_best_f"])
        self.ax_conv.relim()
        self.ax_conv.autoscale_view()

        # Pheromone
        self.line_pher_max.set_data(iters, history["pher_max"])
        self.line_pher_avg.set_data(iters, history["pher_avg"])
        self.line_pher_min.set_data(iters, history["pher_min"])
        self.ax_pher.relim()
        self.ax_pher.autoscale_view()

        plt.pause(0.05)

    def close(self):
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        plt.ioff()

# =============================================================
# Ant & Graph Classes
# =============================================================
class Ant:
    def __init__(self, ant_id):
        self.id = ant_id
        self.permutation: List[Vehicle] = []
        self.speeds: List[float] = []
        self.fitness = math.inf

    def reset(self):
        self.permutation = []
        self.speeds = []
        self.fitness = math.inf


# ---------- MMASGraph (transition-based) ----------
class MMASGraph:
    def __init__(self, vehicles: List[Vehicle], tau_init: float = TAU_INITIAL):
        self.vehicles = vehicles
        self.n = len(vehicles)
        # map vehicle ID -> index
        self.id_to_idx = {v.id: i for i, v in enumerate(vehicles)}
        self.idx_to_id = {i: v.id for i, v in enumerate(vehicles)}
        # transition pheromone matrix: (n+1) x n
        # rows 0..n-1: vehicle idx -> next vehicle idx
        # row n: start -> first vehicle
        self.tau = np.ones((self.n + 1, self.n)) * tau_init

    def evaporate(self, rho):
        self.tau *= (1 - rho)

    def deposit_for_permutation(self, perm: List[Vehicle], delta: float):
        """Deposit pheromone along transitions defined by perm."""
        if not perm:
            return
        n = self.n
        # start -> first
        first_idx = self.id_to_idx[perm[0].id]
        self.tau[n, first_idx] += delta
        # transitions
        for a, b in zip(perm[:-1], perm[1:]):
            i = self.id_to_idx[a.id]
            j = self.id_to_idx[b.id]
            self.tau[i, j] += delta

    def clamp(self, tmin, tmax):
        np.clip(self.tau, tmin, tmax, out=self.tau)

    def stats(self):
        return float(np.max(self.tau)), float(np.mean(self.tau)), float(np.min(self.tau))


# ---------- Transition-based heuristic ----------
def build_priority_heuristic(vehicles: List[Vehicle], geom: Geometry) -> np.ndarray:
    """
    eta[i][j] = heuristic for choosing vehicle j after vehicle i.
    Row n is the artificial START node.
    """
    n = len(vehicles)
    id_to_idx = {v.id: idx for idx, v in enumerate(vehicles)}
    eta = np.ones((n + 1, n), dtype=float)

    # precompute queue index (closer to stopline -> better)
    queue_index = {}
    for approach, queue in geom.entry_queues.items():
        for idx, v in enumerate(queue):
            queue_index[v.id] = idx

    for i in range(n + 1):  # includes start row
        for v_j in vehicles:
            j = id_to_idx[v_j.id]
            priority = 3.0 if getattr(v_j, "priority_status", False) else 1.0
            q_idx = queue_index.get(v_j.id, 99)
            queue_factor = 1.0 / (1.0 + q_idx)
            eta[i, j] = max(1e-6, priority * queue_factor)

    return eta


# ---------- Deterministic speed baseline for MMAS ----------
def generate_speeds_for_permutation(permutation: List[Vehicle], geom: Geometry):
    """
    Deterministic baseline speeds used for MMAS evaluations.
    Leaders get near-max speed; followers decline smoothly.
    """
    import numpy as _np
    vmin, vmax = config.velocity_range
    speeds = {}

    leader_target = vmax * 0.90
    decay = 0.05  # each follower ~5% slower than previous

    # For each approach queue assign deterministic speeds
    perm_ids = [v.id for v in permutation]
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue

        leader = None
        for v in queue:
            if v.id in perm_ids:
                leader = v
                break
        if leader is None:
            continue

        speeds[leader.id] = float(_np.clip(leader_target, vmin, vmax))
        idx = 1
        for v in queue:
            if v.id == leader.id or v.id not in perm_ids:
                continue
            follower_speed = leader_target * max(0.4, (1 - decay * idx))
            follower_speed = float(_np.clip(follower_speed, vmin, vmax))
            speeds[v.id] = follower_speed
            idx += 1

    # Fill missing vehicles
    final_speeds = []
    for v in permutation:
        if v.id not in speeds:
            speeds[v.id] = float((vmin + vmax) / 2.0)
        final_speeds.append(speeds[v.id])

    return final_speeds


# ---------- Construct anti solution using transition pheromones ----------
def construct_ant_solution(ant: Ant, graph: MMASGraph, eta, alpha, beta, geom, tau_p_dict):
    ant.reset()
    n = graph.n
    vehicles = graph.vehicles
    id_to_idx = graph.id_to_idx

    available = set(v.id for v in vehicles)

    # Choose first vehicle (start row = n)
    probs = []
    cand_list = list(available)
    for vid in cand_list:
        j = id_to_idx[vid]
        pher = graph.tau[n, j] ** alpha
        heur = eta[n, j] ** beta
        probs.append(pher * heur)
    probs = np.array(probs, dtype=float)
    if probs.sum() <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()

    chosen_vid = np.random.choice(cand_list, p=probs)
    ant.permutation.append(next(v for v in vehicles if v.id == chosen_vid))
    available.remove(chosen_vid)

    # Grow permutation using transitions
    while available:
        prev_vid = ant.permutation[-1].id
        i = id_to_idx[prev_vid]
        cand_list = list(available)
        local_probs = []
        for vid in cand_list:
            j = id_to_idx[vid]
            pher = graph.tau[i, j] ** alpha
            heur = eta[i, j] ** beta
            local_probs.append(pher * heur)
        local_probs = np.array(local_probs, dtype=float)
        if local_probs.sum() <= 0:
            local_probs = np.ones_like(local_probs) / len(local_probs)
        else:
            local_probs = local_probs / local_probs.sum()

        next_vid = np.random.choice(cand_list, p=local_probs)
        ant.permutation.append(next(v for v in vehicles if v.id == next_vid))
        available.remove(next_vid)

    # Speeds: deterministic baseline (no randomness)
    ant.speeds = generate_speeds_for_permutation(ant.permutation, geom)

    # Validate speeds using SA/GA validator (if you want strict enforcement here)
    # But avoid injecting randomness: use validate_speeds only if it is deterministic.
    ant.speeds = validate_speeds(ant.permutation, ant.speeds, geom)

    # Evaluate
    obj = evaluate_solution(ant.permutation, ant.speeds, geom, tau_p_dict)
    ant.fitness = obj["f"]


# =============================================================
# Main MMAS Runner
# =============================================================
def run_mmas(max_iter=MAX_ITER,
             num_ants=NUM_ANTS,
             alpha=ALPHA,
             beta=BETA,
             evap=EVAP_RATE,
             tau_min=PHEROMONE_MIN,
             tau_max=PHEROMONE_MAX,
             elitist_weight=ELITIST_WEIGHT,
             visualize_realtime=False,
             verbose=True,
             log_to_csv=False):

    if verbose:
        print("\n=== MMAS START ===")

    # Geometry
    geom = Geometry()
    vehicles = config.pi
    geom.create_entry_queue(vehicles)
    for v in vehicles:
        geom.set_trajectory(v)

    # Conflict times
    all_points = set().union(*(v.path for v in vehicles if v.path))
    tau_p_dict = {p: config.tau for p in all_points}

    # Graph & heuristic
    graph = MMASGraph(vehicles, TAU_INITIAL)
    eta = build_priority_heuristic(vehicles, geom)

    ants = [Ant(i) for i in range(num_ants)]

    best_perm = None
    best_speeds = None
    best_cost = math.inf
    eval_count = 0

    history = {
        "best_f": [],
        "iter_best_f": [],
        "pher_max": [],
        "pher_avg": [],
        "pher_min": []
    }

    visualizer = MMASVisualizer() if visualize_realtime else None

    # Main iterations
    for it in range(1, max_iter + 1):
        iter_best = math.inf

        for ant in ants:
            construct_ant_solution(ant, graph, eta, alpha, beta, geom, tau_p_dict)
            eval_count += 1

            if ant.fitness < iter_best:
                iter_best = ant.fitness

            if ant.fitness < best_cost:
                best_cost = ant.fitness
                best_perm = copy.deepcopy(ant.permutation)
                best_speeds = list(ant.speeds)

        # Pheromone update
        graph.evaporate(evap)
        delta = (1 / (1 + best_cost)) * elitist_weight
        graph.deposit_for_permutation(best_perm, delta)
        graph.clamp(tau_min, tau_max)

        # Log
        pmax, pavg, pmin = graph.stats()
        history["best_f"].append(best_cost)
        history["iter_best_f"].append(iter_best)
        history["pher_max"].append(pmax)
        history["pher_avg"].append(pavg)
        history["pher_min"].append(pmin)

        if visualize_realtime:
            visualizer.update(history)

        if verbose and (it % 10 == 0):
            print(f"Iter {it}/{max_iter}: Best {best_cost:.2f}")

    if visualize_realtime:
        visualizer.close()

    if verbose:
        print(f"\nMMAS FINISHED\nBest f = {best_cost:.2f}\nEvaluations = {eval_count}")

    return best_perm, best_speeds, best_cost, history, geom, tau_p_dict, eval_count

# =============================================================
# Standalone Run
# =============================================================
if __name__ == "__main__":
    best_perm, best_speeds, best_cost, history, geom, tau_p_dict, evals = run_mmas(
        visualize_realtime=True,
        verbose=True
    )

    print("\n--- Final Best Solution ---")
    print("Permutation:", [v.id for v in best_perm])
    print("Speeds:", [f"{s:.2f}" for s in best_speeds])
    print("Best objective:", best_cost)
