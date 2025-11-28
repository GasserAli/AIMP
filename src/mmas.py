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
NUM_ANTS = 150
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


class MMASGraph:
    def __init__(self, vehicles: List[Vehicle], tau_init: float = TAU_INITIAL):
        self.vehicles = vehicles
        self.n = len(vehicles)
        self.vehicle_ids = [v.id for v in vehicles]
        self.tau = np.ones((self.n, self.n)) * tau_init

    def evaporate(self, rho):
        self.tau *= (1 - rho)

    def deposit_for_permutation(self, perm: List[Vehicle], delta: float):
        for pos, v in enumerate(perm):
            idx = self.vehicle_ids.index(v.id)
            self.tau[pos, idx] += delta

    def clamp(self, tmin, tmax):
        np.clip(self.tau, tmin, tmax, out=self.tau)

    def stats(self):
        return float(np.max(self.tau)), float(np.mean(self.tau)), float(np.min(self.tau))

# =============================================================
# Heuristic: PRIORITY + QUEUE INDEX
# =============================================================
def build_priority_heuristic(vehicles: List[Vehicle], geom: Geometry) -> np.ndarray:
    """
    Builds the (n × n) heuristic matrix eta[pos][veh_idx].

    Heuristic components:
      - Emergency vehicles get higher heuristic values.
      - Vehicles close to stopline (low queue index) get higher values.
      - Earlier permutation positions (pos close to 0) amplify the heuristic.

    NOTE:
      This function does NOT include time-to-first-conflict unless requested.
    """

    n = len(vehicles)
    eta = np.ones((n, n), dtype=float)



    # -----------------------------------------------------------
    # Precompute queue index: position of vehicle in its approach queue
    # Closer to stopline → lower index → higher weight
    # -----------------------------------------------------------
    queue_index = {}
    for approach, queue in geom.entry_queues.items():
        for idx, v in enumerate(queue):
            queue_index[v.id] = idx

    # -----------------------------------------------------------
    # Build heuristic matrix
    # eta[position][vehicle]
    # -----------------------------------------------------------
    for pos in range(n):
        # earlier positions get stronger emphasis
        pos_scale = max(0.01, 1.0 - pos / max(1, n))

        for v_idx, v in enumerate(vehicles):

            # Priority factor: emergency > normal
            priority_factor = 3.0 if getattr(v, "priority_status", False) else 1.0

            # Queue index factor
            q_idx = queue_index.get(v.id, 99)    # penalize if not found
            queue_factor = 1.0 / (1.0 + float(q_idx))

            # Final heuristic value
            value = priority_factor * queue_factor * pos_scale

            eta[pos, v_idx] = max(1e-6, value)  # avoid zeros

    return eta


# =============================================================
# Speed Initialization (leader/follower)
# =============================================================
def generate_speeds_for_permutation(permutation: List[Vehicle], geom: Geometry):
    vmin, vmax = config.velocity_range
    speeds = {}

    for approach, queue in geom.entry_queues.items():
        leader = None
        for v in queue:
            if v.id in [x.id for x in permutation]:
                leader = v
                break

        if leader is None:
            continue

        leader_speed = random.uniform(vmin, vmax)
        speeds[leader.id] = leader_speed
        last_s = leader_speed

        for v in queue:
            if v.id == leader.id or v.id not in [x.id for x in permutation]:
                continue
            s_new = random.uniform(vmin, last_s)
            speeds[v.id] = s_new
            last_s = s_new

    final_speeds = []
    for v in permutation:
        if v.id not in speeds:
            speeds[v.id] = random.uniform(vmin, vmax)
        final_speeds.append(speeds[v.id])

    return final_speeds

# =============================================================
# Construct solution for one ant
# =============================================================
def construct_ant_solution(ant: Ant, graph: MMASGraph, eta, alpha, beta, geom, tau_p_dict):
    ant.reset()
    n = graph.n
    available = graph.vehicle_ids.copy()
    vmap = {v.id: v for v in graph.vehicles}

    # Build permutation
    for pos in range(n):
        probs = []
        for vid in available:
            idx = graph.vehicle_ids.index(vid)
            phero = graph.tau[pos, idx] ** alpha
            heur = eta[pos, idx] ** beta
            probs.append(phero * heur)

        probs = np.array(probs)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)

        chosen_idx = np.random.choice(len(available), p=probs)
        chosen_vid = available.pop(chosen_idx)
        ant.permutation.append(vmap[chosen_vid])

    # Speeds
    ant.speeds = generate_speeds_for_permutation(ant.permutation, geom)
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
