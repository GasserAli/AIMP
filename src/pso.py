# File: pso.py
"""
PSO FOR SPEED OPTIMIZATION (AFTER PERMUTATION FIXED)
====================================================

This PSO is the speed-optimization phase after MMAS finds a permutation.
Architecture matches MMAS philosophy:
    - Uses real decoder → collision-free
    - Uses SA’s evaluate_solution & validate_speeds
    - Logs history (best, iter-best, avg, velocity norms)
    - Optional real-time plot like ACO/MMAS
Return signature:
    best_speeds, best_cost, history
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

import config
from sa import evaluate_solution, validate_speeds


# =============================================================
#  Real-time PSO Visualizer (optional)
# =============================================================
class PSOVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_cost, self.ax_vel) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle("Real-Time PSO Optimization (Speeds)", fontsize=14)

        # cost plot
        self.ax_cost.set_title("Best vs Iteration Best (Cost)")
        self.ax_cost.set_xlabel("Iteration")
        self.ax_cost.set_ylabel("Objective f")
        self.ax_cost.grid(True)
        self.line_best, = self.ax_cost.plot([], [], 'b-', label='Best-so-far', linewidth=2)
        self.line_iter, = self.ax_cost.plot([], [], 'g--', label='Iteration Best', linewidth=1.5)
        self.ax_cost.legend()

        # velocity statistics
        self.ax_vel.set_title("Velocity Norm (Swarm Movement)")
        self.ax_vel.set_xlabel("Iteration")
        self.ax_vel.set_ylabel("||velocity||")
        self.ax_vel.grid(True)
        self.line_vel, = self.ax_vel.plot([], [], 'magenta', label='Velocity Norm')
        self.ax_vel.legend()

        plt.tight_layout()

    def update(self, history):
        if not plt.fignum_exists(self.fig.number):
            return

        iters = range(len(history["best_f"]))
        self.line_best.set_data(iters, history["best_f"])
        self.line_iter.set_data(iters, history["iter_best_f"])
        self.ax_cost.relim(); self.ax_cost.autoscale_view()

        self.line_vel.set_data(iters, history["vel_norm"])
        self.ax_vel.relim(); self.ax_vel.autoscale_view()

        plt.pause(0.05)

    def close(self):
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        plt.ioff()


# =============================================================
#  SPEED FITNESS WRAPPER
# =============================================================
def _fitness(permutation, speeds, geom, tau_p_dict):
    """Evaluate speeds with decoder; returns objective f."""
    s_val = validate_speeds(permutation, speeds, geom)
    obj = evaluate_solution(permutation, s_val, geom, tau_p_dict)
    return obj["f"]


# =============================================================
#  FULL PSO IMPLEMENTATION
# =============================================================
def pso_optimize_speeds(permutation,
                        init_speeds,
                        geom,
                        tau_p_dict,
                        swarm_size=20,
                        iters=80,
                        w=0.6,
                        c1=1.4,
                        c2=1.4,
                        visualize=False,
                        verbose=True):
    """
    Optimize speeds for a fixed permutation using PSO.

    MATCHES MMAS ARCHITECTURE (history, logs, real decoder evaluation)
    """

    # speed bounds
    vmin, vmax = config.velocity_range
    dim = len(init_speeds)

    # -----------------------------------------------------------
    #  Initialize swarm
    # -----------------------------------------------------------
    swarm = []
    velocity = []

    for _ in range(swarm_size):
        p = np.array([random.uniform(vmin, vmax) for _ in range(dim)])
        v = np.zeros(dim)
        swarm.append(p)
        velocity.append(v)

    # personal bests
    pbest = [p.copy() for p in swarm]
    pbest_f = [_fitness(permutation, p, geom, tau_p_dict) for p in swarm]

    # global best
    g_idx = np.argmin(pbest_f)
    gbest = pbest[g_idx].copy()
    gbest_f = pbest_f[g_idx]

    # -----------------------------------------------------------
    #  History structure (like MMAS)
    # -----------------------------------------------------------
    history = {
        "best_f": [],
        "iter_best_f": [],
        "vel_norm": []
    }

    # Visualizer
    viz = PSOVisualizer() if visualize else None

    if verbose:
        print("\n=== PSO START ===")
        print(f"Swarm={swarm_size}, Iter={iters}, dim={dim}\n")

    # -----------------------------------------------------------
    #  MAIN PSO LOOP
    # -----------------------------------------------------------
    for it in range(1, iters + 1):
        iter_best = math.inf

        for i in range(swarm_size):

            r1, r2 = random.random(), random.random()

            # velocity update
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (pbest[i] - swarm[i])
                + c2 * r2 * (gbest - swarm[i])
            )

            # update particle
            swarm[i] = swarm[i] + velocity[i]
            swarm[i] = np.clip(swarm[i], vmin, vmax)

            # evaluate fitness
            f = _fitness(permutation, swarm[i], geom, tau_p_dict)

            # update personal best
            if f < pbest_f[i]:
                pbest[i] = swarm[i].copy()
                pbest_f[i] = f

                # update global best
                if f < gbest_f:
                    gbest = swarm[i].copy()
                    gbest_f = f

            # iteration best
            if f < iter_best:
                iter_best = f

        # HISTORY UPDATE
        history["best_f"].append(gbest_f)
        history["iter_best_f"].append(iter_best)
        vel_norm = np.mean([np.linalg.norm(v) for v in velocity])
        history["vel_norm"].append(vel_norm)

        # progress print
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  Iter {it:03d}/{iters}  Best={gbest_f:.2f}  VelNorm={vel_norm:.4f}")

        if viz:
            viz.update(history)

    if viz:
        viz.close()

    if verbose:
        print("\n=== PSO END ===")
        print(f"Final best f = {gbest_f:.2f}")

    return gbest.tolist(), gbest_f, history


# =============================================================
# MAIN: Run MMAS → then PSO on resulting permutation
# =============================================================
if __name__ == "__main__":
    print("\n===================================================")
    print("      PSO SPEED OPTIMIZATION — STANDALONE TEST")
    print("  (Loads permutation from MMAS, then runs PSO)")
    print("===================================================\n")

    try:
        from mmas import run_mmas, MAX_ITER, NUM_ANTS
    except Exception as e:
        print("ERROR: Could not import MMAS:", e)
        exit(1)

    import config
    from geometry import Geometry

    print("Running MMAS first to get best permutation...\n")

    # -------------------------------------------------------
    # Run MMAS to obtain best permutation
    # -------------------------------------------------------
    perm_best, speeds_best_mmas, best_cost_mmas, mmas_history, geom, tau_p_dict, evals = run_mmas(
        max_iter=MAX_ITER,
        num_ants=NUM_ANTS,
        verbose=True,
        log_to_csv=False
    )

    print("\n--- MMAS Completed ---")
    print("Best Permutation:", [v.id for v in perm_best])
    print("MMAS best cost (with initial speeds):", best_cost_mmas)

    # -------------------------------------------------------
    # Prepare initial PSO speeds (random within global limits)
    # -------------------------------------------------------
    vmin, vmax = config.velocity_range
    init_speeds = [random.uniform(vmin, vmax) for _ in perm_best]

    print("\nRunning PSO speed optimization...\n")

    # -------------------------------------------------------
    # Run PSO on the MMAS permutation
    # -------------------------------------------------------
    best_speeds_pso, best_cost_pso, pso_history = pso_optimize_speeds(
        permutation=perm_best,
        init_speeds=init_speeds,
        geom=geom,
        tau_p_dict=tau_p_dict,
        swarm_size=30,
        iters=100,
        visualize=True,
        verbose=True
    )

    print("\n=======================")
    print("   FINAL RESULTS")
    print("=======================\n")

    print("Permutation:", [v.id for v in perm_best])
    print("Speeds from PSO:", [round(s, 2) for s in best_speeds_pso])
    print("MMAS Cost (before PSO):", round(best_cost_mmas, 2))
    print("PSO Cost  (after  PSO):", round(best_cost_pso, 2))
    print("Improvement:", round(best_cost_mmas - best_cost_pso, 4))

    print("\nPSO optimization completed.")

