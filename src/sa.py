import random
import math
from typing import List, Tuple, Dict

from decoder import is_permutation_valid
from objective import objective_from_queues
import config
import matplotlib.pyplot as plt
import os


def permute_swap(pi: List) -> List:
    """Return a new permutation by swapping two random indices."""
    n = len(pi)
    if n < 2:
        return pi[:]
    i, j = random.sample(range(n), 2)
    new = pi[:]
    new[i], new[j] = new[j], new[i]
    return new


def tweak_speeds(pi: List, sigma=1.0):
    """Randomly perturb vehicle velocities (small gaussian noise)."""
    new = []
    for v in pi:
        v2 = v
        # clamp velocities to positive values
        v2.velocity = max(0.1, v.velocity + random.gauss(0, sigma))
        new.append(v2)
    return new


def evaluate(pi: List, d0=10.0, inter=10.0, tau=config.tau) -> Tuple[float, int, Dict]:
    """Return (cost, crashes, detail) where detail contains delays mapping."""
    valid, crashes = is_permutation_valid(pi, distance_to_first_conflict=d0, inter_conflict_distance=inter, safety_time=config.safety_distance)
    # objective expects Vehicles with .delay set by decoder; objective_from_queues calculates f
    res = objective_from_queues(pi, alpha=config.alpha, beta=config.beta)
    return res['f'], (0 if valid else crashes), res


def simulated_annealing(pi0: List, iterations=200, T0=1.0, alpha=0.995):
    """Basic SA that perturbs permutation and occasionally speeds.

    Returns histories: delays_per_iter, cost_per_iter, crashes_per_iter
    """
    random.seed(0)
    current = pi0[:]
    best = current[:]
    best_cost, best_crashes, _ = evaluate(best)

    T = T0
    delays_hist = []
    cost_hist = []
    crash_hist = []

    for it in range(iterations):
        # propose either a swap or speed tweak
        if random.random() < 0.7:
            candidate = permute_swap(current)
        else:
            candidate = tweak_speeds([v for v in current], sigma=1.0)

        cost_c, crashes_c, detail = evaluate(candidate)
        cost_curr, crashes_curr, _ = evaluate(current)

        # Accept if better or probabilistically
        accept = False
        if cost_c < cost_curr:
            accept = True
        else:
            delta = cost_c - cost_curr
            if random.random() < math.exp(-delta / max(T, 1e-9)):
                accept = True

        if accept:
            current = candidate

        # record
        delays_hist.append(detail['delays'])
        cost_hist.append(cost_curr)
        crash_hist.append(crashes_curr)

        # update best
        if cost_curr < best_cost:
            best = current[:]
            best_cost = cost_curr

        T *= alpha

    return {
        'best_perm': best,
        'best_cost': best_cost,
        'delays_per_iter': delays_hist,
        'cost_per_iter': cost_hist,
        'crashes_per_iter': crash_hist,
    }


def main_test():
    import config
    pi = config.pi
    # ensure geometry paths are assigned by main (Geometry.set_trajectory)
    from geometry import Geometry
    g = Geometry()
    for v in pi:
        g.set_trajectory(v)

    out = simulated_annealing(pi, iterations=50)
    print('Best cost:', out['best_cost'])
    print('Cost history (last 10):', out['cost_per_iter'][-10:])
    print('Crashes history (last 10):', out['crashes_per_iter'][-10:])
    plot_results(out, prefix='sa_run')


def plot_results(out: Dict, prefix: str = 'sa_run'):
    """Create and save plots for cost, crashes, and average delay per iteration."""
    costs = out['cost_per_iter']
    crashes = out['crashes_per_iter']
    delays = out['delays_per_iter']

    # compute average delay per iteration
    avg_delays = []
    for d in delays:
        if not d:
            avg_delays.append(0.0)
        else:
            avg_delays.append(sum(d.values()) / len(d))

    # ensure output dir
    out_dir = os.path.join(os.getcwd(), 'sa_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Plot cost
    plt.figure(figsize=(10, 4))
    plt.plot(costs, '-o', markersize=3)
    plt.title('SA: Cost per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.tight_layout()
    cost_file = os.path.join(out_dir, f"{prefix}_cost.png")
    plt.savefig(cost_file)
    plt.close()

    # Plot crashes
    plt.figure(figsize=(10, 4))
    plt.plot(crashes, '-o', markersize=3)
    plt.title('SA: Crashes per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Crashes')
    plt.grid(True)
    plt.tight_layout()
    crash_file = os.path.join(out_dir, f"{prefix}_crashes.png")
    plt.savefig(crash_file)
    plt.close()

    # Plot average delay
    plt.figure(figsize=(10, 4))
    plt.plot(avg_delays, '-o', markersize=3)
    plt.title('SA: Average Delay per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Delay')
    plt.grid(True)
    plt.tight_layout()
    delay_file = os.path.join(out_dir, f"{prefix}_avg_delay.png")
    plt.savefig(delay_file)
    plt.close()

    print(f"Saved plots to {out_dir}:\n  {cost_file}\n  {crash_file}\n  {delay_file}")


if __name__ == '__main__':
    main_test()
