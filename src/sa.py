# ================================================================
# File: sa.py  (FULLY FIXED VERSION)
# ================================================================

import math
import random
import copy
import matplotlib.pyplot as plt
import os
import traceback

# --- Import Project Files ---
import config
import objective
from geometry import Geometry

# IMPORTANT: use the corrected decoder you have already implemented
from decoder import run_decoder      # <--- Ensure decoder.py on disk is the FIXED VERSION

try:
    from visualization import IntersectionVisualization
    animation_enabled = True
    print("Successfully imported visualization module.")
except ImportError:
    print("Warning: visualization.py not found or class missing. Animation disabled.")
    animation_enabled = False
# -------------------------------------------------------------------

# --- SA Parameters ---
T_INITIAL = 100.0
T_MIN = 1.0
COOLING_RATE = 0.99
MAX_ITER_PER_TEMP = 10               # evaluations per temperature
MAX_TOTAL_ITERATIONS = 100000        # global max evaluations


# ================================================================
# Initial Solution Generator
# ================================================================
def create_initial_solution(geom):
    """
    Generates initial permutation & speeds.
    Respects no-catch-up constraint.
    """
    initial_perm = copy.deepcopy(config.pi)
    random.shuffle(initial_perm)

    initial_speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range

    # Assign ordered speeds per queue
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue

        # Identify first vehicle in queue in permutation
        leader_in_queue = None
        for v in queue:
            if v.id in [p.id for p in initial_perm]:
                leader_in_queue = v
                break

        if leader_in_queue is None:
            continue

        # Leader starts with a high-ish random value
        last_speed = random.uniform(
            v_min_global + 0.5*(v_max_global - v_min_global),
            v_max_global
        )
        initial_speeds_dict[leader_in_queue.id] = last_speed

        # Followers cannot exceed predecessor
        followers_in_queue = [
            v for v in queue 
            if v.id != leader_in_queue.id and v.id in [p.id for p in initial_perm]
        ]

        for v_follower in followers_in_queue:
            current_max = min(v_max_global, last_speed)
            new_speed = random.uniform(v_min_global, current_max)
            initial_speeds_dict[v_follower.id] = new_speed
            last_speed = new_speed

    # Convert dict → speed list following permutation order
    initial_speeds_list = []
    for v in initial_perm:
        if v.id not in initial_speeds_dict:
            initial_speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
        initial_speeds_list.append(initial_speeds_dict[v.id])

    print(f"Initial SA Perm: {[v.id for v in initial_perm]}")
    return initial_perm, initial_speeds_list


# ================================================================
# Speed Validation (C0 Constraint)
# ================================================================
def validate_speeds(permutation, speeds, geom):
    """
    Enforce no-catch-up constraint per lane queue:
    follower_speed ≤ leader_speed.
    """
    speed_dict = {p.id: s for p, s in zip(permutation, speeds)}
    v_min_global, v_max_global = config.velocity_range

    for queue in geom.entry_queues.values():
        if not queue:
            continue

        # Find first vehicle in queue
        leader = None
        for v in queue:
            if v.id in speed_dict:
                leader = v
                break
        if not leader:
            continue

        last_speed = speed_dict[leader.id]

        # Apply constraint down the queue
        for v_follower in queue:
            if v_follower.id not in speed_dict or v_follower.id == leader.id:
                continue

            if speed_dict[v_follower.id] > last_speed:
                speed_dict[v_follower.id] = last_speed

            last_speed = speed_dict[v_follower.id]

    return [speed_dict[v.id] for v in permutation]


# ================================================================
# Neighbor Generator (FIXED)
# ================================================================
def generate_neighbor(perm_current, speeds_current, geom):
    """
    Create a neighbor solution.
    Key FIX: If permutation swap happens, also swap speeds accordingly.
    """
    perm_new = copy.deepcopy(perm_current)
    speeds_new = copy.deepcopy(speeds_current)
    v_min_global, v_max_global = config.velocity_range

    # 50%: SWAP PERMUTATION (AND SPEEDS)
    if random.random() < 0.5 and len(perm_new) > 1:
        idx1, idx2 = random.sample(range(len(perm_new)), 2)

        # swap vehicles
        perm_new[idx1], perm_new[idx2] = perm_new[idx2], perm_new[idx1]

        # swap their speeds too (critical fix)
        speeds_new[idx1], speeds_new[idx2] = speeds_new[idx2], speeds_new[idx1]

    # 50%: MUTATE SPEED
    else:
        idx = random.randrange(len(speeds_new))
        
        # option A: small local mutation
        if random.random() < 0.7:
            delta = random.uniform(-1.5, 1.5)
            speeds_new[idx] += delta

        # option B: reassign speed completely (exploration)
        else:
            speeds_new[idx] = random.uniform(v_min_global, v_max_global)

        # clamp to global bounds
        speeds_new[idx] = max(v_min_global, min(speeds_new[idx], v_max_global))

    # enforce no-catch-up
    speeds_new = validate_speeds(perm_new, speeds_new, geom)
    return perm_new, speeds_new


# ================================================================
# Evaluate a Solution
# ================================================================
def evaluate_solution(permutation, speeds, geom, tau_p_dict, return_full_schedule=False):
    """
    Runs decoder + objective. Handles exceptions robustly.
    """
    try:
        # Call decoder. If full schedule requested, decoder returns (decoder_results, scheduled_times, t_ear)
        if return_full_schedule:
            decoder_results, scheduled_times, t_ear = run_decoder(
                permutation=permutation,
                speeds=speeds,
                geom=geom,
                tau_p_dict=tau_p_dict,
                return_full_schedule=True
            )
        else:
            decoder_results = run_decoder(
                permutation=permutation,
                speeds=speeds,
                geom=geom,
                tau_p_dict=tau_p_dict,
                return_full_schedule=False
            )

        # Compute objective
        obj_dict = objective.calculate_objective(decoder_results, speeds=speeds)

        if return_full_schedule:
            return obj_dict, scheduled_times, t_ear
        return obj_dict

    except Exception as e:
        print(f"Error evaluating solution: {e}")
        traceback.print_exc()

        penalized = [{"id": v.id, "delay": 99999.0, "is_emergency": v.priority_status}
                     for v in permutation]
        obj_pen = objective.calculate_objective(penalized)

        if return_full_schedule:
            return obj_pen, {}, {}
        return obj_pen


# ================================================================
# Plotting Results
# ================================================================
def plot_results(history):
    costs = history['costs']
    temps = history['temps']
    total_delays = history['total_delays']
    emergency_delays = history['emergency_delays']
    avg_delays = history['avg_delays']

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SA Performance Metrics", fontsize=14, fontweight='bold')

    def plot(ax, y, label, color):
        ax.plot(y, color=color)
        ax.set_title(label)
        ax.grid(True)

    plot(axs[0,0], costs, "Objective f", "blue")
    plot(axs[0,1], avg_delays, "Average Delay", "orange")
    plot(axs[1,0], total_delays, "Total Delay f_all", "green")
    plot(axs[1,1], emergency_delays, "Emergency Delay f_em", "red")

    plt.tight_layout()
    plt.show()


# ================================================================
# Simulated Annealing Main Loop
# ================================================================
def run_sa(T_init=T_INITIAL, T_min=T_MIN, cool_rate=COOLING_RATE,
           iter_per_temp=MAX_ITER_PER_TEMP, max_iter=MAX_TOTAL_ITERATIONS,
           initial_solution=None, verbose=True):

    if verbose:
        print("=== Starting Simulated Annealing ===")

    # prepare geometry
    geom = Geometry()
    geom.create_entry_queue(config.pi)
    for v in config.pi:
        geom.set_trajectory(v)

    # collect all conflict points
    all_points = set().union(*(v.path for v in config.pi if v.path))
    tau_p_dict = {p: config.tau for p in all_points}

    # initial solution
    if initial_solution:
        perm_current, speeds_current = initial_solution
    else:
        perm_current, speeds_current = create_initial_solution(geom)

    obj_dict_current = evaluate_solution(perm_current, speeds_current, geom, tau_p_dict)
    obj_current = obj_dict_current["f"]

    perm_best, speeds_best, obj_best = perm_current, speeds_current, obj_current

    T = T_init
    iter_count = 0

    history = {
        "costs": [], "temps": [], "total_delays": [],
        "emergency_delays": [], "avg_delays": []
    }

    # log first point
    def record():
        history["costs"].append(obj_current)
        history["temps"].append(T)
        history["total_delays"].append(obj_dict_current.get("fall", 0))
        history["emergency_delays"].append(obj_dict_current.get("fem", 0))
        delays = obj_dict_current.get("delays", {})
        if delays:
            history["avg_delays"].append(sum(delays.values()) / len(delays))
        else:
            history["avg_delays"].append(0)

    record()

    if verbose:
        print(f"Initial objective: {obj_current:.2f}")

    # SA loop
    while T > T_min and iter_count < max_iter:

        for _ in range(iter_per_temp):
            if iter_count >= max_iter:
                break

            perm_new, speeds_new = generate_neighbor(perm_current, speeds_current, geom)
            obj_dict_new = evaluate_solution(perm_new, speeds_new, geom, tau_p_dict)
            obj_new = obj_dict_new["f"]

            iter_count += 1

            ΔE = obj_new - obj_current

            # acceptance rule
            if ΔE < 0 or math.exp(-ΔE / max(T, 1e-9)) > random.random():
                perm_current, speeds_current, obj_current = perm_new, speeds_new, obj_new
                obj_dict_current = obj_dict_new

                if obj_current < obj_best:
                    perm_best, speeds_best, obj_best = perm_current, speeds_current, obj_current
                    if verbose:
                        print(f" New BEST f={obj_best:.2f}")

            record()

        # cool down
        T = T * cool_rate

    if verbose:
        print("\n=== SA FINISHED ===")
        print(f"Best f = {obj_best:.2f}")
        print(f"Iterations = {iter_count}")
        print(f"Best speed range: {min(speeds_best):.3f}–{max(speeds_best):.3f}")
        # Print full list of vehicle IDs and their corresponding final speeds
        try:
            ids = [v.id for v in perm_best]
            speeds_str = [f"{s:.3f}" for s in speeds_best]
            approach_str= [v.approach for v in perm_best]
            print("Final permutation (vehicle IDs):", ids)
            print("Final speeds (m/s):", speeds_str)
            print("Final approaches:", approach_str)
            # Pair them for clearer output
            paired = [f"ID {vid}: {spd} m/s" for vid, spd in zip(ids, speeds_str)]
            print("Final speeds per vehicle:")
            for p in paired:
                print("  ", p)
        except Exception:
            # Fallback: if something unexpected, still return gracefully
            print("Could not print detailed speeds (unexpected error).")
        # --- Compute and print collision statistics from the final best schedule ---
        try:
            # Request full schedule from decoder for the best solution
            obj_full, scheduled_times, t_ear = evaluate_solution(
                perm_best, speeds_best, geom, tau_p_dict, return_full_schedule=True
            )

            # Count penalized/stuck vehicles
            penalty_count = 0
            penalty_ids = []
            for vid, st in scheduled_times.items():
                if st.get('__PENALTY__') == math.inf:
                    penalty_count += 1
                    penalty_ids.append(vid)

            # Count overlapping occupancies per conflict point (pairwise overlaps)
            overlap_count = 0
            overlap_examples = []
            for p in tau_p_dict.keys():
                intervals = []
                for vid, st in scheduled_times.items():
                    if p in st and st[p] != math.inf:
                        arrival = st[p]
                        departure = arrival + tau_p_dict.get(p, config.tau)
                        intervals.append((arrival, departure, vid))
                intervals.sort(key=lambda x: x[0])
                for i in range(len(intervals)):
                    ai, di, vidi = intervals[i]
                    for j in range(i+1, len(intervals)):
                        aj, dj, vidj = intervals[j]
                        if aj < di:  # overlap detected
                            overlap_count += 1
                            if len(overlap_examples) < 5:
                                overlap_examples.append((p, vidi, vidj, ai, di, aj, dj))
                        else:
                            break

            print("\nCollision summary for best solution:")
            print(f"  Unique conflict points in scenario: {len(all_points)}")
            print(f"  Vehicles penalized/stuck: {penalty_count} {penalty_ids}")
            print(f"  Overlapping occupancy pairs detected: {overlap_count}")
            if overlap_examples:
                print("  Example overlaps (point, vid1, vid2, t1_start,t1_end,t2_start,t2_end):")
                for ex in overlap_examples:
                    print("   ", ex)
        except Exception as e:
            print(f"Could not compute collision statistics: {e}")
    return perm_best, speeds_best, obj_best, history, geom, tau_p_dict, iter_count
