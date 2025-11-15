"""Genetic Algorithm (GA) implementation for permutation-based scheduling.

This module provides a lightweight GA that works with the project's
decoder and objective functions. The GA represents individuals as a
permutation of Vehicle objects plus a speed list (one speed per vehicle
in the permutation). The fitness is the objective value 'f' computed by
the existing decoder+objective pipeline.

The implementation choices are intentionally simple and well documented
so you can extend them later (different crossover, mutation, selection
schemes, parallel evaluation, etc.).
"""
import copy
import math
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import config
from geometry import Geometry
from decoder import run_decoder
from objective import calculate_objective
from vehicle import Vehicle

POP_SIZE= 50
GENERATIONS= 100


def validate_speeds(permutation: List[Vehicle], speeds: List[float], geom: Geometry) -> List[float]:
    """Enforce C0 (no-catch-up) on a list of speeds for a given permutation.

    This mirrors the logic used in `sa.py` to keep behavior consistent.
    """
    v_new = list(speeds)
    speed_dict = {p.id: s for p, s in zip(permutation, v_new)}
    for queue in geom.entry_queues.values():
        if not queue:
            continue
        if queue[0].id not in speed_dict:
            continue
        last_speed = speed_dict[queue[0].id]
        for v_follower in queue[1:]:
            if v_follower.id not in speed_dict:
                continue
            follower_speed = speed_dict[v_follower.id]
            if follower_speed > last_speed:
                speed_dict[v_follower.id] = last_speed
            last_speed = speed_dict[v_follower.id]

    validated_speeds_list = [speed_dict[v.id] for v in permutation]
    return validated_speeds_list


def create_initial_population(pop_size: int, geom: Geometry) -> List[Tuple[List[Vehicle], List[float]]]:
    """Create an initial population of (permutation, speeds) individuals.

    Each permutation is a random shuffle of `config.pi`. Speeds are
    generated per-queue to respect the C0 constraint and then validated.
    """
    population = []
    v_min, v_max = config.velocity_range

    base_vehicles = list(config.pi)

    for _ in range(pop_size):
        perm = list(base_vehicles)
        random.shuffle(perm)

        # Generate speeds with queue-awareness
        speeds_dict: Dict[int, float] = {}
        for approach, queue in geom.entry_queues.items():
            if not queue:
                continue
            last_speed = random.uniform(v_min, v_max)
            if queue[0].id in [v.id for v in perm]:
                speeds_dict[queue[0].id] = last_speed
            for follower in queue[1:]:
                if follower.id not in [v.id for v in perm]:
                    continue
                current_max = min(v_max, last_speed)
                current_min = min(v_min, current_max)
                if current_min > current_max:
                    current_min = current_max
                new_speed = random.uniform(current_min, current_max + 1e-9)
                speeds_dict[follower.id] = new_speed
                last_speed = new_speed

        # Ensure every vehicle in the perm has a speed
        speeds_list = []
        for v in perm:
            if v.id not in speeds_dict:
                speeds_dict[v.id] = random.uniform(v_min, v_max)
            speeds_list.append(speeds_dict[v.id])

        # Enforce C0 using our validator
        speeds_list = validate_speeds(perm, speeds_list, geom)
        population.append((perm, speeds_list))

    return population


def davis_order_crossover(parent_a: List[Vehicle], parent_b: List[Vehicle]) -> List[Vehicle]:
    """Perform Davis Order Crossover (DOC) on vehicle permutations.

    Cuts a random subsequence from parent_a and inserts it into a random position,
    then fills remaining positions from parent_b (preserving relative order).
    
    Returns a single offspring permutation.
    """
    n = len(parent_a)
    if n < 2:
        return list(parent_a)

    # Cut a random block from parent_a
    i, j = sorted(random.sample(range(n), 2))
    block = parent_a[i:j + 1]
    
    # Remove block from parent_a to get remainder
    remainder = parent_a[:i] + parent_a[j + 1:]
    
    # Fill remainder with items from parent_b (in order, skipping duplicates in block)
    block_ids = {v.id for v in block}
    filled = []
    for v in parent_b:
        if v.id not in block_ids:
            filled.append(v)
    
    # If not enough items from parent_b, add remaining from parent_a
    filled_ids = {v.id for v in filled}
    for v in parent_a:
        if v.id not in block_ids and v.id not in filled_ids:
            filled.append(v)
            filled_ids.add(v.id)
    
    # Choose insertion position for the block in the filled list
    insert_pos = random.randint(0, len(filled))
    offspring = filled[:insert_pos] + block + filled[insert_pos:]
    
    return offspring


def swap_mutation(perm: List[Vehicle], mutation_rate: float) -> List[Vehicle]:
    """Perform swap mutation: randomly swap two vehicles in the permutation.

    With probability mutation_rate, randomly select two positions and swap them.
    """
    new_perm = list(perm)
    n = len(new_perm)
    if n < 2 or random.random() >= mutation_rate:
        return new_perm
    
    # Randomly select two distinct positions and swap them
    idx1, idx2 = random.sample(range(n), 2)
    new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
    return new_perm


def mutate_speeds(speeds: List[float], mutation_rate: float) -> List[float]:
    v_min, v_max = config.velocity_range
    new_speeds = list(speeds)
    for i in range(len(new_speeds)):
        if random.random() < mutation_rate:
            change = random.uniform(-1.5, 1.5)
            new_speeds[i] = max(v_min, min(v_max, new_speeds[i] + change))
    return new_speeds


def tournament_selection(population: List[Tuple[List[Vehicle], List[float]]],
                         fitnesses: List[float], k: int = 3) -> Tuple[List[Vehicle], List[float]]:
    """Select one individual using tournament selection."""
    selected_indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def evaluate_individual(permutation: List[Vehicle], speeds: List[float], geom: Geometry, tau_p_dict: dict):
    """Evaluate an individual by running decoder+objective and returning the objective dict."""
    try:
        decoder_results = run_decoder(permutation=permutation, speeds=speeds, geom=geom, tau_p_dict=tau_p_dict,
                                      return_full_schedule=False)
        obj = calculate_objective(decoder_results)
        return obj
    except Exception as e:
        # On error, return a heavy penalty objective
        penalized = {"delays": {v.id: 99999.0 for v in permutation}, "fem": 99999.0, "fall": 99999.0, "f": 1e12}
        return penalized


def plot_results(history: dict):
    """Plot GA history similarly to SA plots (2x2 grid).

    history keys expected: 'costs', 'avg_delays', 'total_delays', 'emergency_delays', 'generations', 'pop_size'
    """
    costs = history.get('costs', [])
    avg_delays = history.get('avg_delays', [])
    total_delays = history.get('total_delays', [])
    emergency_delays = history.get('emergency_delays', [])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GA Performance Metrics", fontsize=14, fontweight='bold')

    def simple_plot(ax, data, color, title, ylabel):
        ax.plot(data, '-', color=color)
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    simple_plot(axs[0, 0], costs, 'blue', 'Weighted Objective Cost (f)', 'Cost (f)')
    simple_plot(axs[0, 1], avg_delays, 'orange', 'Average Delay per Vehicle', 'Avg Delay (s)')
    simple_plot(axs[1, 0], total_delays, 'green', 'Total Delay (f_all)', 'Total Delay (s)')
    simple_plot(axs[1, 1], emergency_delays, 'red', 'Emergency Delay (f_em)', 'Emergency Delay (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Close the figure during batch/experiment runs to avoid blocking the process.
    # Use plt.show() interactively if you want to inspect the plots.
    plt.close()


def run_ga(pop_size: int = POP_SIZE,
           generations: int = GENERATIONS,
           pop_scale_with_generations: bool = False,
           crossover_rate: float = 0.8,
           mutation_rate: float = 0.10,
           elitism: int = 5,
           random_seed: int = None,
           return_history: bool = False) -> Tuple[List[Vehicle], List[float], dict]:
    """Run a GA and return best (perm, speeds, objective_dict).

    The function is intentionally conservative with defaults so it can be
    used interactively for testing. For serious experiments, increase sizes.
    """
    if random_seed is not None:
        random.seed(random_seed)

    print("--- Starting Genetic Algorithm (GA) ---")

    # Prepare geometry and tau dict using config.pi as the reference
    geom = Geometry()
    all_vehicles = config.pi
    geom.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom.set_trajectory(v)

    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    if not all_points:
        raise RuntimeError("No conflict points found in geometry; cannot run GA.")
    tau_p_dict = {p: config.tau for p in all_points}

    # Optionally scale population size based on generations
    if pop_scale_with_generations:
        # simple heuristic: pop_size = max(10, generations * 2)
        computed_pop = max(10, generations * 2)
        print(f"pop_scale_with_generations=True, setting pop_size={computed_pop} (generations={generations})")
        pop_size = computed_pop

    # Initialize population
    population = create_initial_population(pop_size, geom)

    # History tracking for plotting
    history = {'costs': [], 'avg_delays': [], 'total_delays': [], 'emergency_delays': [], 'generations': [], 'pop_size': pop_size}

    # Evaluate initial population
    fitnesses = []
    evaluated_objs = []
    for perm, speeds in population:
        obj = evaluate_individual(perm, speeds, geom, tau_p_dict)
        fitnesses.append(obj['f'])
        evaluated_objs.append(obj)

    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_perm, best_speeds = population[best_idx]
    best_obj = evaluated_objs[best_idx]

    print(f"Initial best f: {best_obj['f']:.2f}")
    # Record initial state as generation 0
    delays = best_obj.get('delays', {})
    avg_delay = sum(delays.values()) / len(delays) if delays else 0.0
    history['costs'].append(best_obj['f'])
    history['avg_delays'].append(avg_delay)
    history['total_delays'].append(best_obj.get('fall', 0))
    history['emergency_delays'].append(best_obj.get('fem', 0))
    history['generations'].append(0)

    # Main GA loop
    for gen in range(1, generations + 1):
        new_population = []
        new_objs = []
        new_fitnesses = []

        # Elitism: carry top `elitism` individuals forward
        sorted_idxs = sorted(range(len(population)), key=lambda i: fitnesses[i])
        elites = [population[i] for i in sorted_idxs[:elitism]]
        elite_objs = [evaluated_objs[i] for i in sorted_idxs[:elitism]]

        # Keep elites
        for e in elites:
            new_population.append(e)
        for eo in elite_objs:
            new_objs.append(eo); new_fitnesses.append(eo['f'])

        # Fill rest of new population
        while len(new_population) < pop_size:
            # Selection
            parent_a = tournament_selection(population, fitnesses)
            parent_b = tournament_selection(population, fitnesses)

            perm_a, speeds_a = parent_a
            perm_b, speeds_b = parent_b

            # Crossover (Davis Order Crossover)
            if random.random() < crossover_rate:
                child_perm = davis_order_crossover(perm_a, perm_b)
                # Child speeds: inherit aligned speeds from parent_a then mutate
                # Map by vehicle id so speeds align to the permutation order
                speed_map = {v.id: s for v, s in zip(perm_a, speeds_a)}
                child_speeds = [speed_map.get(v.id, random.uniform(*config.velocity_range)) for v in child_perm]
            else:
                child_perm = list(perm_a)
                child_speeds = list(speeds_a)

            # Mutation (Swap mutation for permutation)
            child_perm = swap_mutation(child_perm, mutation_rate)
            child_speeds = mutate_speeds(child_speeds, mutation_rate)

            # Validate C0
            child_speeds = validate_speeds(child_perm, child_speeds, geom)

            # Evaluate child
            child_obj = evaluate_individual(child_perm, child_speeds, geom, tau_p_dict)
            new_population.append((child_perm, child_speeds))
            new_objs.append(child_obj)
            new_fitnesses.append(child_obj['f'])

        # Replace population
        population = new_population
        evaluated_objs = new_objs
        fitnesses = new_fitnesses

        # Track best
        gen_best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_obj = evaluated_objs[gen_best_idx]
        if gen_best_obj['f'] < best_obj['f']:
            best_obj = gen_best_obj
            best_perm, best_speeds = population[gen_best_idx]
            print(f"Gen {gen}: New best f = {best_obj['f']:.2f}")

        # Record history for this generation (best so far)
        delays = best_obj.get('delays', {})
        avg_delay = sum(delays.values()) / len(delays) if delays else 0.0
        history['costs'].append(best_obj['f'])
        history['avg_delays'].append(avg_delay)
        history['total_delays'].append(best_obj.get('fall', 0))
        history['emergency_delays'].append(best_obj.get('fem', 0))
        history['generations'].append(gen)

        # Occasional progress print
        if gen % max(1, generations // 10) == 0:
            print(f"Generation {gen}/{generations} - current best f: {best_obj['f']:.2f}")

    print("--- GA Finished ---")
    print(f"Best Objective (f): {best_obj['f']:.2f}")
    print(f"Best Perm IDs: {[v.id for v in best_perm]}")

    # Plot GA history
    try:
        plot_results(history)
    except Exception:
        pass

    if return_history:
        return best_perm, best_speeds, best_obj, history
    else:
        return best_perm, best_speeds, best_obj


# if __name__ == '__main__':
#     run_ga(pop_size=20, generations=20, random_seed=1)
