# File: src/ga.py  (UPDATED)
# - Changes:
#   * Use speed penalty (low speed => penalty) so optimizer finds balanced speeds
#   * Bias leader initial speeds upward (to avoid collapsed start)
#   * Improve crossover/mutation for speeds while validating C0 no-catch-up constraint
#   * Old (original) lines are commented out and replaced with new logic
#   * Keep compatibility with evaluate_solution(...) from sa.py which expects (permutation, speeds, geom, tau_p_dict)

import math
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import traceback
import numpy as np
from typing import Tuple

# --- Import Project Files ---
import config
from geometry import Geometry
from vehicle import Vehicle
# The GA uses helper functions from sa.py (evaluate_solution & validate_speeds)
from sa import evaluate_solution, validate_speeds

# =============================================================================
# GA PARAMETERS
# =============================================================================
POPULATION_SIZE = 150
NUM_GENERATIONS = 200
ELITISM_RATE = 0.1
TOURNAMENT_SIZE = 80
MUTATION_RATE_PERM = 0.5
MUTATION_RATE_SPEED = 0.2

# --- MODIFICATION: Added early stopping patience ---
CONVERGENCE_PATIENCE = 25 # Stop if no improvement after 25 generations

# =============================================================================
# GA VISUALIZER CLASS
# =============================================================================
class GAVisualizer:
    """
    Handles real-time visualization of the Genetic Algorithm.
    """
    def __init__(self):
        plt.ion() # Turn on interactive mode
        self.fig, (self.ax_line, self.ax_bar) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-Time Genetic Algorithm Evolution', fontsize=14, fontweight='bold')
        
        # Setup Line Plot (Convergence)
        self.ax_line.set_title("Convergence: Best vs Average Cost")
        self.ax_line.set_xlabel("Generation")
        self.ax_line.set_ylabel("Cost (f)")
        self.ax_line.grid(True)
        self.line_best, = self.ax_line.plot([], [], 'b-', label='Best Fitness', linewidth=2)
        self.line_avg, = self.ax_line.plot([], [], 'r--', label='Avg Fitness', linewidth=1.5)
        self.ax_line.legend(loc='upper right')

        # Setup Bar Plot (Population Fitness)
        self.ax_bar.set_title(f"Current Generation Population (Size: {POPULATION_SIZE})")
        self.ax_bar.set_xlabel("Individual Index")
        self.ax_bar.set_ylabel("Cost (f) - Green is Good")
        
        # Initialize bars
        self.indices = range(POPULATION_SIZE)
        # Placeholder data
        self.bars = self.ax_bar.bar(self.indices, [0]*POPULATION_SIZE)
        
        # Setup Colormap (Green=Low/Good, Red=High/Bad)
        self.cmap = plt.get_cmap('RdYlGn_r') # Reversed: Low=Green, High=Red

        plt.tight_layout()

    def update(self, gen, population, history):
        """Updates the plots with new generation data."""
        if not plt.fignum_exists(self.fig.number): return # Stop if window closed

        # 1. Update Line Plot
        gens = range(len(history['best_f']))
        self.line_best.set_data(gens, history['best_f'])
        self.line_avg.set_data(gens, history['avg_f'])
        
        # Rescale view
        self.ax_line.relim()
        self.ax_line.autoscale_view()

        # 2. Update Bar Plot
        fitness_values = [ind.fitness for ind in population]
        
        # Normalize colors based on current generation's range
        v_min = min(fitness_values)
        v_max = max(fitness_values)
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

        for bar, fitness in zip(self.bars, fitness_values):
            bar.set_height(fitness)
            bar.set_color(self.cmap(norm(fitness)))
        
        # Dynamic Y-limit for bars with some headroom
        self.ax_bar.set_ylim(v_min * 0.95, v_max * 1.05)
        self.ax_bar.set_title(f"Gen {gen}: Population Fitness (Best: {v_min:.2f})")

        # 3. Render
        plt.pause(0.05) # Small pause to allow GUI update

    def close(self):
        plt.ioff() # Turn off interactive mode
        # plt.close(self.fig)


# =============================================================================
# GA CORE COMPONENTS
# =============================================================================

def create_initial_solution(geom):
    """
    Create a single initial (permutation, speeds) pair.
    Changes from original:
      - Leader speeds are biased upward slightly to avoid entire lane starting at v_min.
      - Followers are sampled <= leader (C0 constraint still enforced).
    """
    initial_perm = copy.deepcopy(config.pi)
    random.shuffle(initial_perm)
    
    initial_speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range

    for approach, queue in geom.entry_queues.items():
        if not queue: 
            continue

        leader_in_queue = None
        for v in queue:
            if v.id in [p.id for p in initial_perm]:
                leader_in_queue = v
                break
        if leader_in_queue is None: 
            continue 

        # -----------------------------
        # OLD (original) leader init:
        # last_speed = random.uniform(v_min_global, v_max_global)
        # -----------------------------
        # NEW: bias leader speeds upward (heuristic)
        last_speed = random.uniform(v_min_global + 0.25*(v_max_global - v_min_global), v_max_global)
        initial_speeds_dict[leader_in_queue.id] = last_speed
        
        followers_in_queue = [v for v in queue if v.id != leader_in_queue.id and v.id in [p.id for p in initial_perm]]
        for v_follower in followers_in_queue:
            # keep follower <= predecessor (C0)
            current_max = min(v_max_global, last_speed)
            # OLD follower logic (kept for reference)
            # current_min = min(v_min_global, current_max)
            # if current_min > current_max: current_min = current_max
            # new_speed = random.uniform(current_min, current_max + 1e-9)

            # NEW: sample followers within [v_min, current_max]
            new_speed = random.uniform(v_min_global, current_max)
            initial_speeds_dict[v_follower.id] = new_speed
            last_speed = new_speed

    initial_speeds_list = []
    for v in initial_perm:
        if v.id not in initial_speeds_dict:
            initial_speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
        initial_speeds_list.append(initial_speeds_dict[v.id])

    return initial_perm, initial_speeds_list


class Individual:
    def __init__(self, permutation: list[Vehicle], speeds: list[float]):
        self.permutation = permutation
        self.speeds = speeds
        self.fitness = math.inf
        self.f = math.inf
        self.f_em = math.inf
        self.f_all = math.inf
        self.avg_delay = math.inf

    def calculate_fitness(self, geom, tau_p_dict):
        # NOTE: evaluate_solution expects speeds to be passed to objective internally.
        obj_dict = evaluate_solution(self.permutation, self.speeds, geom, tau_p_dict)
        self.f = obj_dict.get('f', math.inf)
        self.f_em = obj_dict.get('fem', math.inf)
        self.f_all = obj_dict.get('fall', math.inf)
        delays = obj_dict.get('delays', {})
        self.avg_delay = sum(delays.values()) / len(delays) if delays else 0.0
        self.fitness = self.f
        return self.f

def create_initial_population(size, geom) -> list[Individual]:
    population = []
    for _ in range(size):
        perm, speeds = create_initial_solution(geom)
        population.append(Individual(perm, speeds))
    return population

def selection(population: list[Individual], num_to_select) -> list[Individual]:
    selected = []
    for _ in range(num_to_select):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        winner = min(tournament, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected

def crossover(parent1: Individual, parent2: Individual, geom) -> Tuple[Individual, Individual]:
    """
    Permutation crossover (order crossover) is preserved.
    For speeds, instead of copying one parent's speeds, we blend them (average)
    and then enforce validate_speeds to ensure C0 constraint.
    """
    size = len(parent1.permutation)
    p1_perm, p2_perm = parent1.permutation, parent2.permutation
    v_map = {v.id: v for v in p1_perm}
    p1_ids = [v.id for v in p1_perm]
    p2_ids = [v.id for v in p2_perm]

    start, end = sorted(random.sample(range(size), 2))
    child1_ids = [None] * size
    child2_ids = [None] * size

    child1_ids[start:end] = p1_ids[start:end]
    child2_ids[start:end] = p2_ids[start:end]

    p2_idx = end
    c1_idx = end
    while None in child1_ids:
        if p2_ids[p2_idx % size] not in child1_ids:
            child1_ids[c1_idx % size] = p2_ids[p2_idx % size]
            c1_idx += 1
        p2_idx += 1

    p1_idx = end
    c2_idx = end
    while None in child2_ids:
        if p1_ids[p1_idx % size] not in child2_ids:
            child2_ids[c2_idx % size] = p1_ids[p1_idx % size]
            c2_idx += 1
        p1_idx += 1

    child1_perm = [v_map[vid] for vid in child1_ids]
    child2_perm = [v_map[vid] for vid in child2_ids]
    
    # --- OLD speed crossover (original):
    # p1_speeds, p2_speeds = parent1.speeds, parent2.speeds
    # child1_speeds = [(s1 + s2) / 2.0 for s1, s2 in zip(p1_speeds, p2_speeds)]
    # child2_speeds = [(s1 + s2) / 2.0 for s1, s2 in zip(p2_speeds, p1_speeds)]
    #
    # Reason: simple averaging allowed speeds to trend to extremes under speed reward.
    # -------------------------------------------------

    # NEW: Blend speeds with small controlled randomness to avoid runaway to max
    p1_speeds, p2_speeds = parent1.speeds, parent2.speeds
    child1_speeds = []
    child2_speeds = []
    v_min, v_max = config.velocity_range
    for s1, s2 in zip(p1_speeds, p2_speeds):
        # weighted average with slight noise
        w = random.uniform(0.4, 0.6)
        base1 = w * s1 + (1-w) * s2
        base2 = w * s2 + (1-w) * s1
        # small jitter to keep diversity, but not too big to explode speeds
        jitter1 = random.uniform(-0.25, 0.25)
        jitter2 = random.uniform(-0.25, 0.25)
        child1_speeds.append(max(v_min, min(v_max, base1 + jitter1)))
        child2_speeds.append(max(v_min, min(v_max, base2 + jitter2)))

    # Enforce C0 no-catch-up constraint using validate_speeds
    child1_speeds = validate_speeds(child1_perm, child1_speeds, geom)
    child2_speeds = validate_speeds(child2_perm, child2_speeds, geom)

    return Individual(child1_perm, child1_speeds), Individual(child2_perm, child2_speeds)

def mutate(individual: Individual, geom):
    """
    Mutation does:
      - permutation swap (with probability MUTATION_RATE_PERM)
      - speed mutation (with probability MUTATION_RATE_SPEED)
    For speed mutation we:
      - with high prob do a small local adjustment
      - with some small prob reassign a random speed (explore)
    Finally, validate speeds to enforce C0.
    """
    if random.random() < MUTATION_RATE_PERM:
        idx1, idx2 = random.sample(range(len(individual.permutation)), 2)
        # --- OLD:
        # individual.permutation[idx1], individual.permutation[idx2] = \
        #     individual.permutation[idx2], individual.permutation[idx1]
        # ------------------------------------------------------------
        # NEW: swap permutation AND also swap speeds (to keep vehicle-speed pairing)
        individual.permutation[idx1], individual.permutation[idx2] = \
            individual.permutation[idx2], individual.permutation[idx1]
        individual.speeds[idx1], individual.speeds[idx2] = \
            individual.speeds[idx2], individual.speeds[idx1]

    if random.random() < MUTATION_RATE_SPEED:
        idx = random.randrange(len(individual.speeds))
        v_min, v_max = config.velocity_range
        # OLD simple mutation:
        # change = random.uniform(-2.0, 2.0)
        # individual.speeds[idx] = max(v_min, min(individual.speeds[idx] + change, v_max))
        # ------------------------------------------------------------
        # NEW: two-level mutation for fairness
        if random.random() < 0.8:
            # smaller local mutation to allow fine tuning
            change = random.uniform(-1.0, 1.0)
            individual.speeds[idx] = max(v_min, min(individual.speeds[idx] + change, v_max))
        else:
            # exploratory reassign
            individual.speeds[idx] = random.uniform(v_min, v_max)

        # enforce safety constraint after mutation
        individual.speeds = validate_speeds(individual.permutation, individual.speeds, geom)
        
    return individual

# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def run_ga(max_evaluations=None, initial_population=None, verbose=True, visualize_realtime=False):
    """Main Genetic Algorithm (GA) loop."""
    
    visualizer = None
    if visualize_realtime:
        if verbose: print("Initializing Real-Time GA Visualizer...")
        visualizer = GAVisualizer()

    if verbose:
        print("--- Starting Genetic Algorithm ---")
        print("Initializing geometry and parameters...")
        
    geom = Geometry()
    all_vehicles = config.pi
    geom.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom.set_trajectory(v)
    
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    if not all_points:
        print("Error: No vehicles or no paths found. Exiting.")
        return [], [], 0.0, {}, None, None, {}, 0

    tau_p_dict = {p: config.tau for p in all_points}

    if initial_population:
        if verbose: print("Using provided initial population.")
        population = copy.deepcopy(initial_population)
    else:
        population = create_initial_population(POPULATION_SIZE, geom)

    if verbose: print("Evaluating initial population...")
    for ind in population:
        if ind.fitness == math.inf:
            ind.calculate_fitness(geom, tau_p_dict)
    
    eval_count = POPULATION_SIZE
    
    history = {
        'best_f': [], 'avg_f': [], 'best_fem': [],
        'best_fall': [], 'best_avg_delay': []
    }
    
    best_solution = min(population, key=lambda ind: ind.fitness)
    best_fitness = best_solution.fitness
    
    # --- MODIFICATION: Add tracker variables for convergence ---
    last_best_fitness_for_stalling = best_fitness
    generations_without_improvement = 0
    # --- End Modification ---

    history['best_f'].append(best_solution.f)
    history['best_fem'].append(best_solution.f_em)
    history['best_fall'].append(best_solution.f_all)
    history['best_avg_delay'].append(best_solution.avg_delay)
    history['avg_f'].append(np.mean([ind.f for ind in population]))

    if visualizer:
        visualizer.update(0, population, history)

    if verbose: print(f"Initial Best Fitness (f): {best_fitness:.2f}")

    num_elitism = int(POPULATION_SIZE * ELITISM_RATE)
    
    if max_evaluations is not None:
        evals_per_gen = POPULATION_SIZE - num_elitism
        if evals_per_gen <= 0: evals_per_gen = 1
        num_generations = (max_evaluations - POPULATION_SIZE) // evals_per_gen
        if verbose: print(f"Running for {num_generations} generations based on evaluation budget.")
    else:
        num_generations = NUM_GENERATIONS
        if verbose: print(f"Running for {num_generations} generations (or until convergence).")


    for gen in range(num_generations):
        
        population.sort(key=lambda ind: ind.fitness)
        new_population = population[:num_elitism]
        
        num_parents = POPULATION_SIZE - num_elitism
        parents = selection(population, num_parents)
        
        for i in range(0, num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if (i+1) < len(parents) else parents[0] 
            
            child1, child2 = crossover(parent1, parent2, geom)
            
            new_population.append(mutate(child1, geom))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2, geom))

        population = new_population
        
        evals_this_gen = 0
        for ind in population:
            if ind.fitness == math.inf:
                ind.calculate_fitness(geom, tau_p_dict)
                evals_this_gen += 1
                
        eval_count += evals_this_gen

        current_best = min(population, key=lambda ind: ind.fitness)
        new_best_found = False
        
        # Check if the *all-time best* solution was improved
        if current_best.fitness < best_fitness:
            best_solution = copy.deepcopy(current_best)
            best_fitness = best_solution.fitness
            new_best_found = True

        history['best_f'].append(best_solution.f)
        history['best_fem'].append(best_solution.f_em)
        history['best_fall'].append(best_solution.f_all)
        history['best_avg_delay'].append(best_solution.avg_delay)
        history['avg_f'].append(np.mean([ind.f for ind in population]))
        
        avg_fitness_current = history['avg_f'][-1]

        if visualizer:
            visualizer.update(gen + 1, population, history)

        if verbose:
            if new_best_found:
                print(f"  Gen {gen+1}: * NEW BEST: {best_fitness:.2f} (Avg: {avg_fitness_current:.2f})")
            else:
                print(f"  Gen {gen+1}:   Best: {best_fitness:.2f} (Avg: {avg_fitness_current:.2f})")

        # --- MODIFICATION: Add convergence check ---
        if best_fitness < last_best_fitness_for_stalling:
            last_best_fitness_for_stalling = best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if max_evaluations is None and generations_without_improvement >= CONVERGENCE_PATIENCE:
            if verbose:
                print("\n--- STOPPING EARLY (Convergence) ---")
                print(f"No improvement in {CONVERGENCE_PATIENCE} generations.")
                print(f"Stopping at generation {gen + 1}.")
            break
        # --- End Modification ---

        if max_evaluations is not None and eval_count >= max_evaluations:
            if verbose: print(f"Termination: Reached evaluation limit ({eval_count}).")
            break
            
    if verbose:
        print("\n--- GA Finished ---")
        print(f"Total generations: {gen + 1}")
        print(f"Total evaluations: {eval_count}")
        print(f"Best Objective (f): {best_fitness:.2f}")
        # Print vehicle speeds for the best solution
        try:
            perm_ids = [v.id for v in best_solution.permutation]
            speeds_str = [f"{s:.3f}" for s in best_solution.speeds]
            print("Final permutation (vehicle IDs):", perm_ids)
            print("Final speeds (m/s):", speeds_str)
            print("Final speeds per vehicle:")
            for vid, sp in zip(perm_ids, speeds_str):
                print("  ", f"ID {vid}: {sp} m/s")
        except Exception:
            print("Could not print detailed GA speeds (unexpected error).")
    
    best_solution_obj_dict = {
        "f": best_solution.f,
        "fem": best_solution.f_em,
        "fall": best_solution.f_all,
        "delays": {} 
    }

    if visualizer:
        print("Closing real-time visualization...")
        visualizer.close()

    return (best_solution.permutation, best_solution.speeds, best_fitness, 
            history, geom, tau_p_dict, best_solution_obj_dict, eval_count)

# =============================================================================
# PLOTTING
# =============================================================================

def plot_ga_performance_dashboard(history_data):
    """
    Create a 2x2 grid of GA performance plots (Final Summary).
    """
    if not history_data:
        print("No history to plot.")
        return

    generations = range(len(history_data['best_f']))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GA Performance Dashboard (Final Results)", fontsize=14, fontweight='bold')

    axs[0, 0].plot(generations, history_data['best_f'], 'b-', label='Best Fitness (f)')
    axs[0, 0].plot(generations, history_data['avg_f'], 'r--', label='Average Fitness (f)')
    axs[0, 0].set_title('Best vs. Average Objective Cost (f)')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Cost (f)')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True)

    axs[0, 1].plot(generations, history_data['best_avg_delay'], 'g-', label='Best Sol. Avg Delay')
    axs[0, 1].set_title('Average Delay of Best Solution')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Avg Delay (s)')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True)

    axs[1, 0].plot(generations, history_data['best_fall'], 'k-', label='Best Sol. Total Delay (f_all)')
    axs[1, 0].set_title('Total Delay of Best Solution')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('Total Delay (s)')
    axs[1, 0].legend(loc='upper left')
    axs[1, 0].grid(True)

    axs[1, 1].plot(generations, history_data['best_fem'], 'm-', label='Best Sol. Emergency Delay (f_em)')
    axs[1, 1].set_title('Emergency Delay of Best Solution')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Emergency Delay (s)')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
