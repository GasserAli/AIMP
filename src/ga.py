# File: src/ga.py
import math
import random
import copy
import matplotlib.pyplot as plt
import traceback
import numpy as np
from typing import Tuple

# --- Import Project Files ---
import config
from geometry import Geometry
from vehicle import Vehicle
# We only import what's needed from sa.py
from sa import evaluate_solution, validate_speeds

# =============================================================================
# GA PARAMETERS
# =============================================================================
POPULATION_SIZE = 50       # Number of solutions in each generation
NUM_GENERATIONS = 100      # Number of generations to run
ELITISM_RATE = 0.1         # Percentage of top solutions to carry over
TOURNAMENT_SIZE = 3        # Number of individuals to select for tournament
MUTATION_RATE_PERM = 0.1   # Probability of a permutation mutation (swap)
MUTATION_RATE_SPEED = 0.1  # Probability of a speed mutation (adjust)

# =============================================================================
# GA CORE COMPONENTS
# =============================================================================

def create_initial_solution(geom):
    """
    Generates a valid initial solution (permutation, speeds).
    Respects C0 constraint.
    """
    initial_perm = copy.deepcopy(config.pi)
    random.shuffle(initial_perm)
    
    initial_speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range

    for approach, queue in geom.entry_queues.items():
        if not queue: continue
        
        leader_in_queue = None
        for v in queue:
            if v.id in [p.id for p in initial_perm]:
                leader_in_queue = v
                break
        
        if leader_in_queue is None:
            continue 

        last_speed = random.uniform(v_min_global, v_max_global)
        initial_speeds_dict[leader_in_queue.id] = last_speed
        
        followers_in_queue = [v for v in queue if v.id != leader_in_queue.id and v.id in [p.id for p in initial_perm]]
        
        for v_follower in followers_in_queue:
             current_max = min(v_max_global, last_speed)
             current_min = min(v_min_global, current_max)
             if current_min > current_max: current_min = current_max
             new_speed = random.uniform(current_min, current_max + 1e-9)
             initial_speeds_dict[v_follower.id] = new_speed
             last_speed = new_speed

    initial_speeds_list = []
    for v in initial_perm:
        if v.id not in initial_speeds_dict:
             initial_speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
        initial_speeds_list.append(initial_speeds_dict[v.id])

    # print(f"  Initial Perm (IDs): {[v.id for v in initial_perm]}") # Keep this commented for cleaner logs
    return initial_perm, initial_speeds_list


class Individual:
    """Represents a single solution (chromosome) in the population."""
    def __init__(self, permutation: list[Vehicle], speeds: list[float]):
        self.permutation = permutation
        self.speeds = speeds
        # --- MODIFICATION: Store all cost components ---
        self.fitness = math.inf  # Main objective 'f'
        self.f = math.inf
        self.f_em = math.inf
        self.f_all = math.inf
        self.avg_delay = math.inf

    def calculate_fitness(self, geom, tau_p_dict):
        """Evaluates the solution and stores its fitness and all components."""
        obj_dict = evaluate_solution(self.permutation, self.speeds, geom, tau_p_dict)
        
        # --- MODIFICATION: Store all components ---
        self.f = obj_dict.get('f', math.inf)
        self.f_em = obj_dict.get('fem', math.inf)
        self.f_all = obj_dict.get('fall', math.inf)
        
        delays = obj_dict.get('delays', {})
        self.avg_delay = sum(delays.values()) / len(delays) if delays else 0.0
        
        # Fitness is the main objective 'f'
        self.fitness = self.f
        return self.f

def create_initial_population(size, geom) -> list[Individual]:
    """Creates a list of random, valid Individuals."""
    population = []
    for _ in range(size):
        perm, speeds = create_initial_solution(geom)
        population.append(Individual(perm, speeds))
    return population

def selection(population: list[Individual], num_to_select) -> list[Individual]:
    """Selects parents using k-tournament selection."""
    selected = []
    for _ in range(num_to_select):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        winner = min(tournament, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected

def crossover(parent1: Individual, parent2: Individual, geom) -> Tuple[Individual, Individual]:
    """Performs crossover on permutation (OX1) and speeds (average)."""
    
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
    
    p1_speeds, p2_speeds = parent1.speeds, parent2.speeds
    child1_speeds = [(s1 + s2) / 2.0 for s1, s2 in zip(p1_speeds, p2_speeds)]
    child2_speeds = [(s1 + s2) / 2.0 for s1, s2 in zip(p2_speeds, p1_speeds)]
    
    child1_speeds = validate_speeds(child1_perm, child1_speeds, geom)
    child2_speeds = validate_speeds(child2_perm, child2_speeds, geom)

    return Individual(child1_perm, child1_speeds), Individual(child2_perm, child2_speeds)

def mutate(individual: Individual, geom):
    """Applies mutation to permutation (swap) and speeds (adjust)."""
    
    if random.random() < MUTATION_RATE_PERM:
        idx1, idx2 = random.sample(range(len(individual.permutation)), 2)
        individual.permutation[idx1], individual.permutation[idx2] = \
            individual.permutation[idx2], individual.permutation[idx1]

    if random.random() < MUTATION_RATE_SPEED:
        idx = random.randrange(len(individual.speeds))
        change = random.uniform(-2.0, 2.0)
        
        v_min, v_max = config.velocity_range
        individual.speeds[idx] = max(v_min, min(individual.speeds[idx] + change, v_max))
        
        individual.speeds = validate_speeds(individual.permutation, individual.speeds, geom)
        
    return individual

# =============================================================================
# MAIN GA FUNCTION
# =============================================================================

def run_ga(max_evaluations=None):
    """Main Genetic Algorithm (GA) loop."""
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

    print(f"Creating initial population (Size: {POPULATION_SIZE})...")
    population = create_initial_population(POPULATION_SIZE, geom)
    
    print("Evaluating initial population...")
    for ind in population:
        ind.calculate_fitness(geom, tau_p_dict)
    
    eval_count = POPULATION_SIZE
    
    # --- MODIFICATION: Expanded history tracking ---
    history = {
        'best_f': [], 'avg_f': [], 'best_fem': [],
        'best_fall': [], 'best_avg_delay': []
    }
    
    best_solution = min(population, key=lambda ind: ind.fitness)
    best_fitness = best_solution.fitness
    
    # Populate history for generation 0
    history['best_f'].append(best_solution.f)
    history['best_fem'].append(best_solution.f_em)
    history['best_fall'].append(best_solution.f_all)
    history['best_avg_delay'].append(best_solution.avg_delay)
    history['avg_f'].append(np.mean([ind.f for ind in population]))

    print(f"Initial Best Fitness (f): {best_fitness:.2f}")

    num_elitism = int(POPULATION_SIZE * ELITISM_RATE)
    
    if max_evaluations is not None:
        evals_per_gen = POPULATION_SIZE - num_elitism
        if evals_per_gen <= 0: evals_per_gen = 1
        num_generations = (max_evaluations - POPULATION_SIZE) // evals_per_gen
        print(f"Running for {num_generations} generations based on evaluation budget.")
    else:
        num_generations = NUM_GENERATIONS
        print(f"Running for {num_generations} generations.")


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
        if current_best.fitness < best_fitness:
            best_solution = copy.deepcopy(current_best)
            best_fitness = best_solution.fitness
            new_best_found = True

        # --- MODIFICATION: Update full history ---
        history['best_f'].append(best_solution.f)
        history['best_fem'].append(best_solution.f_em)
        history['best_fall'].append(best_solution.f_all)
        history['best_avg_delay'].append(best_solution.avg_delay)
        history['avg_f'].append(np.mean([ind.f for ind in population]))
        
        avg_fitness_current = history['avg_f'][-1]

        # Print update every generation
        if new_best_found:
            print(f"  Gen {gen+1}: * NEW BEST: {best_fitness:.2f} (Avg: {avg_fitness_current:.2f})")
        else:
            print(f"  Gen {gen+1}:   Best: {best_fitness:.2f} (Avg: {avg_fitness_current:.2f})")

        if max_evaluations is not None and eval_count >= max_evaluations:
            print(f"Termination: Reached evaluation limit ({eval_count}).")
            break

    print("\n--- GA Finished ---")
    print(f"Total generations: {gen + 1}")
    print(f"Total evaluations: {eval_count}")
    print(f"Best Objective (f): {best_fitness:.2f}")
    
    # Get the final objective dictionary for the bar plot
    best_solution_obj_dict = {
        "f": best_solution.f,
        "fem": best_solution.f_em,
        "fall": best_solution.f_all,
        "delays": {} # Not needed for plot, but good to have
    }

    return (best_solution.permutation, best_solution.speeds, best_fitness, 
            history, geom, tau_p_dict, best_solution_obj_dict, eval_count)

# =============================================================================
# PLOTTING
# =============================================================================

def plot_ga_performance_dashboard(history_data):
    """
    --- NEW 2x2 PLOT ---
    Create a 2x2 grid of GA performance plots.
    """
    if not history_data:
        print("No history to plot.")
        return

    generations = range(len(history_data['best_f']))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GA Performance Dashboard", fontsize=14, fontweight='bold')

    # --- 1. Best vs. Avg Objective Cost (f) ---
    axs[0, 0].plot(generations, history_data['best_f'], 'b-', label='Best Fitness (f)')
    axs[0, 0].plot(generations, history_data['avg_f'], 'r--', label='Average Fitness (f)')
    axs[0, 0].set_title('Best vs. Average Objective Cost (f)')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Cost (f)')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True)

    # --- 2. Best Solution's Average Delay ---
    axs[0, 1].plot(generations, history_data['best_avg_delay'], 'g-', label='Best Sol. Avg Delay')
    axs[0, 1].set_title('Average Delay of Best Solution')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Avg Delay (s)')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True)

    # --- 3. Best Solution's Total Delay (f_all) ---
    axs[1, 0].plot(generations, history_data['best_fall'], 'k-', label='Best Sol. Total Delay (f_all)')
    axs[1, 0].set_title('Total Delay of Best Solution')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('Total Delay (s)')
    axs[1, 0].legend(loc='upper left')
    axs[1, 0].grid(True)

    # --- 4. Best Solution's Emergency Delay (f_em) ---
    axs[1, 1].plot(generations, history_data['best_fem'], 'm-', label='Best Sol. Emergency Delay (f_em)')
    axs[1, 1].set_title('Emergency Delay of Best Solution')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Emergency Delay (s)')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()