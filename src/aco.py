# File: src/aco.py
"""
Ant Colony Optimization (ACO) - Ant System Implementation
for Intersection Traffic Optimization

This module implements the classic Ant System (AS) variant of ACO
to optimize vehicle permutation and speed assignment.
"""

import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import traceback
import csv
import os
from datetime import datetime
from typing import List, Dict, Tuple

# --- Import Project Files ---
import config
from geometry import Geometry
from vehicle import Vehicle
from sa import evaluate_solution, validate_speeds

# =============================================================================
# ACO PARAMETERS (Ant System)
# =============================================================================
NUM_ANTS = 50                    # m: Number of ants per iteration
NUM_ITERATIONS = 100             # Number of ACO iterations
ALPHA = 1.0                      # α: Pheromone importance
BETA = 2.0                       # β: Heuristic importance  
RHO = 0.3                        # ρ: Evaporation rate (0 < ρ < 1)
Q = 100.0                        # Q: Pheromone deposit constant
TAU_INITIAL = 0.1                # Initial pheromone level
ELITIST_WEIGHT = 2.0             # Weight for best-so-far solution (elitist AS)

# Early stopping
CONVERGENCE_PATIENCE = 20        # Stop if no improvement for this many iterations

# =============================================================================
# ACO VISUALIZER CLASS
# =============================================================================
class ACOVisualizer:
    """Handles real-time visualization of ACO progress."""
    
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_conv, self.ax_pher) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-Time Ant Colony Optimization', fontsize=14, fontweight='bold')
        
        # Convergence plot
        self.ax_conv.set_title("Convergence: Best vs Iteration Average")
        self.ax_conv.set_xlabel("Iteration")
        self.ax_conv.set_ylabel("Cost (f)")
        self.ax_conv.grid(True)
        self.line_best, = self.ax_conv.plot([], [], 'b-', label='Best Cost', linewidth=2)
        self.line_iter_best, = self.ax_conv.plot([], [], 'g--', label='Iteration Best', linewidth=1.5)
        self.ax_conv.legend(loc='upper right')
        
        # Pheromone statistics plot
        self.ax_pher.set_title("Pheromone Statistics")
        self.ax_pher.set_xlabel("Iteration")
        self.ax_pher.set_ylabel("Pheromone Level")
        self.ax_pher.grid(True)
        self.line_pher_max, = self.ax_pher.plot([], [], 'r-', label='Max Pheromone', linewidth=1.5)
        self.line_pher_avg, = self.ax_pher.plot([], [], 'orange', label='Avg Pheromone', linewidth=1.5)
        self.line_pher_min, = self.ax_pher.plot([], [], 'brown', label='Min Pheromone', linewidth=1.5)
        self.ax_pher.legend(loc='upper right')
        
        plt.tight_layout()
    
    def update(self, iteration, history):
        """Updates visualization with current iteration data."""
        if not plt.fignum_exists(self.fig.number):
            return
        
        iters = range(len(history['best_f']))
        
        # Update convergence plot
        self.line_best.set_data(iters, history['best_f'])
        self.line_iter_best.set_data(iters, history['iter_best_f'])
        self.ax_conv.relim()
        self.ax_conv.autoscale_view()
        
        # Update pheromone plot
        self.line_pher_max.set_data(iters, history['pher_max'])
        self.line_pher_avg.set_data(iters, history['pher_avg'])
        self.line_pher_min.set_data(iters, history['pher_min'])
        self.ax_pher.relim()
        self.ax_pher.autoscale_view()
        
        plt.pause(0.05)
    
    def close(self):
        """Closes the visualization window."""
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        plt.ioff()


# =============================================================================
# ANT CLASS
# =============================================================================
class Ant:
    """Represents a single ant that constructs a solution."""
    
    def __init__(self, ant_id):
        self.ant_id = ant_id
        self.permutation = []     # Vehicle permutation (list of Vehicle objects)
        self.speeds = []          # Speed assignment (list of floats)
        self.fitness = math.inf   # Objective value (lower is better)
        self.visited = set()      # Track visited vehicles
    
    def reset(self):
        """Resets ant for new solution construction."""
        self.permutation = []
        self.speeds = []
        self.fitness = math.inf
        self.visited = set()


# =============================================================================
# ACO GRAPH CLASS
# =============================================================================
class ACOGraph:
    """
    Represents the pheromone graph for ACO.
    
    For permutation problems, we use a fully connected graph where:
    - Nodes represent positions in the permutation
    - Edges (i, j) represent placing vehicle j at position i
    """
    
    def __init__(self, vehicles: List[Vehicle]):
        self.vehicles = vehicles
        self.n = len(vehicles)
        self.vehicle_ids = [v.id for v in vehicles]
        
        # Pheromone matrix: tau[position][vehicle_id]
        # tau[i][j] = pheromone for placing vehicle j at position i
        self.tau = np.ones((self.n, self.n)) * TAU_INITIAL
        
        # Heuristic matrix: eta[position][vehicle_id]
        # Based on vehicle priority (emergency vehicles preferred early)
        self.eta = self._init_heuristic()
    
    def _init_heuristic(self) -> np.ndarray:
        """
        Initialize heuristic information.
        
        Emergency vehicles get higher heuristic values for earlier positions.
        This encourages ants to place emergency vehicles early in permutation.
        """
        eta = np.ones((self.n, self.n))
        
        for pos in range(self.n):
            for v_idx, vehicle in enumerate(self.vehicles):
                if vehicle.priority_status:  # Emergency vehicle
                    # Higher value for earlier positions
                    eta[pos][v_idx] = 2.0 * (1.0 - pos / self.n)
                else:
                    # Normal vehicles have uniform moderate heuristic
                    eta[pos][v_idx] = 1.0
        
        return eta
    
    def get_pheromone_stats(self) -> Tuple[float, float, float]:
        """Returns (max, avg, min) pheromone levels."""
        return float(np.max(self.tau)), float(np.mean(self.tau)), float(np.min(self.tau))
    
    def evaporate_pheromone(self, rho: float):
        """
        Apply pheromone evaporation to all edges.
        
        τ_ij ← (1 - ρ) * τ_ij
        """
        self.tau *= (1.0 - rho)
        # Prevent pheromone from becoming too small
        self.tau = np.maximum(self.tau, TAU_INITIAL * 0.01)
    
    def deposit_pheromone(self, ant: Ant, delta_tau: float):
        """
        Deposit pheromone along the path taken by an ant.
        
        For Ant System: Δτ_ij = Q / L_k
        where L_k is the tour length (fitness) of ant k
        """
        for position, vehicle in enumerate(ant.permutation):
            v_idx = self.vehicle_ids.index(vehicle.id)
            self.tau[position][v_idx] += delta_tau


# =============================================================================
# SOLUTION CONSTRUCTION
# =============================================================================
def construct_ant_solution(ant: Ant, graph: ACOGraph, geom: Geometry, 
                          alpha: float, beta: float, tau_p_dict: Dict) -> None:
    """
    Constructs a complete solution for one ant using probabilistic selection.
    
    The ant builds a permutation by selecting vehicles sequentially based on:
    - Pheromone levels (τ)
    - Heuristic information (η)
    
    Selection probability: p_ij = (τ_ij^α * η_ij^β) / Σ(τ_il^α * η_il^β)
    """
    ant.reset()
    available_vehicles = list(graph.vehicles)
    
    v_min_global, v_max_global = config.velocity_range
    
    # Build permutation position by position
    for position in range(graph.n):
        # Calculate selection probabilities
        probs = []
        
        for vehicle in available_vehicles:
            v_idx = graph.vehicle_ids.index(vehicle.id)
            
            # Pheromone factor
            tau_val = graph.tau[position][v_idx]
            # Heuristic factor
            eta_val = graph.eta[position][v_idx]
            
            # Combined attractiveness
            attractiveness = (tau_val ** alpha) * (eta_val ** beta)
            probs.append(attractiveness)
        
        # Normalize to probabilities
        prob_sum = sum(probs)
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]
        else:
            # Fallback: uniform random
            probs = [1.0 / len(available_vehicles)] * len(available_vehicles)
        
        # Select vehicle using roulette wheel selection
        selected_vehicle = random.choices(available_vehicles, weights=probs, k=1)[0]
        
        ant.permutation.append(selected_vehicle)
        ant.visited.add(selected_vehicle.id)
        available_vehicles.remove(selected_vehicle)
    
    # Generate speeds respecting C0 constraint (using SA's approach)
    ant.speeds = _generate_speeds_for_permutation(ant.permutation, geom)
    
    # Validate and evaluate
    ant.speeds = validate_speeds(ant.permutation, ant.speeds, geom)
    obj_dict = evaluate_solution(ant.permutation, ant.speeds, geom, tau_p_dict)
    ant.fitness = obj_dict['f']


def _generate_speeds_for_permutation(permutation: List[Vehicle], geom: Geometry) -> List[float]:
    """
    Generate speed assignments respecting the C0 (no-catch-up) constraint.
    Similar to SA's initial solution generation.
    """
    speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range
    
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue
        
        # Find leader in this queue
        leader_in_queue = None
        for v in queue:
            if v.id in [p.id for p in permutation]:
                leader_in_queue = v
                break
        
        if leader_in_queue is None:
            continue
        
        # Assign random speed to leader
        last_speed = random.uniform(v_min_global, v_max_global)
        speeds_dict[leader_in_queue.id] = last_speed
        
        # Assign speeds to followers (must be ≤ leader's speed)
        followers_in_queue = [v for v in queue 
                             if v.id != leader_in_queue.id and v.id in [p.id for p in permutation]]
        
        for v_follower in followers_in_queue:
            current_max = min(v_max_global, last_speed)
            current_min = min(v_min_global, current_max)
            if current_min > current_max:
                current_min = current_max
            
            new_speed = random.uniform(current_min, current_max + 1e-9)
            speeds_dict[v_follower.id] = new_speed
            last_speed = new_speed
    
    # Build speed list matching permutation order
    speeds_list = []
    for v in permutation:
        if v.id not in speeds_dict:
            speeds_dict[v.id] = random.uniform(v_min_global, v_max_global)
        speeds_list.append(speeds_dict[v.id])
    
    return speeds_list


# =============================================================================
# CSV LOGGING FUNCTIONS
# =============================================================================
def save_aco_iteration_log(filename: str, iteration_data: List[Dict]):
    """
    Saves detailed iteration-by-iteration ACO performance data to CSV.
    
    Parameters
    ----------
    filename : str
        Output CSV filename
    iteration_data : List[Dict]
        List of dictionaries containing iteration metrics
    """
    if not iteration_data:
        return
    
    try:
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=iteration_data[0].keys())
            writer.writeheader()
            writer.writerows(iteration_data)
        print(f"  Successfully saved iteration log to: {filename}")
    except IOError as e:
        print(f"  ERROR: Could not save iteration log CSV. {e}")


def save_aco_run_summary(filename: str, run_summary: Dict):
    """
    Saves ACO run summary with parameters and final results.
    
    Parameters
    ----------
    filename : str
        Output CSV filename
    run_summary : Dict
        Dictionary containing run parameters and results
    """
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, mode='a', newline='') as f:
            if not file_exists:
                # Create header
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Num_Ants', 'Max_Iterations', 'Iterations_Run',
                    'Alpha', 'Beta', 'Rho', 'Q', 'Tau_Initial', 'Elitist_Weight',
                    'Convergence_Patience', 'Best_Fitness', 'Emergency_Delay', 
                    'Total_Delay', 'Avg_Delay_Per_Vehicle', 'Total_Evaluations',
                    'Early_Stopped', 'Runtime_Seconds'
                ])
            
            # Write run data
            writer = csv.writer(f)
            writer.writerow([
                run_summary['timestamp'],
                run_summary['num_ants'],
                run_summary['max_iterations'],
                run_summary['iterations_run'],
                run_summary['alpha'],
                run_summary['beta'],
                run_summary['rho'],
                run_summary['q'],
                run_summary['tau_initial'],
                run_summary['elitist_weight'],
                run_summary['convergence_patience'],
                run_summary['best_fitness'],
                run_summary['emergency_delay'],
                run_summary['total_delay'],
                run_summary['avg_delay_per_vehicle'],
                run_summary['total_evaluations'],
                run_summary['early_stopped'],
                run_summary['runtime_seconds']
            ])
        print(f"  Successfully appended run summary to: {filename}")
    except IOError as e:
        print(f"  ERROR: Could not save run summary CSV. {e}")


# =============================================================================
# MAIN ACO ALGORITHM
# =============================================================================
def run_aco(max_iterations: int = None,
            visualize_realtime: bool = False,
            verbose: bool = True,
            log_to_csv: bool = False,
            csv_prefix: str = "aco_run") -> Tuple:
    """
    Run the Ant Colony Optimization algorithm.
    
    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of iterations. Uses NUM_ITERATIONS if None.
    visualize_realtime : bool
        Whether to show real-time visualization
    verbose : bool
        Whether to print progress messages
    log_to_csv : bool
        Whether to log iteration data and run summary to CSV files
    csv_prefix : str
        Prefix for CSV filenames (default: "aco_run")
    
    Returns
    -------
    tuple : (best_perm, best_speeds, best_fitness, history, geom, tau_p_dict, best_obj_dict, eval_count)
        - best_perm: Best vehicle permutation found
        - best_speeds: Best speed assignment found
        - best_fitness: Best objective value
        - history: Dictionary with convergence data
        - geom: Geometry object
        - tau_p_dict: Tau values for all conflict points
        - best_obj_dict: Detailed objective breakdown
        - eval_count: Total number of solution evaluations
    """
    if max_iterations is None:
        max_iterations = NUM_ITERATIONS
    
    if verbose:
        print("\n" + "="*70)
        print("ANT COLONY OPTIMIZATION (Ant System)")
        print("="*70)
        print(f"  Number of Ants:        {NUM_ANTS}")
        print(f"  Max Iterations:        {max_iterations}")
        print(f"   (pheromone weight):  {ALPHA}")
        print(f"   (heuristic weight):  {BETA}")
        print(f"   (evaporation):       {RHO}")
        print(f"  q (deposit constant):  {Q}")
        print("="*70 + "\n")
    
    # Initialize geometry and vehicles
    geom = Geometry()
    all_vehicles = config.pi
    geom.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom.set_trajectory(v)
    
    # Get tau values for conflict points
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    tau_p_dict = {p: config.tau for p in all_points}
    
    # Initialize ACO graph
    graph = ACOGraph(all_vehicles)
    
    # Initialize ant colony
    ants = [Ant(i) for i in range(NUM_ANTS)]
    
    # Best solution tracking
    best_perm = None
    best_speeds = None
    best_fitness = math.inf
    best_obj_dict = {}
    
    # History tracking
    history = {
        'best_f': [],           # Best-so-far fitness per iteration
        'iter_best_f': [],      # Best fitness in current iteration
        'pher_max': [],         # Max pheromone level
        'pher_avg': [],         # Average pheromone level
        'pher_min': [],         # Min pheromone level
        'best_avg_delay': [],   # Average delay of best solution
        'best_fall': [],        # Total delay of best solution
        'best_fem': []          # Emergency delay of best solution
    }
    
    # Visualization
    visualizer = None
    if visualize_realtime:
        visualizer = ACOVisualizer()
    
    # Early stopping
    no_improvement_count = 0
    eval_count = 0
    
    # CSV logging data collection
    iteration_data = []
    start_time = datetime.now()
    
    # Main ACO loop
    for iteration in range(max_iterations):
        # Phase 1: Solution Construction
        iter_best_fitness = math.inf
        iter_best_ant = None
        
        for ant in ants:
            construct_ant_solution(ant, graph, geom, ALPHA, BETA, tau_p_dict)
            eval_count += 1
            
            # Track iteration best
            if ant.fitness < iter_best_fitness:
                iter_best_fitness = ant.fitness
                iter_best_ant = ant
            
            # Track global best
            if ant.fitness < best_fitness:
                best_fitness = ant.fitness
                best_perm = copy.deepcopy(ant.permutation)
                best_speeds = copy.copy(ant.speeds)
                
                # Get detailed objective breakdown
                best_obj_dict = evaluate_solution(best_perm, best_speeds, geom, tau_p_dict)
                
                no_improvement_count = 0
                
                if verbose:
                    print(f"  Iter {iteration+1}: NEW BEST = {best_fitness:.2f}")
        
        # Phase 2: Pheromone Evaporation
        graph.evaporate_pheromone(RHO)
        
        # Phase 3: Pheromone Deposition (Ant System)
        # Each ant deposits pheromone proportional to solution quality
        for ant in ants:
            if ant.fitness < math.inf:  # Valid solution
                delta_tau = Q / ant.fitness
                graph.deposit_pheromone(ant, delta_tau)
        
        # Elitist strategy: Extra pheromone for best-so-far solution
        if best_perm is not None:
            # Create temporary ant with best solution
            elite_ant = Ant(-1)
            elite_ant.permutation = best_perm
            elite_ant.fitness = best_fitness
            
            delta_tau_elite = (Q / best_fitness) * ELITIST_WEIGHT
            graph.deposit_pheromone(elite_ant, delta_tau_elite)
        
        # Record history
        history['best_f'].append(best_fitness)
        history['iter_best_f'].append(iter_best_fitness)
        
        pher_max, pher_avg, pher_min = graph.get_pheromone_stats()
        history['pher_max'].append(pher_max)
        history['pher_avg'].append(pher_avg)
        history['pher_min'].append(pher_min)
        
        if best_obj_dict:
            avg_delay = best_obj_dict['fall'] / len(all_vehicles)
            history['best_avg_delay'].append(avg_delay)
            history['best_fall'].append(best_obj_dict['fall'])
            history['best_fem'].append(best_obj_dict['fem'])
        else:
            history['best_avg_delay'].append(0)
            history['best_fall'].append(0)
            history['best_fem'].append(0)
        
        # Collect iteration data for CSV logging
        if log_to_csv:
            iter_data = {
                'iteration': iteration + 1,
                'best_f': best_fitness,
                'iter_best_f': iter_best_fitness,
                'pher_max': pher_max,
                'pher_avg': pher_avg,
                'pher_min': pher_min,
                'best_avg_delay': history['best_avg_delay'][-1],
                'best_fall': history['best_fall'][-1],
                'best_fem': history['best_fem'][-1],
                'eval_count': eval_count
            }
            iteration_data.append(iter_data)
        
        # Update visualization
        if visualizer:
            visualizer.update(iteration, history)
        
        # Early stopping check
        # no_improvement_count += 1
        # if no_improvement_count >= CONVERGENCE_PATIENCE:
        #     if verbose:
        #         print(f"\n  Early stopping: No improvement for {CONVERGENCE_PATIENCE} iterations")
        #     break
        
        # Progress update
        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration+1}/{max_iterations}: Best = {best_fitness:.2f}, "
                  f"Iter Best = {iter_best_fitness:.2f}")
    
    # Close visualizer
    if visualizer:
        visualizer.close()
    
    # Calculate runtime
    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()
    
    # Save CSV logs if requested
    if log_to_csv:
        # Save iteration log
        iter_log_filename = f"{csv_prefix}_iterations.csv"
        save_aco_iteration_log(iter_log_filename, iteration_data)
        
        # Create and save run summary
        run_summary = {
            'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_ants': NUM_ANTS,
            'max_iterations': max_iterations,
            'iterations_run': len(history['best_f']),
            'alpha': ALPHA,
            'beta': BETA,
            'rho': RHO,
            'q': Q,
            'tau_initial': TAU_INITIAL,
            'elitist_weight': ELITIST_WEIGHT,
            'convergence_patience': CONVERGENCE_PATIENCE,
            'best_fitness': best_fitness,
            'emergency_delay': best_obj_dict.get('fem', 0) if best_obj_dict else 0,
            'total_delay': best_obj_dict.get('fall', 0) if best_obj_dict else 0,
            'avg_delay_per_vehicle': best_obj_dict.get('fall', 0) / len(all_vehicles) if best_obj_dict else 0,
            'total_evaluations': eval_count,
            'early_stopped': len(history['best_f']) < max_iterations,
            'runtime_seconds': runtime_seconds
        }
        
        run_summary_filename = f"{csv_prefix}_summary.csv"
        save_aco_run_summary(run_summary_filename, run_summary)
        
        if verbose:
            print(f"\n  CSV logs saved:")
            print(f"    - {iter_log_filename}")
            print(f"    - {run_summary_filename}")
    
    if verbose:
        print("\n" + "="*70)
        print("ACO COMPLETE")
        print("="*70)
        print(f"  Best Objective:     {best_fitness:.2f}")
        print(f"  Total Evaluations:  {eval_count}")
        print(f"  Iterations Run:     {len(history['best_f'])}")
        if best_obj_dict:
            print(f"  Emergency Delay:    {best_obj_dict['fem']:.2f}")
            print(f"  Total Delay:        {best_obj_dict['fall']:.2f}")
            print(f"  Avg Delay/Vehicle:  {best_obj_dict['fall']/len(all_vehicles):.2f}")
        print("="*70 + "\n")
    
    return (best_perm, best_speeds, best_fitness, history, 
            geom, tau_p_dict, best_obj_dict, eval_count)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_aco_performance_dashboard(history: Dict):
    """
    Creates a comprehensive 2x2 dashboard showing ACO performance metrics.
    """
    print("\nDisplaying ACO Performance Dashboard...")
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ACO Performance Dashboard (Ant System)', fontsize=16, fontweight='bold')
    
    iterations = range(len(history['best_f']))
    
    # Top-left: Convergence
    ax = axs[0, 0]
    ax.plot(iterations, history['best_f'], 'b-', label='Best-So-Far', linewidth=2)
    ax.plot(iterations, history['iter_best_f'], 'g--', label='Iteration Best', linewidth=1.5, alpha=0.7)
    ax.set_title('Convergence Curve')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Cost (f)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Delay Components
    ax = axs[0, 1]
    ax.plot(iterations, history['best_avg_delay'], 'g-', label='Avg Delay per Vehicle', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(iterations, history['best_fem'], 'r--', label='Emergency Delay', linewidth=1.5, alpha=0.7)
    ax.set_title('Delay Components (Best Solution)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Avg Delay (s)', color='g')
    ax2.set_ylabel('Emergency Delay (s)', color='r')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Pheromone Statistics
    ax = axs[1, 0]
    ax.plot(iterations, history['pher_max'], 'r-', label='Max Pheromone', linewidth=2)
    ax.plot(iterations, history['pher_avg'], 'orange', label='Avg Pheromone', linewidth=2)
    ax.plot(iterations, history['pher_min'], 'brown', label='Min Pheromone', linewidth=2)
    ax.set_title('Pheromone Levels Evolution')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pheromone Level (τ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Total Delay Evolution
    ax = axs[1, 1]
    ax.plot(iterations, history['best_fall'], 'c-', label='Total Delay (f_all)', linewidth=2)
    ax.set_title('Total Delay Evolution')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Delay (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# INITIAL POPULATION CREATION (for consistency with GA/SA patterns)
# =============================================================================
def create_initial_population(num_ants: int, geom: Geometry) -> List[Ant]:
    """
    Creates an initial population of ants with random solutions.
    This can be used for warm-starting ACO or for experiments.
    """
    ants = []
    graph = ACOGraph(config.pi)
    
    # Create tau_p_dict
    all_vehicles = config.pi
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    tau_p_dict = {p: config.tau for p in all_points}
    
    for i in range(num_ants):
        ant = Ant(i)
        construct_ant_solution(ant, graph, geom, ALPHA, BETA, tau_p_dict)
        ants.append(ant)
    
    return ants


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Testing ACO module independently...")
    
    # Run ACO with visualization and CSV logging
    (best_perm, best_speeds, best_fitness, history, 
     geom, tau_p_dict, best_obj_dict, eval_count) = run_aco(
        max_iterations=None,
        visualize_realtime=True,
        verbose=True,
        log_to_csv=True,
        csv_prefix="aco_test_run"
    )
    
    print("\n--- BEST SOLUTION FOUND ---")
    print(f"Permutation (IDs): {[v.id for v in best_perm]}")
    print(f"Speeds: {[f'{s:.2f}' for s in best_speeds]}")
    print(f"Objective: {best_fitness:.2f}")
    
    # Show performance dashboard
    plot_aco_performance_dashboard(history)
