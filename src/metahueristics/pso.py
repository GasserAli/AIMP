# File: pso.py
"""
Particle Swarm Optimization (PSO) for Intersection Traffic Optimization

This module implements PSO to optimize BOTH vehicle permutation AND speeds.
Can be used standalone or as a hybrid with ACO (ACO finds permutation, PSO optimizes speeds).
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Tuple
from datetime import datetime

# --- Import Project Files ---
import config
from engine.geometry import Geometry
from engine.vehicle import Vehicle
from metahueristics.sa import evaluate_solution, validate_speeds, create_initial_solution

# =============================================================================
# PSO PARAMETERS
# =============================================================================
SWARM_SIZE = 20                  # Number of particles (20-50)
NUM_ITERATIONS = 100             # Number of PSO iterations (50-200)
W = 0.5                         # Inertia weight (0.4-0.9)
C1 = 1.4                         # Cognitive parameter (1.0-2.0)
C2 = 1.0                         # Social parameter (1.0-2.0)
CONVERGENCE_PATIENCE = 60        # Early stopping patience

# PSO Mode
OPTIMIZE_SPEEDS_ONLY = False     # True: optimize speeds for fixed permutation
                                 # False: optimize both permutation and speeds

# =============================================================================
# PSO VISUALIZER CLASS
# =============================================================================
class PSOVisualizer:
    """Handles real-time visualization of PSO progress."""
    
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_cost, self.ax_vel) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Real-Time Particle Swarm Optimization', fontsize=14, fontweight='bold')
        
        # Cost plot
        self.ax_cost.set_title("Convergence: Best vs Iteration Best")
        self.ax_cost.set_xlabel("Iteration")
        self.ax_cost.set_ylabel("Objective Cost (f)")
        self.ax_cost.grid(True)
        self.line_best, = self.ax_cost.plot([], [], 'b-', label='Best Cost', linewidth=2)
        self.line_iter, = self.ax_cost.plot([], [], 'g--', label='Iteration Best', linewidth=1.5)
        self.ax_cost.legend(loc='upper right')
        
        # Velocity statistics plot
        self.ax_vel.set_title("Swarm Velocity (Movement Activity)")
        self.ax_vel.set_xlabel("Iteration")
        self.ax_vel.set_ylabel("Average Velocity Magnitude")
        self.ax_vel.grid(True)
        self.line_vel, = self.ax_vel.plot([], [], 'magenta', label='Velocity Norm', linewidth=2)
        self.ax_vel.legend(loc='upper right')
        
        plt.tight_layout()
    
    def update(self, history):
        """Updates visualization with current iteration data."""
        if not plt.fignum_exists(self.fig.number):
            return
        
        iters = range(len(history['best_f']))
        
        # Update cost plot
        self.line_best.set_data(iters, history['best_f'])
        self.line_iter.set_data(iters, history['iter_best_f'])
        self.ax_cost.relim()
        self.ax_cost.autoscale_view()
        
        # Update velocity plot
        self.line_vel.set_data(iters, history['vel_norm'])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()
        
        plt.pause(0.05)
    
    def close(self):
        """Closes the visualization window."""
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        plt.ioff()


# =============================================================================
# PARTICLE CLASS
# =============================================================================
class Particle:
    """Represents a single particle in the swarm."""
    
    def __init__(self, particle_id, permutation, speeds):
        self.particle_id = particle_id
        self.permutation = permutation      # Current permutation
        self.speeds = speeds                # Current speeds
        self.fitness = math.inf             # Current fitness
        
        # Personal best
        self.best_permutation = copy.deepcopy(permutation)
        self.best_speeds = copy.copy(speeds)
        self.best_fitness = math.inf
        
        # Velocity (for speeds only)
        self.velocity = np.zeros(len(speeds))


# =============================================================================
# SPEED OPTIMIZATION FUNCTIONS
# =============================================================================
def generate_baseline_speeds(permutation: List[Vehicle], geom: Geometry) -> List[float]:
    """
    Generate deterministic baseline speeds for a permutation.
    Similar to ACO's speed assignment strategy.
    """
    speeds_dict = {}
    v_min_global, v_max_global = config.velocity_range
    
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue
        
        # Get vehicles from this queue that are in the permutation
        queue_vehicles = [v for v in queue if v in permutation]
        if not queue_vehicles:
            continue
        
        # Leader gets high speed, followers get decreasing speeds
        for idx, vehicle in enumerate(queue_vehicles):
            if vehicle.priority_status:  # Emergency vehicle
                speeds_dict[vehicle.id] = v_max_global
            else:
                # Interpolate based on position in queue
                position_factor = 1.0 - (idx / max(len(queue_vehicles), 1))
                speed = v_min_global + position_factor * (v_max_global - v_min_global)
                speeds_dict[vehicle.id] = speed
    
    # Build speed list matching permutation order
    speeds_list = [speeds_dict.get(v.id, v_min_global) for v in permutation]
    
    return speeds_list


def optimize_speeds_with_pso(permutation: List[Vehicle], 
                             init_speeds: List[float],
                             geom: Geometry,
                             tau_p_dict: Dict,
                             swarm_size: int = SWARM_SIZE,
                             num_iterations: int = NUM_ITERATIONS,
                             w: float = W,
                             c1: float = C1,
                             c2: float = C2,
                             visualize: bool = False,
                             verbose: bool = True) -> Tuple:
    """
    Optimize speeds for a FIXED permutation using PSO.
    This is the "speeds-only" mode for hybrid ACO+PSO.
    
    Returns
    -------
    tuple : (best_speeds, best_fitness, history)
    """
    if verbose:
        print("\n=== PSO Speed Optimization ===")
        print(f"Swarm Size: {swarm_size}, Iterations: {num_iterations}")
        print(f"Optimizing speeds for {len(permutation)} vehicles")
    
    v_min, v_max = config.velocity_range
    dim = len(init_speeds)
    
    # Initialize swarm
    swarm = []
    velocity = []
    
    baseline = np.array(init_speeds)
    
    for i in range(swarm_size):
        # Initialize positions around baseline with small perturbation
        position = baseline + np.random.uniform(-2.0, 2.0, size=dim)
        position = np.clip(position, v_min, v_max)
        swarm.append(position)
        
        # Initialize velocities
        vel = np.zeros(dim)
        velocity.append(vel)
    
    # Personal bests
    pbest = [p.copy() for p in swarm]
    pbest_fitness = []
    
    for p in pbest:
        validated_speeds = validate_speeds(permutation, p.tolist(), geom)
        obj_dict = evaluate_solution(permutation, validated_speeds, geom, tau_p_dict)
        pbest_fitness.append(obj_dict['f'])
    
    # Global best
    g_idx = np.argmin(pbest_fitness)
    gbest = pbest[g_idx].copy()
    gbest_fitness = pbest_fitness[g_idx]
    
    # History
    history = {
        'best_f': [],
        'iter_best_f': [],
        'vel_norm': []
    }
    
    # Visualizer
    visualizer = PSOVisualizer() if visualize else None
    
    # Main PSO loop
    for iteration in range(num_iterations):
        iter_best = math.inf
        
        for i in range(swarm_size):
            r1, r2 = random.random(), random.random()
            
            # Update velocity
            velocity[i] = (w * velocity[i] +
                          c1 * r1 * (pbest[i] - swarm[i]) +
                          c2 * r2 * (gbest - swarm[i]))
            
            # Update position
            swarm[i] = swarm[i] + velocity[i]
            swarm[i] = np.clip(swarm[i], v_min, v_max)
            
            # Validate and store back to ensure constraints are met
            validated_speeds = validate_speeds(permutation, swarm[i].tolist(), geom)
            swarm[i] = np.array(validated_speeds)  # Store validated speeds back
            
            # Evaluate fitness
            obj_dict = evaluate_solution(permutation, validated_speeds, geom, tau_p_dict)
            f = obj_dict['f']
            
            # Update personal best
            if f < pbest_fitness[i]:
                pbest[i] = swarm[i].copy()  # swarm[i] now contains validated speeds
                pbest_fitness[i] = f
                
                # Update global best
                if f < gbest_fitness:
                    gbest = swarm[i].copy()  # swarm[i] now contains validated speeds
                    gbest_fitness = f
            
            # Track iteration best
            if f < iter_best:
                iter_best = f
        
        # Update history
        history['best_f'].append(gbest_fitness)
        history['iter_best_f'].append(iter_best)
        vel_norm = np.mean([np.linalg.norm(v) for v in velocity])
        history['vel_norm'].append(vel_norm)
        
        # Progress
        if verbose and (iteration % 10 == 0 or iteration == 0):
            print(f"  Iter {iteration+1}/{num_iterations}: Best={gbest_fitness:.2f}, "
                  f"IterBest={iter_best:.2f}, VelNorm={vel_norm:.4f}")
        
        if visualizer:
            visualizer.update(history)
    
    if visualizer:
        visualizer.close()
    
    if verbose:
        print(f"\nPSO Speed Optimization Complete: Best={gbest_fitness:.2f}")
    
    return gbest.tolist(), gbest_fitness, history


# =============================================================================
# MAIN PSO ALGORITHM (Full Optimization)
# =============================================================================
def run_pso(max_iterations: int = None,
           initial_solution: Tuple = None,
           visualize_realtime: bool = False,
           verbose: bool = True) -> Tuple:
    """
    Run Particle Swarm Optimization for full problem (permutation + speeds).
    
    Note: PSO is not well-suited for permutation optimization. 
    This implementation uses discrete swaps for permutations and 
    continuous PSO for speeds.
    
    Parameters
    ----------
    max_iterations : int, optional
        Maximum iterations (uses NUM_ITERATIONS if None)
    initial_solution : tuple, optional
        (permutation, speeds) to start from
    visualize_realtime : bool
        Show real-time visualization
    verbose : bool
        Print progress messages
    
    Returns
    -------
    tuple : (best_perm, best_speeds, best_fitness, history, geom, tau_p_dict, best_obj_dict, eval_count)
    """
    if max_iterations is None:
        max_iterations = NUM_ITERATIONS
    
    if verbose:
        print("\n" + "="*70)
        print("PARTICLE SWARM OPTIMIZATION (PSO)")
        print("="*70)
        print(f"  Swarm Size:      {SWARM_SIZE}")
        print(f"  Max Iterations:  {max_iterations}")
        print(f"  Inertia (w):     {W}")
        print(f"  Cognitive (c1):  {C1}")
        print(f"  Social (c2):     {C2}")
        print("="*70 + "\n")
    
    # Initialize geometry and vehicles
    geom = Geometry()
    all_vehicles = config.pi
    geom.create_entry_queue(all_vehicles)
    for v in all_vehicles:
        geom.set_trajectory(v)
    
    # Get tau values
    all_points = set().union(*(v.path for v in all_vehicles if v.path))
    tau_p_dict = {p: config.tau for p in all_points}
    
    # Initialize swarm
    particles = []
    
    for i in range(SWARM_SIZE):
        if i == 0 and initial_solution:
            # Use provided initial solution for first particle
            perm, speeds = initial_solution
        else:
            # Generate random solution
            perm, speeds = create_initial_solution(geom)
        
        particle = Particle(i, perm, speeds)
        
        # Evaluate initial fitness
        obj_dict = evaluate_solution(particle.permutation, particle.speeds, geom, tau_p_dict)
        particle.fitness = obj_dict['f']
        particle.best_fitness = particle.fitness
        
        particles.append(particle)
    
    # Find global best
    global_best_particle = min(particles, key=lambda p: p.best_fitness)
    global_best_fitness = global_best_particle.best_fitness
    global_best_perm = copy.deepcopy(global_best_particle.best_permutation)
    global_best_speeds = copy.copy(global_best_particle.best_speeds)
    global_best_obj_dict = evaluate_solution(global_best_perm, global_best_speeds, geom, tau_p_dict)
    
    # History
    history = {
        'best_f': [],
        'iter_best_f': [],
        'vel_norm': [],
        'best_fem': [],
        'best_fall': [],
        'best_avg_delay': []
    }
    
    # Visualizer
    visualizer = PSOVisualizer() if visualize_realtime else None
    
    # Early stopping
    no_improvement_count = 0
    eval_count = SWARM_SIZE  # Initial evaluations
    
    # Main PSO loop
    for iteration in range(max_iterations):
        iter_best_fitness = math.inf
        
        for particle in particles:
            # --- Update Speeds using PSO velocity ---
            r1, r2 = random.random(), random.random()
            
            particle.velocity = (W * particle.velocity +
                               C1 * r1 * (np.array(particle.best_speeds) - np.array(particle.speeds)) +
                               C2 * r2 * (np.array(global_best_speeds) - np.array(particle.speeds)))
            
            new_speeds = np.array(particle.speeds) + particle.velocity
            v_min, v_max = config.velocity_range
            new_speeds = np.clip(new_speeds, v_min, v_max)
            particle.speeds = new_speeds.tolist()
            
            # --- Update Permutation using discrete swaps ---
            # With small probability, perform swap inspired by best solutions
            if random.random() < 0.3:
                # Swap two random positions
                if len(particle.permutation) > 1:
                    idx1, idx2 = random.sample(range(len(particle.permutation)), 2)
                    particle.permutation[idx1], particle.permutation[idx2] = \
                        particle.permutation[idx2], particle.permutation[idx1]
            
            # Validate and evaluate
            particle.speeds = validate_speeds(particle.permutation, particle.speeds, geom)
            obj_dict = evaluate_solution(particle.permutation, particle.speeds, geom, tau_p_dict)
            particle.fitness = obj_dict['f']
            eval_count += 1
            
            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_permutation = copy.deepcopy(particle.permutation)
                particle.best_speeds = copy.copy(particle.speeds)
            
            # Update global best
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_perm = copy.deepcopy(particle.permutation)
                global_best_speeds = copy.copy(particle.speeds)
                global_best_obj_dict = obj_dict
                no_improvement_count = 0
                
                if verbose:
                    print(f"  Iter {iteration+1}: NEW BEST = {global_best_fitness:.2f}")
            
            # Track iteration best
            if particle.fitness < iter_best_fitness:
                iter_best_fitness = particle.fitness
        
        # Update history
        history['best_f'].append(global_best_fitness)
        history['iter_best_f'].append(iter_best_fitness)
        
        avg_vel = np.mean([np.linalg.norm(p.velocity) for p in particles])
        history['vel_norm'].append(avg_vel)
        
        if global_best_obj_dict:
            avg_delay = global_best_obj_dict['fall'] / len(all_vehicles)
            history['best_avg_delay'].append(avg_delay)
            history['best_fall'].append(global_best_obj_dict['fall'])
            history['best_fem'].append(global_best_obj_dict['fem'])
        
        # Progress
        if verbose and iteration % 10 == 0:
            print(f"  Iter {iteration+1}/{max_iterations}: Best={global_best_fitness:.2f}, "
                  f"IterBest={iter_best_fitness:.2f}")
        
        if visualizer:
            visualizer.update(history)
        
        # Early stopping
        no_improvement_count += 1
        if no_improvement_count >= CONVERGENCE_PATIENCE:
            if verbose:
                print(f"\nEarly stopping: No improvement for {CONVERGENCE_PATIENCE} iterations")
            break
    
    if visualizer:
        visualizer.close()
    
    if verbose:
        print("\n--- PSO Finished ---")
        print(f"Total evaluations: {eval_count}")
        print(f"Best Objective (f): {global_best_fitness:.2f}")
        print(f"Best Permutation (IDs): {[v.id for v in global_best_perm]}")
        print(f"Best Speeds: {[round(s, 2) for s in global_best_speeds]}")
    
    return (global_best_perm, global_best_speeds, global_best_fitness, history,
            geom, tau_p_dict, global_best_obj_dict, eval_count)


# =============================================================================
# PLOTTING
# =============================================================================
def plot_pso_performance_dashboard(history: Dict):
    """Creates a 2x2 dashboard showing PSO performance metrics."""
    print("\nDisplaying PSO Performance Dashboard...")
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PSO Performance Dashboard', fontsize=16, fontweight='bold')
    
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
    
    # Top-right: Velocity Evolution
    ax = axs[0, 1]
    if 'vel_norm' in history and history['vel_norm']:
        # vel_norm might be shorter than total iterations (e.g., in hybrid mode)
        vel_iterations = range(len(history['vel_norm']))
        ax.plot(vel_iterations, history['vel_norm'], 'magenta', linewidth=2)
        ax.set_title('Swarm Velocity Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Velocity Norm')
    else:
        ax.text(0.5, 0.5, 'Velocity data not available\n(ACO phase only)', ha='center', va='center')
        ax.set_title('Swarm Velocity Evolution')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Delay Components (if available)
    ax = axs[1, 0]
    if 'best_avg_delay' in history and history['best_avg_delay']:
        ax.plot(iterations, history['best_avg_delay'], 'g-', label='Avg Delay', linewidth=2)
        ax2 = ax.twinx()
        if 'best_fem' in history and history['best_fem']:
            ax2.plot(iterations, history['best_fem'], 'r--', label='Emergency Delay', linewidth=1.5)
            ax2.set_ylabel('Emergency Delay (s)', color='r')
            ax2.legend(loc='upper right')
        ax.set_title('Delay Components')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Delay (s)', color='g')
        ax.legend(loc='upper left')
    else:
        ax.text(0.5, 0.5, 'Delay data not available', ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Total Delay (if available)
    ax = axs[1, 1]
    if 'best_fall' in history and history['best_fall']:
        ax.plot(iterations, history['best_fall'], 'c-', linewidth=2)
        ax.set_title('Total Delay Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Delay (s)')
    else:
        ax.text(0.5, 0.5, 'Total delay data not available', ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("Testing PSO module...")
    
    # Test 1: Standalone PSO
    print("\n### TEST 1: Standalone PSO ###")
    (best_perm, best_speeds, best_fitness, history,
     geom, tau_p_dict, best_obj_dict, eval_count) = run_pso(
        max_iterations=50,
        visualize_realtime=True,
        verbose=True
    )
    
    plot_pso_performance_dashboard(history)
