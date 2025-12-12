# File: src/metahueristics/dragonfly.py
"""
Two-Stage Dragonfly Algorithm for Autonomous Intersection Management

This module implements a decoupled 2-stage approach:
- Stage 1 (Discrete): Optimized Discrete Dragonfly Algorithm (ODDA) for permutation
- Stage 2 (Continuous): Standard Dragonfly Algorithm for speed optimization

References:
- Paper 1: Dragonfly Algorithm (DA) - Mirjalili, 2016
- Paper 2: Optimized Discrete Dragonfly Algorithm (ODDA) for TSP
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
from typing import List, Dict, Tuple, Set

# --- Import Project Files ---
import config
from engine.geometry import Geometry
from engine.vehicle import Vehicle
from metahueristics.sa import evaluate_solution, validate_speeds

# =============================================================================
# DRAGONFLY ALGORITHM PARAMETERS
# =============================================================================

# --- Stage 1: Discrete DA (Permutation Optimization) ---
DISCRETE_SWARM_SIZE = 30              # Number of dragonflies for permutation
DISCRETE_MAX_ITERATIONS = 150         # Iterations for Stage 1

# --- Stage 2: Continuous DA (Speed Optimization) ---
CONTINUOUS_SWARM_SIZE = 30            # Number of dragonflies for speeds
CONTINUOUS_MAX_ITERATIONS = 150       # Iterations for Stage 2

# --- Common DA Parameters ---
# Weights for swarm behaviors (typical range: 0.1 - 2.0)
WEIGHT_SEPARATION = 2.0               # s: Separation weight
WEIGHT_ALIGNMENT = 2.0                # a: Alignment weight
WEIGHT_COHESION = 2.0                 # c: Cohesion weight
WEIGHT_FOOD = 1.0                     # f: Food attraction weight
WEIGHT_ENEMY = 1.0                    # e: Enemy repulsion weight
WEIGHT_INERTIA = 0.5                  # w: Inertia weight (decreases over time) - REDUCED to prevent explosion
INERTIA_MIN = 0.1                     # Minimum inertia weight - REDUCED to prevent explosion

# Neighborhood parameters
NEIGHBOR_RADIUS = 0.3                 # Radius for finding neighbors (fraction of swarm)

# Local search parameters (Stage 1 only)
LOCAL_SEARCH_PROB = 0.3               # Probability of applying local search
LOCAL_SEARCH_ITERATIONS = 10          # Number of local search iterations

# Velocity control parameters (to prevent exponential growth)
MAX_VELOCITY_LENGTH = 100             # Maximum number of swaps in velocity


# =============================================================================
# SWAP SEQUENCE OPERATORS (for Discrete DA)
# =============================================================================

def calculate_swap_sequence(perm_a: List[Vehicle], perm_b: List[Vehicle]) -> List[Tuple[int, int]]:
    """
    Calculate the swap sequence (⊖ operator) needed to transform perm_a into perm_b.
    
    Returns a list of (i, j) tuples representing swaps.
    This implements the subtraction operation: perm_b ⊖ perm_a
    
    Algorithm:
    1. Create a working copy of perm_a
    2. For each position, if the vehicle doesn't match perm_b, find and swap it
    3. Record each swap operation
    """
    swap_sequence = []
    working_perm = [v.id for v in perm_a]
    target_perm = [v.id for v in perm_b]
    
    for i in range(len(working_perm)):
        if working_perm[i] != target_perm[i]:
            # Find where the target vehicle is currently located
            target_id = target_perm[i]
            j = working_perm.index(target_id)
            
            # Swap positions i and j
            working_perm[i], working_perm[j] = working_perm[j], working_perm[i]
            swap_sequence.append((i, j))
    
    return swap_sequence


def merge_swap_sequences(seq1: List[Tuple[int, int]], seq2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge two swap sequences (⊕ operator).
    
    Simply concatenates the two sequences.
    The combined sequence applies seq1 first, then seq2.
    """
    return seq1 + seq2


def apply_swap_sequence(perm: List[Vehicle], swap_seq: List[Tuple[int, int]], 
                       probability: float = 1.0) -> List[Vehicle]:
    """
    Apply a swap sequence to a permutation (⊗ operator).
    
    Parameters:
    -----------
    perm : List[Vehicle]
        The permutation to modify
    swap_seq : List[Tuple[int, int]]
        The sequence of swaps to apply
    probability : float
        Probability of applying each swap (for stochastic behavior)
    
    Returns:
    --------
    List[Vehicle] : The modified permutation
    """
    result = copy.deepcopy(perm)
    
    for i, j in swap_seq:
        # Apply swap with given probability
        if random.random() < probability:
            if 0 <= i < len(result) and 0 <= j < len(result):
                result[i], result[j] = result[j], result[i]
    
    return result


def scale_swap_sequence(swap_seq: List[Tuple[int, int]], weight: float) -> List[Tuple[int, int]]:
    """
    Scale a swap sequence by a weight (used in motion equation).
    
    Weight controls how many swaps are kept:
    - weight = 1.0: keep all swaps
    - weight = 0.5: keep ~50% of swaps
    - weight = 0.0: keep no swaps
    """
    if weight <= 0:
        return []
    if weight >= 1.0:
        return swap_seq
    
    # Probabilistically keep swaps based on weight
    scaled_seq = []
    for swap in swap_seq:
        if random.random() < weight:
            scaled_seq.append(swap)
    
    return scaled_seq


# =============================================================================
# DETERMINISTIC SPEED ASSIGNMENT (for Stage 1 Proxy Cost)
# =============================================================================

def _assign_speeds_deterministic(permutation: List[Vehicle], geom: Geometry) -> List[float]:
    """
    Assign speeds DETERMINISTICALLY based on vehicle priority and queue position.
    
    This is used in Stage 1 to generate proxy speeds for evaluating permutations.
    
    Strategy:
    1. Process vehicles in spawn order (FIFO within each queue)
    2. Leader gets max speed, followers get progressively lower speeds
    3. Emergency vehicles always get max speed
    4. Strictly enforce C0 constraint (v_k <= v_{k-1})
    """
    speeds_dict = {}
    v_min, v_max = config.velocity_range
    
    for approach, queue in geom.entry_queues.items():
        if not queue:
            continue
        
        # Get vehicles in this queue, preserving physical spawn order (FIFO)
        queue_vehicles = [v for v in queue if v in permutation]
        if not queue_vehicles:
            continue
        
        # Heuristic: Leader gets max speed to clear intersection; followers slow down
        previous_speed = v_max + 1.0
        
        for idx, vehicle in enumerate(queue_vehicles):
            # Calculate target speed based on rank in the physical queue
            ratio = idx / max(1, len(queue_vehicles) - 1)
            target_speed = v_max - ratio * (v_max - v_min)
            
            if vehicle.priority_status:
                target_speed = v_max
            
            # Enforce C0 (No Catch Up)
            actual_speed = min(target_speed, previous_speed)
            
            speeds_dict[vehicle.id] = actual_speed
            previous_speed = actual_speed
    
    # Return list matching the order of the 'permutation' input
    return [speeds_dict.get(v.id, (v_min + v_max) / 2) for v in permutation]


# =============================================================================
# DISCRETE DRAGONFLY CLASS (for Permutation Optimization)
# =============================================================================

class DiscreteDragonfly:
    """
    Represents a dragonfly for discrete optimization (permutation).
    
    Position: A permutation of vehicles
    Velocity: A swap sequence (list of swaps)
    """
    
    def __init__(self, dragonfly_id: int, vehicles: List[Vehicle]):
        self.id = dragonfly_id
        self.position = random.sample(vehicles, len(vehicles))  # Random permutation
        self.velocity = []  # Swap sequence (initially empty)
        self.fitness = math.inf
        self.proxy_speeds = []  # Speeds assigned by deterministic heuristic
    
    def evaluate(self, geom: Geometry, tau_p_dict: Dict) -> float:
        """
        Evaluate this dragonfly's fitness using proxy speeds.
        """
        # Assign speeds deterministically
        self.proxy_speeds = _assign_speeds_deterministic(self.position, geom)
        
        # Validate speeds
        self.proxy_speeds = validate_speeds(self.position, self.proxy_speeds, geom)
        
        # Evaluate solution
        obj_dict = evaluate_solution(self.position, self.proxy_speeds, geom, tau_p_dict)
        self.fitness = obj_dict['f']
        
        return self.fitness
    
    def update_velocity(self, separation: List[Tuple[int, int]], 
                       alignment: List[Tuple[int, int]],
                       cohesion: List[Tuple[int, int]],
                       food: List[Tuple[int, int]],
                       enemy: List[Tuple[int, int]],
                       weights: Dict[str, float]):
        """
        Update velocity using the discrete motion equation (Eq 14 from Paper 2):
        ΔΠ_{t+1} = (s*S ⊕ a*A ⊕ c*C ⊕ f*F ⊕ e*E) ⊕ w*ΔΠ_t
        """
        s, a, c, f, e, w = (weights['s'], weights['a'], weights['c'], 
                            weights['f'], weights['e'], weights['w'])
        
        # Scale each component by its weight
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Scaling sequences...")
        scaled_S = scale_swap_sequence(separation, s)
        scaled_A = scale_swap_sequence(alignment, a)
        scaled_C = scale_swap_sequence(cohesion, c)
        scaled_F = scale_swap_sequence(food, f)
        scaled_E = scale_swap_sequence(enemy, e)
        scaled_inertia = scale_swap_sequence(self.velocity, w)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Scaled lengths - S:{len(scaled_S)}, A:{len(scaled_A)}, C:{len(scaled_C)}, F:{len(scaled_F)}, E:{len(scaled_E)}, Inertia:{len(scaled_inertia)}")
        
        # Merge all components (⊕ operator)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merging S+A...")
        new_velocity = merge_swap_sequences(scaled_S, scaled_A)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merged S+A, length={len(new_velocity)}")
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merging +C...")
        new_velocity = merge_swap_sequences(new_velocity, scaled_C)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merged +C, length={len(new_velocity)}")
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merging +F...")
        new_velocity = merge_swap_sequences(new_velocity, scaled_F)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merged +F, length={len(new_velocity)}")
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merging +E...")
        new_velocity = merge_swap_sequences(new_velocity, scaled_E)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merged +E, length={len(new_velocity)}")
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Merging +Inertia (length={len(scaled_inertia)})...")
        new_velocity = merge_swap_sequences(new_velocity, scaled_inertia)
        # print(f"[DEBUG-VELOCITY] DF {self.id}: Final velocity length={len(new_velocity)}")
        
        # CRITICAL FIX: Limit velocity length to prevent exponential growth
        if len(new_velocity) > MAX_VELOCITY_LENGTH:
            # Randomly sample swaps to keep velocity bounded
            new_velocity = random.sample(new_velocity, MAX_VELOCITY_LENGTH)
            # print(f"[DEBUG-VELOCITY] DF {self.id}: Velocity limited to {MAX_VELOCITY_LENGTH} swaps")
        
        self.velocity = new_velocity
    
    def update_position(self):
        """
        Update position by applying velocity (swap sequence) to current position.
        """
        # Apply swap sequence with high probability
        self.position = apply_swap_sequence(self.position, self.velocity, probability=0.8)


# =============================================================================
# CONTINUOUS DRAGONFLY CLASS (for Speed Optimization)
# =============================================================================

class ContinuousDragonfly:
    """
    Represents a dragonfly for continuous optimization (speeds).
    
    Position: A vector of speeds [v1, v2, ..., vN]
    Velocity: A velocity vector [Δv1, Δv2, ..., ΔvN]
    """
    
    def __init__(self, dragonfly_id: int, num_vehicles: int, v_min: float, v_max: float):
        self.id = dragonfly_id
        self.num_vehicles = num_vehicles
        self.v_min = v_min
        self.v_max = v_max
        
        # Initialize with random speeds
        self.position = np.random.uniform(v_min, v_max, num_vehicles)
        self.velocity = np.zeros(num_vehicles)
        self.fitness = math.inf
    
    def evaluate(self, permutation: List[Vehicle], geom: Geometry, tau_p_dict: Dict) -> float:
        """
        Evaluate this dragonfly's fitness with the fixed permutation.
        """
        # Convert position to list
        speeds = self.position.tolist()
        
        # Enforce constraints
        speeds = self._enforce_constraints(permutation, speeds, geom)
        self.position = np.array(speeds)
        
        # Evaluate solution
        obj_dict = evaluate_solution(permutation, speeds, geom, tau_p_dict)
        self.fitness = obj_dict['f']
        
        return self.fitness
    
    def _enforce_constraints(self, permutation: List[Vehicle], speeds: List[float], 
                            geom: Geometry) -> List[float]:
        """
        Enforce speed constraints:
        1. Clamp to [v_min, v_max]
        2. Enforce C0 constraint (no catch-up in same lane)
        """
        # Clamp speeds
        speeds = [max(self.v_min, min(self.v_max, s)) for s in speeds]
        
        # Enforce C0 using existing validation function
        speeds = validate_speeds(permutation, speeds, geom)
        
        return speeds
    
    def update_velocity(self, separation: np.ndarray, alignment: np.ndarray,
                       cohesion: np.ndarray, food: np.ndarray, enemy: np.ndarray,
                       weights: Dict[str, float]):
        """
        Update velocity using the continuous motion equation (Eq 3.6 from Paper 1):
        Δv_{t+1} = (s*S + a*A + c*C + f*F + e*E) + w*Δv_t
        """
        s, a, c, f, e, w = (weights['s'], weights['a'], weights['c'], 
                            weights['f'], weights['e'], weights['w'])
        
        # Calculate new velocity
        self.velocity = (s * separation + a * alignment + c * cohesion + 
                        f * food + e * enemy + w * self.velocity)
    
    def update_position(self, permutation: List[Vehicle], geom: Geometry):
        """
        Update position using velocity and enforce constraints.
        """
        # Update position: x_{t+1} = x_t + Δx_{t+1}
        self.position = self.position + self.velocity
        
        # Enforce constraints
        speeds = self.position.tolist()
        speeds = self._enforce_constraints(permutation, speeds, geom)
        self.position = np.array(speeds)


# =============================================================================
# SWARM BEHAVIOR CALCULATIONS
# =============================================================================

def calculate_neighbors(dragonflies: List, current_idx: int, radius_fraction: float = 0.3) -> List[int]:
    """
    Find neighboring dragonflies based on fitness similarity.
    
    Returns indices of neighbors within the specified radius.
    """
    if len(dragonflies) <= 1:
        return []
    
    current_fitness = dragonflies[current_idx].fitness
    
    # Calculate fitness distances
    distances = []
    for i, df in enumerate(dragonflies):
        if i != current_idx and df.fitness < math.inf:
            # Normalized fitness distance
            dist = abs(df.fitness - current_fitness) / (current_fitness + 1e-9)
            distances.append((i, dist))
    
    if not distances:
        return []
    
    # Sort by distance and select closest neighbors
    distances.sort(key=lambda x: x[1])
    num_neighbors = max(1, int(len(dragonflies) * radius_fraction))
    neighbors = [idx for idx, _ in distances[:num_neighbors]]
    
    return neighbors


def calculate_discrete_separation(current: DiscreteDragonfly, 
                                  neighbors: List[DiscreteDragonfly]) -> List[Tuple[int, int]]:
    """
    Calculate separation for discrete dragonflies.
    
    Separation: Move away from neighbors by inverting their swap sequences.
    """
    if not neighbors:
        return []
    
    # Aggregate swap sequences from all neighbors
    all_swaps = []
    for neighbor in neighbors:
        # Calculate difference and invert it (move away)
        diff = calculate_swap_sequence(neighbor.position, current.position)
        all_swaps.extend(diff)
    
    return all_swaps


def calculate_discrete_alignment(current: DiscreteDragonfly,
                                neighbors: List[DiscreteDragonfly]) -> List[Tuple[int, int]]:
    """
    Calculate alignment for discrete dragonflies.
    
    Alignment: Match velocities with neighbors.
    """
    if not neighbors:
        return []
    
    # Average neighbor velocities
    all_swaps = []
    for neighbor in neighbors:
        all_swaps.extend(neighbor.velocity)
    
    return all_swaps


def calculate_discrete_cohesion(current: DiscreteDragonfly,
                               neighbors: List[DiscreteDragonfly]) -> List[Tuple[int, int]]:
    """
    Calculate cohesion for discrete dragonflies.
    
    Cohesion: Move toward the center of the neighbor group.
    For permutations, we use the best neighbor as the "center".
    """
    if not neighbors:
        return []
    
    # Find best neighbor
    best_neighbor = min(neighbors, key=lambda df: df.fitness)
    
    # Move toward best neighbor
    cohesion_swaps = calculate_swap_sequence(current.position, best_neighbor.position)
    
    return cohesion_swaps


def calculate_continuous_separation(current: ContinuousDragonfly,
                                   neighbors: List[ContinuousDragonfly]) -> np.ndarray:
    """
    Calculate separation for continuous dragonflies.
    
    Separation: S = -Σ(X - X_i) where X_i are neighbor positions.
    """
    if not neighbors:
        return np.zeros(current.num_vehicles)
    
    separation = np.zeros(current.num_vehicles)
    for neighbor in neighbors:
        separation -= (current.position - neighbor.position)
    
    return separation


def calculate_continuous_alignment(current: ContinuousDragonfly,
                                  neighbors: List[ContinuousDragonfly]) -> np.ndarray:
    """
    Calculate alignment for continuous dragonflies.
    
    Alignment: A = (Σ V_i) / N - V where V_i are neighbor velocities.
    """
    if not neighbors:
        return np.zeros(current.num_vehicles)
    
    avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
    alignment = avg_velocity - current.velocity
    
    return alignment


def calculate_continuous_cohesion(current: ContinuousDragonfly,
                                 neighbors: List[ContinuousDragonfly]) -> np.ndarray:
    """
    Calculate cohesion for continuous dragonflies.
    
    Cohesion: C = (Σ X_i) / N - X where X_i are neighbor positions.
    """
    if not neighbors:
        return np.zeros(current.num_vehicles)
    
    center = np.mean([n.position for n in neighbors], axis=0)
    cohesion = center - current.position
    
    return cohesion


# =============================================================================
# LOCAL SEARCH (for Discrete DA)
# =============================================================================

def steepest_ascent_hill_climbing(dragonfly: DiscreteDragonfly, geom: Geometry,
                                  tau_p_dict: Dict, max_iterations: int = 10) -> None:
    """
    Perform local search using steepest ascent hill climbing.
    
    Try random swaps and keep improvements.
    """
    current_fitness = dragonfly.fitness
    
    for _ in range(max_iterations):
        # Try a random swap
        if len(dragonfly.position) < 2:
            break
        
        i, j = random.sample(range(len(dragonfly.position)), 2)
        
        # Create neighbor
        neighbor_perm = copy.deepcopy(dragonfly.position)
        neighbor_perm[i], neighbor_perm[j] = neighbor_perm[j], neighbor_perm[i]
        
        # Evaluate neighbor
        neighbor_speeds = _assign_speeds_deterministic(neighbor_perm, geom)
        neighbor_speeds = validate_speeds(neighbor_perm, neighbor_speeds, geom)
        obj_dict = evaluate_solution(neighbor_perm, neighbor_speeds, geom, tau_p_dict)
        neighbor_fitness = obj_dict['f']
        
        # Accept if better
        if neighbor_fitness < current_fitness:
            dragonfly.position = neighbor_perm
            dragonfly.proxy_speeds = neighbor_speeds
            dragonfly.fitness = neighbor_fitness
            current_fitness = neighbor_fitness


# =============================================================================
# VISUALIZATION
# =============================================================================

class DragonflyVisualizer:
    """Handles real-time visualization of Dragonfly Algorithm progress."""
    
    def __init__(self, two_stage: bool = True):
        self.two_stage = two_stage
        plt.ion()
        
        if two_stage:
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle('Two-Stage Dragonfly Algorithm Progress', fontsize=14, fontweight='bold')
            
            # Stage 1 plots
            self.ax1.set_title("Stage 1: Permutation Optimization (Proxy Cost)")
            self.ax1.set_xlabel("Iteration")
            self.ax1.set_ylabel("Cost (f)")
            self.ax1.grid(True)
            self.line1_best, = self.ax1.plot([], [], 'b-', label='Best', linewidth=2)
            self.line1_avg, = self.ax1.plot([], [], 'g--', label='Avg', linewidth=1.5)
            self.ax1.legend()
            
            self.ax2.set_title("Stage 1: Fitness Distribution")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("Fitness")
            self.ax2.grid(True)
            
            # Stage 2 plots
            self.ax3.set_title("Stage 2: Speed Optimization (Refined Cost)")
            self.ax3.set_xlabel("Iteration")
            self.ax3.set_ylabel("Cost (f)")
            self.ax3.grid(True)
            self.line2_best, = self.ax3.plot([], [], 'b-', label='Best', linewidth=2)
            self.line2_avg, = self.ax3.plot([], [], 'g--', label='Avg', linewidth=1.5)
            self.ax3.legend()
            
            self.ax4.set_title("Stage 2: Fitness Distribution")
            self.ax4.set_xlabel("Iteration")
            self.ax4.set_ylabel("Fitness")
            self.ax4.grid(True)
        else:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
            self.fig.suptitle('Dragonfly Algorithm Progress', fontsize=14, fontweight='bold')
            
            self.ax1.set_title("Convergence")
            self.ax1.set_xlabel("Iteration")
            self.ax1.set_ylabel("Cost (f)")
            self.ax1.grid(True)
            self.line1_best, = self.ax1.plot([], [], 'b-', label='Best', linewidth=2)
            self.line1_avg, = self.ax1.plot([], [], 'g--', label='Avg', linewidth=1.5)
            self.ax1.legend()
            
            self.ax2.set_title("Fitness Distribution")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("Fitness")
            self.ax2.grid(True)
        
        plt.tight_layout()
    
    def update_stage1(self, iteration: int, history: Dict):
        """Update Stage 1 visualization."""
        if not plt.fignum_exists(self.fig.number):
            return
        
        iters = range(len(history['best']))
        self.line1_best.set_data(iters, history['best'])
        self.line1_avg.set_data(iters, history['avg'])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Fitness distribution
        self.ax2.clear()
        self.ax2.boxplot([history['all_fitness'][i] for i in range(0, len(history['all_fitness']), 
                                                                     max(1, len(history['all_fitness']) // 10))],
                         positions=range(0, len(history['all_fitness']), 
                                       max(1, len(history['all_fitness']) // 10)))
        self.ax2.set_title("Stage 1: Fitness Distribution")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Fitness")
        self.ax2.grid(True)
        
        plt.pause(0.01)
    
    def update_stage2(self, iteration: int, history: Dict):
        """Update Stage 2 visualization."""
        if not plt.fignum_exists(self.fig.number):
            return
        
        iters = range(len(history['best']))
        self.line2_best.set_data(iters, history['best'])
        self.line2_avg.set_data(iters, history['avg'])
        self.ax3.relim()
        self.ax3.autoscale_view()
        
        # Fitness distribution
        self.ax4.clear()
        self.ax4.boxplot([history['all_fitness'][i] for i in range(0, len(history['all_fitness']), 
                                                                     max(1, len(history['all_fitness']) // 10))],
                         positions=range(0, len(history['all_fitness']), 
                                       max(1, len(history['all_fitness']) // 10)))
        self.ax4.set_title("Stage 2: Fitness Distribution")
        self.ax4.set_xlabel("Iteration")
        self.ax4.set_ylabel("Fitness")
        self.ax4.grid(True)
        
        plt.pause(0.01)
    
    def close(self):
        """Close visualization window."""
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        plt.ioff()


# =============================================================================
# CSV LOGGING FUNCTIONS
# =============================================================================

def save_da_iteration_log(filename: str, iteration_data: List[Dict]):
    """
    Saves detailed iteration-by-iteration DA performance data to CSV.
    
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


def save_da_run_summary(filename: str, run_summary: Dict):
    """
    Saves DA run summary with parameters and final results.
    
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
                    'Timestamp', 'Discrete_Swarm_Size', 'Discrete_Max_Iterations',
                    'Continuous_Swarm_Size', 'Continuous_Max_Iterations',
                    'Weight_Separation', 'Weight_Alignment', 'Weight_Cohesion',
                    'Weight_Food', 'Weight_Enemy', 'Weight_Inertia', 'Inertia_Min',
                    'Neighbor_Radius', 'Local_Search_Prob', 'Local_Search_Iterations',
                    'Stage1_Best_Fitness', 'Stage2_Best_Fitness', 'Final_Best_Fitness',
                    'Emergency_Delay', 'Total_Delay', 'Avg_Delay_Per_Vehicle',
                    'Total_Evaluations', 'Runtime_Seconds'
                ])
            
            # Write run data
            writer = csv.writer(f)
            writer.writerow([
                run_summary['timestamp'],
                run_summary['discrete_swarm_size'],
                run_summary['discrete_max_iterations'],
                run_summary['continuous_swarm_size'],
                run_summary['continuous_max_iterations'],
                run_summary['weight_separation'],
                run_summary['weight_alignment'],
                run_summary['weight_cohesion'],
                run_summary['weight_food'],
                run_summary['weight_enemy'],
                run_summary['weight_inertia'],
                run_summary['inertia_min'],
                run_summary['neighbor_radius'],
                run_summary['local_search_prob'],
                run_summary['local_search_iterations'],
                run_summary['stage1_best_fitness'],
                run_summary['stage2_best_fitness'],
                run_summary['final_best_fitness'],
                run_summary['emergency_delay'],
                run_summary['total_delay'],
                run_summary['avg_delay_per_vehicle'],
                run_summary['total_evaluations'],
                run_summary['runtime_seconds']
            ])
        print(f"  Successfully appended run summary to: {filename}")
    except IOError as e:
        print(f"  ERROR: Could not save run summary CSV. {e}")


# =============================================================================
# TWO-STAGE DRAGONFLY OPTIMIZER
# =============================================================================

class TwoStageDragonflyOptimizer:
    """
    Main optimizer class that orchestrates both stages.
    
    Stage 1: Optimize permutation using Discrete DA
    Stage 2: Optimize speeds using Continuous DA with fixed permutation
    """
    
    def __init__(self, verbose: bool = True, visualize: bool = False, log_to_csv: bool = False, csv_prefix: str = "da_run"):
        self.verbose = verbose
        self.visualize = visualize
        self.log_to_csv = log_to_csv
        self.csv_prefix = csv_prefix
        self.visualizer = None
        
        # Initialize geometry and vehicles
        self.geom = Geometry()
        self.all_vehicles = config.pi
        self.geom.create_entry_queue(self.all_vehicles)
        for v in self.all_vehicles:
            self.geom.set_trajectory(v)
        
        # Get tau values for conflict points
        all_points = set().union(*(v.path for v in self.all_vehicles if v.path))
        self.tau_p_dict = {p: config.tau for p in all_points}
        
        # Best solution tracking
        self.best_permutation = None
        self.best_speeds = None
        self.best_fitness = math.inf
        self.best_obj_dict = {}
        
        # History tracking
        self.stage1_history = {'best': [], 'avg': [], 'all_fitness': []}
        self.stage2_history = {'best': [], 'avg': [], 'all_fitness': []}
    
    def optimize_permutation(self) -> Tuple[List[Vehicle], float]:
        """
        Stage 1: Optimize permutation using Discrete Dragonfly Algorithm.
        
        Returns:
        --------
        Tuple[List[Vehicle], float] : (best_permutation, best_fitness)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 1: DISCRETE DRAGONFLY ALGORITHM (Permutation Optimization)")
            print("="*70)
            print(f"  Swarm Size:        {DISCRETE_SWARM_SIZE}")
            print(f"  Max Iterations:    {DISCRETE_MAX_ITERATIONS}")
            print(f"  Optimization:      Permutation (speeds assigned deterministically)")
            print("="*70 + "\n")
        
        # Initialize swarm
        #print("[DEBUG] Stage 1 - Step 1: Initializing discrete swarm...")
        swarm = [DiscreteDragonfly(i, self.all_vehicles) for i in range(DISCRETE_SWARM_SIZE)]
        #print(f"[DEBUG] Stage 1 - Step 1: Created {len(swarm)} discrete dragonflies")
        
        # Evaluate initial swarm
        #print("[DEBUG] Stage 1 - Step 2: Evaluating initial swarm...")
        for df in swarm:
            df.evaluate(self.geom, self.tau_p_dict)
        #print("[DEBUG] Stage 1 - Step 2: Initial swarm evaluation complete")
        
        # Track best and worst (food and enemy)
        #print("[DEBUG] Stage 1 - Step 3: Identifying food (best) and enemy (worst)...")
        food = copy.deepcopy(min(swarm, key=lambda df: df.fitness))  # Best dragonfly (deep copy)
        enemy = copy.deepcopy(max(swarm, key=lambda df: df.fitness))  # Worst dragonfly (deep copy)
        #print(f"[DEBUG] Stage 1 - Step 3: Food fitness = {food.fitness:.2f}, Enemy fitness = {enemy.fitness:.2f}")
        
        # Main optimization loop
        #print("[DEBUG] Stage 1 - Step 4: Starting main optimization loop...")
        for iteration in range(DISCRETE_MAX_ITERATIONS):
            
            # Update inertia weight (linearly decrease)
            w = WEIGHT_INERTIA - (WEIGHT_INERTIA - INERTIA_MIN) * (iteration / DISCRETE_MAX_ITERATIONS)
            
            weights = {
                's': WEIGHT_SEPARATION,
                'a': WEIGHT_ALIGNMENT,
                'c': WEIGHT_COHESION,
                'f': WEIGHT_FOOD,
                'e': WEIGHT_ENEMY,
                'w': w
            }
            
            # Update each dragonfly
            for idx, df in enumerate(swarm):
                
                # Find neighbors
                neighbor_indices = calculate_neighbors(swarm, idx, NEIGHBOR_RADIUS)
                neighbors = [swarm[i] for i in neighbor_indices]
                
                if neighbors:
                    # Calculate swarm behaviors
                    separation = calculate_discrete_separation(df, neighbors)
                    alignment = calculate_discrete_alignment(df, neighbors)
                    cohesion = calculate_discrete_cohesion(df, neighbors)
                else:
                    separation = alignment = cohesion = []
                
                # Calculate food and enemy attraction/repulsion
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Calculating food attraction...")
                food_attraction = calculate_swap_sequence(df.position, food.position)
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Calculating enemy repulsion...")
                enemy_repulsion = calculate_swap_sequence(enemy.position, df.position)
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Food/Enemy calculated (F:{len(food_attraction)}, E:{len(enemy_repulsion)})")
                
                # Update velocity
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Updating velocity...")
                df.update_velocity(separation, alignment, cohesion, 
                                 food_attraction, enemy_repulsion, weights)
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Velocity updated (new velocity length: {len(df.velocity)})")
                
                # Update position
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Updating position...")
                df.update_position()
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Position updated")
                
                # Evaluate new position
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Evaluating fitness...")
                df.evaluate(self.geom, self.tau_p_dict)
                #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Fitness = {df.fitness:.2f}")
                
                # Local search with probability
                if random.random() < LOCAL_SEARCH_PROB:
                    #print(f"[DEBUG-DETAIL] Iter {iteration+1}, DF {idx}: Applying local search...")
                    steepest_ascent_hill_climbing(df, self.geom, self.tau_p_dict, 
                                                 LOCAL_SEARCH_ITERATIONS)
            
            # Update food and enemy
            current_best = min(swarm, key=lambda df: df.fitness)
            if current_best.fitness < food.fitness:
                food = copy.deepcopy(current_best)  # Deep copy to preserve best solution
                if self.verbose:
                    print(f"  Iter {iteration+1}: NEW BEST = {food.fitness:.2f}")
            
            enemy = copy.deepcopy(max(swarm, key=lambda df: df.fitness))  # Update enemy (deep copy)
            
            # Record history
            all_fitness = [df.fitness for df in swarm if df.fitness < math.inf]
            if all_fitness:
                self.stage1_history['best'].append(food.fitness)
                self.stage1_history['avg'].append(np.mean(all_fitness))
                self.stage1_history['all_fitness'].append(all_fitness)
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update_stage1(iteration, self.stage1_history)
            
            # Progress update
            if self.verbose and (iteration + 1) % 10 == 0:
                avg_fitness = np.mean(all_fitness) if all_fitness else math.inf
                print(f"  Iter {iteration+1}/{DISCRETE_MAX_ITERATIONS}: "
                      f"Best = {food.fitness:.2f}, Avg = {avg_fitness:.2f}")
        
        if self.verbose:
            print(f"\n  Stage 1 Complete: Best Permutation Cost = {food.fitness:.2f}")
            print(f"  Best Permutation (IDs): {[v.id for v in food.position]}\n")
        return food.position, food.fitness
    
    def optimize_speeds(self, fixed_permutation: List[Vehicle]) -> Tuple[List[float], float]:
        """
        Stage 2: Optimize speeds using Continuous Dragonfly Algorithm.
        
        Parameters:
        -----------
        fixed_permutation : List[Vehicle]
            The permutation to use (from Stage 1)
        
        Returns:
        --------
        Tuple[List[float], float] : (best_speeds, best_fitness)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE 2: CONTINUOUS DRAGONFLY ALGORITHM (Speed Optimization)")
            print("="*70)
            print(f"  Swarm Size:        {CONTINUOUS_SWARM_SIZE}")
            print(f"  Max Iterations:    {CONTINUOUS_MAX_ITERATIONS}")
            print(f"  Optimization:      Speeds (permutation fixed)")
            print("="*70 + "\n")
        
        v_min, v_max = config.velocity_range
        num_vehicles = len(fixed_permutation)
        
        # Initialize swarm
        swarm = [ContinuousDragonfly(i, num_vehicles, v_min, v_max) 
                for i in range(CONTINUOUS_SWARM_SIZE)]
        #print(f"[DEBUG] Stage 2 - Step 1: Created {len(swarm)} continuous dragonflies")
        
        # Evaluate initial swarm
        #print("[DEBUG] Stage 2 - Step 2: Evaluating initial swarm...")
        for df in swarm:
            df.evaluate(fixed_permutation, self.geom, self.tau_p_dict)
        #print("[DEBUG] Stage 2 - Step 2: Initial swarm evaluation complete")
        
        # Track best and worst (food and enemy)
        #print("[DEBUG] Stage 2 - Step 3: Identifying food (best) and enemy (worst)...")
        food = copy.deepcopy(min(swarm, key=lambda df: df.fitness))  # Best dragonfly (deep copy)
        enemy = copy.deepcopy(max(swarm, key=lambda df: df.fitness))  # Worst dragonfly (deep copy)
        #print(f"[DEBUG] Stage 2 - Step 3: Food fitness = {food.fitness:.2f}, Enemy fitness = {enemy.fitness:.2f}")
        
        # Main optimization loop
        #print("[DEBUG] Stage 2 - Step 4: Starting main optimization loop...")
        for iteration in range(CONTINUOUS_MAX_ITERATIONS):
            
            # Update inertia weight (linearly decrease)
            w = WEIGHT_INERTIA - (WEIGHT_INERTIA - INERTIA_MIN) * (iteration / CONTINUOUS_MAX_ITERATIONS)
            
            weights = {
                's': WEIGHT_SEPARATION,
                'a': WEIGHT_ALIGNMENT,
                'c': WEIGHT_COHESION,
                'f': WEIGHT_FOOD,
                'e': WEIGHT_ENEMY,
                'w': w
            }
            
            # Update each dragonfly
            for idx, df in enumerate(swarm):
                # Find neighbors
                neighbor_indices = calculate_neighbors(swarm, idx, NEIGHBOR_RADIUS)
                neighbors = [swarm[i] for i in neighbor_indices]
                
                if neighbors:
                    # Calculate swarm behaviors
                    separation = calculate_continuous_separation(df, neighbors)
                    alignment = calculate_continuous_alignment(df, neighbors)
                    cohesion = calculate_continuous_cohesion(df, neighbors)
                else:
                    separation = np.zeros(num_vehicles)
                    alignment = np.zeros(num_vehicles)
                    cohesion = np.zeros(num_vehicles)
                
                # Calculate food and enemy attraction/repulsion
                food_attraction = food.position - df.position
                enemy_repulsion = df.position - enemy.position
                
                # Update velocity
                df.update_velocity(separation, alignment, cohesion,
                                 food_attraction, enemy_repulsion, weights)
                
                # Update position
                df.update_position(fixed_permutation, self.geom)
                
                # Evaluate new position
                df.evaluate(fixed_permutation, self.geom, self.tau_p_dict)
            
            # Update food and enemy
            current_best = min(swarm, key=lambda df: df.fitness)
            if current_best.fitness < food.fitness:
                food = copy.deepcopy(current_best)  # Deep copy to preserve best solution
                if self.verbose:
                    print(f"  Iter {iteration+1}: NEW BEST = {food.fitness:.2f}")
            
            enemy = copy.deepcopy(max(swarm, key=lambda df: df.fitness))  # Update enemy (deep copy)
            
            # Record history
            all_fitness = [df.fitness for df in swarm if df.fitness < math.inf]
            if all_fitness:
                self.stage2_history['best'].append(food.fitness)
                self.stage2_history['avg'].append(np.mean(all_fitness))
                self.stage2_history['all_fitness'].append(all_fitness)
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update_stage2(iteration, self.stage2_history)
            
            # Progress update
            if self.verbose and (iteration + 1) % 10 == 0:
                avg_fitness = np.mean(all_fitness) if all_fitness else math.inf
                print(f"  Iter {iteration+1}/{CONTINUOUS_MAX_ITERATIONS}: "
                      f"Best = {food.fitness:.2f}, Avg = {avg_fitness:.2f}")
        
        #print("[DEBUG] Stage 2 - Step 5: Optimization loop complete")
        if self.verbose:
            print(f"\n  Stage 2 Complete: Best Speed Cost = {food.fitness:.2f}")
            print(f"  Best Speeds: {[f'{s:.2f}' for s in food.position]}\n")
        
        #print(f"[DEBUG] Stage 2 - COMPLETE: Returning best speeds with fitness {food.fitness:.2f}")
        return food.position.tolist(), food.fitness
    
    def optimize(self) -> Tuple[List[Vehicle], List[float], float, Dict]:
        """
        Run the complete two-stage optimization.
        
        Returns:
        --------
        Tuple : (best_permutation, best_speeds, best_fitness, results_dict)
        """
        #print("[DEBUG] MAIN OPTIMIZE - Starting two-stage optimization...")
        start_time = datetime.now()
        
        # Initialize visualizer if requested
        if self.visualize:
            #print("[DEBUG] MAIN OPTIMIZE - Initializing visualizer...")
            self.visualizer = DragonflyVisualizer(two_stage=True)
        
        # Stage 1: Optimize permutation
        #print("[DEBUG] MAIN OPTIMIZE - Calling Stage 1 (optimize_permutation)...")
        stage1_permutation, stage1_fitness = self.optimize_permutation()
        #print(f"[DEBUG] MAIN OPTIMIZE - Stage 1 returned with fitness: {stage1_fitness:.2f}")
        
        # Evaluate Stage 1 solution with its proxy speeds for global comparison
        stage1_speeds = _assign_speeds_deterministic(stage1_permutation, self.geom)
        stage1_speeds = validate_speeds(stage1_permutation, stage1_speeds, self.geom)
        stage1_obj_dict = evaluate_solution(stage1_permutation, stage1_speeds, self.geom, self.tau_p_dict)
        stage1_actual_fitness = stage1_obj_dict['f']
        
        if self.verbose:
            print(f"\n  Stage 1 Actual Fitness (with proxy speeds): {stage1_actual_fitness:.2f}")
        
        # Stage 2: Optimize speeds with fixed permutation
        #print("[DEBUG] MAIN OPTIMIZE - Calling Stage 2 (optimize_speeds)...")
        stage2_speeds, stage2_fitness = self.optimize_speeds(stage1_permutation)
        #print(f"[DEBUG] MAIN OPTIMIZE - Stage 2 returned with fitness: {stage2_fitness:.2f}")
        
        # Evaluate Stage 2 solution
        stage2_obj_dict = evaluate_solution(stage1_permutation, stage2_speeds, self.geom, self.tau_p_dict)
        stage2_actual_fitness = stage2_obj_dict['f']
        
        # GLOBAL BEST: Compare Stage 1 and Stage 2, keep the better one
        if stage1_actual_fitness < stage2_actual_fitness:
            # Stage 1 solution is better
            self.best_permutation = stage1_permutation
            self.best_speeds = stage1_speeds
            self.best_fitness = stage1_actual_fitness
            self.best_obj_dict = stage1_obj_dict
            if self.verbose:
                print(f"  >> GLOBAL BEST from Stage 1: {self.best_fitness:.2f}")
        else:
            # Stage 2 solution is better (or equal)
            self.best_permutation = stage1_permutation
            self.best_speeds = stage2_speeds
            self.best_fitness = stage2_actual_fitness
            self.best_obj_dict = stage2_obj_dict
            if self.verbose:
                print(f"  >> GLOBAL BEST from Stage 2: {self.best_fitness:.2f}")
        
        #print(f"[DEBUG] MAIN OPTIMIZE - Final evaluation complete: f={self.best_obj_dict.get('f', 0):.2f}")
        
        # Calculate runtime
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        #print(f"[DEBUG] MAIN OPTIMIZE - Total runtime: {runtime_seconds:.2f} seconds")
        
        # Close visualizer
        if self.visualizer:
            #print("[DEBUG] MAIN OPTIMIZE - Closing visualizer...")
            self.visualizer.close()
        
        # Save CSV logs if requested
        if self.log_to_csv:
            #print("[DEBUG] MAIN OPTIMIZE - Saving CSV logs...")
            # Combine iteration data from both stages
            stage1_iter_data = []
            for i in range(len(self.stage1_history['best'])):
                stage1_iter_data.append({
                    'stage': 1,
                    'iteration': i + 1,
                    'best_fitness': self.stage1_history['best'][i],
                    'avg_fitness': self.stage1_history['avg'][i]
                })
            
            stage2_iter_data = []
            for i in range(len(self.stage2_history['best'])):
                stage2_iter_data.append({
                    'stage': 2,
                    'iteration': i + 1,
                    'best_fitness': self.stage2_history['best'][i],
                    'avg_fitness': self.stage2_history['avg'][i]
                })
            
            # Save iteration logs
            if stage1_iter_data:
                save_da_iteration_log(f"{self.csv_prefix}_stage1_iterations.csv", stage1_iter_data)
            if stage2_iter_data:
                save_da_iteration_log(f"{self.csv_prefix}_stage2_iterations.csv", stage2_iter_data)
            
            # Create and save run summary
            total_evals = (DISCRETE_SWARM_SIZE * DISCRETE_MAX_ITERATIONS + 
                          CONTINUOUS_SWARM_SIZE * CONTINUOUS_MAX_ITERATIONS)
            
            run_summary = {
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'discrete_swarm_size': DISCRETE_SWARM_SIZE,
                'discrete_max_iterations': DISCRETE_MAX_ITERATIONS,
                'continuous_swarm_size': CONTINUOUS_SWARM_SIZE,
                'continuous_max_iterations': CONTINUOUS_MAX_ITERATIONS,
                'weight_separation': WEIGHT_SEPARATION,
                'weight_alignment': WEIGHT_ALIGNMENT,
                'weight_cohesion': WEIGHT_COHESION,
                'weight_food': WEIGHT_FOOD,
                'weight_enemy': WEIGHT_ENEMY,
                'weight_inertia': WEIGHT_INERTIA,
                'inertia_min': INERTIA_MIN,
                'neighbor_radius': NEIGHBOR_RADIUS,
                'local_search_prob': LOCAL_SEARCH_PROB,
                'local_search_iterations': LOCAL_SEARCH_ITERATIONS,
                'stage1_best_fitness': stage1_fitness,
                'stage2_best_fitness': self.best_fitness,
                'final_best_fitness': self.best_fitness,
                'emergency_delay': self.best_obj_dict.get('fem', 0),
                'total_delay': self.best_obj_dict.get('fall', 0),
                'avg_delay_per_vehicle': self.best_obj_dict.get('fall', 0) / len(self.all_vehicles),
                'total_evaluations': total_evals,
                'runtime_seconds': runtime_seconds
            }
            
            save_da_run_summary(f"{self.csv_prefix}_summary.csv", run_summary)
            
            if self.verbose:
                print(f"\n  CSV logs saved:")
                print(f"    - {self.csv_prefix}_stage1_iterations.csv")
                print(f"    - {self.csv_prefix}_stage2_iterations.csv")
                print(f"    - {self.csv_prefix}_summary.csv")
        
        # Prepare results
        #print("[DEBUG] MAIN OPTIMIZE - Preparing final results dictionary...")
        results = {
            'best_permutation': self.best_permutation,
            'best_speeds': self.best_speeds,
            'best_fitness': self.best_fitness,
            'stage1_proxy_fitness': stage1_fitness,
            'stage1_actual_fitness': stage1_actual_fitness,
            'stage2_fitness': stage2_actual_fitness,
            'obj_dict': self.best_obj_dict,
            'stage1_history': self.stage1_history,
            'stage2_history': self.stage2_history,
            'runtime_seconds': runtime_seconds,
            'timestamp': start_time
        }
        
        if self.verbose:
            print("\n" + "="*70)
            print("TWO-STAGE DRAGONFLY ALGORITHM COMPLETE")
            print("="*70)
            print(f"  Stage 1 Best (Proxy):       {stage1_fitness:.2f}")
            print(f"  Stage 1 Best (Actual):      {stage1_actual_fitness:.2f}")
            print(f"  Stage 2 Best (Optimized):   {stage2_actual_fitness:.2f}")
            print(f"  >> GLOBAL BEST:             {self.best_fitness:.2f}")
            print(f"  Emergency Delay:            {self.best_obj_dict.get('fem', 0):.2f}")
            print(f"  Total Delay:                {self.best_obj_dict.get('fall', 0):.2f}")
            print(f"  Runtime:                    {runtime_seconds:.2f} seconds")
            print("="*70 + "\n")
        
        #print("[DEBUG] MAIN OPTIMIZE - COMPLETE: Returning final results")
        return self.best_permutation, self.best_speeds, self.best_fitness, results
    

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Testing Two-Stage Dragonfly Algorithm...")
    
    # Run optimizer with CSV logging enabled
    optimizer = TwoStageDragonflyOptimizer(verbose=True, visualize=False, 
                                          log_to_csv=True, csv_prefix="da_test_run")
    best_perm, best_speeds, best_fitness, results = optimizer.optimize()
    
    print("\n--- BEST SOLUTION FOUND ---")
    print(f"Permutation (IDs): {[v.id for v in best_perm]}")
    print(f"Speeds: {[f'{s:.2f}' for s in best_speeds]}")
    print(f"Objective: {best_fitness:.2f}")
    
    # Keep plots open
    # input("\nPress Enter to close plots and exit...")
