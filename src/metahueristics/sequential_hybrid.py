# File: sequential_hybrid.py
"""
Sequential Hybrid Optimization: ACO -> PSO
==========================================

This module implements a sequential hybrid approach where:
1. Phase 1: ACO optimizes the vehicle permutation
2. Phase 2: PSO fine-tunes the speed assignment for the ACO-found permutation

The two-phase strategy leverages:
- ACO's strength in discrete optimization (vehicle ordering)
- PSO's strength in continuous optimization (speed tuning)
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
from metahueristics.sa import evaluate_solution, validate_speeds

# =============================================================================
# HYBRID PARAMETERS
# =============================================================================
ACO_ITERATIONS = 100             # Number of ACO iterations for permutation
PSO_ITERATIONS = 100              # Number of PSO iterations for speed tuning
CONVERGENCE_PATIENCE = 20        # Early stopping patience

# =============================================================================
# MAIN SEQUENTIAL HYBRID ALGORITHM
# =============================================================================
def run_sequential_hybrid(aco_iterations: int = None,
                         pso_iterations: int = None,
                         verbose: bool = True) -> Tuple:
    """
    Run Sequential Hybrid Optimization: ACO -> PSO.
    
    Phase 1: ACO finds optimal permutation with deterministic speeds
    Phase 2: PSO optimizes speeds for the ACO-found permutation
    
    Parameters
    ----------
    aco_iterations : int, optional
        Number of ACO iterations (uses ACO_ITERATIONS if None)
    pso_iterations : int, optional
        Number of PSO iterations (uses PSO_ITERATIONS if None)
    verbose : bool
        Print progress messages
    
    Returns
    -------
    tuple : (best_perm, best_speeds, best_fitness, history, geom, tau_p_dict, 
             best_obj_dict, total_evals, aco_iterations, pso_iterations)
    """
    if aco_iterations is None:
        aco_iterations = ACO_ITERATIONS
    if pso_iterations is None:
        pso_iterations = PSO_ITERATIONS
    
    if verbose:
        print("\n" + "="*70)
        print("SEQUENTIAL HYBRID OPTIMIZATION (ACO -> PSO)")
        print("="*70)
        print(f"  Phase 1: ACO optimizes permutation ({aco_iterations} iterations)")
        print(f"  Phase 2: PSO fine-tunes speeds ({pso_iterations} iterations)")
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
    
    # =============================================================================
    # PHASE 1: ACO for Permutation Optimization
    # =============================================================================
    try:
        from metahueristics.aco import run_aco
        
        if verbose:
            print("="*70)
            print("PHASE 1: ACO - Optimizing Vehicle Permutation")
            print("="*70)
        
        (aco_perm, aco_speeds, aco_fitness, aco_history,
         aco_geom, aco_tau, aco_obj_dict, aco_evals) = run_aco(
            max_iterations=aco_iterations,
            visualize_realtime=False,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nPhase 1 Complete: ACO Best = {aco_fitness:.2f}")
            print(f"  Permutation found (IDs): {[v.id for v in aco_perm[:10]]}...")
            print(f"  Evaluations used: {aco_evals}")
        
    except ImportError as e:
        if verbose:
            print(f"ERROR: ACO module not available: {e}")
            print("  Using random permutation for Phase 2.")
        
        from metahueristics.sa import create_initial_solution
        aco_perm, aco_speeds = create_initial_solution(geom)
        aco_obj_dict = evaluate_solution(aco_perm, aco_speeds, geom, tau_p_dict)
        aco_fitness = aco_obj_dict['f']
        aco_history = {'best_f': [aco_fitness], 'iter_best_f': [aco_fitness]}
        aco_evals = 1
    
    # =============================================================================
    # PHASE 2: PSO for Speed Optimization
    # =============================================================================
    try:
        from metahueristics.pso import optimize_speeds_with_pso
        
        if verbose:
            print("\n" + "="*70)
            print("PHASE 2: PSO - Fine-tuning Speed Assignment")
            print("="*70)
        
        (pso_speeds, pso_fitness, pso_history) = optimize_speeds_with_pso(
            permutation=aco_perm,
            init_speeds=aco_speeds,
            geom=geom,
            tau_p_dict=tau_p_dict,
            num_iterations=pso_iterations,
            visualize=False,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nPhase 2 Complete: PSO Best = {pso_fitness:.2f}")
        
    except ImportError as e:
        if verbose:
            print(f"ERROR: PSO module not available: {e}")
            print("  Using ACO speeds as final result.")
        
        pso_speeds = aco_speeds
        pso_fitness = aco_fitness
        pso_history = {'best_f': [], 'iter_best_f': [], 'vel_norm': []}
    
    # =============================================================================
    # Final Evaluation and Results
    # =============================================================================
    final_obj_dict = evaluate_solution(aco_perm, pso_speeds, geom, tau_p_dict)
    final_fitness = final_obj_dict['f']
    
    # Combine histories for visualization
    combined_history = {
        'best_f': aco_history.get('best_f', []) + pso_history.get('best_f', []),
        'iter_best_f': aco_history.get('iter_best_f', []) + pso_history.get('iter_best_f', []),
        'vel_norm': pso_history.get('vel_norm', []),
        'best_fem': aco_history.get('best_fem', []) + pso_history.get('best_fem', []),
        'best_fall': aco_history.get('best_fall', []) + pso_history.get('best_fall', []),
        'best_avg_delay': aco_history.get('best_avg_delay', []) + pso_history.get('best_avg_delay', [])
    }
    
    # Calculate total evaluations
    # ACO: NUM_ANTS per iteration
    try:
        from metahueristics.aco import NUM_ANTS
        aco_evals_total = aco_iterations * NUM_ANTS
    except:
        aco_evals_total = aco_evals
    
    # PSO: SWARM_SIZE per iteration
    try:
        from metahueristics.pso import SWARM_SIZE
        pso_evals_total = pso_iterations * SWARM_SIZE
    except:
        pso_evals_total = pso_iterations * 30  # Default swarm size
    
    total_evals = aco_evals_total + pso_evals_total
    
    # Calculate improvements
    aco_to_pso_improvement = aco_fitness - pso_fitness
    aco_to_pso_improvement_pct = (aco_to_pso_improvement / aco_fitness * 100) if aco_fitness > 0 else 0
    
    if verbose:
        print("\n" + "="*70)
        print("SEQUENTIAL HYBRID RESULTS")
        print("="*70)
        print(f"  Phase 1 (ACO) Final:   {aco_fitness:.2f}")
        print(f"  Phase 2 (PSO) Final:   {pso_fitness:.2f}")
        print(f"  Overall Best:          {final_fitness:.2f}")
        print(f"  ACO->PSO Improvement:   {aco_to_pso_improvement:.2f} ({aco_to_pso_improvement_pct:.1f}%)")
        print(f"  Total Evaluations:     {total_evals}")
        print(f"    - ACO Phase:  {aco_evals_total}")
        print(f"    - PSO Phase:  {pso_evals_total}")
        
        if final_obj_dict:
            print(f"\n  Final Solution Metrics:")
            print(f"    Emergency Delay: {final_obj_dict.get('fem', 0):.2f}s")
            print(f"    Total Delay:     {final_obj_dict.get('fall', 0):.2f}s")
            print(f"    Avg Delay:       {final_obj_dict.get('fall', 0) / len(all_vehicles):.2f}s/vehicle")
            print(f" best perm IDs: {[v.id for v in aco_perm[:]]}")
            print(f" best speeds: {[round(s,2) for s in pso_speeds[:]]}...")
        print("="*70)
    
    return (aco_perm, pso_speeds, final_fitness, combined_history,
            geom, tau_p_dict, final_obj_dict, total_evals, 
            aco_iterations, pso_iterations)


# =============================================================================
# ENHANCED VISUALIZATION FOR SEQUENTIAL HYBRID
# =============================================================================
def plot_sequential_hybrid_dashboard(history: Dict, aco_iterations: int, pso_iterations: int):
    """
    Creates two comprehensive dashboards showing Sequential Hybrid performance.
    Figure 1: Convergence and phase analysis
    Figure 2: Delay components and performance metrics
    
    Parameters
    ----------
    history : Dict
        Combined history from run_sequential_hybrid
    aco_iterations : int
        Number of ACO iterations (for phase boundary)
    pso_iterations : int
        Number of PSO iterations (for phase boundary)
    """
    print("\nDisplaying Sequential Hybrid (ACO->PSO) Performance Dashboards...")
    
    # Calculate phase boundary
    phase_boundary = min(aco_iterations, len(history.get('best_f', [])))
    iterations = range(len(history['best_f']))
    
    # =============================================================================
    # FIGURE 1: Convergence and Phase Analysis (2x2)
    # =============================================================================
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig1.suptitle('Sequential Hybrid: Convergence Analysis (ACO -> PSO)', 
                fontsize=16, fontweight='bold')
    
    # =========================================================================
    # Top-left: Two-Phase Convergence
    # =========================================================================
    ax = fig1.add_subplot(gs1[0, 0])
    ax.plot(iterations, history['best_f'], 'b-', label='Best-So-Far', linewidth=2.5)
    ax.plot(iterations, history['iter_best_f'], 'g--', label='Iteration Best', 
           linewidth=1.5, alpha=0.7)
    
    # Phase boundary line
    ax.axvline(x=phase_boundary, color='red', linestyle=':', linewidth=2.5, 
              label='ACO->PSO Transition')
    
    # Phase labels with background
    aco_mid = phase_boundary // 2
    pso_mid = phase_boundary + (len(iterations) - phase_boundary) // 2
    
    ax.text(aco_mid, ax.get_ylim()[1] * 0.95, 'Phase 1: ACO\n(Permutation)', 
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='skyblue', 
                    edgecolor='blue', linewidth=2, alpha=0.8))
    
    if pso_mid < len(iterations):
        ax.text(pso_mid, ax.get_ylim()[1] * 0.95, 'Phase 2: PSO\n(Speed Tuning)', 
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', 
                        edgecolor='green', linewidth=2, alpha=0.8))
    
    ax.set_title('Two-Phase Convergence Curve', fontsize=13, fontweight='bold')
    ax.set_xlabel('Combined Iteration', fontsize=11)
    ax.set_ylabel('Objective Cost (f)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Top-right: Phase-wise Improvement Analysis
    # =========================================================================
    ax = fig1.add_subplot(gs1[0, 1])
    
    if len(history['best_f']) > phase_boundary > 0:
        initial_cost = history['best_f'][0]
        aco_final = history['best_f'][phase_boundary - 1]
        pso_final = history['best_f'][-1]
        
        aco_improvement = initial_cost - aco_final
        pso_improvement = aco_final - pso_final
        total_improvement = initial_cost - pso_final
        
        phases = ['Initial', 'After ACO', 'After PSO']
        values = [initial_cost, aco_final, pso_final]
        colors = ['lightcoral', 'skyblue', 'lightgreen']
        
        bars = ax.bar(phases, values, color=colors, edgecolor='black', linewidth=2, width=0.6)
        
        # Value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Improvement arrows
        # ACO improvement
        ax.annotate('', xy=(1, aco_final), xytext=(0, initial_cost),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2.5))
        ax.text(0.5, (initial_cost + aco_final) / 2, 
               f'ACO:\n-{aco_improvement:.2f}',
               va='center', ha='center', fontsize=9, color='blue', 
               fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # PSO improvement
        ax.annotate('', xy=(2, pso_final), xytext=(1, aco_final),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
        ax.text(1.5, (aco_final + pso_final) / 2, 
               f'PSO:\n-{pso_improvement:.2f}',
               va='center', ha='center', fontsize=9, color='green', 
               fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Phase-wise Solution Quality', fontsize=13, fontweight='bold')
        ax.set_ylabel('Objective Cost (f)', fontsize=11)
        ax.set_ylim(0, initial_cost * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for phase comparison', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # =========================================================================
    # Bottom-left: PSO Velocity Evolution
    # =========================================================================
    ax = fig1.add_subplot(gs1[1, 0])
    
    if 'vel_norm' in history and history['vel_norm']:
        vel_iterations = range(phase_boundary, phase_boundary + len(history['vel_norm']))
        ax.plot(vel_iterations, history['vel_norm'], 'magenta', linewidth=2.5, 
               label='Swarm Velocity', marker='o', markersize=3, alpha=0.7)
        ax.axvline(x=phase_boundary, color='red', linestyle=':', linewidth=2, alpha=0.5)
        
        ax.set_title('PSO Swarm Velocity (Phase 2 Only)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Combined Iteration', fontsize=11)
        ax.set_ylabel('Average Velocity Norm', fontsize=11)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Velocity data not available\n(PSO phase)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Bottom-right: Average Improvement Rate
    # =========================================================================
    ax = fig1.add_subplot(gs1[1, 1])
    
    if 'best_fall' in history and history['best_fall'] and any(history['best_fall']):

        # Only plot delay data where it exists (might not cover all iterations)
        delay_iterations = list(range(1, len(history['best_fall']) + 1))
        ax.plot(delay_iterations, history['best_fall'], 'c-', linewidth=2.5, 
               label='Total Delay', marker='s', markersize=3, alpha=0.7)
        ax.axvline(x=phase_boundary, color='red', linestyle=':', linewidth=2, 
                  label='ACO->PSO', alpha=0.5)
        
        if 'best_fem' in history and history['best_fem'] and any(history['best_fem']):
            fem_iterations = list(range(1, len(history['best_fem']) + 1))
            ax2 = ax.twinx()
            ax2.plot(fem_iterations, history['best_fem'], 'r--', linewidth=2, 
                    label='Emergency Delay', marker='^', markersize=3, alpha=0.7)
            ax2.set_ylabel('Emergency Delay (s)', color='r', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper right', fontsize=9)
        
        ax.set_title('Delay Components Evolution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Combined Iteration', fontsize=11)
        ax.set_ylabel('Total Delay (s)', color='c', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='c')
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Delay data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # =============================================================================
    # FIGURE 2: Delay Components and Performance Summary
    # =============================================================================
    fig2 = plt.figure(figsize=(14, 6))
    gs2 = fig2.add_gridspec(1, 1)
    
    fig2.suptitle('Sequential Hybrid: Performance Summary', 
                fontsize=16, fontweight='bold')
    
    # =========================================================================
    # Summary Statistics Table
    # =========================================================================
    ax = fig2.add_subplot(gs2[0])
    ax.axis('off')
    
    if len(history['best_f']) > phase_boundary:
        initial_cost = history['best_f'][0]
        aco_final = history['best_f'][phase_boundary - 1] if phase_boundary > 0 else initial_cost
        pso_final = history['best_f'][-1]
        
        aco_improvement = initial_cost - aco_final
        pso_improvement = aco_final - pso_final
        total_improvement = initial_cost - pso_final
        
        aco_pct = (aco_improvement / initial_cost * 100) if initial_cost > 0 else 0
        pso_pct = (pso_improvement / aco_final * 100) if aco_final > 0 else 0
        total_pct = (total_improvement / initial_cost * 100) if initial_cost > 0 else 0
        
        # Prepare table
        table_data = [
            ['Metric', 'Initial', 'After ACO', 'After PSO'],
            ['Cost (f)', f'{initial_cost:.2f}', f'{aco_final:.2f}', f'{pso_final:.2f}'],
            ['Iterations', '0', f'{aco_iterations}', f'{aco_iterations + pso_iterations}'],
            ['Improvement', '—', f'{aco_improvement:.2f}', f'{total_improvement:.2f}'],
            ['Improvement %', '—', f'{aco_pct:.1f}%', f'{total_pct:.1f}%'],
            ['Phase Contrib.', '—', f'{aco_improvement:.2f}', f'{pso_improvement:.2f}'],
        ]
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.28, 0.24, 0.24, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white', fontsize=10)
        
        # Style metric column
        for i in range(1, len(table_data)):
            cell = table[(i, 0)]
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold', fontsize=9)
        
        # Highlight improvement rows
        for i in [3, 4, 5]:
            if i < len(table_data):
                for j in range(1, 4):
                    cell = table[(i, j)]
                    cell.set_facecolor('#FFF9C4')
        
        ax.set_title('Performance Summary', fontweight='bold', fontsize=13, pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================
if __name__ == "__main__":
    print("Testing Sequential Hybrid (ACO->PSO) module...")
    
    # Run sequential hybrid optimization
    (best_perm, best_speeds, best_fitness, history,
     geom, tau_p_dict, best_obj_dict, total_evals,
     aco_iters, pso_iters) = run_sequential_hybrid(
        aco_iterations=100,
        pso_iterations=50,
        verbose=True
    )
    
    print("\n--- BEST SOLUTION FOUND ---")
    print(f"Permutation (IDs): {[v.id for v in best_perm]}")
    print(f"Speeds: {[f'{s:.2f}' for s in best_speeds]}")
    print(f"Objective: {best_fitness:.2f}")
    print(f"bes permutation: {[v.id for v in best_perm]}")
    print(f"best speeds: {[f'{s:.2f}' for s in best_speeds]}")
    # Show performance dashboard
    plot_sequential_hybrid_dashboard(history, aco_iters, pso_iters)
