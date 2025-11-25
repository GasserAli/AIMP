# Ant Colony Optimization (ACO) Implementation

## Overview

This implementation provides a complete **Ant System (AS)** variant of Ant Colony Optimization for solving the intersection traffic optimization problem. The algorithm finds optimal vehicle permutations and speed assignments to minimize total delay while respecting traffic constraints.

## File Structure

- **`src/aco.py`**: Complete ACO implementation with Ant System algorithm
- **`src/main_with_viz.py`**: Integrated runner supporting ACO alongside SA and GA

## Algorithm Details

### Implementation: Ant System (AS)

The classic Ant System approach with the following key features:

1. **Pheromone-based construction**: Ants build solutions probabilistically based on pheromone trails and heuristic information
2. **Evaporation**: Pheromone trails decay over time to avoid premature convergence
3. **Elitist strategy**: Best-so-far solution receives extra pheromone reinforcement
4. **Early stopping**: Automatically stops if no improvement for specified patience

### Key Parameters

Located in `src/aco.py`:

```python
NUM_ANTS = 50                    # m: Number of ants per iteration
NUM_ITERATIONS = 100             # Maximum iterations
ALPHA = 1.0                      # α: Pheromone importance (typically 1.0)
BETA = 2.0                       # β: Heuristic importance (typically 2-5)
RHO = 0.1                        # ρ: Evaporation rate (0.05-0.2)
Q = 100.0                        # Q: Pheromone deposit constant
TAU_INITIAL = 0.1                # Initial pheromone level
ELITIST_WEIGHT = 2.0             # Weight for best-so-far pheromone
CONVERGENCE_PATIENCE = 20        # Early stopping patience
```

### Algorithm Components

#### 1. **Graph Initialization** (`ACOGraph` class)
- **Pheromone matrix** `τ[position][vehicle_id]`: Represents the desirability of placing vehicle j at position i
- **Heuristic matrix** `η[position][vehicle_id]`: Emergency vehicles get higher values for earlier positions

#### 2. **Ant Construction** (`Ant` class)
- Each ant maintains: permutation, speeds, fitness, visited set
- Ants build solutions incrementally using probabilistic selection

#### 3. **Solution Construction** (`construct_ant_solution`)
Selection probability for placing vehicle j at position i:

```
p_ij = (τ_ij^α × η_ij^β) / Σ(τ_il^α × η_il^β)
```

Where:
- `τ_ij`: Pheromone level
- `η_ij`: Heuristic information
- `α, β`: Control relative importance

#### 4. **Pheromone Evaporation** (`evaporate_pheromone`)
```
τ_ij ← (1 - ρ) × τ_ij
```

#### 5. **Pheromone Deposition** (`deposit_pheromone`)
Each ant deposits pheromone inversely proportional to solution quality:

```
Δτ_ij = Q / L_k
```

Where `L_k` is the ant's tour cost (fitness)

#### 6. **Elitist Update**
Best-so-far solution receives additional pheromone:

```
Δτ_ij^elite = (Q / L_best) × ELITIST_WEIGHT
```

## Usage

### 1. Standalone Execution

Run ACO directly:

```bash
cd src
python aco.py
```

This will:
- Run ACO for 50 iterations (configured in `__main__`)
- Show real-time visualization
- Display performance dashboard
- Print best solution found

### 2. Via Main Runner

Edit `src/main_with_viz.py`:

```python
OPTIMIZATION_ALGORITHM = 'ACO'  # Single run
# or
OPTIMIZATION_ALGORITHM = 'ACO_ANALYSIS'  # Statistical analysis (multiple runs)

VISUALIZATION_METHOD = 'none'  # Or 'matplotlib'/'web' for animation
```

Then run:

```bash
cd src
python main_with_viz.py
```

### Available Modes

| Mode | Description |
|------|-------------|
| `'ACO'` | Single ACO run with optional visualization |
| `'ACO_ANALYSIS'` | N statistical runs (default 5) with convergence plots |
| `'BOTH'` | Compare SA vs GA (ACO can be added) |
| `'EXPERIMENT'` | Full statistical comparison (can extend to include ACO) |

### 3. Function API

```python
from aco import run_aco, plot_aco_performance_dashboard

# Run ACO
(best_perm, best_speeds, best_fitness, history, 
 geom, tau_p_dict, best_obj_dict, eval_count) = run_aco(
    max_iterations=100,
    visualize_realtime=True,
    verbose=True,
    log_to_csv=True,              # Enable CSV logging
    csv_prefix="my_aco_run"       # Prefix for CSV files
)

# Plot results
plot_aco_performance_dashboard(history)
```

### 4. CSV Logging

ACO includes comprehensive CSV logging to track algorithm parameters and performance metrics.

#### Enable Logging

```python
# Standalone
run_aco(
    max_iterations=50,
    log_to_csv=True,
    csv_prefix="aco_experiment_1"
)
```

#### Generated Files

Two CSV files are created:

1. **`{prefix}_iterations.csv`**: Per-iteration metrics
   - Columns: `iteration`, `best_f`, `iter_best_f`, `pher_max`, `pher_avg`, `pher_min`, `best_avg_delay`, `best_fall`, `best_fem`, `eval_count`
   - One row per iteration showing convergence and pheromone dynamics

2. **`{prefix}_summary.csv`**: Run-level summary
   - Columns: `Timestamp`, `Num_Ants`, `Max_Iterations`, `Iterations_Run`, `Alpha`, `Beta`, `Rho`, `Q`, `Tau_Initial`, `Elitist_Weight`, `Convergence_Patience`, `Best_Fitness`, `Emergency_Delay`, `Total_Delay`, `Avg_Delay_Per_Vehicle`, `Total_Evaluations`, `Early_Stopped`, `Runtime_Seconds`
   - Appends one row per run for multi-run experiments

#### Example Output

**Iteration Log** (`aco_test_run_iterations.csv`):
```csv
iteration,best_f,iter_best_f,pher_max,pher_avg,pher_min,best_avg_delay,best_fall,best_fem,eval_count
1,557.0,557.0,1.606,0.253,0.09,10.41,520.33,36.67,50
2,541.0,541.0,3.952,0.395,0.081,10.04,502.0,39.0,100
...
```

**Run Summary** (`aco_test_run_summary.csv`):
```csv
Timestamp,Num_Ants,Max_Iterations,Iterations_Run,Alpha,Beta,Rho,Q,...
2025-11-25 22:08:43,50,50,50,1.0,2.0,0.1,100.0,...
```

#### Integration with main_with_viz.py

The main runner also logs to:
- `experiment_summary_log.csv`: Includes ACO parameters alongside SA/GA
- `experiment_raw_data_log.csv`: Includes ACO results for comparative analysis

#### CSV Data Uses

- **Performance tracking**: Monitor convergence across multiple runs
- **Parameter tuning**: Compare different α, β, ρ settings
- **Statistical analysis**: Analyze ACO performance with pandas/R
- **Reproducibility**: Document exact parameters and results
- **Reporting**: Generate tables and charts for papers/reports

## Output

### Console Output

```
======================================================================
ANT COLONY OPTIMIZATION (Ant System)
======================================================================
  Number of Ants:        50
  Max Iterations:        100
  α (pheromone weight):  1.0
  β (heuristic weight):  2.0
  ρ (evaporation):       0.1
  Q (deposit constant):  100.0
======================================================================

  Iter 1: NEW BEST = 712.36
  Iter 2: NEW BEST = 558.10
  ...
  Early stopping: No improvement for 20 iterations

======================================================================
ACO COMPLETE
======================================================================
  Best Objective:     526.24
  Total Evaluations:  2250
  Iterations Run:     45
  Emergency Delay:    38.60
  Total Delay:        487.64
  Avg Delay/Vehicle:  9.75
======================================================================
```

### Return Values

```python
best_perm          # List[Vehicle]: Best vehicle permutation
best_speeds        # List[float]: Best speed assignment
best_fitness       # float: Best objective value (lower is better)
history            # Dict: Convergence data
geom               # Geometry: Problem geometry
tau_p_dict         # Dict: Conflict point parameters
best_obj_dict      # Dict: Detailed objective breakdown
eval_count         # int: Total fitness evaluations
```

### History Dictionary

```python
history = {
    'best_f':           # Best-so-far fitness per iteration
    'iter_best_f':      # Best fitness in current iteration
    'pher_max':         # Max pheromone level
    'pher_avg':         # Average pheromone level
    'pher_min':         # Min pheromone level
    'best_avg_delay':   # Average delay per vehicle (best solution)
    'best_fall':        # Total delay (best solution)
    'best_fem':         # Emergency delay (best solution)
}
```

## Performance Dashboard

The `plot_aco_performance_dashboard()` creates a 2×2 grid:

1. **Top-left**: Convergence curve (best-so-far vs iteration-best)
2. **Top-right**: Delay components over iterations
3. **Bottom-left**: Pheromone statistics evolution
4. **Bottom-right**: Total delay evolution

## Real-time Visualization

When `visualize_realtime=True`:
- Live convergence plot
- Pheromone level statistics
- Updates every iteration

## Parameter Tuning Guidelines

### α (Pheromone Importance)
- **Typical range**: 0.5 - 2.0
- **Default**: 1.0
- **Higher values**: More exploitation of learned paths
- **Lower values**: More exploration

### β (Heuristic Importance)
- **Typical range**: 1.0 - 5.0
- **Default**: 2.0
- **Higher values**: More greedy behavior (follow heuristic)
- **Lower values**: More random exploration

### ρ (Evaporation Rate)
- **Typical range**: 0.01 - 0.3
- **Default**: 0.1
- **Higher values**: Faster forgetting (more exploration)
- **Lower values**: Stronger memory (more exploitation)

### Q (Deposit Constant)
- **Typical range**: 1.0 - 1000.0
- **Default**: 100.0
- Scale according to typical fitness values

### Number of Ants
- **Typical range**: 10 - 100
- **Default**: 50
- **Trade-off**: More ants = better exploration but slower iterations

## Comparison with SA and GA

| Aspect | ACO | SA | GA |
|--------|-----|----|----|
| **Population-based** | ✓ (colony) | ✗ (single point) | ✓ (population) |
| **Constructive** | ✓ (builds solution) | ✗ (modifies solution) | Hybrid |
| **Memory** | ✓ (pheromone trails) | ✗ | ✗ |
| **Parallelizable** | ✓✓ (ants independent) | ✗ | ✓ (individuals) |
| **Convergence** | Moderate | Fast (early), slow (late) | Steady |

## Tips for Best Results

1. **Initial runs**: Use default parameters first
2. **If converging too fast**: Increase ρ (evaporation), decrease ELITIST_WEIGHT
3. **If not converging**: Increase α (pheromone importance), increase ELITIST_WEIGHT
4. **For large problems**: Increase NUM_ANTS
5. **For quick tests**: Decrease NUM_ANTS and NUM_ITERATIONS

## Constraint Handling

ACO respects all problem constraints:
- **C0 (No-catch-up)**: Speed validation after solution construction
- **Conflict resolution**: Through decoder (same as SA/GA)
- **Emergency priority**: Built into heuristic matrix (η)

## Integration with Existing Code

ACO seamlessly integrates with the existing codebase:
- Uses same `evaluate_solution()` from `sa.py`
- Uses same `validate_speeds()` for constraint enforcement
- Uses same `Geometry` and `Vehicle` classes
- Compatible with all visualization methods

## Example: Statistical Analysis

```python
# In main_with_viz.py
OPTIMIZATION_ALGORITHM = 'ACO_ANALYSIS'
NUM_EXPERIMENT_RUNS = 10

# Run
python src/main_with_viz.py
```

This will:
1. Run ACO 10 times with natural stopping
2. Compute mean, std, best cost
3. Generate box plots and distributions
4. Save results to CSV
5. Show dashboard for best run

## Files Generated

When running ACO_ANALYSIS:
- `experiment_summary_log.csv`: Statistical summary
- `experiment_raw_data_log.csv`: All run results

## Troubleshooting

### ACO not improving
- **Check**: α and β balance (try β=3-5 for more heuristic influence)
- **Increase**: Number of ants or iterations
- **Adjust**: Evaporation rate (try ρ=0.15-0.2)

### Too slow
- **Decrease**: NUM_ANTS (try 20-30)
- **Enable**: Early stopping (already default)
- **Disable**: Real-time visualization

### Premature convergence
- **Increase**: ρ (evaporation)
- **Decrease**: ELITIST_WEIGHT
- **Decrease**: α (pheromone importance)

## References

- **Dorigo, M., & Stützle, T. (2004)**. *Ant Colony Optimization*. MIT Press.
- **Dorigo, M., Maniezzo, V., & Colorni, A. (1996)**. Ant system: optimization by a colony of cooperating agents.

## License

Part of the Intersection Traffic Optimization project.
