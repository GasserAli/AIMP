# ACO Hyperparameter Tuning Guide

## ✅ Fixed Issues

1. **ACO now optimizes PERMUTATION ONLY** - Speeds are assigned deterministically
2. **ALPHA = 1.0** (was 0.0) - Pheromone now has influence
3. **BETA = 2.0** (was missing) - Heuristic importance added
4. **RHO = 0.3** (was 0) - Pheromone now evaporates properly

## ACO Hyperparameters Explained

### 1. ALPHA (α) - Pheromone Importance
**What it does:** Controls how much ants follow pheromone trails left by previous successful ants.

- **Low α (0.5-1.0)**: More exploration, less exploitation of known good solutions
- **High α (1.5-2.0)**: More exploitation, ants converge faster to best-known solutions
- **α = 0**: Ignores pheromone completely (random search with heuristic only)

**Recommended values to test:** `[0.5, 1.0, 1.5, 2.0]`
**Default:** `1.0` (balanced)

---

### 2. BETA (β) - Heuristic Importance
**What it does:** Controls how much ants follow the greedy heuristic (domain knowledge).

In our case, the heuristic prioritizes:
- Emergency vehicles early in permutation
- Balanced distribution of vehicles

- **Low β (1.0-2.0)**: Less greedy, more random exploration
- **High β (3.0-5.0)**: More greedy, follows heuristic strongly

**Recommended values to test:** `[1.0, 2.0, 3.0, 5.0]`
**Default:** `2.0` (balanced)

---

### 3. RHO (ρ) - Evaporation Rate
**What it does:** Controls how quickly pheromone trails decay over time.

- **Low ρ (0.1-0.2)**: Slow evaporation, pheromone persists longer (more exploitation)
- **High ρ (0.4-0.5)**: Fast evaporation, encourages exploration of new solutions
- **ρ = 0**: No evaporation (pheromone accumulates forever - BAD!)

**Recommended values to test:** `[0.1, 0.2, 0.3, 0.4, 0.5]`
**Default:** `0.3` (balanced)

---

### 4. Q - Pheromone Deposit Constant
**What it does:** Controls how much pheromone is deposited by each ant.

Formula: `Δτ = Q / fitness` (better solutions deposit more pheromone)

- **Low Q (1-10)**: Small pheromone deposits
- **High Q (100-1000)**: Large pheromone deposits

**Recommended values to test:** `[10, 50, 100, 200]`
**Default:** `100.0` (works well for most problems)

---

### 5. NUM_ANTS - Colony Size
**What it does:** Number of ants (solutions) constructed per iteration.

- **Small colony (10-30)**: Faster iterations, less diverse exploration
- **Large colony (50-100)**: Slower iterations, more diverse exploration

**Trade-off:** More ants = more evaluations per iteration = slower but better exploration

**Recommended values to test:** `[20, 50, 100]`
**Default:** `50` (good balance)

---

### 6. NUM_ITERATIONS - Total Iterations
**What it does:** How many iterations the algorithm runs.

- **Few iterations (50-100)**: Fast but may not converge
- **Many iterations (200-500)**: Slower but better convergence

**Note:** With 50 ants and 100 iterations = 5,000 evaluations total

**Recommended values to test:** `[50, 100, 200, 500]`
**Default:** `100` (reasonable for testing)

---

### 7. ELITIST_WEIGHT - Best Solution Boost
**What it does:** Extra pheromone deposited by the best-so-far solution.

- **Low weight (1.0-1.5)**: Minimal elitism
- **High weight (2.0-5.0)**: Strong elitism, faster convergence

**Recommended values to test:** `[1.0, 2.0, 3.0]`
**Default:** `2.0` (moderate elitism)

---

## 📊 Recommended Testing Strategy

### Phase 1: Baseline Test
```python
ALPHA = 1.0
BETA = 2.0
RHO = 0.3
Q = 100.0
NUM_ANTS = 50
NUM_ITERATIONS = 100
ELITIST_WEIGHT = 2.0
```

### Phase 2: Exploration vs Exploitation
Test different α/β combinations:
```python
# More exploration
ALPHA = 0.5, BETA = 1.0

# Balanced (current)
ALPHA = 1.0, BETA = 2.0

# More exploitation
ALPHA = 2.0, BETA = 3.0
```

### Phase 3: Evaporation Rate
Test different ρ values:
```python
RHO = 0.1  # Slow evaporation (exploitation)
RHO = 0.3  # Balanced (current)
RHO = 0.5  # Fast evaporation (exploration)
```

### Phase 4: Colony Size
Test different ant counts:
```python
NUM_ANTS = 20   # Small, fast
NUM_ANTS = 50   # Medium (current)
NUM_ANTS = 100  # Large, diverse
```

### Phase 5: Fine-tuning
Based on results from phases 1-4, adjust:
- Q (if convergence is too fast/slow)
- ELITIST_WEIGHT (if you want more/less greedy behavior)
- NUM_ITERATIONS (if not converging or converging too early)

---

## 🎯 What Values to Use for Comparison

For fair comparison with SA and GA, use:

```python
# Good balance for 10-vehicle intersection
NUM_ANTS = 50              # Similar to GA population size
NUM_ITERATIONS = 100       # Similar to GA generations
ALPHA = 1.0                # Balanced pheromone influence
BETA = 2.0                 # Moderate heuristic influence
RHO = 0.3                  # Moderate evaporation
Q = 100.0                  # Standard deposit amount
ELITIST_WEIGHT = 2.0       # Moderate elitism
```

This gives ~5,000 evaluations (50 ants × 100 iterations), comparable to SA and GA.

---

## 🔬 Experimental Design

Run each configuration **10 times** (use `OPTIMIZATION_ALGORITHM = 'ACO_ANALYSIS'`) and compare:

1. **Mean objective value** (lower is better)
2. **Standard deviation** (lower is more consistent)
3. **Convergence speed** (iterations to best solution)
4. **Final best solution quality**

---

## ⚠️ Important Notes

1. **ACO optimizes PERMUTATION ONLY** - Vehicle speeds are assigned deterministically based on:
   - Emergency vehicles get maximum speed
   - Normal vehicles get speed based on position (earlier = faster)
   
2. **Stochastic algorithm** - Results will vary between runs. Always run multiple times for statistical significance.

3. **No free lunch** - Best parameters depend on your specific problem instance. Test systematically!

4. **Computation cost** - Total evaluations = NUM_ANTS × NUM_ITERATIONS
   - Keep this similar to SA/GA for fair comparison

---

## 📈 Quick Parameter Sensitivity Test

To quickly see parameter effects, try these extreme cases:

```python
# Pure random (no learning)
ALPHA = 0.0, BETA = 0.0, RHO = 1.0  # Should perform poorly

# Pure greedy (no learning)
ALPHA = 0.0, BETA = 5.0, RHO = 1.0  # Similar to greedy heuristic

# Pure pheromone (no heuristic)
ALPHA = 2.0, BETA = 0.0, RHO = 0.1  # Learns but ignores domain knowledge

# Balanced (recommended)
ALPHA = 1.0, BETA = 2.0, RHO = 0.3  # Uses both learning and heuristic
```

Compare these to see how each component contributes to performance!
