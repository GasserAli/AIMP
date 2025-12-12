# Dragonfly Algorithm Debug Guide

## Overview
This guide explains the debug print statements added to `dragonfly.py` to help identify bottlenecks and track execution flow.

## Debug Print Levels

### 1. **High-Level Stage Tracking** (`[DEBUG]` prefix)
These prints track major stages and steps:

**Stage 1 (Discrete DA - Permutation Optimization):**
- `[DEBUG] Stage 1 - Step 1: Initializing discrete swarm...`
- `[DEBUG] Stage 1 - Step 2: Evaluating initial swarm...`
- `[DEBUG] Stage 1 - Step 3: Identifying food (best) and enemy (worst)...`
- `[DEBUG] Stage 1 - Step 4: Starting main optimization loop...`
- `[DEBUG] Stage 1 - Step 4.1: Updating dragonflies (iteration X)...` (every 10 iterations)
- `[DEBUG] Stage 1 - Step 4.2: Applying local search...` (when triggered)
- `[DEBUG] Stage 1 - Step 4.3: Updating food and enemy...` (every 10 iterations)
- `[DEBUG] Stage 1 - Step 5: Optimization loop complete`
- `[DEBUG] Stage 1 - NEW BEST FOUND at iteration X: Y.YY`

**Stage 2 (Continuous DA - Speed Optimization):**
- Similar structure to Stage 1

**Main Optimize Method:**
- `[DEBUG] MAIN OPTIMIZE - Starting two-stage optimization...`
- `[DEBUG] MAIN OPTIMIZE - Calling Stage 1...`
- `[DEBUG] MAIN OPTIMIZE - Stage 1 returned with fitness: X.XX`
- `[DEBUG] MAIN OPTIMIZE - Calling Stage 2...`
- `[DEBUG] MAIN OPTIMIZE - Stage 2 returned with fitness: X.XX`
- `[DEBUG] MAIN OPTIMIZE - Evaluating final solution...`
- `[DEBUG] MAIN OPTIMIZE - Total runtime: X.XX seconds`
- `[DEBUG] MAIN OPTIMIZE - COMPLETE: Returning final results`

### 2. **Detailed Loop Tracking** (`[DEBUG-DETAIL]` prefix)
These prints show detailed operations for **iteration 0 only** to catch early issues:

For each dragonfly in the first iteration:
```
[DEBUG-DETAIL] Iter 1, DF 0: Starting update...
[DEBUG-DETAIL] Iter 1, DF 0: Finding neighbors...
[DEBUG-DETAIL] Iter 1, DF 0: Found N neighbors
[DEBUG-DETAIL] Iter 1, DF 0: Calculating separation...
[DEBUG-DETAIL] Iter 1, DF 0: Calculating alignment...
[DEBUG-DETAIL] Iter 1, DF 0: Calculating cohesion...
[DEBUG-DETAIL] Iter 1, DF 0: Swarm behaviors calculated (S:X, A:Y, C:Z)
[DEBUG-DETAIL] Iter 1, DF 0: Calculating food attraction...
[DEBUG-DETAIL] Iter 1, DF 0: Calculating enemy repulsion...
[DEBUG-DETAIL] Iter 1, DF 0: Food/Enemy calculated (F:X, E:Y)
[DEBUG-DETAIL] Iter 1, DF 0: Updating velocity...
[DEBUG-DETAIL] Iter 1, DF 0: Velocity updated (new velocity length: X)
[DEBUG-DETAIL] Iter 1, DF 0: Updating position...
[DEBUG-DETAIL] Iter 1, DF 0: Position updated
[DEBUG-DETAIL] Iter 1, DF 0: Evaluating fitness...
[DEBUG-DETAIL] Iter 1, DF 0: Fitness = X.XX
[DEBUG-DETAIL] Iter 1, DF 0: Applying local search... (if triggered)
[DEBUG-DETAIL] Iter 1, DF 0: Local search complete
```

### 3. **Velocity Update Tracking** (`[DEBUG-VELOCITY]` prefix)
**Currently COMMENTED OUT** but can be enabled by uncommenting lines in `DiscreteDragonfly.update_velocity()`:

```python
# Uncomment these lines in update_velocity() method (lines ~254-276)
print(f"[DEBUG-VELOCITY] DF {self.id}: Scaling sequences...")
print(f"[DEBUG-VELOCITY] DF {self.id}: Scaled lengths - S:{len(scaled_S)}, A:{len(scaled_A)}, C:{len(scaled_C)}, F:{len(scaled_F)}, E:{len(scaled_E)}, Inertia:{len(scaled_inertia)}")
print(f"[DEBUG-VELOCITY] DF {self.id}: Merging S+A...")
print(f"[DEBUG-VELOCITY] DF {self.id}: Merged S+A, length={len(new_velocity)}")
# ... etc
```

## Identifying Bottlenecks

### Suspected Issue: Velocity Explosion
Based on the stack trace showing the code stuck in `merge_swap_sequences`, the likely issue is:

**Problem:** The velocity (swap sequence) grows exponentially with each iteration because:
1. Each iteration merges 5-6 swap sequences (S, A, C, F, E, Inertia)
2. The inertia term includes the previous velocity
3. This creates exponential growth: velocity gets longer → next iteration merges even longer sequences → etc.

**To Diagnose:**
1. Run the code and watch for `[DEBUG-DETAIL]` prints
2. Look at "new velocity length" values - if they grow rapidly (100s, 1000s), that's the problem
3. Enable `[DEBUG-VELOCITY]` prints to see exactly which merge operation is slow

**Potential Solutions:**
1. **Limit velocity length:** Cap the maximum number of swaps in a velocity
2. **Sample swaps:** Instead of keeping all swaps, randomly sample a fixed number
3. **Reduce inertia weight:** Lower the `w` parameter to reduce velocity accumulation
4. **Clear velocity periodically:** Reset velocity every N iterations

## Running with Debug Output

```bash
cd "d:\Uni\9-Ninth semester\MCTR1021-Optimization\Project\AIMP\src"
python metahueristics/dragonfly.py
```

The output will show:
- High-level progress every 10 iterations
- Detailed operation tracking for iteration 1 only
- New best solutions as they're found

## Filtering Debug Output

To see only specific debug levels:
```bash
# Only high-level stage tracking
python metahueristics/dragonfly.py | findstr "[DEBUG] "

# Only detailed iteration 1 tracking
python metahueristics/dragonfly.py | findstr "[DEBUG-DETAIL]"

# Only velocity updates (if enabled)
python metahueristics/dragonfly.py | findstr "[DEBUG-VELOCITY]"
```

## Next Steps

1. **Run the code** and observe the first iteration output
2. **Check velocity lengths** - if they're growing exponentially, implement velocity limiting
3. **Enable velocity prints** if you need more granular tracking
4. **Profile specific operations** that appear slow
