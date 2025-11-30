# Intersection Optimization — Visualization & Optimizers

This README explains how to run the project with visualization, how to run and tune the Genetic Algorithm (GA), how to run and compare GA vs Simulated Annealing (SA), and how to run batch/headless experiments.

Last updated: November 16, 2025

---

## Contents
- Quick start
- Visualization options (matplotlib, web, none)
- Running SA and GA
- Tuning GA parameters
- CLI usage and recommended output format
- Comparison protocol (GA vs SA)
- Batch experiments and examples
- Troubleshooting

---

## Quick start

1. Open the repository root and edit or run `main_with_viz.py`.
2. Choose visualization, optimizer and optional parameters (see CLI or variables).
3. Run:
   ```bash
   python main_with_viz.py
   ```


# --- 1. CHOOSE ALGORITHM --- in the main_with_viz.py
# 'SA':          Single run of SA (uses sa.MAX_TOTAL_ITERATIONS).
# 'GA':          Single run of GA (uses ga.NUM_GENERATIONS).
# 'SA_ANALYSIS': N-run statistical analysis of SA (natural stop).
# 'GA_ANALYSIS': N-run statistical analysis of GA (natural stop).
# 'BOTH':        Single run SA vs. GA (uses COMPARISON_EVALUATION_BUDGET).
# 'EXPERIMENT':  N-run statistical comparison of SA vs. GA (natural stops).

# --- 2. CHOOSE VISUALIZATION ---
# 'matplotlib', 'web', 'none'
# (Ignored for 'EXPERIMENT', 'SA_ANALYSIS', 'GA_ANALYSIS' modes)
VISUALIZATION_METHOD = 'matplotlib' 

# --- 3. ALGORITHM PARAMETERS ---
# Budget for *direct comparison modes only* ('BOTH')
COMPARISON_EVALUATION_BUDGET = 5000 
# Number of runs for 'SA_ANALYSIS', 'GA_ANALYSIS', 'EXPERIMENT'
NUM_EXPERIMENT_RUNS = 50  # 
RANDOM_SEED = 42 

OPTIMIZATION_ALGORITHM = 'GA' 
---

## Visualization options

Set method in `main_with_viz.py` or via CLI (see CLI section):

- 'matplotlib' — Local matplotlib animation (interactive window).
- 'web' — Flask + D3 web visualization (open browser at printed URL).
- 'none' — Headless mode (no visualization; best for batch runs).

Notes:
- Matplotlib may block execution until closed.
- Web requires Flask and template files under `templates/`.
- For headless servers use `'none'`.

---

## Running SA and GA

You can run either optimizer. Default variables in `config.py`.

Examples (headless):
- Run SA:
  ```bash
  python main_with_viz.py --optimizer sa --viz none
  ```
- Run GA:
  ```bash
  python main_with_viz.py --optimizer ga --viz none
  ```

Interactive (web) example:
```bash
python main_with_viz.py --optimizer ga --viz web
# Open the printed URL (e.g. http://localhost:5000)
```

---

## CLI arguments (recommended)

If not present, add an argparse block (example below) to `main_with_viz.py` so you can run without editing the file.

Suggested arguments:
- --viz {matplotlib,web,none}
- --optimizer {sa,ga}
- --seed N
- --out FILE (save JSON summary)
- GA overrides: --ga-pop, --ga-gen, --ga-cr, --ga-mr

Example:
```bash
python main_with_viz.py --optimizer ga --viz none --ga-pop 200 --ga-gen 500 --seed 123 --out results/ga_run1.json
```

---

## GA: parameters and tuning

Place tunable GA parameters inside `ga.py` or `config.py` (or override via CLI):

Example GA params:
```python
GA_PARAMS = {
    "POPULATION_SIZE": 100,
    "GENERATIONS": 200,
    "CROSSOVER_RATE": 0.8,
    "MUTATION_RATE": 0.02,
    "ELITISM": True,
    "ELITE_SIZE": 2,
    "TOURNAMENT_SIZE": 5,
    "RANDOM_SEED": None,
}
```

Tuning tips:
- Population size: larger increases exploration, costs evaluation time.
- Generations: more generations usually improve quality; combine with early stopping.
- Mutation rate: raise if premature convergence; lower if search becomes too random.
- Crossover rate: high values favor recombination; tune with mutation.
- Elitism: preserves best individuals — useful in noisy fitness landscapes.
- Use multiple seeds and aggregate statistics.

---

## SA: parameters and tuning

Typical SA parameters (in `main_with_viz.py` or `config.py`):
```python
T_INITIAL = 1000.0
T_MIN = 1.0
COOLING_RATE = 0.99
MAX_ITER_PER_TEMP = 20
MAX_TOTAL_ITERATIONS = 100000
RANDOM_SEED = None
```

Tuning tips:
- Increase T_INITIAL or reduce COOLING_RATE to improve escaping local minima.
- Increase MAX_ITER_PER_TEMP to allow more local search per temperature.
- Use fixed seeds for reproducibility when comparing methods.

---

## Recommended output format

Save run summaries to JSON/CSV for later analysis. Example JSON schema:
```json
{
  "optimizer": "ga",
  "seed": 42,
  "best_cost": 123.45,
  "time_seconds": 12.3,
  "evaluations": 3456,
  "iterations": 200,
  "best_solution": [ ... ],
  "history": [ ... ]  // optional per-iteration best cost or population stats
}
```

Have the runner write this file when `--out` is provided.

---

## Compare GA vs SA — protocol

1. Fix vehicle/config settings in `config.py`.
2. Choose common seeds or a seed list.
3. Run N repeats per optimizer (N >= 10 recommended).
4. Save summaries (JSON) with history, time, and final cost.
5. Compute metrics: best, mean, median, std, time-to-best.
6. Plot convergence curves using the saved histories.



Interpretation guide:
- Use time-to-best and quality to judge practical performance.
- Compare convergence curves for per-iteration or per-second performance.
- Use statistical tests (e.g., Wilcoxon) if needed.

---

## Batch experiments and automation

- Use 'none' visualization for batch runs.
- Parallelize independent runs across cores or machines.
- Save histories and final solutions to disk for offline plotting.

Example CI-friendly run (single run):
```bash
python main_with_viz.py --optimizer ga --viz none --seed 123 --ga-pop 150 --ga-gen 300 --out results/ga_123.json
```

---

## Example: add CLI parsing to main_with_viz.py

Patch suggestion (insert into `main_with_viz.py` near top and use parsed values later):



## Troubleshooting

- "Could not import matplotlib" — run `pip install matplotlib`.
- "Could not import Flask" — run `pip install flask`.
- Web server won't start — check port 5000 usage or change port in `visualization_utils.py`.
- GA seems stuck — increase mutation or population, or add diversity (restart runs).
- Ensure both GA and SA use the same evaluation function and constraints for fair comparison.

---

## Minimal sanity checks before experiments

- Fix `RANDOM_SEED` or pass `--seed` for reproducibility.
- Confirm `config.py` vehicle list and evaluation function are correct.
- Verify output (JSON) contains `best_cost` and `history` if you need convergence curves.

---

## Support

If problems persist:
- Inspect console logs.
- Confirm CLI arguments are parsed and applied.
- Check that `ga.py` consumes GA params from `GA_PARAMS`.
- Open an issue or attach run logs and sample JSON outputs.

--- 

End of README