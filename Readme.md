# Autonomous Intersection Management Platform (AIMP)
## Metaheuristic Optimization for Traffic Scheduling at Autonomous Intersections

**Course:** Optimization Techniques (MCTR1021) (Winter 2025)
**Institution:** German University in Cairo (GUC)  
**Last Updated:** January 2025

---

## 📖 Project Overview

### What is AIMP?

The Autonomous Intersection Management Platform (AIMP) is a comprehensive research and simulation platform designed to optimize vehicle scheduling at smart intersections using advanced metaheuristic algorithms. As autonomous vehicles become more prevalent, traditional traffic light systems become obsolete. This project explores how intelligent scheduling algorithms can minimize delays while prioritizing emergency vehicles.

### Course Context & Learning Objectives

This project is developed as part of the **Optimization Techniques** course, where we apply theoretical optimization concepts to real-world problems. Key learning objectives include:

1. **Algorithm Implementation**: Implementing and comparing multiple metaheuristic algorithms (SA, GA, ACO, PSO)
2. **Performance Analysis**: Conducting rigorous statistical comparisons of algorithm performance
3. **Constraint Handling**: Managing complex safety constraints (collision avoidance, speed limits)
4. **Multi-Objective Optimization**: Balancing competing objectives (emergency vehicle priority vs. overall delay)
5. **Visualization**: Creating interactive visualizations to communicate optimization results

### Why This Problem?

Autonomous intersection management is a critical problem in smart city infrastructure:

- **Real-World Impact**: Optimized intersections can reduce delays by 30-50% compared to traditional traffic lights
- **Safety Critical**: Vehicle scheduling must guarantee zero collisions
- **Computational Challenge**: The problem is NP-hard (combinatorial explosion as vehicles increase)
- **Multi-Objective**: Must balance individual delays, total delay, and emergency vehicle priority
- **Research Relevance**: Active area of research in autonomous vehicles and smart transportation

### Project Goals

1. **Implement 5 metaheuristic algorithms** from scratch (SA, GA, ACO, PSO, Hybrid)
2. **Compare algorithm performance** on standardized metrics (objective value, convergence speed, consistency)
3. **Visualize solutions** with interactive animations showing vehicle movements
4. **Analyze results** using statistical tests (ANOVA, t-tests) to determine best algorithm
5. **Document findings** in a comprehensive report with reproducible experiments


## 📁 Project Structure

```
AIMP/
├── src/                          # Source code
│   ├── config.py                 # 🔧 Configuration: vehicles, parameters
│   ├── main_with_viz.py          # ⭐ MAIN ENTRY POINT (run this!)
│   ├── main.py                   # Basic test runner (optional)
│   ├── analyze.py                # Statistical analysis tools
│   ├── utils.py                  # Helper functions
│   │
│   ├── engine/                   # Core intersection logic (DO NOT MODIFY)
│   │   ├── vehicle.py            # Vehicle class definition
│   │   ├── geometry.py           # Intersection geometry & conflict points
│   │   ├── decoder.py            # Solution validation & collision detection
│   │   └── objective.py          # Objective function calculation
│   │
│   ├── metaheuristics/           # 🧬 Optimization algorithms (CUSTOMIZE HERE)
│   │   ├── sa.py                 # Simulated Annealing
│   │   ├── ga.py                 # Genetic Algorithm
│   │   ├── aco.py                # Ant Colony Optimization
│   │   ├── pso.py                # Particle Swarm Optimization
│   │   └── sequential_hybrid.py  # Hybrid ACO+PSO approach
│   │
│   └── visualization/            # Visualization components
│       ├── visualization.py      # Matplotlib animation
│       ├── visualization_utils.py # Helper functions
│       ├── visualization_server.py # Flask web server
│       └── templates/
│           └── intersection.html # Web-based D3.js visualization
│
├── experiment_summary_log.csv    # 📊 Output: Experiment results summary
├── experiment_raw_data_log.csv   # 📊 Output: Detailed run data
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ACO_HYPERPARAMETERS.md       # ACO parameter tuning guide
└── .gitignore                    # Git ignore rules
```

**Key Files to Know:**

| File | Purpose | When to Use |
|------|---------|-------------|
| [`config.py`](src/config.py) | Define vehicles, parameters | Change problem instance |
| [`main_with_viz.py`](src/main_with_viz.py) | Run algorithms, experiments | **Always use this** |
| [`metaheuristics/*.py`](src/metaheuristics/) | Algorithm implementations | Tune parameters |
| [`*.csv`](experiment_summary_log.csv) | Experiment outputs | Analyze results |

---

## 🚀 Getting Started

### 1. Installation

**Prerequisites:**
- Python 3.8 or higher
- pip (Python package manager)

**Install Dependencies:**

```bash
# Clone the repository
git clone <repository-url>
cd AIMP

# Install required packages
pip install -r requirements.txt
```

**Required Packages:**
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `flask` - Web server for interactive visualization
- `pandas` - Data analysis (optional)
- `scipy` - Statistical tests (optional)

**Verify Installation:**

```bash
# Quick test (should print vehicle queue info)
cd src
python main.py
```

---

## 📚 Step-by-Step Usage Guide

### STEP 1: Define the Problem Instance

**File:** [`src/config.py`](src/config.py)

This file defines your intersection scenario (vehicles, speeds, priorities).

#### Vehicle Parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `vehicle_id` | int | Unique identifier | `1, 2, 3, ...` |
| `approach` | str | Entry direction | `"N"`, `"E"`, `"S"`, `"W"` |
| `maneuver` | str | Movement type | `"S"` (straight), `"L"` (left), `"R"` (right) |
| `priority_status` | bool | Emergency vehicle? | `True` / `False` |
| `velocity` | tuple | Speed range (min, max) | `(20, 50)` |


### STEP 2: Choose Algorithm and Run Mode

**File:** [`src/main_with_viz.py`](src/main_with_viz.py)  
**Lines:** 137-144

#### Configuration Settings:

```python
# =============================================================================
# CONFIGURATION
# =============================================================================

# --- 1. CHOOSE ALGORITHM ---
OPTIMIZATION_ALGORITHM = 'GA'  # Change this!

# --- 2. CHOOSE VISUALIZATION ---
VISUALIZATION_METHOD = 'matplotlib'  # or 'web', 'none'

# --- 3. EXPERIMENT SETTINGS ---
NUM_EXPERIMENT_RUNS = 30  # For analysis modes
RANDOM_SEED = 42          # For reproducibility (None = random)
```

#### Available Run Modes:

| Mode | Code | Purpose | Use Case |
|------|------|---------|----------|
| **Single Algorithm Runs** |
| Simulated Annealing | `'SA'` | Run SA once with visualization | Test SA, see animation |
| Genetic Algorithm | `'GA'` | Run GA once with visualization | Test GA, see animation |
| Ant Colony Optimization | `'ACO'` | Run ACO once with visualization | Test ACO, see animation |
| Particle Swarm Optimization | `'PSO'` | Run PSO once with visualization | Test PSO, see animation |
| Hybrid ACO+PSO | `'HYBRID'` | Run hybrid once with visualization | Test hybrid approach |
| **Statistical Analysis (N Runs)** |
| SA Analysis | `'SA_ANALYSIS'` | Run SA N times, compute stats | Evaluate SA performance |
| GA Analysis | `'GA_ANALYSIS'` | Run GA N times, compute stats | Evaluate GA performance |
| ACO Analysis | `'ACO_ANALYSIS'` | Run ACO N times, compute stats | Evaluate ACO performance |
| PSO Analysis | `'PSO_ANALYSIS'` | Run PSO N times, compute stats | Evaluate PSO performance |
| Hybrid Analysis | `'HYBRID_ANALYSIS'` | Run Hybrid N times, compute stats | Evaluate Hybrid performance |
| **Comparative Experiments** |
| Three-Way Comparison | `'EXPERIMENT'` | Compare SA vs GA vs Hybrid (N runs each) | **Final report comparison** |
| Two-Way (Budget) | `'BOTH'` | Compare SA vs GA with fixed evaluation budget | Fair comparison |

#### Visualization Options:

| Option | Description | Best For |
|--------|-------------|----------|
| `'matplotlib'` | Interactive animation window | Single runs, debugging |
| `'web'` | Flask web server (browser-based) | Presentations, demos |
| `'none'` | No visualization (fastest) | Batch experiments, analysis |

---

### STEP 3: Run the Platform

#### Basic Single Run:

```bash
cd src
python main_with_viz.py
```

**What Happens:**
1. Algorithm runs (progress printed to console)
2. Best solution found is displayed
3. Performance plots appear (close to continue)
4. Animation shows vehicle movements (if visualization enabled)

#### Console Output Example:

```
=== GENETIC ALGORITHM ===
Population Size:   50
Max Generations:  200
Crossover Rate:   0.8
Mutation Rate:    0.2
========================

Generation 1/200: Best=543.21, Avg=678.90
Generation 10/200: Best=421.34, Avg=489.12
  ✓ NEW BEST = 421.34
Generation 20/200: Best=398.56, Avg=445.23
  ✓ NEW BEST = 398.56
...
Generation 200/200: Best=345.67, Avg=378.90

======================================================================
GENETIC ALGORITHM COMPLETE
======================================================================
  Best Objective:     345.67
  Total Evaluations:  10050
  Runtime:            42.3 seconds
  
Best Solution Found:
  Emergency Delay (f_em):  12.34s
  Total Delay (f_all):     234.56s
  Weighted Objective (f):  345.67
  Permutation (IDs):       [4, 1, 7, 2, 5, 3, 6, 8]
  Speeds:                  [45.2, 50.0, 38.7, 42.1, 35.5, 48.3, 40.0, 37.2]
======================================================================

Displaying GA performance plots...
  (Close all plot windows to continue to animation)
```

---

### STEP 4: Understanding Outputs

#### 📊 Performance Dashboard (Plots)

When algorithm completes, you'll see a **2×2 dashboard** (or 2×3 for comparisons):

**Single Algorithm Dashboard:**

| Plot | What It Shows | Interpretation |
|------|---------------|----------------|
| **Top-Left: Convergence** | Best objective value over iterations | Faster drop = faster convergence |
| **Top-Right: Diversity/Temp** | Population diversity (GA) or Temperature (SA) | High = exploring, Low = exploiting |
| **Bottom-Left: Delay Components** | Emergency vs Total delay over time | Shows priority balancing |
| **Bottom-Right: Speed Distribution** | Histogram of assigned speeds | Check if speeds are reasonable |

**Three-Way Comparison Dashboard (EXPERIMENT mode):**

| Plot | What It Shows |
|------|---------------|
| **Convergence Overlay** | All algorithms on same axes (see which converges fastest) |
| **Final Quality Bar Chart** | Winner highlighted in gold |
| **Efficiency Metrics** | Cost reduction per 1000 evaluations |
| **Delay Breakdowns** | Emergency, total, average delays for all algorithms |

**How to Read Convergence Plots:**

```
Good Convergence:
Objective
   500│ ╲
      │  ╲___
   400│      ╲___
      │          ╲____
   300│               ╲___________  (plateau = converged)
      └────────────────────────────>
         Iterations

Bad Convergence (stuck):
Objective
   500│ ╲
      │  ╲
   450│   ╲_____________________ (flat too early)
      │
   400│
      └────────────────────────────>
         Iterations
```

#### 📁 CSV Output Files

**1. Summary Log: [`experiment_summary_log.csv`](experiment_summary_log.csv)**

One row per experiment (aggregate statistics):

```csv
Timestamp,Algorithm,Mean Cost (f),Std Deviation (f),Avg Run Time (s),Best Cost Overall,Num Runs,Avg Evals Used
2025-01-15 10:30:00,GA,345.67,8.23,42.1,332.45,30,10050
2025-01-15 11:00:00,SA,389.12,12.45,38.9,365.23,30,8234
2025-01-15 11:30:00,HYBRID,328.91,6.78,58.2,312.56,30,15000
```

**Columns Explained:**
- **Mean Cost**: Average objective across all runs
- **Std Deviation**: Consistency (lower = more reliable)
- **Best Cost Overall**: Best solution found in any run
- **Avg Evals Used**: Average fitness evaluations (computational cost)

**2. Raw Data Log: [`experiment_raw_data_log.csv`](experiment_raw_data_log.csv)**

One row per individual run:

```csv
Run,Algorithm,Objective,Evals,Time,Permutation,Speeds
1,GA,345.67,10000,42.2,[4,1,7,2,5,3,6],[45.2,38.7,50.0,...]
2,GA,352.13,10000,41.8,[4,7,1,2,5,3,6],[43.1,40.2,48.5,...]
3,GA,338.45,10000,43.1,[1,4,7,2,5,3,6],[50.0,45.2,38.7,...]
...
```

**Use Cases:**
- **Summary Log**: Quick comparison of algorithms (mean, std, best)
- **Raw Data Log**: Detailed analysis (variance, outliers, convergence patterns)

#### 🎬 Animation Output

**Matplotlib Mode:**

A window appears showing:
- Intersection layout with conflict points
- Vehicles moving along their paths
- Color coding: Blue=normal, Red=emergency
- Curved trajectories for left turns

**Controls:**
- Toolbar: Zoom, pan, save image
- Close window to exit

**Web Mode:**

Browser opens at `http://localhost:5000` showing:
- Interactive D3.js animation
- Vehicle info on hover
- Playback controls
- Modern responsive design



## 🧪 Running Experiments for Your Report

### Recommended Experiment Workflow:

#### 1. Quick Test (5 minutes)

Test each algorithm works:

```python
# main_with_viz.py
OPTIMIZATION_ALGORITHM = 'GA'  # Test SA, GA, ACO, PSO, HYBRID
NUM_EXPERIMENT_RUNS = 1
VISUALIZATION_METHOD = 'matplotlib'
```

Run: `python main_with_viz.py`

**Check:**
- ✅ Algorithm completes without errors
- ✅ Plots display correctly
- ✅ Animation shows vehicles moving

#### 2. Small-Scale Analysis (30 minutes)

Get initial performance data:

```python
OPTIMIZATION_ALGORITHM = 'EXPERIMENT'  # Compares SA, GA, Hybrid
NUM_EXPERIMENT_RUNS = 10
VISUALIZATION_METHOD = 'none'  # Faster
```

**Output:**
- Summary statistics (mean, std)
- Box plots comparing algorithms
- CSV logs for detailed analysis

#### 3. Full Experimental Run (2-4 hours)

Final data for report:

```python
OPTIMIZATION_ALGORITHM = 'EXPERIMENT'
NUM_EXPERIMENT_RUNS = 30  # 50 for publication-quality
VISUALIZATION_METHOD = 'none'
RANDOM_SEED = 42  # Reproducibility
```

**Run overnight or on powerful machine**

**Deliverables:**
- `experiment_summary_log.csv` (aggregate stats)
- `experiment_raw_data_log.csv` (per-run data)
- Performance dashboards (save as PNG)
- Statistical test results (ANOVA, t-tests)

#### 4. Individual Algorithm Deep Dive

For each algorithm in your report section:

```python
OPTIMIZATION_ALGORITHM = 'GA_ANALYSIS'  # or SA_ANALYSIS, ACO_ANALYSIS
NUM_EXPERIMENT_RUNS = 30
VISUALIZATION_METHOD = 'none'
```

**Analyze:**
- Convergence patterns
- Parameter sensitivity
- Failure cases
- Best/worst runs

---

## 📊 Interpreting Results for Your Report

### Key Metrics to Report:

| Metric | What It Measures | How to Get It |
|--------|------------------|---------------|
| **Mean Objective** | Average solution quality | From `experiment_summary_log.csv` |
| **Std Deviation** | Consistency/reliability | From summary log (lower = better) |
| **Best Found** | Best solution across all runs | From summary log |
| **Convergence Rate** | How fast algorithm improves | Plot slope analysis |
| **Computational Cost** | Total evaluations/runtime | From summary log |
| **Success Rate** | % of runs finding "good" solution | Count runs below threshold |

### Statistical Tests:

**ANOVA (Are algorithms different?):**
- **p < 0.05**: Yes, significant difference
- **p ≥ 0.05**: No significant difference

**Pairwise t-tests (Which is better?):**
- Compare each pair of algorithms
- Report p-values and winners
- Use Bonferroni correction if needed

### Sample Report Table:

```
Table 1: Algorithm Performance Comparison (30 runs each)

Algorithm  | Mean (f) | Std Dev | Best (f) | Worst (f) | Avg Time (s) | Avg Evals
-----------|----------|---------|----------|-----------|--------------|----------
SA         | 389.12   | 12.45   | 365.23   | 418.90    | 38.9         | 8234
GA         | 345.67   | 8.23    | 332.45   | 367.12    | 42.1         | 10050
ACO        | 368.45   | 10.12   | 352.67   | 391.23    | 51.3         | 20000
PSO        | 356.78   | 11.34   | 341.23   | 382.45    | 35.6         | 3000
Hybrid     | 328.91*  | 6.78*   | 312.56*  | 344.12    | 58.2         | 15000

* Statistically significant best (p < 0.01)
```

### Sample Report Figures:

**Figure 1: Convergence Comparison**
- Overlay plot showing all algorithms
- Caption: "GA converges fastest, Hybrid achieves best final solution"

**Figure 2: Distribution Box Plots**
- Shows variance and outliers
- Caption: "Hybrid has lowest variance (most reliable)"

**Figure 3: Efficiency Analysis**
- Cost reduction per 1000 evaluations
- Caption: "PSO most efficient early, GA catches up later"

---


## 📖 Algorithm Background (For Report Introduction)

### Simulated Annealing (SA)

**Inspiration:** Metallurgical annealing process  
**Type:** Single-solution, local search with probabilistic acceptance  
**Strengths:** Simple, escapes local optima, good for continuous optimization  
**Weaknesses:** Sensitive to cooling schedule, single-threaded exploration  
**Best for:** Problems with smooth objective landscapes

**Key Papers:**
- Kirkpatrick et al. (1983) - "Optimization by Simulated Annealing"

### Genetic Algorithm (GA)

**Inspiration:** Natural evolution and genetics  
**Type:** Population-based, evolutionary  
**Strengths:** Maintains diversity, explores multiple regions, good global search  
**Weaknesses:** Many parameters to tune, higher computational cost  
**Best for:** Large discrete search spaces, multi-modal problems

**Key Papers:**
- Goldberg (1989) - "Genetic Algorithms in Search, Optimization and Machine Learning"

### Ant Colony Optimization (ACO)

**Inspiration:** Foraging behavior of ants  
**Type:** Population-based, pheromone-guided construction  
**Strengths:** Exploits problem structure, good for combinatorial problems  
**Weaknesses:** Many parameters, can converge prematurely  
**Best for:** Sequencing problems, graph problems (TSP, routing)

**Key Papers:**
- Dorigo & Stützle (2004) - "Ant Colony Optimization"

### Particle Swarm Optimization (PSO)

**Inspiration:** Flocking behavior of birds/fish  
**Type:** Population-based, velocity-driven  
**Strengths:** Fast convergence, few parameters, good for continuous optimization  
**Weaknesses:** Struggles with discrete variables, can get stuck in local optima  
**Best for:** Continuous optimization, speed fine-tuning

**Key Papers:**
- Kennedy & Eberhart (1995) - "Particle Swarm Optimization"

### Hybrid ACO+PSO

**Type:** Sequential two-phase approach  
**Rationale:** Leverage ACO's strength in permutation optimization and PSO's strength in continuous optimization  
**Strengths:** Combines best of both worlds, separates problem concerns  
**Weaknesses:** Longer runtime, more parameters  
**Best for:** Mixed integer-continuous optimization problems

---


## 🎓 Citation & References

If you use this platform in your research or report:

```bibtex
@software{aimp2025,
  title={Autonomous Intersection Management Platform: Metaheuristic Optimization for Traffic Scheduling},
  author={[Your Names]},
  institution={German University in Cairo},
  course={Optimization Techniques (OPT25)},
  year={2025},
  url={[Repository URL]}
}
```

**Related Work:**
- Dresner & Stone (2004) - "Multiagent coordination for traffic light control"
- Li et al. (2013) - "Conflict-free cooperative adaptive cruise control"
- [Add your references]

---


**Good luck with your project! 🚀**

---

**End of README** | Version 3.0 - Course Project Edition | January 2025