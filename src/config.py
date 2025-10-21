"""
config.py
----------

Base configuration file for the Intersection Scheduling and Delay Minimization project.

This module defines all *fixed* simulation and optimization parameters that are shared
across the project. It includes:
    - Intersection geometry parameters
    - Vehicle and safety constants
    - Objective weighting factors
    - Simulation-level settings (time step, tolerances, etc.)

References:
-----------
Based on the problem formulation described in:
"Priority-Aware Autonomous Intersection Management" — Team 36, GUC (October 2025)
"""

# ==============================================================
# 1. GLOBAL PROJECT SETTINGS
# ==============================================================

# Random seed for reproducibility (used when generating test data or random permutations)
SEED = 42

# Time unit: seconds
# Distance unit: meters
# Speed unit: meters/second

# ==============================================================
# 2. INTERSECTION GEOMETRY PARAMETERS
# ==============================================================

# Approaches (directions)
APPROACHES = ["N", "E", "S", "W"]

# Maneuvers (Straight, Left, Right)
MANEUVERS = ["S", "L", "R"]

# Number of total conflict points (crossings + merges)
N_CONFLICTS = 20

# Conflict-point identifiers (16 crossing + 4 merging)
CONFLICT_POINTS = [
    # Central crossing points (C1–C16)
    *[f"C{i}" for i in range(1, 17)],
    # Merge points (MN, ME, MS, MW)
    "MN", "ME", "MS", "MW"
]

# Geometric tolerances
GEOMETRIC_TOLERANCE = 1e-2  # tolerance for point overlap (in meters)

# ==============================================================
# 3. SAFETY AND MOTION PARAMETERS
# ==============================================================

# Safe spacing between consecutive vehicles on the same approach (meters)
SAFE_SPACING = 7.5  # corresponds to ~0.5 car length + buffer

# Per-conflict-point time headway (clearance time, seconds)
# This is a typical constant headway across all conflict points unless geometry-specific data is used
DEFAULT_HEADWAY = 1.5  # seconds

# Bounds on per-vehicle constant speed (m/s)
# Typical CAV intersection speed range (10–60 km/h)
V_MIN = 2.8   # ≈ 10 km/h
V_MAX = 16.7  # ≈ 60 km/h

# Default spawn distance before first conflict point (m)
# Used when estimating earliest arrival times
SPAWN_DISTANCE = 25.0

# ==============================================================
# 4. OBJECTIVE WEIGHTING PARAMETERS
# ==============================================================

# α and β define the weighting between emergency and normal vehicles
ALPHA = 10.0   # Weight for emergency vehicle delay
BETA = 1.0     # Weight for all other vehicles

# ==============================================================
# 5. DECODER AND SIMULATION SETTINGS
# ==============================================================

# Minimum time increment used for scheduling updates (s)
TIME_STEP = 0.1

# Maximum number of vehicles in simulation (for array preallocation)
MAX_VEHICLES = 100

# Default penalty factor for constraint violations (used later by objective)
# — This is *not* an optimizer hyperparameter, only a decoder safeguard.
PENALTY_FACTOR = 1e4

# ==============================================================
# 6. GEOMETRY DESCRIPTION PLACEHOLDER
# ==============================================================

"""
The geometry is defined externally in `geometryy.py`, which will generate:

- PATHS: dictionary {vehicle_id → [conflict_points]}
- MERGE_MAP: dictionary defining which maneuvers merge at each Mo
- CONFLICT_MAP: dictionary defining which pairs of maneuvers share each Ci
- DISTANCES: per-vehicle distance to first conflict point
"""

# ==============================================================
# 7. VEHICLE SETUP PARAMETERS
# ==============================================================

# Example number of vehicles (placeholder for initialization)
N_VEHICLES = 12

# Fraction or IDs of emergency vehicles (E ⊂ V)
# These are fixed and not optimization variables
EMERGENCY_VEHICLES = [0, 5]  # Example: vehicles with IDs 0 and 5 are emergency vehicles

# Example vehicle dictionary structure (to be filled later by geometryy.py or data input)
VEHICLE_DATA_TEMPLATE = {
    "id": None,            # integer ID
    "approach": None,      # one of {"N", "E", "S", "W"}
    "maneuver": None,      # one of {"S", "L", "R"}
    "path": [],            # ordered list of conflict points (filled later)
    "distance": SPAWN_DISTANCE,
    "is_emergency": False
}

CONFLICT_SPACING = 6.0  # meters

# ==============================================================
# 8. PRINT CONFIG SUMMARY (for debugging)
# ==============================================================

def print_config_summary():
    """Prints a formatted summary of key configuration parameters."""
    print("\n=== INTERSECTION CONFIGURATION SUMMARY ===")
    print(f"Conflict Points     : {len(CONFLICT_POINTS)} ({CONFLICT_POINTS})")
    print(f"Approaches          : {APPROACHES}")
    print(f"Maneuvers           : {MANEUVERS}")
    print(f"Safe Spacing (m)    : {SAFE_SPACING}")
    print(f"Speed Range (m/s)   : [{V_MIN}, {V_MAX}]")
    print(f"Default Headway (s) : {DEFAULT_HEADWAY}")
    print(f"Objective Weights   : α={ALPHA}, β={BETA}")
    print(f"Emergency Vehicles  : {EMERGENCY_VEHICLES}")
    print(f"Max Vehicles        : {MAX_VEHICLES}")
    print("==========================================\n")


# ==============================================================
# End of config.py
# ==============================================================
