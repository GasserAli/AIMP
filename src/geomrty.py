"""
geometryy.py
-------------

Defines the *logical* intersection structure for the unsignalized four-way intersection.

This version does NOT use physical coordinates or box dimensions.
Instead, it defines:
    - Lanes (approaches)
    - Allowed maneuvers per approach
    - Conflict points per maneuver
    - Merge relationships
    - Vehicle queues per lane

All distances and safety parameters are imported from `config.py`.
"""

# ==============================================================
# Imports
# ==============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import config

# ==============================================================
# 1. BASIC DATA STRUCTURES
# ==============================================================

@dataclass
class ConflictPoint:
    """Symbolic definition of a single conflict or merge point."""
    pid: str                    # ID (e.g., "C1", "MN")
    headway: float = config.DEFAULT_HEADWAY  # time headway τ_p [s]


@dataclass
class Path:
    """Ordered conflict-point sequence for one approach/maneuver."""
    approach: str                # One of {"N", "E", "S", "W"}
    maneuver: str                # One of {"S", "L", "R"}
    conflicts: List[str]         # List of conflict point IDs


@dataclass
class Vehicle:
    """Simplified logical vehicle definition."""
    vid: int
    approach: str
    maneuver: str
    path: List[str]
    queue_index: int
    distance_to_first_conflict: float


@dataclass
class LaneQueue:
    """Queue of vehicles for a given approach."""
    approach: str
    vehicles: List[Vehicle] = field(default_factory=list)

    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to the queue (appended at the end)."""
        self.vehicles.append(vehicle)

    def pop_front(self):
        """Pop the first vehicle (when it enters the intersection)."""
        if self.vehicles:
            return self.vehicles.pop(0)
        return None

    def __len__(self):
        return len(self.vehicles)


@dataclass
class Approach:
    """Logical lane entry into the intersection."""
    name: str                    # "N", "E", "S", "W"
    maneuvers: Dict[str, Path] = field(default_factory=dict)
    queue: LaneQueue = field(default_factory=lambda: LaneQueue(""))

# ==============================================================
# 2. CONSTANTS
# ==============================================================

CONFLICT_SPACING = 6.0  # Logical distance between consecutive conflict points (m)

CROSSING_POINTS = [f"C{i}" for i in range(1, 17)]
MERGE_POINTS = ["MN", "ME", "MS", "MW"]
CONFLICT_POINTS = CROSSING_POINTS + MERGE_POINTS

# ==============================================================
# 3. PATH DEFINITIONS PER APPROACH / MANEUVER
# ==============================================================

PATHS: Dict[Tuple[str, str], Path] = {
    # North approach
    ("N", "S"): Path("N", "S", ["C1", "C2", "C3", "C4", "MS"]),
    ("N", "L"): Path("N", "L", ["C5", "C9", "C13", "MW"]),
    ("N", "R"): Path("N", "R", ["ME"]),

    # East approach
    ("E", "S"): Path("E", "S", ["C4", "C8", "C12", "C16", "MW"]),
    ("E", "L"): Path("E", "L", ["C3", "C7", "C11", "MN"]),
    ("E", "R"): Path("E", "R", ["MS"]),

    # South approach
    ("S", "S"): Path("S", "S", ["C13", "C14", "C15", "C16", "MN"]),
    ("S", "L"): Path("S", "L", ["C9", "C10", "C11", "ME"]),
    ("S", "R"): Path("S", "R", ["MW"]),

    # West approach
    ("W", "S"): Path("W", "S", ["C1", "C5", "C9", "C13", "ME"]),
    ("W", "L"): Path("W", "L", ["C2", "C6", "C10", "MS"]),
    ("W", "R"): Path("W", "R", ["MN"])
}

# ==============================================================
# 4. MERGE RELATIONSHIPS
# ==============================================================

MERGE_MAP = {
    "MN": [("S", "S"), ("E", "L")],
    "ME": [("W", "S"), ("N", "L")],
    "MS": [("N", "S"), ("E", "R")],
    "MW": [("E", "S"), ("S", "L")]
}

# ==============================================================
# 5. LANE / APPROACH REGISTRY WITH QUEUES
# ==============================================================

APPROACHES: Dict[str, Approach] = {}

for approach in config.APPROACHES:
    allowed = {m: PATHS[(approach, m)] for m in config.MANEUVERS if (approach, m) in PATHS}
    lane_queue = LaneQueue(approach)
    APPROACHES[approach] = Approach(approach, allowed, lane_queue)

# ==============================================================
# 6. UTILITY FUNCTIONS
# ==============================================================

def generate_vehicle_path(approach: str, maneuver: str) -> List[str]:
    """Return the conflict sequence for a given (approach, maneuver)."""
    if (approach, maneuver) not in PATHS:
        raise ValueError(f"No path defined for {approach}:{maneuver}")
    return PATHS[(approach, maneuver)].conflicts


def distance_to_first_conflict(queue_index: int) -> float:
    """Compute distance from spawn point to first conflict."""
    return (queue_index - 1) * config.SAFE_SPACING


def travel_distance(num_conflicts: int) -> float:
    """Approximate total path length given number of conflict points."""
    return num_conflicts * CONFLICT_SPACING


def add_vehicle_to_lane(approach: str, maneuver: str, vid: int):
    """
    Add a new vehicle to the queue of a given approach.
    Automatically assigns queue index, distance, and path.
    """
    lane = APPROACHES[approach].queue
    queue_index = len(lane) + 1
    path = generate_vehicle_path(approach, maneuver)
    dist = distance_to_first_conflict(queue_index)
    v = Vehicle(vid=vid, approach=approach, maneuver=maneuver,
                path=path, queue_index=queue_index,
                distance_to_first_conflict=dist)
    lane.add_vehicle(v)
    return v


def get_lane_queues() -> Dict[str, List[int]]:
    """Return queue composition (vehicle IDs) for each lane."""
    summary = {}
    for name, app in APPROACHES.items():
        summary[name] = [v.vid for v in app.queue.vehicles]
    return summary

# ==============================================================
# 7. DEBUG / SUMMARY PRINTING
# ==============================================================

def print_geometry_summary():
    print("\n=== LOGICAL INTERSECTION GEOMETRY ===")
    print(f"Approaches: {list(APPROACHES.keys())}")
    print(f"Conflict points: {len(CONFLICT_POINTS)} total ({CROSSING_POINTS} + {MERGE_POINTS})")
    for key, path in PATHS.items():
        print(f"{key}: {path.conflicts}")
    print("======================================\n")


def print_queue_summary():
    print("\n=== CURRENT LANE QUEUES ===")
    for name, app in APPROACHES.items():
        if len(app.queue) == 0:
            print(f"{name}: (empty)")
        else:
            q_ids = [v.vid for v in app.queue.vehicles]
            print(f"{name}: Vehicle IDs → {q_ids}")
    print("======================================\n")


# ==============================================================
# 8. SELF-TEST
# ==============================================================

if __name__ == "__main__":
    print_geometry_summary()

    # Example: Add vehicles to each lane
    add_vehicle_to_lane("N", "S", 0)
    add_vehicle_to_lane("N", "L", 1)
    add_vehicle_to_lane("E", "R", 2)
    add_vehicle_to_lane("S", "S", 3)
    add_vehicle_to_lane("S", "L", 4)
    add_vehicle_to_lane("W", "R", 5)

    print_queue_summary()

    # Example outputs
    print("Example: (E,L) path =", generate_vehicle_path("E", "L"))
    print("Example: Vehicle 3 distance =", distance_to_first_conflict(3))
