# FILE: visualization.py
#
# --- MODIFIED: This file now uses D3.js/HTML-style "smooth" animation logic ---
# It no longer uses the decoder's time-accurate schedule.
# This avoids the "teleporting" (waiting) effect.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Patch
import numpy as np
import config 
import math
from typing import List, Dict, Tuple

try:
    from engine.vehicle import Vehicle
    from engine.geometry import Geometry 
except ImportError:
    print("Error: Could not import Vehicle or Geometry class from vehicle.py/geometry.py")
    class Vehicle: pass
    class Geometry: pass

# --- Constants for Layout (Copied from original visualization.py) ---
VEHICLE_RADIUS = 2.0 
QUEUE_SPACING_VIS_FACTOR = 4.0 
LANE_WIDTH = 20.0 
QUEUE_VIS_BUFFER = 60.0
MIN_X = -30 - QUEUE_VIS_BUFFER
MAX_X = 100 + QUEUE_VIS_BUFFER
MIN_Y = -30 - QUEUE_VIS_BUFFER
MAX_Y = 100 + QUEUE_VIS_BUFFER
EXTENT_X = MAX_X - MIN_X
EXTENT_Y = MAX_Y - MIN_Y

# --- Hardcoded Coordinate Map (Copied from original visualization.py) ---
POINT_COORDINATES = {
    'C1': (0, 60),  'C2': (20, 60),  'C3': (40, 60),  'C4': (60, 60),
    'C5': (0, 40),  'C6': (30, 40),  'C7': (60, 40),  'C8': (20, 30),
    'C9': (40, 30), 'C11': (0, 20), 'C10': (30, 20), 'C12': (60, 20),
    'C13': (0, 0),  'C14': (20, 0),  'C15': (40, 0),  'C16': (60, 0),
    'M_W': (-20, 60), 'M_S': (0, -20), 'M_E': (80, 0), 'M_N': (60, 80),
    'S_N': (0, 80),   'S_E': (80, 60),   'S_S': (60, -20), 'S_W': (-20, 0),
}
# --- END MAP ---


# --- Helper Functions (Copied from original visualization.py) ---
# These functions build the geometric path (the 'polyline')
# They are used by the animator to know the (x,y) coordinates
# and the total distance.

def get_trajectory_coords(vehicle: 'Vehicle', path_names: List[str]) -> List[Tuple[float, float]]:
    """
    Builds the full (x,y) coordinate path for a vehicle,
    including the 'p_behind' queue point and the final exit point.
    """
    coords = []
    approach = getattr(vehicle, 'approach', None)
    base_key = f"S_{approach}" if approach else None
    
    if not (base_key and base_key in POINT_COORDINATES):
        print(f"Warning: Missing base start {base_key} for vehicle {getattr(vehicle,'id','UNK')}. Using (0,0).")
        coords.append((0.0, 0.0))
        return coords 

    p_base = POINT_COORDINATES[base_key] # e.g., (0, 80)
    
    QUEUE_LENGTH = 10.0 
    
    if approach == 'N': 
        p_behind = (p_base[0], p_base[1] + QUEUE_LENGTH)
    elif approach == 'S': 
        p_behind = (p_base[0], p_base[1] - QUEUE_LENGTH)
    elif approach == 'E': 
        p_behind = (p_base[0] + QUEUE_LENGTH, p_base[1])
    elif approach == 'W': 
        p_behind = (p_base[0] - QUEUE_LENGTH, p_base[1])
    else:
        p_behind = (p_base[0], p_base[1]) 

    # 1. Add p_behind (coords[0])
    coords.append(p_behind)
    
    # 2. Add p_base (coords[1])
    if tuple(np.round(p_base, 8)) != tuple(np.round(p_behind, 8)):
        coords.append(p_base)

    missing_points = []
    
    # 3. Add all conflict/merge points from the vehicle's path
    for name in path_names:
        if name in POINT_COORDINATES:
            if not coords or tuple(np.round(coords[-1], 8)) != tuple(np.round(POINT_COORDINATES[name], 8)):
                coords.append(POINT_COORDINATES[name])
        else:
            missing_points.append(name)
            if coords:
                coords.append(coords[-1])

    # 4. Add the final exit point
    if len(coords) >= 1 and path_names and 'M_' in path_names[-1]:
        merge_point_name = path_names[-1]
        p_merge = np.array(coords[-1]) 
        EXIT_LENGTH = 60.0 
        p_final_exit = None

        if merge_point_name == 'M_S': 
            p_final_exit = (p_merge[0], p_merge[1] - EXIT_LENGTH) 
        elif merge_point_name == 'M_N': 
            p_final_exit = (p_merge[0], p_merge[1] + EXIT_LENGTH) 
        elif merge_point_name == 'M_E': 
            p_final_exit = (p_merge[0] + EXIT_LENGTH, p_merge[1]) 
        elif merge_point_name == 'M_W': 
            p_final_exit = (p_merge[0] - EXIT_LENGTH, p_merge[1]) 
        
        if p_final_exit and tuple(np.round(p_final_exit, 8)) != tuple(np.round(p_merge, 8)):
            coords.append(p_final_exit)

    if missing_points:
        print(f"Warning: V {getattr(vehicle,'id','UNK')} path points not in map: {missing_points}.")

    if len(coords) < 2:
        coords.append(coords[-1])

    return coords

def calculate_segment_distances(coords: List[Tuple[float, float]]) -> List[float]:
    """Calculates true distances between consecutive points."""
    distances = [0.0]
    if len(coords) < 2: return distances
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i]); p2 = np.array(coords[i+1])
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist if dist > 1e-6 else 0.0)
    return distances

def calculate_cumulative_distances(segment_distances: List[float]) -> np.ndarray:
    """Calculates cumulative distance, forcing distance 0.0 at p_base (coords[1])."""
    cumulative = np.cumsum(segment_distances)
    offset = cumulative[1] if len(cumulative) > 1 else 0.0
    return cumulative - offset

def get_point_at_distance(coords, segment_distances, cumulative_distances, target_distance):
    """Finds (x, y) at a distance along path (linear interp)."""
    if not coords or len(coords) < 2: return None

    # Handle queue positions
    if target_distance <= cumulative_distances[0]: 
        start_coord = np.array(coords[0]) # p_behind
        next_coord = np.array(coords[1])  # p_base (distance 0)
        direction_vec = next_coord - start_coord 
        norm = np.linalg.norm(direction_vec)
        if norm > 1e-6:
            unit_direction_away = -(direction_vec / norm)
            pos = next_coord + unit_direction_away * abs(target_distance)
            return tuple(pos)
        else: return coords[0]

    # Handle exit positions
    if target_distance >= cumulative_distances[-1]:
        end_coord = np.array(coords[-1])
        if len(coords) > 1:
            prev_coord = np.array(coords[-2])
            direction_vec = end_coord - prev_coord
            norm = np.linalg.norm(direction_vec)
            if norm > 1e-6:
                unit_direction_exit = direction_vec / norm
                overshoot = target_distance - cumulative_distances[-1]
                pos = end_coord + unit_direction_exit * overshoot
                return tuple(pos)
        return coords[-1]

    # Handle in-between points
    for i in range(1, len(cumulative_distances)):
        d_prev = cumulative_distances[i-1]
        d_curr = cumulative_distances[i]
        if d_prev - 1e-9 <= target_distance <= d_curr + 1e-9:
            segment_len = segment_distances[i]
            if segment_len < 1e-6: return coords[i]
            ratio = max(0.0, min(1.0, (target_distance - d_prev) / segment_len))
            p_prev = np.array(coords[i-1])
            p_curr = np.array(coords[i])
            interpolated_point = p_prev + ratio * (p_curr - p_prev)
            return tuple(interpolated_point)
            
    return coords[-1]


# ---
# --- MODIFIED ANIMATION LOGIC (D3.js style) ---
# ---
class VehicleAnimator:
    """Manages state and animation of one vehicle."""
    
    # --- MODIFICATION: Simplified __init__ ---
    def __init__(self, vehicle: 'Vehicle', ax, color, queue_pos, start_time, speeds_dict):
        self.id = vehicle.id
        self.path_names = vehicle.path
        self.is_emergency = vehicle.priority_status
        self.speed = speeds_dict.get(self.id, config.velocity_range[0])
        self.valid = False

        # 1. Build the full (x,y) path
        self.trajectory_coords = get_trajectory_coords(vehicle, self.path_names)
        if not self.trajectory_coords or len(self.trajectory_coords) < 2:
             print(f"Error: Insufficient coords for V {self.id} (Path: {self.path_names}).")
             return

        # 2. Calculate distance metrics for the path
        self.segment_distances = calculate_segment_distances(self.trajectory_coords)
        self.cumulative_distances = calculate_cumulative_distances(self.segment_distances)

        # 3. Define the animation parameters (D3-style)
        
        # Start distance is the visual queue position (negative value)
        self.start_dist = -queue_pos * (config.safety_distance * QUEUE_SPACING_VIS_FACTOR)
        
        # Determine end distance: by default it's the end of the exit ramp.
        # Prefer to stop the animation when the vehicle ENTERS the destination
        # lane (merge point `M_*`) so it disappears as soon as it reaches that
        # lane; if no merge point exists in the path, fall back to the last
        # conflict point, otherwise to the exit ramp.
        self.end_dist = self.cumulative_distances[-1]

        # 1) Prefer merge point (destination lane marker 'M_*')
        merge_name = None
        for name in self.path_names:
            if isinstance(name, str) and name.startswith('M_'):
                merge_name = name
                break

        def _set_end_dist_from_label(lbl):
            if lbl and lbl in POINT_COORDINATES:
                p = POINT_COORDINATES[lbl]
                # Find matching coordinate index in trajectory_coords
                for idx, coord in enumerate(self.trajectory_coords):
                    if np.allclose(coord, p, atol=1e-6):
                        try:
                            return float(self.cumulative_distances[idx])
                        except Exception:
                            return None
            return None

        # Try merge point first
        merged_dist = _set_end_dist_from_label(merge_name)
        if merged_dist is not None:
            self.end_dist = merged_dist
        else:
            # 2) Fallback: last conflict point (name starting with 'C')
            last_conflict = None
            for name in reversed(self.path_names):
                if isinstance(name, str) and name.startswith('C'):
                    last_conflict = name
                    break
            conflict_dist = _set_end_dist_from_label(last_conflict)
            if conflict_dist is not None:
                self.end_dist = conflict_dist
        
        self.total_dist_to_travel = self.end_dist - self.start_dist
        safe_speed = max(self.speed, 1.0) # Avoid divide by zero
        
        # Total duration of this vehicle's *movement*
        self.duration = self.total_dist_to_travel / safe_speed
        
        # The *absolute time* in the animation when this vehicle starts
        self.start_time_anim = start_time
        
        # The *absolute time* in the animation when this vehicle finishes
        self.end_time_anim = self.start_time_anim + self.duration

        # 4. Create the Matplotlib patch
        dot_radius = VEHICLE_RADIUS * 1.1 if self.is_emergency else VEHICLE_RADIUS
        self.patch = Circle((0, 0), dot_radius, color=color, zorder=10)
        ax.add_patch(self.patch)
        self.text = ax.text(0, 0, str(self.id), ha='center', va='center',
                             fontsize=7, color='white', zorder=11)

        self.valid = True
        self.set_position(0.0) # Set to initial position

    # --- NEW: helper to report status at absolute time t ---
    def status_at_time(self, t: float) -> str:
        """Return 'in_queue'|'crossing'|'finished' for given animation time t."""
        if t < self.start_time_anim - 1e-6:
            return "in_queue"
        if t > self.end_time_anim + 1e-6:
            return "finished"
        return "crossing"

    # --- MODIFICATION: No longer need build_key_frames ---

    # --- MODIFICATION: New get_distance_at_time logic ---
    def get_distance_at_time(self, t):
        """
        NEW LOGIC: Calculates distance based on a simple, constant-speed
        linear interpolation, just like the D3/HTML version.
        't' is the current *absolute* animation time.
        """
        if t < self.start_time_anim:
            return self.start_dist # Hasn't started yet, stay in queue
        
        if t > self.end_time_anim:
            return self.end_dist # Animation is finished
            
        # It's currently moving. Calculate its progress.
        elapsed_time = t - self.start_time_anim
        
        # Avoid division by zero if duration is somehow 0
        if self.duration < 1e-6:
            return self.end_dist
            
        progress_ratio = elapsed_time / self.duration
        
        target_distance = self.start_dist + progress_ratio * self.total_dist_to_travel
        return target_distance

    def set_position(self, t):
        """Updates patch and text based on time."""
        if not self.valid: return
        
        # Get the target distance based on the new logic
        target_dist = self.get_distance_at_time(t)
        
        # Use the *existing* helper to find the (x,y) for that distance
        pos = get_point_at_distance(self.trajectory_coords, self.segment_distances,
                                     self.cumulative_distances, target_dist)
        if pos:
            self.patch.set_center(pos)
            self.text.set_position(pos)
            # Vehicle is visible from its start time until its end time
            visible = (t >= self.start_time_anim - 0.1 and t <= self.end_time_anim + 0.1)
            self.patch.set_visible(visible)
            self.text.set_visible(visible)
        else:
            self.patch.set_visible(False); self.text.set_visible(False)


class IntersectionVisualization:
    """Manages the entire matplotlib visualization."""
    
    APPROACH_COLORS = {
        'N': '#3399FF', 'E': '#FFCC33', 'S': '#33FF66',
        'W': '#FF66B2', 'DEFAULT': '#E0E0E0'
    }
    EMERGENCY_COLOR = 'red'

    def __init__(self, algorithm_name="Unknown"):
        self.algorithm_name = algorithm_name
        self.fig = plt.figure(figsize=(18, 10))
        # Create grid: left panel (dashboard) + right area (intersection)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.02)
        
        # Main intersection axes (right side)
        self.ax = self.fig.add_subplot(gs[0, 1])
        self.ax.set_xlim(MIN_X, MAX_X)
        self.ax.set_ylim(MIN_Y, MAX_Y)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_title("Intersection Traffic Optimization - Live Simulation", 
                         color='white', fontsize=16, fontweight='bold', pad=15)
        
        self.vehicle_animators: Dict[int, VehicleAnimator] = {}
        self.vehicle_status: Dict[int, Dict] = {}   # Track info per vehicle
        self.info_ax = None                          # Dashboard panel axis
        self.info_block = None                       # Text object for the panel
        self.ani = None
        self.time_text = self.ax.text(0.02, 0.97, '', color='#ffffff', 
                                      transform=self.ax.transAxes, fontsize=12, 
                                      fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.5', 
                                              facecolor='#2a2a2a', alpha=0.8))
        self.t_max = 0.0
        
        try:
            self.geom_for_drawing = Geometry()
        except Exception as e:
            print(f"Error initializing Geometry for drawing: {e}")
            self.geom_for_drawing = None
        
        self.setup_intersection_layout()
        self.setup_info_panel()   # Create the dashboard panel

    # --- Dashboard Panel (Left Side) ---
    def setup_info_panel(self):
        """Sets up the left-side dashboard for comprehensive vehicle status and statistics."""
        try:
            # Remove existing info_ax if present
            if self.info_ax is not None:
                try:
                    self.info_ax.remove()
                except Exception:
                    pass

            # Dashboard panel on left side using GridSpec
            self.info_ax = self.fig.add_subplot(self.fig.axes[0].get_gridspec()[0, 0])
            self.info_ax.set_xlim(0, 1)
            self.info_ax.set_ylim(0, 1)
            self.info_ax.set_facecolor('#0d1117')
            self.info_ax.axis('off')
            
            # Add subtle border
            self.info_ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, 
                                                  fill=False, 
                                                  edgecolor='#30363d', 
                                                  linewidth=1.5, 
                                                  transform=self.info_ax.transAxes))
            
            # Initial header
            header = "═══════════════════════════════\n    TRAFFIC CONTROL DASHBOARD\n═══════════════════════════════"
            self.info_block = self.info_ax.text(0.05, 0.97, header, va='top', ha='left',
                                                fontsize=9, color='#c9d1d9', 
                                                fontfamily='monospace', fontweight='bold')
        except Exception as e:
            print(f"Warning: could not create dashboard panel: {e}")
            self.info_ax = None
            self.info_block = None

    def setup_intersection_layout(self):
        """
        Draws roads, lanes, queues, and trajectories.
        (Copied from original visualization.py, no changes)
        """
        road_color = '#606060'; line_color = '#FFFFFF'; queue_color = '#404040'
        ylims = (MIN_Y, MAX_Y); xlims = (MIN_X, MAX_X)
        
        self.ax.add_patch(Rectangle((-10, MIN_Y), 80, EXTENT_Y, color=road_color, zorder=0))
        self.ax.add_patch(Rectangle((MIN_X, -10), EXTENT_X, 80, color=road_color, zorder=0))

        # Vertical Lines
        self.ax.plot([0, 0], ylims, color=line_color, ls='--', lw=0.5, zorder=1)
        self.ax.plot([30, 30], ylims, color=line_color, ls='-', lw=1.0, zorder=1)
        self.ax.plot([60, 60], ylims, color=line_color, ls='--', lw=0.5, zorder=1)
        
        # Horizontal Lines
        self.ax.plot(xlims, [0, 0], color=line_color, ls='--', lw=0.5, zorder=1)
        self.ax.plot(xlims, [30, 30], color=line_color, ls='-', lw=1.0, zorder=1)
        self.ax.plot(xlims, [60, 60], color=line_color, ls='--', lw=0.5, zorder=1)

        q_len = 20.0 
        self.ax.add_patch(Rectangle((-10, 70.5), 40, q_len, color=queue_color, alpha=0.3, zorder=1))
        self.ax.add_patch(Rectangle((30, -10.5 - q_len), 40, q_len, color=queue_color, alpha=0.3, zorder=1))
        self.ax.add_patch(Rectangle((70.5, 30), q_len, 40, color=queue_color, alpha=0.3, zorder=1))
        self.ax.add_patch(Rectangle((-10.5 - q_len, -10), q_len, 40, color=queue_color, alpha=0.3, zorder=1))

        if self.geom_for_drawing:
            try:
                traj_colors = {'S': '#00FFFF', 'L': '#FF00FF', 'R': '#00FF00'}
                if 'Vehicle' in globals():
                    dummy_v = Vehicle(vehicle_id=0, approach='N', maneuver='S', priority_status=False, velocity=(0,0))
                else: return
                
                for approach in ['N', 'E', 'S', 'W']:
                    for maneuver in ['S', 'L', 'R']:
                        dummy_v.approach = approach
                        dummy_v.maneuver = maneuver
                        self.geom_for_drawing.set_trajectory(dummy_v)
                        
                        if dummy_v.path:
                            coords = get_trajectory_coords(dummy_v, dummy_v.path)
                            if coords and len(coords) >= 2:
                                x_coords, y_coords = zip(*coords)
                                self.ax.plot(x_coords, y_coords, color=traj_colors[maneuver], 
                                             linestyle='--', linewidth=0.7, alpha=0.4, zorder=2)
            except Exception as e:
                print(f"Warning: Could not draw trajectories - {e}")
        
        for name, (x, y) in POINT_COORDINATES.items():
            if 'S_' not in name and 'M_' not in name:
                self.ax.plot(x, y, 'o', color='#FFFFE0', markersize=3, alpha=0.5, zorder=2)
                
        arrow_props = dict(facecolor='white', edgecolor='none', width=0.5, head_width=2.5, head_length=2.5, zorder=2)
        text_props = dict(color='white', fontsize=10, ha='center', va='center')
        
        self.ax.text(10, 90, "NORTH (In)", **text_props)
        self.ax.arrow(10, 85, 0, -10, **arrow_props) 
        self.ax.text(90, 50, "EAST (In)", rotation=-90, **text_props)
        self.ax.arrow(85, 50, -10, 0, **arrow_props)
        self.ax.text(50, -40, "SOUTH (In)", **text_props)
        self.ax.arrow(50, -35, 0, 10, **arrow_props) 
        self.ax.text(-30, 10, "WEST (In)", rotation=90, **text_props)
        self.ax.arrow(-35, 10, 10, 0, **arrow_props)
        
        self.ax.axis('off')

    # ---
    # --- MODIFIED LOADING LOGIC (D3.js style) ---
    # ---
    def load_schedule(self, best_perm, speeds_dict):
        """
        Loads vehicles and creates animators based on D3-style logic.
        This no longer requires 'final_schedule', 'final_tear', or 'tau_p_dict'.
        """
        if not speeds_dict:
            print("Animation Error: Missing speeds data.")
            return

        # Build queue ordering (just like D3)
        # We group vehicles by approach, then sort by permutation index
        approach_queues = {'N': [], 'E': [], 'S': [], 'W': []}
        
        for v in best_perm:
            if v.approach in approach_queues:
                approach_queues[v.approach].append(v)
        
        # We need to find the queue *index* for each vehicle.
        queue_positions = {}
        for approach, q_list in approach_queues.items():
            # Find all vehicles in this queue, *in the order* they appear in best_perm
            vehicles_in_queue = [v for v in best_perm if v.approach == approach]
            for i, v in enumerate(vehicles_in_queue):
                queue_positions[v.id] = i # 0, 1, 2...

        self.vehicle_animators.clear()
        vehicles_loaded = 0
        self.t_max = 0.0 # Reset max time

        # NEW: initialize vehicle_status dictionary from best_perm
        self.vehicle_status.clear()
        for vehicle in best_perm:
            self.vehicle_status[vehicle.id] = {
                "id": vehicle.id,
                "approach": getattr(vehicle, "approach", "?"),
                "maneuver": getattr(vehicle, "maneuver", getattr(vehicle, "path", ["?"])[0]) if hasattr(vehicle, "maneuver") else "?",
                "speed": speeds_dict.get(vehicle.id, None),
                "is_emergency": getattr(vehicle, "priority_status", False),
                "status": "in_queue",
                "delay": 0.0,  # Will be updated during animation
                "start_time": 0.0,
                "end_time": 0.0
            }

        for vehicle in best_perm:
            if vehicle.priority_status:
                v_color = self.EMERGENCY_COLOR
            else:
                v_color = self.APPROACH_COLORS.get(vehicle.approach, self.APPROACH_COLORS['DEFAULT'])

            if vehicle.id in speeds_dict:
                q_pos_index = queue_positions.get(vehicle.id, 0)
                
                # --- NEW: Calculate stagger time (like D3) ---
                # Stagger start time by 1.0 second per vehicle in queue
                stagger_start_time = q_pos_index * 1.0 
                
                animator = VehicleAnimator(vehicle, self.ax, v_color,
                                           q_pos_index, stagger_start_time, speeds_dict)
                
                if animator.valid:
                    self.vehicle_animators[vehicle.id] = animator
                    vehicles_loaded += 1
                    # Update total animation time
                    self.t_max = max(self.t_max, animator.end_time_anim)
                    # Store timing info for delay calculation
                    if vehicle.id in self.vehicle_status:
                        self.vehicle_status[vehicle.id]["start_time"] = animator.start_time_anim
                        self.vehicle_status[vehicle.id]["end_time"] = animator.end_time_anim
                else:
                    print(f"Debug: Failed to initialize animator for V {vehicle.id}.")
            else:
                print(f"Debug: Missing speed for V {vehicle.id}, skipping animator.")

        if self.t_max <= 1e-6 and self.vehicle_animators:
            self.t_max = 10.0 # Fallback duration

        # Ensure info panel exists and show initial snapshot
        if self.info_ax is None or self.info_block is None:
            self.setup_info_panel()
        self.update_info_panel(0.0)   # show initial statuses

        print(f"Smooth animation loaded: {vehicles_loaded} vehicles. Total Duration: {self.t_max:.2f}s.")


    def init_anim(self):
        """Init function for the animation."""
        self.time_text.set_text('Time: 0.00s')
        patches = [self.time_text]
        for v_id in sorted(self.vehicle_animators.keys()):
            v_anim = self.vehicle_animators[v_id]
            v_anim.set_position(0.0)
            patches.append(v_anim.patch); patches.append(v_anim.text)
        # update info panel snapshot at t=0
        self.update_info_panel(0.0)
        return patches

    def update(self, t):
        """Update function called for each frame."""
        self.time_text.set_text(f'Time: {t:.2f}s')
        patches = [self.time_text]
        # Iterate over a copy of keys because we may remove finished vehicles
        for v_id in sorted(list(self.vehicle_animators.keys())):
            v_anim = self.vehicle_animators[v_id]

            # If the vehicle has finished its movement (passed end_time),
            # remove its patch and text from axes and from the animator map.
            if t > v_anim.end_time_anim + 1e-6:
                try:
                    v_anim.patch.remove()
                except Exception:
                    pass
                try:
                    v_anim.text.remove()
                except Exception:
                    pass
                # Mark status finished but keep in vehicle_status for panel
                if v_id in self.vehicle_status:
                    self.vehicle_status[v_id]['status'] = "finished"
                del self.vehicle_animators[v_id]
                continue

            v_anim.set_position(t)
            patches.append(v_anim.patch); patches.append(v_anim.text)

            # update vehicle_status live fields
            if v_id in self.vehicle_status:
                self.vehicle_status[v_id]['status'] = v_anim.status_at_time(t)
                self.vehicle_status[v_id]['speed'] = getattr(v_anim, "speed", self.vehicle_status[v_id]['speed'])
                # Calculate delay: time spent waiting before crossing
                if t >= v_anim.start_time_anim:
                    self.vehicle_status[v_id]['delay'] = max(0.0, v_anim.start_time_anim)

        # UPDATE the info panel after all vehicles updated
        self.update_info_panel(t)

        return patches

    # --- Update Dashboard with Rich Formatting ---
    def update_info_panel(self, current_time: float):
        """Refresh the dashboard with 2x2 grid layout showing each approach."""
        if self.info_block is None or not self.vehicle_status:
            return

        # Clear previous colored text objects
        if hasattr(self, '_colored_texts'):
            for txt in self._colored_texts:
                try:
                    txt.remove()
                except:
                    pass
        self._colored_texts = []

        # Header
        y_pos = 0.98
        header = f"══════════════════════════════════\n TRAFFIC CONTROL DASHBOARD\n Algorithm: {self.algorithm_name}\n══════════════════════════════════"
        self.info_block.set_text(header)
        self.info_block.set_position((0.02, y_pos))
        y_pos -= 0.09
        
        # Legend Section (compact)
        legend_txt = self.info_ax.text(0.02, y_pos, 
                                       "Legend: Red★=Emerg | S/L/R=Man. | W/X/D=Status",
                                       va='top', ha='left', fontsize=6, 
                                       color='#6e7681', fontfamily='monospace')
        self._colored_texts.append(legend_txt)
        y_pos -= 0.028
        
        # Summary Statistics Section
        total = len(self.vehicle_status)
        in_queue = sum(1 for v in self.vehicle_status.values() if v.get('status') == 'in_queue')
        crossing = sum(1 for v in self.vehicle_status.values() if v.get('status') == 'crossing')
        finished = sum(1 for v in self.vehicle_status.values() if v.get('status') == 'finished')
        emergency_count = sum(1 for v in self.vehicle_status.values() if v.get('is_emergency'))
        total_delay = sum(v.get('delay', 0.0) for v in self.vehicle_status.values())
        
        progress_pct = min(100, (current_time / self.t_max * 100)) if self.t_max > 0 else 0
        
        stats_text = f"Time: {current_time:>4.1f}s/{self.t_max:.1f}s {('█'*int(progress_pct/5))+('░'*(20-int(progress_pct/5)))} {progress_pct:.0f}%\nVeh:{total} Wait:{in_queue} Cross:{crossing} Done:{finished} Em:{emergency_count}\nTotal Delay: {total_delay:.1f}s"
        
        txt = self.info_ax.text(0.02, y_pos, stats_text, va='top', ha='left',
                               fontsize=6.5, color='#8b949e', fontfamily='monospace', linespacing=1.5)
        self._colored_texts.append(txt)
        y_pos -= 0.075
        
        # Organize vehicles by approach
        approaches = {'N': [], 'E': [], 'S': [], 'W': []}
        for vid, info in self.vehicle_status.items():
            # Update dynamic fields
            animator = self.vehicle_animators.get(vid)
            if animator:
                info['speed'] = getattr(animator, "speed", info.get('speed'))
                info['status'] = animator.status_at_time(current_time)
                if current_time >= animator.start_time_anim:
                    info['delay'] = max(0.0, animator.start_time_anim)
            else:
                if info.get('status') != "finished" and current_time > self.t_max + 0.5:
                    info['status'] = "finished"
            
            app = info.get('approach', '?')
            if app in approaches:
                approaches[app].append((vid, info))
        
        # Sort each approach by vehicle ID
        for app in approaches:
            approaches[app].sort(key=lambda x: x[0])
        
        # 2x2 Grid Layout
        approach_labels = {'N': 'NORTH', 'E': 'EAST', 'S': 'SOUTH', 'W': 'WEST'}
        approach_colors_label = {'N': '#3399FF', 'E': '#FFCC33', 'S': '#33FF66', 'W': '#FF66B2'}
        
        # Grid positions: (x_offset, y_offset) for each quadrant
        grid_layout = {
            'N': (0.02, 0.0),     # Top-left
            'E': (0.52, 0.0),     # Top-right
            'S': (0.02, -0.42),   # Bottom-left
            'W': (0.52, -0.42)    # Bottom-right
        }
        
        for app in ['N', 'E', 'S', 'W']:
            x_base, y_offset = grid_layout[app]
            y_start = y_pos + y_offset
            
            # Approach header
            header_txt = self.info_ax.text(x_base, y_start, f"┌─ {approach_labels[app]} ─┐",
                                          va='top', ha='left', fontsize=8,
                                          color=approach_colors_label[app],
                                          fontfamily='monospace', fontweight='bold')
            self._colored_texts.append(header_txt)
            
            # Column headers
            col_header_txt = self.info_ax.text(x_base, y_start - 0.03, "ID Man Status",
                                              va='top', ha='left', fontsize=6.5,
                                              color='#6e7681', fontfamily='monospace')
            self._colored_texts.append(col_header_txt)
            
            # Render vehicles for this approach
            y_vehicle = y_start - 0.055
            for vid, info in approaches[app]:
                # ID color: red if emergency, white otherwise
                if info.get('is_emergency'):
                    id_color = '#ff4444'  # Red
                    id_str = f"{vid:>2}★"
                else:
                    id_color = '#c9d1d9'  # White
                    id_str = f"{vid:>2} "
                
                # Maneuver - single letter
                maneuver_raw = info.get('maneuver', '?')
                if isinstance(maneuver_raw, str) and len(maneuver_raw) > 0:
                    maneuver = maneuver_raw[0]  # Just S, L, or R
                else:
                    maneuver = '?'
                
                # Status with compact text
                st = info.get('status', 'UNK')
                
                # Calculate progress for this vehicle (0-100%)
                animator = self.vehicle_animators.get(vid)
                if animator and st == 'crossing':
                    # Vehicle is crossing - show progress
                    elapsed = current_time - animator.start_time_anim
                    progress = min(100, (elapsed / animator.duration * 100)) if animator.duration > 0 else 0
                    # Mini progress bar (3 blocks for space)
                    filled = int(progress / 33.33)  # 0-3
                    bar = ('█' * filled) + ('░' * (3 - filled))
                    status_text = f"X {bar}"  # X = crossing
                    status_color = '#58a6ff'  # Blue
                elif st == 'finished':
                    status_text = 'D ███'  # D = done
                    status_color = '#3fb950'  # Green
                else:  # in_queue
                    status_text = 'W ░░░'  # W = waiting
                    status_color = '#ff4444'  # Red
                
                # Vehicle ID
                id_txt = self.info_ax.text(x_base, y_vehicle, id_str, va='top', ha='left',
                                          fontsize=7, color=id_color,
                                          fontfamily='monospace', fontweight='bold')
                self._colored_texts.append(id_txt)
                
                # Maneuver
                maneuver_txt = self.info_ax.text(x_base + 0.055, y_vehicle, maneuver,
                                                va='top', ha='left', fontsize=7,
                                                color='#8b949e', fontfamily='monospace')
                self._colored_texts.append(maneuver_txt)
                
                # Status with progress
                status_txt = self.info_ax.text(x_base + 0.095, y_vehicle, status_text,
                                              va='top', ha='left', fontsize=7,
                                              color=status_color, fontfamily='monospace',
                                              fontweight='bold')
                self._colored_texts.append(status_txt)
                
                y_vehicle -= 0.025
                
                # Stop if quadrant is full
                if y_vehicle < (y_start - 0.40):
                    break

    def start_animation(self, interval=50, repeat=True):
        """
        Start the matplotlib animation.
        
        Parameters:
        -----------
        interval : int
            Delay between frames in milliseconds (default: 50ms = 20 fps)
        repeat : bool
            Whether to loop the animation (default: True)
        """
        if not self.vehicle_animators:
            print("Error: No vehicles to animate. Load a schedule first.")
            return
        
        # Calculate number of frames based on duration and interval
        fps = 1000.0 / interval  # frames per second
        total_frames = int(self.t_max * fps) + 10  # Add buffer frames
        
        print(f"Starting animation: {total_frames} frames @ {fps:.1f} fps")
        print(f"Animation duration: {self.t_max:.2f} seconds")
        
        # Create time array for the animation
        time_array = np.linspace(0, self.t_max, total_frames)
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=time_array,
            init_func=self.init_anim,
            interval=interval,
            repeat=repeat,
            blit=False  # Set to False for better compatibility with info panel
        )
        
        plt.show()
