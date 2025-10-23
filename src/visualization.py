# FILE: visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Patch
import numpy as np
import config # Needed for safety_distance
import math
from typing import List, Dict, Tuple

# *** Ensure Vehicle class is correctly imported ***
try:
    from vehicle import Vehicle
    from geometry import Geometry # Import Geometry to get paths for drawing
except ImportError:
    print("Error: Could not import Vehicle or Geometry class from vehicle.py/geometry.py")
    class Vehicle: pass
    class Geometry: pass # Dummy class

# --- Constants for Layout (Derived from your map) ---
VEHICLE_RADIUS = 2.0 # INCREASED Visual size of vehicle points
QUEUE_SPACING_VIS_FACTOR = 4.0 # INCREASED to separate queued cars visually
LANE_WIDTH = 20.0 # From your coordinate grid

# --- Extended Plot limits to show queues ---
QUEUE_VIS_BUFFER = 60.0
# Adjust plot limits based on the new, wider road geometry
MIN_X = -30 - QUEUE_VIS_BUFFER
MAX_X = 100 + QUEUE_VIS_BUFFER
MIN_Y = -30 - QUEUE_VIS_BUFFER
MAX_Y = 100 + QUEUE_VIS_BUFFER
EXTENT_X = MAX_X - MIN_X
EXTENT_Y = MAX_Y - MIN_Y

# --- *** YOUR HARDCODED COORDINATE MAP *** ---
# Lane-specific points (S_X_L) are now permanently omitted.
POINT_COORDINATES = {
    # Conflict Points (C1-C16)
    'C1': (0, 60),  'C2': (20, 60),  'C3': (40, 60),  'C4': (60, 60),
    'C5': (0, 40),  'C6': (30, 40),  'C7': (60, 40),  'C8': (20, 30),
    'C9': (40, 30), 'C11': (0, 20), 'C10': (30, 20), 'C12': (60, 20),
    'C13': (0, 0),  'C14': (20, 0),  'C15': (40, 0),  'C16': (60, 0),

    # Merge/Exit Points
    'M_W': (-20, 60), 'M_S': (0, -20), 'M_E': (80, 0), 'M_N': (60, 80),

    # Start Points (Queue Origins) - Single point per approach
    'S_N': (0, 80),   # North Approach
    'S_E': (80, 60),   # East Approach
    'S_S': (60, -20), # South Approach
    'S_W': (-20, 0),   # West Approach
}
# --- END MAP ---


# --- Helper Functions ---
def get_trajectory_coords(vehicle: 'Vehicle', path_names: List[str]) -> List[Tuple[float, float]]:
    """
    FIXED: Ensures straight queue stacking by inserting a p_behind point (coords[0])
    and extends the trajectory 60 units past the final merge point along the correct road centerline.
    """
    coords = []
    approach = getattr(vehicle, 'approach', None)
    base_key = f"S_{approach}" if approach else None
    
    if not (base_key and base_key in POINT_COORDINATES):
        print(f"Warning: Missing base start {base_key} for vehicle {getattr(vehicle,'id','UNK')}. Using (0,0).")
        coords.append((0.0, 0.0))
        return coords 

    p_base = POINT_COORDINATES[base_key] # e.g., (0, 80)
    
    # --- FIX 1: DEFINE THE STRAIGHT QUEUE SEGMENT (For stacking) ---
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

    # 1. Add the point BEHIND the queue line (p_behind). (coords[0])
    coords.append(p_behind)
    
    # 2. Add the actual queue start point (S_X). (coords[1])
    if tuple(np.round(p_base, 8)) != tuple(np.round(p_behind, 8)):
        coords.append(p_base)
    # --- END FIX 1 ---

    missing_points = []
    
    # 3. Add the actual path points from geometry.py.
    for name in path_names:
        if name in POINT_COORDINATES:
            # Avoid duplicate consecutive points
            if not coords or tuple(np.round(coords[-1], 8)) != tuple(np.round(POINT_COORDINATES[name], 8)):
                coords.append(POINT_COORDINATES[name])
        else:
            missing_points.append(name)
            if coords:
                coords.append(coords[-1])

    # --- FIX 2: CENTERLINE EXIT EXTENSION (Further out) ---
    if len(coords) >= 1 and path_names and 'M_' in path_names[-1]:
        merge_point_name = path_names[-1]
        p_merge = np.array(coords[-1]) 
        
        # Extended to 60.0 units for a longer exit path
        EXIT_LENGTH = 60.0 

        p_final_exit = None

        # Determine the exit path explicitly based on the merge point's direction
        if merge_point_name == 'M_S': 
            p_final_exit = (p_merge[0], p_merge[1] - EXIT_LENGTH) 
        elif merge_point_name == 'M_N': 
            p_final_exit = (p_merge[0], p_merge[1] + EXIT_LENGTH) 
        elif merge_point_name == 'M_E': 
            p_final_exit = (p_merge[0] + EXIT_LENGTH, p_merge[1]) 
        elif merge_point_name == 'M_W': 
            p_final_exit = (p_merge[0] - EXIT_LENGTH, p_merge[1]) 
        
        # Append the final exit point to the coordinates list
        if p_final_exit and tuple(np.round(p_final_exit, 8)) != tuple(np.round(p_merge, 8)):
            coords.append(p_final_exit)
    # --- END FIX 2 ---

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
        # We now use the true distance for ALL segments.
        distances.append(dist if dist > 1e-6 else 0.0)
    return distances

def calculate_cumulative_distances(segment_distances: List[float]) -> np.ndarray:
    """
    FIXED: Calculates cumulative distance, forcing distance to start at 0.0 at p_base (coords[1]).
    The distance at coords[0] (p_behind) will be negative, enabling queue calculation.
    """
    cumulative = np.cumsum(segment_distances)
    
    # CRITICAL FIX: Offset the entire cumulative array so that the distance 
    # at coords[1] (p_base) becomes 0.0.
    offset = cumulative[1] if len(cumulative) > 1 else 0.0
    
    return cumulative - offset

def get_point_at_distance(coords, segment_distances, cumulative_distances, target_distance):
    """Finds (x, y) at a distance along path (linear interp)."""
    if not coords or len(coords) < 2: return None

    # This section handles queue positions (target_distance <= cumulative_distances[0] which is negative)
    if target_distance <= cumulative_distances[0]: 
        start_coord = np.array(coords[0]) # p_behind
        next_coord = np.array(coords[1])  # p_base (distance 0)
        direction_vec = next_coord - start_coord # Vector TOWARDS intersection 
        norm = np.linalg.norm(direction_vec)
        
        if norm > 1e-6:
            unit_direction_towards = direction_vec / norm # Unit vector TOWARDS intersection 
            unit_direction_away = -unit_direction_towards # Unit vector AWAY from intersection
            
            # CRITICAL FIX for Overlap: Position is calculated relative to p_base (distance 0) 
            # and offset AWAY from the intersection by abs(target_distance).
            pos = next_coord + unit_direction_away * abs(target_distance)
            return tuple(pos)
        else: return coords[0]

    if target_distance >= cumulative_distances[-1]: # Beyond end point
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


class VehicleAnimator:
    """Manages state and animation of one vehicle."""
    def __init__(self, vehicle: 'Vehicle', ax, color, queue_pos, t_ear, schedule, speeds_dict, tau_p_dict):
        self.id = vehicle.id
        self.path_names = vehicle.path
        self.is_emergency = vehicle.priority_status
        self.speed = speeds_dict.get(self.id, config.velocity_range[0])
        self.valid = False

        # This will now *always* return a path starting from the base S_ point
        self.trajectory_coords = get_trajectory_coords(vehicle, self.path_names)

        if not self.trajectory_coords or len(self.trajectory_coords) < 2:
             print(f"Error: Insufficient coords for V {self.id} (Path: {self.path_names}).")
             return

        self.segment_distances = calculate_segment_distances(self.trajectory_coords)
        self.cumulative_distances = calculate_cumulative_distances(self.segment_distances)
        # d0_vis should now be the distance of the p_base to C1/C2 segment (coords[1] to coords[2])
        self.d0_vis = self.segment_distances[2] if len(self.segment_distances) > 2 else 0
        # queue_pos is now 0, 1, 2... for the *single approach* queue
        self.queue_offset = queue_pos * (config.safety_distance * QUEUE_SPACING_VIS_FACTOR)

        dot_radius = VEHICLE_RADIUS * 1.1 if self.is_emergency else VEHICLE_RADIUS
        self.patch = Circle((0, 0), dot_radius, color=color, zorder=10)
        ax.add_patch(self.patch)
        self.text = ax.text(0, 0, str(self.id), ha='center', va='center',
                             fontsize=7, color='white', zorder=11)

        self.key_frames = [] # (time, distance)
        self.build_key_frames(t_ear, schedule, tau_p_dict)

        if not self.key_frames:
             print(f"Debug: V {self.id} invalid - Failed to build keyframes.")
             return

        self.valid = True
        self.set_position(0.0)

    def build_key_frames(self, t_ear, schedule, tau_p_dict):
            """Builds (time, distance) keyframes from the schedule, respecting queue offset and smoothing movement."""
            if not self.path_names or len(self.trajectory_coords) < 2:
                return

            # Start with the vehicle's position BEFORE the queue is resolved.
            self.key_frames = [(0.0, -self.queue_offset)]

            safe_speed = max(self.speed, 1e-6)
            
            # The distance the vehicle needs to travel to reach the start line (distance 0)
            distance_to_start_line = self.queue_offset 
            time_to_travel_to_start = distance_to_start_line / safe_speed if safe_speed > 1e-6 else 0.0

            # determine time when vehicle reaches S point (distance 0)
            physical_arrival_time = self.key_frames[0][0] + time_to_travel_to_start 

            # The actual time at S_point is the later of: 
            # 1. When it physically gets there based on queue position (physical_arrival_time)
            # 2. When it is scheduled to be ready to go (t_ear).
            time_at_S_point = max(physical_arrival_time, t_ear)
            time_at_S_point = max(time_at_S_point, self.key_frames[0][0] + 1e-6)
            
            # Append the keyframe for the arrival at the start point (distance 0.0)
            self.key_frames.append((time_at_S_point, 0.0))

            last_scheduled_time_kf = time_at_S_point
            last_scheduled_dist_kf = 0.0 # Vehicle starts at distance 0.0 after queueing

            # Add scheduled arrivals/departures for each named point in path_names
            for i, point_name in enumerate(self.path_names):
                if point_name in schedule:
                    
                    current_dist_name_match = -1
                    # Search for the point's coordinate in the final generated trajectory list
                    if point_name in POINT_COORDINATES:
                        coord_to_find = POINT_COORDINATES[point_name]
                        for j, coord in enumerate(self.trajectory_coords):
                            if tuple(np.round(coord, 8)) == tuple(np.round(coord_to_find, 8)):
                                current_dist_name_match = self.cumulative_distances[j]
                                break
                    
                    if current_dist_name_match == -1: continue 

                    current_dist = current_dist_name_match 
                    t_arrival = float(schedule[point_name])
                    t_departure = t_arrival + float(tau_p_dict.get(point_name, config.tau))
                    
                    # --- FIX FOR TELEPORTATION: INSERT INTERMEDIATE POINT ---
                    # Calculate time needed to cover distance at max speed
                    dist_diff = current_dist - last_scheduled_dist_kf
                    time_at_max_speed = dist_diff / safe_speed if safe_speed > 1e-6 else 0.0
                    
                    # Calculate the earliest possible time to reach this point
                    t_earliest_arrival = last_scheduled_time_kf + time_at_max_speed

                    # If scheduled arrival is significantly later than earliest arrival, 
                    # insert a keyframe at the point of max speed/constant travel to smooth the path.
                    if t_arrival > t_earliest_arrival + 0.1: # Use 0.1s buffer for smoothing
                        # Insert a keyframe at the earliest possible arrival time
                        self.key_frames.append((t_earliest_arrival, current_dist))
                        last_scheduled_time_kf = t_earliest_arrival
                    
                    # The arrival time must be strictly non-decreasing
                    t_arrival = max(t_arrival, last_scheduled_time_kf + 1e-6)

                    # arrival keyframe
                    self.key_frames.append((t_arrival, current_dist))
                    # departure keyframe (if dwell)
                    if t_departure > t_arrival + 1e-6:
                        self.key_frames.append((t_departure, current_dist))
                    
                    last_scheduled_time_kf = max(last_scheduled_time_kf, t_departure)
                    last_scheduled_dist_kf = current_dist # Update the last distance

            # Ensure there's a keyframe that reaches the final coordinate of the path
            final_coord_index = len(self.trajectory_coords) - 1
            final_dist = self.cumulative_distances[final_coord_index] if final_coord_index < len(self.cumulative_distances) else None

            if final_dist is not None:
                # If last recorded distance is less than final_dist, add a linear travel keyframe
                last_time, last_dist = self.key_frames[-1]
                if final_dist > last_dist + 1e-6:
                    dist_to_final = final_dist - last_dist
                    # Guarantee a positive travel time (avoid division by zero / instantaneous teleport)
                    time_needed = dist_to_final / safe_speed if safe_speed > 1e-9 else dist_to_final / (1e-3)
                    arrival_to_final_time = last_time + max(time_needed, 0.05)  # at least 50ms travel
                    # If arrival would be earlier than previous time (shouldn't), force monotonicity:
                    arrival_to_final_time = max(arrival_to_final_time, last_time + 1e-6)
                    self.key_frames.append((arrival_to_final_time, final_dist))

                # Add a small trailing time so the vehicle doesn't freeze right at the exit
                # This segment is now longer due to the extension in get_trajectory_coords
                tail_time = self.key_frames[-1][0] + 2.0
                self.key_frames.append((tail_time, final_dist))

            # Clean and monotonicize keyframes (remove duplicates and ensure strictly non-decreasing time)
            cleaned_keyframes = []
            if self.key_frames:
                cleaned_keyframes.append(self.key_frames[0])
                for i in range(1, len(self.key_frames)):
                    t_prev_clean, d_prev_clean = cleaned_keyframes[-1]
                    t_curr_orig, d_curr_orig = self.key_frames[i]
                    t_curr_clean = max(t_curr_orig, t_prev_clean + 1e-9)
                    # drop spurious duplicates (same time & same dist)
                    if t_curr_clean > t_prev_clean + 1e-9 or abs(d_curr_orig - d_prev_clean) > 1e-6:
                        cleaned_keyframes.append((t_curr_clean, d_curr_orig))

            self.key_frames = cleaned_keyframes
    def get_distance_at_time(self, t):
        """Interpolates distance along path at time t."""
        if not self.key_frames or t < self.key_frames[0][0]: return -self.queue_offset
        for i in range(len(self.key_frames) - 1):
            t0, d0 = self.key_frames[i]; t1, d1 = self.key_frames[i+1]
            if t0 - 1e-9 <= t <= t1 + 1e-9:
                if t1 <= t0 + 1e-9: return d1
                safe_t = max(t0, min(t, t1))
                time_diff = t1 - t0
                if time_diff < 1e-9: return d1
                ratio = (safe_t - t0) / time_diff
                distance = d0 + ratio * (d1 - d0)
                return distance
        return self.key_frames[-1][1]

    def set_position(self, t):
        """Updates patch and text based on time."""
        if not self.valid: return
        target_dist = self.get_distance_at_time(t)
        pos = get_point_at_distance(self.trajectory_coords, self.segment_distances,
                                     self.cumulative_distances, target_dist)
        if pos:
            self.patch.set_center(pos)
            self.text.set_position(pos)
            visible = (self.key_frames and t >= self.key_frames[0][0] - 0.1 and t <= self.key_frames[-1][0] + 0.1)
            self.patch.set_visible(visible)
            self.text.set_visible(visible)
        else:
            self.patch.set_visible(False); self.text.set_visible(False)


class IntersectionVisualization:
    """Manages the entire matplotlib visualization."""
    
    # --- Color definitions for approaches ---
    APPROACH_COLORS = {
        'N': '#3399FF', # Blue
        'E': '#FFCC33', # Yellow
        'S': '#33FF66', # Green
        'W': '#FF66B2', # Pink
        'DEFAULT': '#E0E0E0' # White/Gray for any fallback
    }
    EMERGENCY_COLOR = 'red'

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlim(MIN_X, MAX_X)
        self.ax.set_ylim(MIN_Y, MAX_Y)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#202020')
        self.ax.set_title("Intersection Animation - Best Solution Schedule", color='white', y=1.02, fontsize=16)
        self.vehicle_animators: Dict[int, VehicleAnimator] = {}
        self.ani = None
        self.time_text = self.ax.text(0.02, 0.97, '', color='white', transform=self.ax.transAxes, fontsize=14)
        self.t_max = 0.0
        
        try:
            self.geom_for_drawing = Geometry()
        except Exception as e:
            print(f"Error initializing Geometry for drawing: {e}")
            self.geom_for_drawing = None
        
        self.setup_intersection_layout()

    def setup_intersection_layout(self):
        """
        Draws roads, lanes, queues, and trajectories based on the
        hardcoded coordinates being CENTERLINES, visually scaling the roads and removing unused lines.
        """
        road_color = '#606060'; line_color = '#FFFFFF'; queue_color = '#404040'
        ylims = (MIN_Y, MAX_Y); xlims = (MIN_X, MAX_X)
        
        # --- ROAD GEOMETRY ---
        # Vertical Road Area: x= -10 to x=70
        self.ax.add_patch(Rectangle((-10, MIN_Y), 80, EXTENT_Y, color=road_color, zorder=0))
        
        # Horizontal Road Area: y=-10 to y=70
        self.ax.add_patch(Rectangle((MIN_X, -10), EXTENT_X, 80, color=road_color, zorder=0))


        # --- LANE LINES (Only keeping primary queue lines and median) ---
        
        # Vertical Lines
        self.ax.plot([0, 0], ylims, color=line_color, ls='--', lw=0.5, zorder=1)   # x=0 centerline (Primary N/S queue line)
        # self.ax.plot([20, 20], ylims, color=line_color, ls='--', lw=0.5, zorder=1) # x=20 (REMOVED)
        self.ax.plot([30, 30], ylims, color=line_color, ls='-', lw=1.0, zorder=1)  # x=30 solid median
        # self.ax.plot([40, 40], ylims, color=line_color, ls='--', lw=0.5, zorder=1) # x=40 (REMOVED)
        self.ax.plot([60, 60], ylims, color=line_color, ls='--', lw=0.5, zorder=1) # x=60 centerline (Primary S/N queue line)
        
        # Horizontal Lines
        self.ax.plot(xlims, [0, 0], color=line_color, ls='--', lw=0.5, zorder=1)   # y=0 centerline (Primary W/E queue line)
        # self.ax.plot(xlims, [20, 20], color=line_color, ls='--', lw=0.5, zorder=1) # y=20 (REMOVED)
        self.ax.plot(xlims, [30, 30], color=line_color, ls='-', lw=1.0, zorder=1)  # y=30 solid median
        # self.ax.plot(xlims, [40, 40], color=line_color, ls='--', lw=0.5, zorder=1) # y=40 (REMOVED)
        self.ax.plot(xlims, [60, 60], color=line_color, ls='--', lw=0.5, zorder=1) # y=60 centerline (Primary E/W queue line)


        # --- NEW QUEUE BOXES (Visually confirms queue area is defined by outer lines and median) ---
        q_len = 20.0 
        
        # N Queue Area (x=-10 to 30, above y=70)
        self.ax.add_patch(Rectangle((-10, 70.5), 40, q_len, color=queue_color, alpha=0.3, zorder=1))
        
        # S Queue Area (x=30 to 70, below y=-10)
        self.ax.add_patch(Rectangle((30, -10.5 - q_len), 40, q_len, color=queue_color, alpha=0.3, zorder=1))
        
        # E Queue Area (y=30 to 70, right of x=70)
        self.ax.add_patch(Rectangle((70.5, 30), q_len, 40, color=queue_color, alpha=0.3, zorder=1))
        
        # W Queue Area (y=-10 to 30, left of x=-10)
        self.ax.add_patch(Rectangle((-10.5 - q_len, -10), q_len, 40, color=queue_color, alpha=0.3, zorder=1))


        # --- Draw Faint Trajectories ---
        if self.geom_for_drawing:
            try:
                traj_colors = {'S': '#00FFFF', 'L': '#FF00FF', 'R': '#00FF00'} # Cyan, Magenta, Green
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
        
        # --- Draw Conflict Points ---
        for name, (x, y) in POINT_COORDINATES.items():
            if 'S_' not in name and 'M_' not in name:
                self.ax.plot(x, y, 'o', color='#FFFFE0', markersize=3, alpha=0.5, zorder=2)
                
        # --- Labels and Arrows (Showing Directions) ---
        arrow_props = dict(facecolor='white', edgecolor='none', width=0.5, head_width=2.5, head_length=2.5, zorder=2)
        text_props = dict(color='white', fontsize=10, ha='center', va='center')
        
        # N Approach (centerline x=0, x=20)
        self.ax.text(10, 90, "NORTH (In)", **text_props)
        self.ax.arrow(10, 85, 0, -10, **arrow_props) 
        
        # E Approach (centerline y=60, y=40)
        self.ax.text(90, 50, "EAST (In)", rotation=-90, **text_props)
        self.ax.arrow(85, 50, -10, 0, **arrow_props)
        
        # S Approach (centerline x=60, x=40)
        self.ax.text(50, -40, "SOUTH (In)", **text_props)
        self.ax.arrow(50, -35, 0, 10, **arrow_props) 
        
        # W Approach (centerline y=0, y=20)
        self.ax.text(-30, 10, "WEST (In)", rotation=90, **text_props)
        self.ax.arrow(-35, 10, 10, 0, **arrow_props)

        # --- Legend for Vehicle Colors (Approach/Emergency) ---
        color_legend_patches = [
             Patch(color=self.EMERGENCY_COLOR, label='Emergency (RED)'),
             Patch(color=self.APPROACH_COLORS['N'], label='N Approach (BLUE)'),
             Patch(color=self.APPROACH_COLORS['E'], label='E Approach (YELLOW)'),
             Patch(color=self.APPROACH_COLORS['S'], label='S Approach (GREEN)'),
             Patch(color=self.APPROACH_COLORS['W'], label='W Approach (PINK)')
        ]
        
        # Combine Path Legend and Color Legend
        path_legend_patches = [
             Patch(color='#00FFFF', label='Straight Path'),
             Patch(color='#FF00FF', label='Left Turn Path'),
             Patch(color='#00FF00', label='Right Turn Path')
        ]
        
        # Create a single legend using handles
        all_legend_handles = path_legend_patches + color_legend_patches
        self.ax.legend(handles=all_legend_handles, loc='lower right', fontsize='small', ncol=2)
        
        self.ax.axis('off')

    def load_schedule(self, best_perm, final_schedule, final_tear, speeds_dict, tau_p_dict):
        """Loads schedule and creates VehicleAnimator objects."""
        if not final_schedule or not final_tear or not speeds_dict:
            print("Animation Error: Missing schedule, t_ear, or speeds data.")
            return

        # Build queue ordering using Geometry's entry queues so drawing order matches simulation
        queue_positions = {}
        queues = {'N': [], 'E': [], 'S': [], 'W': []}

        try:
            geom = Geometry()
            if 'Vehicle' in globals():
                 geom.create_entry_queue(config.pi)
            
            for approach, q_list in geom.entry_queues.items():
                if approach in queues:
                    ids = [getattr(v, 'id', v) for v in q_list]
                    queues[approach] = ids
        except Exception as e:
            print(f"Warning: Geometry.create_entry_queue failed: {e}. Falling back to config.pi ordering.")
            for v_cfg in config.pi:
                if v_cfg.approach in queues:
                    queues[v_cfg.approach].append(v_cfg.id)

        # Assign queue positions (0,1,2...) for each approach queue
        for approach, q_ids in queues.items():
            for pos, v_id in enumerate(q_ids):
                queue_positions[v_id] = pos

        self.vehicle_animators.clear()
        vehicles_loaded = 0

        for i, vehicle in enumerate(best_perm):
            # Color logic
            if vehicle.priority_status:
                v_color = self.EMERGENCY_COLOR
            else:
                v_color = self.APPROACH_COLORS.get(vehicle.approach, self.APPROACH_COLORS['DEFAULT'])

            # Each vehicle must have entries in final_schedule/final_tear and speeds_dict
            if vehicle.id in final_schedule and vehicle.id in final_tear and vehicle.id in speeds_dict:
                q_pos = queue_positions.get(vehicle.id, 0)
                t_ear_val = final_tear[vehicle.id]
                schedule_val = final_schedule[vehicle.id]
                animator = VehicleAnimator(vehicle, self.ax, v_color,
                                           q_pos, t_ear_val, schedule_val, speeds_dict, tau_p_dict)
                if animator.valid:
                    self.vehicle_animators[vehicle.id] = animator
                    vehicles_loaded += 1
                else:
                    print(f"Debug: Failed to initialize animator for V {vehicle.id}.")
            else:
                print(f"Debug: Missing schedule/tear/speed for V {vehicle.id}, skipping animator.")

        # Compute t_max from all animators (duration)
        self.t_max = 0.0
        for v_anim in self.vehicle_animators.values():
            if v_anim.key_frames:
                self.t_max = max(self.t_max, v_anim.key_frames[-1][0])
        if self.t_max <= 1e-6 and self.vehicle_animators:
            self.t_max = 10.0

        print(f"Animation loaded: {vehicles_loaded} vehicles. Duration: {self.t_max:.2f}s.")


    def init_anim(self):
        """Init function for the animation."""
        self.time_text.set_text('Time: 0.00s')
        patches = [self.time_text]
        for v_id in sorted(self.vehicle_animators.keys()):
            v_anim = self.vehicle_animators[v_id]
            v_anim.set_position(0.0)
            patches.append(v_anim.patch); patches.append(v_anim.text)
        return patches

    def update(self, t):
        """Update function called for each frame."""
        self.time_text.set_text(f'Time: {t:.2f}s')
        patches = [self.time_text]
        for v_id in sorted(self.vehicle_animators.keys()):
            v_anim = self.vehicle_animators[v_id]
            v_anim.set_position(t)
            patches.append(v_anim.patch); patches.append(v_anim.text)
        return patches

    def start_animation(self):
        """Starts the matplotlib animation."""
        if not self.vehicle_animators or self.t_max <= 0:
            print("Cannot start animation: No valid vehicles/schedule or zero duration.")
            return

        fps = 30
        total_frames = int(self.t_max * fps) + 1
        print(f"Starting animation: {total_frames} frames at {fps} FPS.")

        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=np.linspace(0, self.t_max, total_frames),
            init_func=self.init_anim,
            blit=False,
            interval=max(1, int(1000/fps)),
            repeat=False
        )
        try:
            plt.show() # Blocks until window closed
        except Exception as e:
            print(f"Error displaying animation window: {e}")