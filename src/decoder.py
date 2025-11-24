# File: decoder.py
import config
from geometry import Geometry
import math
from typing import List, Dict, Tuple, Union
from vehicle import Vehicle

def run_decoder(permutation: List[Vehicle],
                segment_speeds_matrix: List[List[float]],  # [vehicle_idx][segment_idx]
                geom: Geometry,
                tau_p_dict: Dict[str, float],
                return_full_schedule: bool = False):
    """
    Scheduling decoder with segment-wise vehicle speeds.
    
    Segments are mapped as:
    - Segment 0: Start → Conflict 1
    - Segment 1: Conflict 1 → Conflict 2
    - Segment 2: Conflict 2 → Conflict 3
    - Segment 3: Conflict 3 → Conflict 4
    - Segment 4: Conflict 4+ → End/Merge
    
    If a path has MORE than 5 points, later segments reuse Segment 4 speed.
    """
    
    t_ear = {}
    scheduled_times = {}
    path_pointers = {}
    vehicle_state = {}
    availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}
    conflict_count_per_point = {p: 0 for p in tau_p_dict.keys()}
    conflict_pairs = []
    
    # Map vehicle ID to segment speeds
    speeds_by_segment = {v.id: segment_speeds_matrix[idx] for idx, v in enumerate(permutation)}
    
    d0 = 4.0
    reference_speed = 9.0
    
    # --- Calculate t_ear using FIRST SEGMENT speed ---
    for queue in geom.entry_queues.values():
        if not queue:
            continue
        last_t_ear = -math.inf
        
        for idx, v in enumerate(queue):
            if v.id not in speeds_by_segment:
                continue
            
            # Use speed for segment 0 (start to first conflict)
            first_segment_speed = speeds_by_segment[v.id][0]
            safe_speed = max(first_segment_speed, 1e-6)
            
            if idx == 0:
                current_t_ear = d0 / safe_speed
            else:
                time_gap = config.safety_distance / safe_speed
                current_t_ear = last_t_ear + time_gap
            
            t_ear[v.id] = current_t_ear
            last_t_ear = current_t_ear
    
    # Initialize vehicle state
    num_vehicles_in_solution = 0
    for v in permutation:
        if v.id in speeds_by_segment:
            num_vehicles_in_solution += 1
            path_pointers[v.id] = 0
            vehicle_state[v.id] = 'WAITING' if v.id in t_ear else 'FINISHED'
            scheduled_times[v.id] = {}
    
    # --- 4. Dynamic Scheduling Loop (Iterative based on dependencies) ---
    num_finished = sum(1 for v_id, state in vehicle_state.items() if state == 'FINISHED')
    active_vehicles_ids = {v.id for v in permutation if vehicle_state.get(v.id) == 'WAITING'}
    loop_limit = num_vehicles_in_solution * max((len(v.path) for v in permutation if v.path and v.id in speeds_by_segment), default=1) * 2
    loop_count = 0
    
    while num_finished < num_vehicles_in_solution and loop_count < loop_limit:
        made_progress_this_iteration = False
        processed_vehicle_ids = set()
        
        for v in permutation:
            v_id = v.id
            if v_id not in active_vehicles_ids or vehicle_state.get(v_id) == 'FINISHED':
                continue
            
            processed_vehicle_ids.add(v_id)
            path_idx = path_pointers.get(v_id, 0)
            
            if path_idx >= len(v.path):
                if vehicle_state.get(v_id) != 'FINISHED':
                    vehicle_state[v_id] = 'FINISHED'; num_finished += 1; made_progress_this_iteration = True
                continue
            
            p = v.path[path_idx]
            
            if p not in tau_p_dict:
                if vehicle_state.get(v_id) != 'FINISHED':
                    vehicle_state[v_id] = 'FINISHED'; num_finished += 1; made_progress_this_iteration = True
                continue
            
            tau = tau_p_dict[p]
            
            # --- Calculate t_reach using SEGMENT SPEED ---
            if path_idx == 0:
                t_reach = t_ear.get(v_id, math.inf)
            else:
                p_prev = v.path[path_idx - 1]
                
                if p_prev not in scheduled_times.get(v_id, {}):
                    continue
                
                tau_prev = tau_p_dict.get(p_prev, config.tau)
                t_prev_departure = scheduled_times[v_id][p_prev] + tau_prev
                
                # NEW: Map segment index capped at 4 (segments 0-4 only)
                # If path_idx > 4, use segment 4 speed for all remaining segments
                segment_idx = min(path_idx, 4)  # Cap at index 4 (5th segment)
                segment_speed = speeds_by_segment[v_id][segment_idx]
                safe_segment_speed = max(segment_speed, 1e-6)
                
                distance = config.inter_conflict_distance
                travel_time = distance / safe_segment_speed
                
                t_reach = t_prev_departure + travel_time
            
            # --- Schedule at conflict point ---
            t_available = availability_clocks[p]
            t_scheduled = max(t_reach, t_available)
            
            scheduled_times.setdefault(v_id, {})[p] = t_scheduled
            availability_clocks[p] = t_scheduled + tau
            path_pointers[v_id] = path_idx + 1
            vehicle_state[v_id] = 'RUNNING'
            made_progress_this_iteration = True
        
        active_vehicles_ids = {v_id for v_id, state in vehicle_state.items() if state == 'WAITING' or state == 'RUNNING'}
        loop_count += 1
        
        # Deadlock detection (same as before)
        if not made_progress_this_iteration and num_finished < num_vehicles_in_solution:
            stuck_vehicles_count = 0
            for v_id_check in active_vehicles_ids:
                if v_id_check in processed_vehicle_ids:
                    stuck_vehicles_count += 1
                    if vehicle_state.get(v_id_check) != 'FINISHED':
                        if v_id_check not in t_ear:
                            t_ear[v_id_check] = 0
                        scheduled_times.setdefault(v_id_check, {})['__PENALTY__'] = math.inf
                        vehicle_state[v_id_check] = 'FINISHED'
                        num_finished += 1
            
            if stuck_vehicles_count > 0:
                break
    
    # Conflict detection (same as before)
    safety_time = config.tau
    for point in tau_p_dict.keys():
        arrivals = []
        for v in permutation:
            if v.id in scheduled_times and point in scheduled_times[v.id]:
                arrivals.append((v.id, scheduled_times[v.id][point]))
        
        arrivals.sort(key=lambda x: x[1])
        
        for i in range(len(arrivals) - 1):
            v_id_1, t_1 = arrivals[i]
            v_id_2, t_2 = arrivals[i + 1]
            time_gap = t_2 - t_1
            
            if time_gap < safety_time:
                conflict_count_per_point[point] += 1
                conflict_pairs.append((v_id_1, v_id_2, point, time_gap))
    
    # --- Delay calculation using segment speeds ---
    decoder_results = []
    for v in permutation:
        if v.id not in speeds_by_segment:
            continue
        
        delay = 0.0
        
        if scheduled_times.get(v.id, {}).get('__PENALTY__') == math.inf:
            delay = 9999.0
        elif v.id not in t_ear:
            delay = 9999.0
        elif not v.path:
            delay = 0.0
        else:
            path_headway_sum = sum(tau_p_dict.get(p, config.tau) for p in v.path)
            num_segments = len(v.path) - 1
            
            # Calculate free-flow time using segment speeds
            free_flow_travel = 0.0
            for seg_idx in range(num_segments):
                # Cap segment index at 4
                capped_seg_idx = min(seg_idx, 4)
                seg_speed = speeds_by_segment[v.id][capped_seg_idx]
                safe_seg_speed = max(seg_speed, 1e-6)
                free_flow_travel += config.inter_conflict_distance / safe_seg_speed
            
            t_free = t_ear[v.id] + path_headway_sum + free_flow_travel
            
            last_p = v.path[-1]
            if last_p not in scheduled_times.get(v.id, {}):
                delay = 9999.0
            else:
                t_exit = scheduled_times[v.id][last_p] + tau_p_dict.get(last_p, config.tau)
                delay = max(0.0, t_exit - t_free)
        
        decoder_results.append({
            "id": v.id,
            "delay": delay,
            "is_emergency": v.priority_status,
            "conflicts_at_point": {point: count for point, count in conflict_count_per_point.items() if count > 0}
        })
    
    if return_full_schedule:
        return decoder_results, scheduled_times, t_ear, conflict_pairs
    else:
        return decoder_results