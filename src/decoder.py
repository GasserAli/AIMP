# File: decoder.py

import config
from geometry import Geometry



def run_decoder(permutation, speeds, geom, tau_p_dict):
    """
    Implements the Decoder (Algorithm 1) from the report.
    Given a solution (Π, v), it computes a feasible schedule
    and returns the delay for each vehicle.
    
    This function respects C0, C1, C2, and C3.
    """
    
    # --- 1. Initialization ---
    
    # Store dynamic data keyed by vehicle.id, not on the object
    t_ear = {}            # (t_v^ear) Earliest arrival at *first* conflict point
    scheduled_times = {}  # {v.id: {p: t_v,p}} Scheduled arrival time at *each* point
    path_pointers = {}    # {v.id: int} Tracks which point in its path a vehicle is at
    vehicle_state = {}    # {v.id: 'WAITING', 'RUNNING', 'FINISHED'}
    
    # (A[p]) Availability clocks for all conflict points
    availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}
    
    # Get physical queues (for C0/t_ear calculation)
    # This must use the *original* config.pi order, not the permutation
    geom_init = Geometry()
    geom_init.create_entry_queue(config.pi) 
    
    # --- 2. Calculate t_ear (Earliest Arrival) for all vehicles ---
    # This implements C0 (no-catch-up) and C1 (earliest reachability)
    
    speeds_dict = {p.id: s for p, s in zip(permutation, speeds)}
    
    for queue in geom_init.entry_queues.values():
        if not queue:
            continue
            
        # Get data for the leader in the physical queue
        v_leader = queue[0]
        v_leader_speed = speeds_dict[v_leader.id]
        
        # d0 = distance from stop-line to first point
        # (This is a simplified d0, you can make this a config map later)
        d0 = 10.0 # meters (Assume 10m to first point for all leaders)
        
        t_ear[v_leader.id] = d0 / max(v_leader_speed, 1e-6)
        
        last_t_ear = t_ear[v_leader.id]
        last_speed = v_leader_speed
        
        # Calculate for all followers
        for v_follower in queue[1:]:
            v_follower_speed = speeds_dict[v_follower.id]
            
            # Follower's free-flow time = time of car in front
            # + safety time (dist / speed)
            safety_time = config.safety_distance / max(v_follower_speed, 1e-6)
            
            # C0 (no-catch-up) is enforced by SA's validate_speeds.
            # C1: Earliest time follower can reach first point
            t_ear[v_follower.id] = last_t_ear + safety_time
            
            last_t_ear = t_ear[v_follower.id]
            last_speed = v_follower_speed
            
    # --- 3. Initialize Decoder State ---
    for v in permutation:
        path_pointers[v.id] = 0
        vehicle_state[v.id] = 'WAITING'
        scheduled_times[v.id] = {}

    # --- 4. Run Scheduling Loop (Algorithm 1) ---
    # We iterate based on the permutation Π (DV1)
    
    # Keep track of vehicles that are finished
    num_finished = 0
    while num_finished < len(permutation):
        
        # Find the next eligible vehicle *based on the permutation order*
        v = None
        for vehicle_in_perm in permutation:
            if vehicle_state[vehicle_in_perm.id] != 'FINISHED':
                v = vehicle_in_perm
                break # Found the highest-priority (earliest in Π) unfinished vehicle
        
        if v is None:
            break # Should not happen, but for safety

        # Get vehicle's current state
        v_id = v.id
        path_idx = path_pointers[v_id]
        
        # Check if vehicle is done
        if path_idx >= len(v.path):
            vehicle_state[v_id] = 'FINISHED'
            num_finished += 1
            continue

        # Get the next conflict point this vehicle needs
        p = v.path[path_idx]
        tau = tau_p_dict[p]

        # --- Calculate Scheduled Time t_v,p ---
        # [cite_start]This is the core of Algorithm 1 [cite: 173]

        # 1. Earliest time it can reach this point (C1)
        if path_idx == 0:
            # First point: must be >= t_ear
            t_reach = t_ear[v_id]
        else:
            # Subsequent points: must be >= time from *previous* point (C3)
            p_prev = v.path[path_idx - 1]
            tau_prev = tau_p_dict[p_prev]
            t_prev_scheduled = scheduled_times[v_id][p_prev]
            
            # (Note: This assumes 0 travel time between conflict points)
            # (A more advanced decoder would add d(p_prev, p) / v_speed)
            t_reach = t_prev_scheduled + tau_prev
        
        # 2. Time the conflict point is available (C2)
        t_available = availability_clocks[p]

        # 3. Schedule the time (t_v,p)
        t_scheduled = max(t_reach, t_available)
        
        # --- Update State ---
        scheduled_times[v_id][p] = t_scheduled
        availability_clocks[p] = t_scheduled + tau
        path_pointers[v_id] += 1
        vehicle_state[v_id] = 'RUNNING'
        
    # --- 5. Calculate Final Delays ---
    decoder_results = []
    
    for v in permutation:
        if not v.path:
            decoder_results.append({"id": v.id, "delay": 0.0, "is_emergency": v.is_emergency})
            continue

        # [cite_start]t_v^free = t_v^ear + sum(tau_p_v,i) [cite: 136]
        path_headway_sum = sum(tau_p_dict[p] for p in v.path)
        t_free = t_ear[v.id] + path_headway_sum
        
        # [cite_start]t_v^exit = t_v,p_v,K_v + tau_p_v,K_v [cite: 135]
        last_p = v.path[-1]
        t_exit = scheduled_times[v.id][last_p] + tau_p_dict[last_p]
        
        # [cite_start]delay_v = max(0, t_v^exit - t_v^free) [cite: 138]
        delay = max(0.0, t_exit - t_free)
        
        decoder_results.append({
            "id": v.id,
            "delay": delay,
            "is_emergency": v.priority_status
        })

    return decoder_results