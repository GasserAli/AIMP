import config
from geometry import Geometry
import math
from typing import List, Dict, Tuple, Union
from vehicle import Vehicle


def run_decoder(permutation: List[Vehicle],
                speeds: List[float],
                geom: Geometry, # Needs original queues for t_ear calc
                tau_p_dict: Dict[str, float],
                return_full_schedule: bool = False):
    """
    Corrected and hardened scheduling decoder based on Algorithm 1 logic.

    Key fixes compared to original:
      - t_ear (earliest arrival at first conflict point) is computed using
        a fixed FREE_FLOW_SPEED (the maximum allowed) and the vehicle's
        queue position. It does NOT depend on the optimizer-chosen speed.
      - Free-flow benchmark travel time (t_free) is calculated using the
        fixed FREE_FLOW_SPEED so it remains a stable baseline.
      - Actual travel times between conflict points continue to use the
        optimizer-chosen speeds, so speed choices affect actual performance
        (and therefore the computed delay), but cannot be used to cheat
        the benchmark.
      - Defensive programming for division-by-zero while keeping min speed
        tiny but non-zero for calculations.

    Parameters:
    - permutation (List[Vehicle]): Current order of vehicles to consider.
    - speeds (List[float]): List of speeds corresponding to vehicles in the permutation.
    - geom (Geometry): Geometry object with original queues (based on config.pi).
    - tau_p_dict (Dict[str, float]): Headway time required at each conflict point.
    - return_full_schedule (bool): If True, returns detailed schedule besides delays.

    Returns:
    - If False: decoder_results (List[dict]) - List of {'id', 'delay', 'is_emergency'}.
    - If True: (decoder_results, scheduled_times, t_ear)
    """

    # --- 1. Initialization ---
    t_ear = {}            # Earliest arrival time at the *first* conflict point for each vehicle
    scheduled_times = {}  # {v_id: {point: arrival_time}}
    path_pointers = {}    # {v_id: index_in_path}
    vehicle_state = {}    # {v_id: 'WAITING', 'RUNNING', 'FINISHED'}
    # A[p]: Time when conflict point 'p' becomes free
    availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}

    # Use the passed-in geom object, assumed to have queues based on config.pi
    geom_init = geom

    # Map vehicle IDs to speeds for quick lookup
    speeds_dict = {p.id: s for p, s in zip(permutation, speeds)}

    # --- 2. Calculate Corrected t_ear (Earliest Arrival at First Point) ---
    # IMPORTANT: t_ear must NOT depend on the decision-variable speed.

    FREE_FLOW_SPEED = config.velocity_range[1] if hasattr(config, 'velocity_range') else max(config.velocity_range)
    if FREE_FLOW_SPEED <= 0:
        FREE_FLOW_SPEED = 1.0  # fallback guard

    # Time headway between queued vehicles measured at free-flow speed
    QUEUE_HEADWAY_TIME = config.safety_distance / FREE_FLOW_SPEED
    d0 = 10.0  # meters (Distance for queue leader to reach first conflict point)

    for approach, queue in geom_init.entry_queues.items():
        if not queue:
            continue

        # Leader baseline arrival time using free-flow speed
        leader_time = d0 / FREE_FLOW_SPEED

        for idx, v in enumerate(queue):
            # Only set t_ear for vehicles that are present in the current solution
            if v.id not in speeds_dict:
                # We intentionally skip vehicles not present in the current decision
                # so that the decoder handles partial solutions too.
                continue

            # Earliest arrival is determined by queue index and free-flow baseline
            current_t_ear = leader_time + idx * QUEUE_HEADWAY_TIME

            t_ear[v.id] = current_t_ear

    # --- 3. Initialize Decoder State for vehicles in the current permutation ---
    num_vehicles_in_solution = 0
    for v in permutation:
        if v.id in speeds_dict: # Ensure vehicle is part of the current solution
            num_vehicles_in_solution += 1
            path_pointers[v.id] = 0
            # Mark as WAITING if we have a t_ear entry, otherwise mark FINISHED
            vehicle_state[v.id] = 'WAITING' if v.id in t_ear else 'FINISHED'
            scheduled_times[v.id] = {}

    # --- 4. Dynamic Scheduling Loop (Iterative based on dependencies) ---
    num_finished = sum(1 for v_id, state in vehicle_state.items() if state == 'FINISHED')
    # Start loop with vehicles that are part of the solution and not already finished
    active_vehicles_ids = {v.id for v in permutation if vehicle_state.get(v.id) == 'WAITING'}

    # Safety counter to prevent infinite loops in case of true deadlock
    loop_limit = num_vehicles_in_solution * max((len(v.path) for v in permutation if v.path and v.id in speeds_dict), default=1) * 3
    loop_count = 0

    while num_finished < num_vehicles_in_solution and loop_count < loop_limit:

        made_progress_this_iteration = False # Flag to detect stalls
        processed_vehicle_ids = set() # Track vehicles processed in this loop pass

        # Process vehicles based on their order in the current permutation
        for v in permutation:
            v_id = v.id
            # Skip if not in solution, already finished, or not ready to be processed yet
            if v_id not in active_vehicles_ids or vehicle_state.get(v_id) == 'FINISHED':
                continue

            processed_vehicle_ids.add(v_id) # Mark as considered in this pass

            path_idx = path_pointers.get(v_id, 0)

            # Check if vehicle path is complete
            if path_idx >= len(v.path):
                if vehicle_state.get(v_id) != 'FINISHED':
                    vehicle_state[v_id] = 'FINISHED'; num_finished += 1; made_progress_this_iteration = True
                continue # Finished

            p = v.path[path_idx] # Current conflict point target

            # Safety check: ensure point 'p' is valid and has a headway time defined
            if p not in tau_p_dict:
                if vehicle_state.get(v_id) != 'FINISHED':
                    vehicle_state[v_id] = 'FINISHED'; num_finished += 1; made_progress_this_iteration = True
                continue

            tau = tau_p_dict[p] # Headway required at point 'p'

            # --- Calculate Earliest Reach Time (t_reach) for point 'p' ---
            if path_idx == 0:
                # First point: earliest reach is t_ear (precomputed) -- use infinity if missing
                t_reach = t_ear.get(v_id, math.inf)
            else:
                # Subsequent points: earliest reach depends on previous point's departure + travel time
                p_prev = v.path[path_idx - 1]

                # Check if the previous point has been scheduled yet
                if p_prev not in scheduled_times.get(v_id, {}):
                    # Cannot schedule 'p' yet, must process p_prev first. Skip for now.
                    continue # Try scheduling this vehicle's point 'p' in the next iteration

                tau_prev = tau_p_dict.get(p_prev, config.tau)
                t_prev_departure = scheduled_times[v_id][p_prev] + tau_prev

                # Calculate travel time from p_prev to p using optimized (decision) speed
                distance = config.inter_conflict_distance # Assumes uniform distance
                speed = speeds_dict.get(v_id, 1e-6)
                travel_time = distance / max(speed, 1e-6)

                t_reach = t_prev_departure + travel_time

            # --- Determine Scheduled Time (t_scheduled) ---
            # Earliest time the *current point* is free (based on availability clocks)
            t_available = availability_clocks[p]
            # Scheduled time is the later of when the vehicle arrives *and* when the point is free
            t_scheduled = max(t_reach, t_available)

            # --- Update State ---
            scheduled_times.setdefault(v_id, {})[p] = t_scheduled
            # Point 'p' is now busy until departure
            availability_clocks[p] = t_scheduled + tau
            path_pointers[v_id] = path_idx + 1 # Advance vehicle to next point in its path
            vehicle_state[v_id] = 'RUNNING' # Mark as actively moving
            made_progress_this_iteration = True # We successfully scheduled a point

        # Update the set of active vehicles for the next pass
        active_vehicles_ids = {v_id for v_id, state in vehicle_state.items() if state == 'WAITING' or state == 'RUNNING'}
        loop_count += 1

        # Safety break: If a full pass makes no progress among active vehicles, declare deadlock/penalty
        if not made_progress_this_iteration and num_finished < num_vehicles_in_solution:
             stuck_vehicles_count = 0
             for v_id_check in active_vehicles_ids:
                 if v_id_check in processed_vehicle_ids:
                     stuck_vehicles_count += 1
                     # Penalize the stuck vehicle: mark finished with high penalty
                     if vehicle_state.get(v_id_check) != 'FINISHED':
                         if v_id_check not in t_ear: t_ear[v_id_check] = 0
                         scheduled_times.setdefault(v_id_check, {})['__PENALTY__'] = math.inf
                         vehicle_state[v_id_check] = 'FINISHED'
                         num_finished += 1

             # If any vehicles were identified as stuck, break the loop
             if stuck_vehicles_count > 0:
                 break

    # --- 5. Calculate Final Delays ---
    decoder_results = []

    for v in permutation:
        if v.id not in speeds_dict: continue # Only process vehicles in the current solution

        delay = 0.0 # Default delay

        # Handle penalized vehicles
        if scheduled_times.get(v.id, {}).get('__PENALTY__') == math.inf:
             delay = 9999.0
        elif v.id not in t_ear: # Handle vehicles that never started properly
             delay = 9999.0
        elif not v.path: # Handle vehicles with empty paths
            delay = 0.0
        else:
            # --- Compute free-flow benchmark using FIXED FREE_FLOW_SPEED ---
            path_headway_sum = sum(tau_p_dict.get(p, config.tau) for p in v.path)
            num_segments = len(v.path) - 1 # Number of inter-point movements

            # Free-flow travel time at MAX SPEED (fixed benchmark)
            free_flow_travel = 0
            if num_segments > 0:
                free_flow_travel = num_segments * config.inter_conflict_distance / FREE_FLOW_SPEED

            t_free = t_ear[v.id] + path_headway_sum + free_flow_travel

            # Actual exit time (based on scheduled times and optimized speeds)
            last_p = v.path[-1]
            if last_p not in scheduled_times.get(v.id, {}):
                 # This implies the vehicle never finished its path in the simulation
                 delay = 9999.0 # High penalty
            else:
                t_exit = scheduled_times[v.id][last_p] + tau_p_dict.get(last_p, config.tau)
                delay = max(0.0, t_exit - t_free)

        decoder_results.append({
            "id": v.id,
            "delay": delay,
            "is_emergency": v.priority_status
        })

    # --- 6. Return Data ---
    if return_full_schedule:
        return decoder_results, scheduled_times, t_ear
    else:
        return decoder_results


# Optional quick test harness when running this file directly (will not run during imports)
if __name__ == '__main__':
    print('Decoder module loaded. This file contains the corrected run_decoder implementation.')
