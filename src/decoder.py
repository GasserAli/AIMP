import config
from geometry import Geometry
import math
from typing import List, Dict
from vehicle import Vehicle


def run_decoder(permutation: List[Vehicle],
                speeds: List[float],
                geom: Geometry,
                tau_p_dict: Dict[str, float],
                return_full_schedule: bool = False):

    # -----------------------------------------------------------
    # 1. INIT
    # -----------------------------------------------------------
    t_ear = {}
    scheduled_times = {}
    path_pointers = {}
    vehicle_state = {}

    # OLD:
    # availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}

    # NEW: (same idea)
    availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}

    # NEW: For real collision detection
    conflict_point_usage = {p: [] for p in tau_p_dict.keys()}

    geom_init = geom
    speeds_dict = {p.id: s for p, s in zip(permutation, speeds)}

    # -----------------------------------------------------------
    # 2. EARLIEST ARRIVAL (t_ear)
    # -----------------------------------------------------------

    # OLD:
    # t_ear used to depend on chosen speed → WRONG

    # NEW:
    # t_ear depends ONLY on queue order using MAX SPEED
    FREE_FLOW_SPEED = config.velocity_range[1]
    if FREE_FLOW_SPEED <= 0:
        FREE_FLOW_SPEED = 1.0

    QUEUE_HEADWAY_TIME = config.safety_distance / FREE_FLOW_SPEED
    d0 = 10.0

    for approach, queue in geom_init.entry_queues.items():
        if not queue:
            continue

        leader_time = d0 / FREE_FLOW_SPEED

        for idx, v in enumerate(queue):

            # OLD:
            # if v.id not in speeds_dict: continue  # same logic keeps

            if v.id not in speeds_dict:
                continue

            current_t_ear = leader_time + idx * QUEUE_HEADWAY_TIME
            t_ear[v.id] = current_t_ear

    # -----------------------------------------------------------
    # 3. INITIAL STATE
    # -----------------------------------------------------------
    num_vehicles_in_solution = 0

    for v in permutation:
        if v.id in speeds_dict:
            num_vehicles_in_solution += 1

            path_pointers[v.id] = 0

            # OLD:
            # vehicle_state[v.id] = "WAITING"

            # NEW:
            vehicle_state[v.id] = 'WAITING' if v.id in t_ear else 'FINISHED'

            scheduled_times[v.id] = {}

    # -----------------------------------------------------------
    # 4. MAIN SCHEDULING LOOP
    # -----------------------------------------------------------
    num_finished = sum(1 for st in vehicle_state.values() if st == 'FINISHED')
    active_vehicles_ids = {v.id for v in permutation if vehicle_state.get(v.id) == 'WAITING'}

    loop_limit = num_vehicles_in_solution * max((len(v.path) for v in permutation if v.path), default=1) * 3
    loop_count = 0

    while num_finished < num_vehicles_in_solution and loop_count < loop_limit:
        made_progress = False
        processed_vehicle_ids = set()

        for v in permutation:
            v_id = v.id

            if v_id not in active_vehicles_ids:
                continue

            if vehicle_state[v_id] == 'FINISHED':
                continue

            processed_vehicle_ids.add(v_id)

            path_idx = path_pointers.get(v_id, 0)
            if path_idx >= len(v.path):

                # OLD: vehicle_state[v]=FINISHED
                vehicle_state[v_id] = 'FINISHED'
                num_finished += 1
                made_progress = True
                continue

            p = v.path[path_idx]

            # OLD:
            # tau = config.tau

            # NEW:
            tau = tau_p_dict[p]

            # -----------------------------------------------------------
            # 4A. Compute earliest reach time
            # -----------------------------------------------------------

            if path_idx == 0:
                t_reach = t_ear.get(v_id, math.inf)

            else:
                p_prev = v.path[path_idx - 1]

                if p_prev not in scheduled_times[v_id]:
                    continue  # must wait for predecessor point

                tau_prev = tau_p_dict.get(p_prev, config.tau)
                t_prev_departure = scheduled_times[v_id][p_prev] + tau_prev

                dist = config.inter_conflict_distance
                speed = speeds_dict.get(v_id, 1e-6)
                travel_time = dist / max(speed, 1e-6)

                t_reach = t_prev_departure + travel_time

            # -----------------------------------------------------------
            # 4B. Conflict point availability
            # -----------------------------------------------------------

            # OLD:
            # t_scheduled = max(t_reach, availability_clocks[p])

            # NEW:
            enter_time = max(t_reach, availability_clocks[p])
            exit_time = enter_time + tau

            # -----------------------------------------------------------
            # 4C. REAL COLLISION DETECTION (NEW)
            # -----------------------------------------------------------
            # OLD DECODER NEVER CHECKED THIS (VERY IMPORTANT)

            for (prev_enter, prev_exit, prev_vid) in conflict_point_usage[p]:

                # NEW OVERLAP LOGIC
                if enter_time < prev_exit and prev_enter < exit_time:
                    # OLD: no check! (unsafe)
                    # NEW: print and shift time until safe
                    print(f"🚨 REAL COLLISION PREVENTED at {p} : {v_id} vs {prev_vid}")

                    # NEW: push forward until no overlap
                    enter_time = prev_exit
                    exit_time = enter_time + tau

            # NEW: store usage
            conflict_point_usage[p].append((enter_time, exit_time, v_id))

            # NEW = actual scheduled time
            t_scheduled = enter_time

            # OLD:
            # scheduled_times[v][p] = t_scheduled

            scheduled_times[v_id][p] = t_scheduled
            availability_clocks[p] = t_scheduled + tau

            path_pointers[v_id] = path_idx + 1
            vehicle_state[v_id] = "RUNNING"
            made_progress = True

        active_vehicles_ids = {vid for vid, st in vehicle_state.items() if st in ("WAITING", "RUNNING")}
        loop_count += 1

        # OLD:
        # no deadlock detection → could freeze

        # NEW: deadlock breaker
        if not made_progress and num_finished < num_vehicles_in_solution:
            for vid in active_vehicles_ids:
                scheduled_times.setdefault(vid, {})["__PENALTY__"] = math.inf
                vehicle_state[vid] = "FINISHED"
                num_finished += 1
            break

    # -----------------------------------------------------------
    # 5. Compute Delay
    # -----------------------------------------------------------
    decoder_results = []

    for v in permutation:
        if v.id not in speeds_dict:
            continue

        v_id = v.id
        delay = 0.0

        # OLD:
        # did not handle penalty correctly
        if scheduled_times.get(v_id, {}).get("__PENALTY__") == math.inf:
            delay = 9999.0

        elif v_id not in t_ear:
            delay = 9999.0

        elif not v.path:
            delay = 0.0

        else:
            path_headway_sum = sum(tau_p_dict[p] for p in v.path)
            num_segments = len(v.path) - 1

            free_flow_travel = 0
            if num_segments > 0:
                free_flow_travel = num_segments * config.inter_conflict_distance / FREE_FLOW_SPEED

            t_free = t_ear[v_id] + path_headway_sum + free_flow_travel

            last_p = v.path[-1]

            if last_p not in scheduled_times[v_id]:
                delay = 9999.0
            else:
                t_exit = scheduled_times[v_id][last_p] + tau_p_dict[last_p]
                delay = max(0.0, t_exit - t_free)

        decoder_results.append({
            "id": v_id,
            "delay": delay,
            "is_emergency": v.priority_status
        })

    if return_full_schedule:
        return decoder_results, scheduled_times, t_ear
    return decoder_results
