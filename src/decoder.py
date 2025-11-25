import config
from geometry import Geometry
import math
from typing import List, Dict, Tuple, Union
from vehicle import Vehicle


def run_decoder(permutation: List[Vehicle],
                speeds: List[float],
                geom: Geometry,
                tau_p_dict: Dict[str, float],
                return_full_schedule: bool = False):

    # --- 1. Initialization ---
    t_ear = {}
    scheduled_times = {}
    path_pointers = {}
    vehicle_state = {}
    availability_clocks = {p: 0.0 for p in tau_p_dict.keys()}
    colliosion_count=0
    # --- ADDED: conflict interval tracking for REAL collision detection ---
    conflict_point_usage = {p: [] for p in tau_p_dict.keys()}

    geom_init = geom
    speeds_dict = {p.id: s for p, s in zip(permutation, speeds)}

    # --- 2. Compute corrected t_ear ---
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
            if v.id not in speeds_dict:
                continue

            current_t_ear = leader_time + idx * QUEUE_HEADWAY_TIME
            t_ear[v.id] = current_t_ear

    # --- 3. Init states ---
    num_vehicles_in_solution = 0
    for v in permutation:
        if v.id in speeds_dict:
            num_vehicles_in_solution += 1
            path_pointers[v.id] = 0
            vehicle_state[v.id] = 'WAITING' if v.id in t_ear else 'FINISHED'
            scheduled_times[v.id] = {}

    # --- 4. Scheduling loop ---
    num_finished = sum(1 for st in vehicle_state.values() if st == 'FINISHED')
    active_vehicles_ids = {v.id for v in permutation if vehicle_state.get(v.id) == 'WAITING'}

    loop_limit = num_vehicles_in_solution * max((len(v.path) for v in permutation if v.path), default=1) * 3
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
                if vehicle_state[v_id] != 'FINISHED':
                    vehicle_state[v_id] = 'FINISHED'
                    num_finished += 1
                    made_progress_this_iteration = True
                continue

            p = v.path[path_idx]

            if p not in tau_p_dict:
                vehicle_state[v_id] = 'FINISHED'
                num_finished += 1
                made_progress_this_iteration = True
                continue

            tau = tau_p_dict[p]

            # --- t_reach ---
            if path_idx == 0:
                t_reach = t_ear.get(v_id, math.inf)
            else:
                p_prev = v.path[path_idx - 1]

                if p_prev not in scheduled_times[v_id]:
                    continue

                tau_prev = tau_p_dict.get(p_prev, config.tau)
                t_prev_departure = scheduled_times[v_id][p_prev] + tau_prev

                dist = config.inter_conflict_distance
                speed = speeds_dict.get(v_id, 1e-6)
                travel_time = dist / max(speed, 1e-6)

                t_reach = t_prev_departure + travel_time

            # --- t_available ---
            t_available = availability_clocks[p]

            # --- ADDED: REAL COLLISION CHECK ---
            enter_time = max(t_reach, t_available)
            exit_time = enter_time + tau

            # Check against previous usages of this conflict point
            for (prev_enter, prev_exit) in conflict_point_usage[p]:
                if enter_time < prev_exit and prev_enter < exit_time:
                    print(f"🚨 REAL COLLISION DETECTED at {p} involving vehicle {v_id}!")
                    colliosion_count+=1
            # After checking, store this interval
            conflict_point_usage[p].append((enter_time, exit_time))
            # --- END REAL COLLISION CHECK ---
            # --- Apply scheduling ---
            t_scheduled = enter_time
            scheduled_times[v_id][p] = t_scheduled
            availability_clocks[p] = t_scheduled + tau

            path_pointers[v_id] = path_idx + 1
            vehicle_state[v_id] = 'RUNNING'
            made_progress_this_iteration = True

        active_vehicles_ids = {vid for vid, st in vehicle_state.items()
                               if st in ('WAITING', 'RUNNING')}
        loop_count += 1

        if not made_progress_this_iteration and num_finished < num_vehicles_in_solution:
            for vid in active_vehicles_ids:
                if vid in processed_vehicle_ids:
                    scheduled_times.setdefault(vid, {})['__PENALTY__'] = math.inf
                    vehicle_state[vid] = 'FINISHED'
                    num_finished += 1
            break

    # --- 5. Compute delays ---
    decoder_results = []

    for v in permutation:
        if v.id not in speeds_dict:
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

            free_flow_travel = 0
            if num_segments > 0:
                free_flow_travel = num_segments * config.inter_conflict_distance / FREE_FLOW_SPEED

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
            "is_emergency": v.priority_status
        })

    if return_full_schedule:
        return decoder_results, scheduled_times, t_ear
    else:
        return decoder_results


if __name__ == '__main__':
    print("Decoder loaded with REAL collision detection logic.")
