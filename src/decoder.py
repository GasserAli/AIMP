from dataclasses import dataclass
from typing import List, Dict, Tuple
from vehicle import Vehicle 
import config 
from geometry import Geometry

# @dataclass
# class Vehicle:
#     vehicle_id: int
#     Approach: str  # 'N','S','E','W'
#     Maneuver: str  # 'S','L','R'
#     velocity: float  # m/s
#     path: List[str]  # list of conflict point ids (e.g., ['C1','C9','Mn'])


from typing import Union


def is_permutation_valid(
    permutation: List[Vehicle],
    distance_to_first_conflict: Union[Dict[int, float], float],
    inter_conflict_distance: Union[Dict[Tuple[str, str], float], float]=10.0,
    safety_time: float = config.tau,
    queue_spacing: float = 10.0,
) -> Tuple[bool, int]:
    """Return True if the permutation is valid (no two vehicles enter same conflict
    point within safety_time), False otherwise.

    Parameters
    - permutation: list of Vehicle objects in the order they'll traverse the intersection
    - distance_to_first_conflict: mapping from vehicle.vehicle_id -> distance (meters) from
      the vehicle's queue start to the first conflict point in its path
    - inter_conflict_distance: mapping from (conflict_i, conflict_j) -> distance (meters)
      between consecutive conflict points along the vehicle's path. If consecutive points
      are the same key, distance should be 0.
    - safety_time: extra time buffer (seconds) that must separate entries into the same conflict

    Assumptions
    - Each vehicle travels its path at the constant speed `velocity`.
    - path is ordered list of conflict point ids the vehicle will traverse in sequence.
    - inter_conflict_distance provehicle_ides distances for consecutive pairs present in any vehicle.path.
    """
    # Initialize conflict count
    count_conflicts = 0
    
    #dict mapping vehicle ids to vehicles
    vehicles = {}

    # First: group vehicles by approach (queue order is the order they appear in permutation)
    approaches: Dict[str, List[Vehicle]] = {}
    for veh in permutation:
        vehicles[veh.vehicle_id] = veh
        approaches.setdefault(veh.approach, []).append(veh)

    # print(approaches)

    # Enforce queue speed constraint: following vehicle must not be faster than the vehicle ahead
    for approach, vehs in approaches.items():
        for i in range(1, len(vehs)):
            v_ahead = vehs[i - 1].velocity
            v_follow = vehs[i].velocity
            if v_follow > v_ahead + 1e-9:
                # Following vehicle is faster than the vehicle ahead in the same queue -> invalid
                return (False, -1)

    # Map conflict point -> list of (vehicle id, arrival_time)
    conflict_arrivals: Dict[str, List[Tuple[int, float]]] = {}

    # We'll compute per-vehicle distance to first conflict. If distance_to_first_conflict is
    # a scalar it represents the distance for the front vehicle in each queue; subsequent
    # vehicles in the same queue are placed further back by `queue_spacing` meters each.
    # If a dict is provehicle_ided, its values are used directly for each vehicle (assumed consistent).

    # Prepare per-vehicle d0 mapping when scalar is used
    per_vehicle_d0: Dict[int, float] = {}
    if not isinstance(distance_to_first_conflict, dict):
        base_d0 = float(distance_to_first_conflict)
        # For each approach, assign d0 for vehicles in queue order (front -> back)
        for approach, vehs in approaches.items():
            for idx, veh in enumerate(vehs):
                per_vehicle_d0[veh.vehicle_id] = base_d0 + idx * float(queue_spacing)

    for veh in permutation:
        if len(veh.path) == 0:
            continue

        # distance from queue start to first conflict
        if isinstance(distance_to_first_conflict, dict):
            d0 = distance_to_first_conflict.get(veh.vehicle_id, None)
            if d0 is None:
                raise ValueError(f"distance_to_first_conflict missing for vehicle {veh.vehicle_id}")
        else:
            d0 = per_vehicle_d0.get(veh.vehicle_id, None)
            if d0 is None:
                # fallback: use base value
                d0 = float(distance_to_first_conflict)

        # time to reach first conflict
        t = d0 / max(veh.velocity, 1e-3)
        print(f"Vehicle {veh.vehicle_id} arrives at conflict {veh.path[0]} at time {t:.2f}s")


        # record arrival to first conflict
        first_conf = veh.path[0]
        conflict_arrivals.setdefault(first_conf, []).append((veh.vehicle_id, t))

        # walk through subsequent conflicts
        for i in range(len(veh.path) - 1):
            a = veh.path[i]
            b = veh.path[i + 1]
            # inter_conflict_distance can be a dict or a scalar (same distance between any pair)
            if isinstance(inter_conflict_distance, dict):
                key = (a, b)
                d = inter_conflict_distance.get(key, None)
                if d is None:
                    # try symmetric key
                    d = inter_conflict_distance.get((b, a), None)
                if d is None:
                    raise ValueError(f"inter_conflict_distance missing for pair {a}->{b}")
            else:
                d = float(inter_conflict_distance)

            dt = d / max(veh.velocity, 1e-3)
            t += dt
            conflict_arrivals.setdefault(b, []).append((veh.vehicle_id, t))
            print(f"Vehicle {veh.vehicle_id} arrives at conflict {a} at time {t:.2f}s")


    # Now check each conflict point for temporal collisions
    for cp, arrivals in conflict_arrivals.items():
        # print(arrivals)
        # sort arrivals by time
        arrivals.sort(key=lambda x: x[1])
        for i in range(len(arrivals) - 1):
            t_curr = arrivals[i][1]
            t_next = arrivals[i + 1][1]
            if t_next - t_curr < safety_time:
                print(f"collision detected at {cp} between vehicles {arrivals[i + 1]} and {arrivals[i]}")
                print(f'previous conflict arrival times \n {conflict_arrivals}')
                count_conflicts += 1
                v_id = arrivals[i+1][0]
                # for vehicle in permutation:
                #     if vehicle.vehicle_id == v_id:
                #         vehicle.set_delay(t_next-t_curr+safety_time)
                vehicles[v_id].set_delay(t_next - t_curr + safety_time)
                flag = True
                for path in vehicles[v_id].path:
                    if path == cp:
                        flag = True
                    if flag:
                        times = conflict_arrivals[path]
                        for vid, time in times:
                            if vid == v_id:
                                conflict_arrivals[path].remove((vid, time))
                                conflict_arrivals[path].append((vid, time + vehicles[v_id].delay))
                        

            print(f'updated conflict arrival times: \n {conflict_arrivals}')
                # conflict detected
                

    return (count_conflicts == 0, count_conflicts)


# if __name__ == '__main__':
#     # Simple unit test
#     velocity = 10
#     v1 = Vehicle(1, 'E', 'L', velocity=velocity, path=['C7', 'C9','C10','C14', 'Ms'])
#     v2 = Vehicle(2, 'E', 'S', velocity=velocity, path=['C4', 'C3','C2', 'C1','Mw'])
#     v3 = Vehicle(3, 'N', 'L', velocity=velocity, path=['C2', 'C6','C7', 'C12','Me'])
#     v4 = Vehicle(4, 'N', 'L', velocity=velocity, path=['C2', 'C6','C7', 'C12','Me'])


#     perm = [v1, v2, v3, v4]
#     # Use scalar distances (same for all vehicles and conflict hops)
#     d0 = 10.0  # meters to first conflict for all vehicles
#     inter = 10.0  # meters between consecutive conflict points for all hops

#     print('Permutation valid?', is_permutation_valid(perm, d0, inter, safety_time=3))
