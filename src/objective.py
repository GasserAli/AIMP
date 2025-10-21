import numpy
from vehicle import Vehicle 

# class Vehicle:
#     def __init__(self, vid, path, times, tear, is_emergency=False):
#         """
#         Parameters
#         ----------
#         vid : int
#             Vehicle ID.
#         path : list[str]
#             List of conflict points (e.g. ['C1','C5','MN']).
#         times : dict[str, float]
#             Time when the vehicle passes each conflict point.
#         tear : float
#             Earliest arrival time at first conflict point.
#         is_emergency : bool
#             True if this vehicle is an emergency vehicle.
#         """
#         self.id = vid
#         self.path = path
#         self.times = times
#         self.tear = tear
#         self.is_emergency = is_emergency

#     def exit_time(self, tau):
#         """Compute exit time (last conflict point time + headway)."""
#         last_p = self.path[-1]
#         return self.times[last_p] + tau[last_p]

#     def free_time(self, tau):
#         """Compute free-flow exit time (if there were no delays)."""
#         return self.tear + sum(tau[p] for p in self.path)

#     def delay(self, tau):
#         """Delay = actual exit - free-flow exit."""
#         return max(0.0, self.exit_time(tau) - self.free_time(tau))


def objective_from_queues(permutation, alpha=1.0, beta=1.0):
    """
    Compute the total and weighted delay for 4 queues of Vehicle objects.

    Parameters
    ----------
    queues : dict[str, list[Vehicle]]
        Dictionary with keys ['N','E','S','W'] each containing a list of Vehicle objects.
    tau : dict[str, float]
        Headway times for each conflict point.
    alpha, beta : float
        Weights for emergency and total delays.

    Returns
    -------
    dict : {"delays", "fem", "fall", "f"}
    """

    all_vehicles = [v for v in permutation]
    delays = {v.vehicle_id: v.delay for v in all_vehicles}

    fem = sum(v.delay for v in all_vehicles if v.priority_status)
    fall = sum(delays.values())
    f = alpha * fem + beta * fall

    return {"delays": delays, "fem": fem, "fall": fall, "f": f}


