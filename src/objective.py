import numpy

class Vehicle:
    def __init__(self, vid, path, times, tear, is_emergency=False):
        """
        Parameters
        ----------
        vid : int
            Vehicle ID.
        path : list[str]
            List of conflict points (e.g. ['C1','C5','MN']).
        times : dict[str, float]
            Time when the vehicle passes each conflict point.
        tear : float
            Earliest arrival time at first conflict point.
        is_emergency : bool
            True if this vehicle is an emergency vehicle.
        """
        self.id = vid
        self.path = path
        self.times = times
        self.tear = tear
        self.is_emergency = is_emergency

    def exit_time(self, tau):
        """Compute exit time (last conflict point time + headway)."""
        last_p = self.path[-1]
        return self.times[last_p] + tau[last_p]

    def free_time(self, tau):
        """Compute free-flow exit time (if there were no delays)."""
        return self.tear + sum(tau[p] for p in self.path)

    def delay(self, tau):
        """Delay = actual exit - free-flow exit."""
        return max(0.0, self.exit_time(tau) - self.free_time(tau))


def objective_from_queues(queues, tau, alpha=1.0, beta=1.0):
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

    all_vehicles = [v for q in queues.values() for v in q]
    delays = {v.id: v.delay(tau) for v in all_vehicles}

    fem = sum(v.delay(tau) for v in all_vehicles if v.is_emergency)
    fall = sum(delays.values())
    f = alpha * fem + beta * fall

    return {"delays": delays, "fem": fem, "fall": fall, "f": f}


if __name__ == "__main__":
    # Headway times
    tau = {p: 1.0 for p in ['C1','C2','C3','C4','C5','C6','C7','C8','MN','ME','MS','MW']}

    # Define 4 queues
    north_queue = [
        Vehicle(1, ['C1','C5','MN'], {'C1':0, 'C5':1, 'MN':2}, tear=0, is_emergency=True)
    ]
    east_queue = [
        Vehicle(2, ['C2','C6','ME'], {'C2':1, 'C6':2, 'ME':3}, tear=0)
    ]
    south_queue = [
        Vehicle(3, ['C3','C7','MS'], {'C3':2, 'C7':3, 'MS':4}, tear=1)
    ]
    west_queue = [
        Vehicle(4, ['C4','C8','MW'], {'C4':3, 'C8':4, 'MW':5}, tear=2)
    ]

    queues = {'N': north_queue, 'E': east_queue, 'S': south_queue, 'W': west_queue}

    result = objective_from_queues(queues, tau, alpha=2.0, beta=1.0)

    print("\n=== Objective Function Test with Queues ===")
    print("Delays:", result["delays"])
    print("Emergency delay (fem):", result["fem"])
    print("Total delay (fall):", result["fall"])
    print("Weighted objective (f):", result["f"])
