# File: objective.py
import config
import numpy as np

def calculate_objective(decoder_results, speeds_matrix=None):
    """
    Computes the weighted objective from a simple list of delay results.
    This function implements the objective function from your report.

    Parameters
    ----------
    decoder_results : list[dict]
        A list of dictionaries. This is the *OUTPUT* of the Decoder.
        Each dictionary must be:
        {"id": int, "delay": float, "is_emergency": bool, "conflicts_at_point": dict (optional)}
    
    speeds_matrix : optional
        A 2D list of speeds for each segment. Used to calculate speed incentives.

    Returns
    -------
    dict : {"delays", "fem", "fall", "conflicts", "conflict_penalty", "f"}
        A dictionary containing the final calculated objective values.
    """
    # Get weights directly from the config file
    alpha = config.alpha
    beta = config.beta
    gamma = config.gamma
    alpha_speed = config.alpha_speed
    
    # Create a dictionary of {vehicle_id: delay}
    delays = {r["id"]: r["delay"] for r in decoder_results}

    # fem = sum of delays for emergency vehicles
    fem = sum(r["delay"] for r in decoder_results if r["is_emergency"])
    
    # fall = sum of delays for all vehicles
    fall = sum(delays.values())
    
    # Count total conflicts
    total_conflicts = 0
    for result in decoder_results:
        if "conflicts_at_point" in result:
            total_conflicts += sum(result["conflicts_at_point"].values())
    
    # Calculate conflict penalty
    conflict_penalty = gamma * total_conflicts
    
    # NEW: Add speed incentive (reward higher speeds)
    # This encourages the optimizer to use speeds closer to the maximum (12 m/s)
    speed_incentive = 0.0
    if speeds_matrix is not None:
        v_min, v_max = config.velocity_range
        # Flatten all segment speeds and compute average
        all_speeds = [s for vehicle_speeds in speeds_matrix for s in vehicle_speeds]
        if all_speeds:
            avg_speed = sum(all_speeds) / len(all_speeds)
            # Reward deviation from minimum towards maximum
            # Formula: (avg_speed - v_min) / (v_max - v_min) gives normalized score 0-1
            # Multiply by a weight to make it meaningful
            speed_reward = (avg_speed - v_min) / (v_max - v_min) if v_max > v_min else 0
            speed_incentive = -alpha_speed * speed_reward  # Negative = reward (minimization)
    
    # f = weighted objective function: minimize emergency delays, all delays, AND conflicts
    f = alpha * fem + beta * fall + conflict_penalty + speed_incentive

    return {
        "delays": delays, 
        "fem": fem, 
        "fall": fall, 
        "conflicts": total_conflicts,
        "conflict_penalty": conflict_penalty,
        "speed_incentive": speed_incentive,
        "f": f
    }


def compute_objective(decoder_results, 
                     alpha: float = 1.0, 
                     beta: float = 1.0, 
                     gamma: float = 100.0):
    """
    Compute weighted objective function with conflict penalties.
    
    Parameters:
    - decoder_results: List of dicts with 'id', 'delay', 'is_emergency', 'conflicts_at_point'
    - alpha: Weight for emergency vehicle delays
    - beta: Weight for all vehicle delays
    - gamma: Weight for each conflict (penalty per conflict)
    
    Returns:
    - dict with 'f' (total objective), 'fem' (emergency delay), 'fall' (total delay), 'conflicts'
    """
    
    delays = {}
    total_conflicts = 0
    
    for result in decoder_results:
        delays[result["id"]] = result["delay"]
        # Count conflicts for this vehicle
        if "conflicts_at_point" in result:
            total_conflicts += sum(result["conflicts_at_point"].values())
    
    # Calculate delay components
    fem = sum(result["delay"] for result in decoder_results if result["is_emergency"])
    fall = sum(result["delay"] for result in decoder_results)
    
    # Add conflict penalty
    conflict_penalty = gamma * total_conflicts
    
    # Total objective: minimize emergency delays, all delays, AND conflicts
    f = alpha * fem + beta * fall + conflict_penalty
    
    return {
        "f": f,
        "fem": fem,
        "fall": fall,
        "conflicts": total_conflicts,
        "conflict_penalty": conflict_penalty,
        "delays": delays
    }


# --- Test Block ---
if __name__ == "__main__":
    
    print("--- Testing Decoupled Objective Function ---")

    # This is a MOCK (simulated) output from a Decoder.
    mock_decoder_output = [
        {"id": 1, "delay": 0.0, "is_emergency": True},
        {"id": 2, "delay": 1.0, "is_emergency": False},
        {"id": 3, "delay": 1.0, "is_emergency": False}
    ]

    print(f"Simulated Decoder output: {mock_decoder_output}")

    # Run the objective function
    result = calculate_objective(mock_decoder_output)

    # Print the final results
    print("\n=== Objective Function Test Results ===")
    print("Delays:", result["delays"])
    print("Emergency delay (fem):", result["fem"])
    print("Total delay (fall):", result["fall"])
    print(f"Weighted objective (f):", result["f"])
    print(f"(Based on config: alpha={config.alpha}, beta={config.beta}")