# File: objective.py
import config

def calculate_objective(decoder_results):
    """
    Computes the weighted objective from a simple list of delay results.
    This function implements the objective function from your report.

    Parameters
    ----------
    decoder_results : list[dict]
        A list of dictionaries. This is the *OUTPUT* of the Decoder.
        Each dictionary must be:
        {"id": int, "delay": float, "is_emergency": bool}
    
    Returns
    -------
    dict : {"delays", "fem", "fall", "f"}
        A dictionary containing the final calculated objective values.
    """
    # Get weights directly from the config file
    alpha = config.alpha
    beta = config.beta

    # Create a dictionary of {vehicle_id: delay}
    delays = {r["id"]: r["delay"] for r in decoder_results}

    # fem = sum of delays for emergency vehicles
    fem = sum(r["delay"] for r in decoder_results if r["is_emergency"])
    
    # fall = sum of delays for all vehicles
    fall = sum(delays.values())
    
    # f = weighted objective function
    f = alpha * fem + beta * fall

    return {"delays": delays, "fem": fem, "fall": fall, "f": f}


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
    print(f"(Based on config: alpha={config.alpha}, beta={config.beta})")