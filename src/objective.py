import config

def calculate_objective(decoder_results, speeds=None):
    alpha = config.alpha
    beta = config.beta
    gamma = getattr(config, "gamma", 0.0)

    delays = {r["id"]: r["delay"] for r in decoder_results}
    fem = sum(r["delay"] for r in decoder_results if r["is_emergency"])
    fall = sum(delays.values())

    # --- Speed penalty ---
    if speeds is not None and len(speeds) > 0:
        v_min, v_max = config.velocity_range
        speed_penalty = sum((v_max - s) for s in speeds) / len(speeds)
    else:
        speed_penalty = 0.0

    # --- Final objective ---
    f = alpha * fem + beta * fall + gamma * speed_penalty

    return {
        "delays": delays,
        "fem": fem,
        "fall": fall,
        "speed_penalty": speed_penalty,
        "f": f
    }

