import config

def calculate_objective(decoder_results, speeds=None):
    """
    Computes weighted objective:
      f = α * fem + β * fall - γ * f_speed   <-- speed is a REWARD now
    """
    alpha = config.alpha
    beta = config.beta
    gamma = getattr(config, "gamma", 0.0)

    # Extract delays
    delays = {r["id"]: r["delay"] for r in decoder_results}
    fem = sum(r["delay"] for r in decoder_results if r["is_emergency"])
    fall = sum(delays.values())

    # Speed reward term (large sum of speeds = large reward)
    if speeds is not None:
        f_speed = sum(speeds)
    else:
        f_speed = 0.0

    # Notice the MINUS sign
    f = alpha * fem + beta * fall - gamma * f_speed

    return {
        "delays": delays,
        "fem": fem,
        "fall": fall,
        "f_speed": f_speed,
        "f": f
    }
