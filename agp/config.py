def build_config(args):
    """Return a dict of all threshold/config constants derived from parsed args.

    Raises ``ValueError`` if the supplied values violate basic invariants.
    """
    very_low = args.very_low_threshold
    low = args.low_threshold
    high = args.high_threshold
    very_high = args.very_high_threshold
    tight_low = args.tight_low
    tight_high = args.tight_high
    bin_minutes = args.bin_minutes
    sensor_interval = args.sensor_interval
    min_samples = args.min_samples

    if not (very_low <= low <= high <= very_high):
        raise ValueError(
            f"Thresholds must satisfy very_low <= low <= high <= very_high, "
            f"got {very_low} <= {low} <= {high} <= {very_high}"
        )
    if tight_low >= tight_high:
        raise ValueError(
            f"tight_low must be less than tight_high, got {tight_low} >= {tight_high}"
        )
    if bin_minutes <= 0:
        raise ValueError(f"bin_minutes must be positive, got {bin_minutes}")
    if sensor_interval <= 0:
        raise ValueError(f"sensor_interval must be positive, got {sensor_interval}")
    if min_samples <= 0:
        raise ValueError(f"min_samples must be positive, got {min_samples}")

    return {
        "VERY_LOW": very_low,
        "LOW": low,
        "HIGH": high,
        "VERY_HIGH": very_high,
        "TIGHT_LOW": tight_low,
        "TIGHT_HIGH": tight_high,
        "BIN_MINUTES": bin_minutes,
        "MIN_SAMPLES_PER_BIN": min_samples,
        "SENSOR_INTERVAL": sensor_interval,
        "ROC_CLIP": 10,  # Physiological constant
    }
