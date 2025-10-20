def convert_timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe string to its equivalent in seconds.

    Args:
        timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', '1d').

    Returns:
        int: Equivalent timeframe in seconds.

    Raises:
        ValueError: If the timeframe format is invalid.
    """
    unit_multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
        'M': 2592000,  # Approximate month as 30 days
    }

    try:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit not in unit_multipliers:
            raise ValueError(f"Invalid timeframe unit: {unit}")
        return value * unit_multipliers[unit]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid timeframe format: {timeframe}") from e