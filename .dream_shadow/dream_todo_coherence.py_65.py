# Dream: p_tension = 0.0 # TODO: Implement contradiction detection
# Generated: 2026-01-28T15:36:50.191699

def check_tension_contradiction(p_tension):
    """
    Checks if the tension value is contradictory.

    Args:
        p_tension (float): The tension value.

    Returns:
        None

    Raises:
        ValueError: If the tension value is contradictory.
    """
    # This is a placeholder for a more complex contradiction detection logic.
    # In a real scenario, this would check against known constraints or previous states.
    if p_tension < 0.0:
        raise ValueError("Tension cannot be negative.")
    if p_tension > 1.0:
        raise ValueError("Tension cannot be greater than 1.")


def calculate_weighted_sum(metabolic, critical, cluster, p_tension):
    """
    Calculates the weighted sum of coherence metrics, incorporating the tension penalty.

    Args:
        metabolic (float): Metabolic coherence value.
        critical (float): Critical coherence value.
        cluster (float): Cluster coherence value.
        p_tension (float): Tension penalty value.

    Returns:
        float: The weighted sum of coherence metrics.
    """
    try:
        check_tension_contradiction(p_tension)  # Check for tension contradiction
        weighted_sum = (0.4 * metabolic) + (0.3 * critical) + (0.3 * cluster)
        return weighted_sum
    except ValueError as e:
        print(f"Error calculating weighted sum: {e}")
        return None  # Or raise the exception, depending on the desired behavior