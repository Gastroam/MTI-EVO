# Dream: priority=0.7 if 'FIXME' in line else 0.5,
# Generated: 2026-01-28T15:38:10.121285

def get_priority(line):
    """
    Determines the priority of a line based on the presence of 'FIXME'.

    Args:
        line (str): The line of code to assess.

    Returns:
        float: The priority score (0.7 if 'FIXME' is present, 0.5 otherwise).
               Returns 0.5 if the input is not a string.
    """
    try:
        if isinstance(line, str) and 'FIXME' in line:
            return 0.7
        elif isinstance(line, str):
            return 0.5
        else:
            return 0.5  # Handle non-string input
    except TypeError:
        return 0.5  # Handle potential TypeError if line is not a string