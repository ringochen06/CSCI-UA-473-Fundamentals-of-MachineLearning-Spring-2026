# Define a function named find_max_price that takes a list of prices as input
def find_max_price(prices):
    """
    Find and return the maximum price from a list of prices.

    This function uses Python's built-in max() function to find the maximum
    value efficiently without modifying the input list.

    Args:
        prices (list): A list of numeric values representing prices.
                       The list will NOT be modified by this function.

    Returns:
        The maximum price value from the input list.

    Raises:
        ValueError: If the input list is empty.
    """
    # Use Python's built-in max() function to find the maximum value
    # This does NOT modify the input list and is more efficient (O(n) vs O(n log n))
    # than sorting the entire list just to find the maximum
    return max(prices)
