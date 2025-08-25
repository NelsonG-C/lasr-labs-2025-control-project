from typing import List


def concatenate_list(dict_list: List[dict[str, str]]) -> str:
    """
    Takes a list of strings and returns a formatted string.
    The string contains the items from the list in numbered format.

    Args:
        string_list (list): List of strings

    Returns:
        str: Formatted string with numbered items separated by newlines
    """
    policy_strategies = []
    for item in dict_list:
        if "policy_strategy" in item:
            policy_strategies.append(item["policy_strategy"])

    # Create numbered format for each policy strategy
    formatted_items = []
    for i, strategy in enumerate(policy_strategies, 1):
        formatted_items.append(f"{i}. {strategy}")

    # Join all items with newlines
    concatenated_string = "\n".join(formatted_items)

    return concatenated_string
