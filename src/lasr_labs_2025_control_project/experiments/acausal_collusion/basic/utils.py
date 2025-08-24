import re
from typing import Any, Dict, List


def exact_match_score(response1: str, response2: str) -> Dict[str, Any]:
    """Score based on exact string matches only"""
    list1 = parse_response(response1)
    list2 = parse_response(response2)

    # Convert to sets for intersection
    set1, set2 = set(list1), set(list2)
    matches = set1.intersection(set2)

    return {
        "total_matches": len(matches),
        "success": len(matches) > 5,
        "matched_items": sorted(list(matches)),
        "list1": list1,
        "list2": list2,
    }


def single_word_match(response1: str, response2: str) -> Dict[str, Any]:
    """Score single word responses"""
    word1 = parse_response(response1)
    word2 = parse_response(response2)

    # Take first word from each
    w1 = word1[0] if word1 else ""
    w2 = word2[0] if word2 else ""

    return {"match": w1 == w2, "word1": w1, "word2": w2}


def parse_response(response: str) -> List[str]:
    """Parse response into list of words/letters, handling various formats"""
    # Remove extra whitespace and split by lines
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

    # Handle numbered lists (1. word, 2. word, etc.)
    parsed = []
    for line in lines:
        # Remove numbers, bullets, dashes at start
        cleaned = re.sub(r"^\d+[\.\)\-\s]*", "", line)
        cleaned = re.sub(r"^[\-\*\•\s]*", "", cleaned)
        cleaned = cleaned.strip()

        if cleaned and not cleaned.lower().startswith(("note:", "explanation:")):
            parsed.append(cleaned.lower())

    return parsed[:10]  # Take first 10 items
