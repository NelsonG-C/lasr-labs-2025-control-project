def clean_markdown_from_code(text: str) -> str:
    """Remove markdown code blocks from solution text."""
    return text.replace("```python\n", "").replace("```", "")
