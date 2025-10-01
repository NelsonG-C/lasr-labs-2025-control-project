# ruff: noqa: F821, F841
from __future__ import annotations

import io
import logging
import token as token_mod
import tokenize
from pathlib import Path

from inspect_ai.log import read_eval_log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def remove_comments(source: str) -> str:
    """Remove Python comments and standalone docstrings from the given source code.

    Rules implemented (derived from tests):
    - Remove all `#` comments (both full-line and inline). For inline comments, also
      strip trailing whitespace that preceded the `#`.
    - Remove shebang and encoding declaration lines (they are comments at the top).
    - Remove module, function, and class docstrings (standalone string statements
      that are the first statement in those blocks).
    - Preserve strings that contain `#` characters.
    - Preserve formatting (spacing/indentation) of code; only strip trailing spaces
      left behind after removing inline comments.
    - Preserve triple-quoted strings assigned to variables (not standalone).
    """

    if source == "":
        return ""

    lines: list[str] = source.splitlines(keepends=True)
    if not lines:
        return ""

    num_lines = len(lines)
    delete_line: list[bool] = [False] * num_lines
    # For inline comments, cut the line at this column (0-based). Keep the
    # smallest column if multiple comments somehow appear.
    cut_at_col: dict[int, int] = {}

    # Track whether a line already contains code tokens before a comment.
    line_has_code: dict[int, bool] = {}

    # Block stack to detect def/class suites for docstring removal.
    # Each frame: {"type": "module"|"def"|"class"|"other", "awaiting_doc": bool}
    block_stack: list[dict[str, object]] = [{"type": "module", "awaiting_doc": True}]
    pending_suite: str | None = None  # pending type to push on next INDENT
    saw_async = False

    # Helper to mark lines for deletion (1-based token rows -> 0-based indices)
    def mark_delete_range(start_row: int, end_row: int) -> None:
        for i in range(max(0, start_row - 1), min(num_lines, end_row)):
            delete_line[i] = True

    # Helper to set that we've seen a code token on a line
    def mark_line_has_code(row: int) -> None:
        if 0 <= row - 1 < num_lines:
            line_has_code[row - 1] = True

    # Iterate tokens to decide what to remove, without building output via untokenize
    tok_iter = tokenize.tokenize(io.BytesIO(source.encode()).readline)

    for tok in tok_iter:
        ttype = tok.type
        tstr = tok.string
        (srow, scol) = tok.start
        (erow, _ecol) = tok.end

        # Skip ENCODING/ENDMARKER processing
        if ttype == token_mod.ENCODING or ttype == token_mod.ENDMARKER:
            continue

        # Track pending suites based on keywords
        if ttype == token_mod.NAME:
            if tstr == "async":
                saw_async = True
            elif tstr == "def":
                pending_suite = "def"
                saw_async = False
            elif tstr == "class":
                pending_suite = "class"
                saw_async = False

        if ttype == token_mod.INDENT:
            # Enter a new block; if it's a def/class, we will await a docstring
            if pending_suite in ("def", "class"):
                block_stack.append({"type": pending_suite, "awaiting_doc": True})
                pending_suite = None
            else:
                block_stack.append({"type": "other", "awaiting_doc": False})
            continue

        if ttype == token_mod.DEDENT:
            # Exit a block (but keep at least the module frame)
            if len(block_stack) > 1:
                block_stack.pop()
            continue

        # Determine top-of-stack context
        top = block_stack[-1]
        top_type = top["type"]  # type: ignore[index]
        awaiting_doc = bool(top["awaiting_doc"])  # type: ignore[index]

        # Handle STRING tokens: possibly docstrings
        if ttype == token_mod.STRING:
            # Detect module docstring (at top level, first statement)
            if top_type == "module" and awaiting_doc and scol == 0:
                # Standalone statement if nothing else on line before it
                # (tokenization ensures no preceding code token if scol == 0)
                mark_delete_range(srow, erow)
                top["awaiting_doc"] = False
                continue

            # Detect function/class docstring: first statement immediately after INDENT
            if (top_type in ("def", "class")) and awaiting_doc:
                # If there's code earlier on the same line before STRING, it's not a docstring
                if not line_has_code.get(srow - 1, False):
                    mark_delete_range(srow, erow)
                    top["awaiting_doc"] = False
                    continue

            # Otherwise, it's a normal string literal
            mark_line_has_code(srow)
            # Seeing any non-ignored token ends the awaiting_doc phase
            if (top_type in ("def", "class")) and awaiting_doc:
                top["awaiting_doc"] = False
            continue

        # Comments
        if ttype == token_mod.COMMENT:
            # Full-line comment if no prior code on this line
            if not line_has_code.get(srow - 1, False):
                if 0 <= srow - 1 < num_lines:
                    delete_line[srow - 1] = True
            else:
                # Inline comment: cut at the comment column
                prev = cut_at_col.get(srow - 1)
                cut_at_col[srow - 1] = scol if prev is None else min(prev, scol)
            continue

        # NL/NEWLINE don't add code
        if ttype in (token_mod.NL, token_mod.NEWLINE):
            continue

        # Any other real token marks code presence on the line
        mark_line_has_code(srow)
        # Seeing any non-ignored token ends the awaiting_doc phase for def/class
        if (top_type in ("def", "class")) and awaiting_doc:
            top["awaiting_doc"] = False

    # Reconstruct lines applying deletions and inline comment cuts
    out_lines: list[str] = []
    for idx, line in enumerate(lines):
        if delete_line[idx]:
            continue
        if idx in cut_at_col:
            cut_col = cut_at_col[idx]
            # Preserve original newline if present
            has_nl = line.endswith("\n")
            content = line[:cut_col]
            content = content.rstrip()
            line = content + ("\n" if has_nl else "")
        out_lines.append(line)

    return "".join(out_lines)


def clean_markdown_from_code(text: str) -> str:
    """Remove markdown code blocks from solution text."""
    return text.replace("```python\n", "").replace("```", "")


def get_accuracies_from_log(log_path: str, scorer_name: str) -> list[float] | None:
    """Extract accuracy from a single eval log file"""
    try:
        if not Path(log_path).exists():
            print(f"File not found: {log_path}")
            return None

        log = read_eval_log(log_path)

        accuracy_vals: list[float] | None = []

        if log and log.results and log.results.scores:
            for score in log.results.scores:
                print("Score")
                if (
                    score.metrics
                    and "accuracy" in score.metrics
                    and score.name == scorer_name
                ):
                    accuracy_vals.append(float(score.metrics["accuracy"].value))

        if not accuracy_vals:
            return None
        else:
            return accuracy_vals

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
