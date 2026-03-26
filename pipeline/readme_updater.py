"""
readme_updater.py
-----------------
Updates the HuggingFace dataset README.md (dataset card) after each run.

Patches:
  1. The "Last Updated" date line
  2. The "Dataset Summary" sequence count
  3. The Data Splits table
  4. A new "Update History" section appended at the bottom

Does NOT touch any other part of the README — all existing content,
code examples, and citations are preserved exactly.
"""

import re
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

README_PATH = Path(__file__).parent.parent / "README.md"


def load_readme(path: Path = README_PATH) -> str:
    with open(path) as f:
        return f.read()


def save_readme(content: str, path: Path = README_PATH) -> None:
    with open(path, "w") as f:
        f.write(content)


def patch_last_updated(content: str, new_date: str) -> str:
    """Update the 'Last Updated' date line."""
    pattern = r"(\*\*Last Updated:\*\*\s*)[\w\s,]+"
    replacement = rf"\g<1>{new_date}"
    updated = re.sub(pattern, replacement, content)
    if updated == content:
        logger.warning("Could not find 'Last Updated' line to patch.")
    return updated


def patch_sequence_count(content: str, new_total: int) -> str:
    """Update the sequence count in the Dataset Summary paragraph."""
    pattern = r"containing ([\d,]+) viral sequences"
    replacement = f"containing {new_total:,} viral sequences"
    updated = re.sub(pattern, replacement, content)
    if updated == content:
        logger.warning("Could not find sequence count to patch.")
    return updated


def patch_splits_table(content: str, train_count: int, test_count: int) -> str:
    """Update the train/test counts in the Data Splits table."""
    train_pattern = r"(\|\s*train\s*\|\s*)[\d,]+(\s*\|)"
    test_pattern  = r"(\|\s*test\s*\|\s*)[\d,]+(\s*\|)"
    content = re.sub(train_pattern, rf"\g<1>{train_count:,}\g<2>", content)
    content = re.sub(test_pattern,  rf"\g<1>{test_count:,}\g<2>",  content)
    return content


def build_update_history_table(run_history: list[dict]) -> str:
    """Build a markdown table of the last 12 runs (most recent first)."""
    recent = list(reversed(run_history))[:12]
    lines = [
        "## Update History",
        "",
        "| Date | Version | Sequences Added | Total | Note |",
        "|------|---------|----------------|-------|------|",
    ]
    for run in recent:
        note = run.get("note", "")
        added = run.get("sequences_added", 0)
        # We don't store running totals in history, so leave total blank
        # unless we add it — for now show the delta
        lines.append(
            f"| {run['date']} | v{run['version']} | +{added:,} | — | {note} |"
        )
    return "\n".join(lines) + "\n"


UPDATE_HISTORY_MARKER = "## Update History"


def patch_update_history(content: str, run_history: list[dict]) -> str:
    """Replace the existing Update History section, or append one if absent."""
    new_section = build_update_history_table(run_history)
    if UPDATE_HISTORY_MARKER in content:
        # Replace everything from the marker to the next ## heading (or EOF)
        pattern = r"## Update History.*?(?=\n## |\Z)"
        updated = re.sub(pattern, new_section.rstrip(), content, flags=re.DOTALL)
    else:
        # Append at end
        updated = content.rstrip() + "\n\n" + new_section
    return updated


def update_readme(state: dict, path: Path = README_PATH) -> None:
    """
    Apply all patches to the README in one pass.

    Args:
        state: The current pipeline_state.json dict (post-run)
        path:  Path to the local README_dataset.md to patch
    """
    if not path.exists():
        logger.warning(f"README not found at {path} — skipping update.")
        return

    content = load_readme(path)

    content = patch_last_updated(content, state["last_update_date"])
    content = patch_sequence_count(content, state["total_sequences"])
    content = patch_splits_table(content, state["train_sequences"], state["test_sequences"])
    content = patch_update_history(content, state["run_history"])

    save_readme(content, path)
    logger.info(f"README updated at {path}")
