"""
readme_updater.py
-----------------
Updates the HuggingFace dataset README.md after each pipeline run.

Patches applied each run:
  1. "Last Updated" date line in the Dataset Summary
  2. Total sequence count in the Dataset Summary
  3. Train/test row counts in the Data Splits table
  4. "Latest Update" section near the top — rich per-run breakdown
  5. "Update History" table — running log of all builds

Also provides push_readme_to_hub() to push the updated README
directly to the HuggingFace dataset repo.

Nothing outside these sections is touched — all citations, code
examples, and field descriptions are preserved exactly.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

README_PATH = Path(__file__).parent.parent / "README_dataset.md"

LATEST_UPDATE_MARKER = "## Latest Update"
UPDATE_HISTORY_MARKER = "## Update History"


def load_readme(path: Path = README_PATH) -> str:
    with open(path) as f:
        return f.read()


def save_readme(content: str, path: Path = README_PATH) -> None:
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Individual field patchers
# ---------------------------------------------------------------------------

def patch_last_updated(content: str, new_date: str) -> str:
    pattern = r"(\*\*Last Updated:\*\*\s*)[\w\s,\-]+"
    updated = re.sub(pattern, rf"\g<1>{new_date}", content)
    if updated == content:
        logger.warning("Could not find 'Last Updated' line to patch.")
    return updated


def patch_sequence_count(content: str, new_total: int) -> str:
    pattern = r"containing ([\d,]+) viral sequences"
    updated = re.sub(pattern, f"containing {new_total:,} viral sequences", content)
    if updated == content:
        logger.warning("Could not find sequence count to patch.")
    return updated


def patch_splits_table(content: str, train_count: int, test_count: int) -> str:
    content = re.sub(r"(\|\s*train\s*\|\s*)[\d,]+(\s*\|)", rf"\g<1>{train_count:,}\g<2>", content)
    content = re.sub(r"(\|\s*test\s*\|\s*)[\d,]+(\s*\|)",  rf"\g<1>{test_count:,}\g<2>",  content)
    return content


# ---------------------------------------------------------------------------
# Latest Update section — rich per-run breakdown
# ---------------------------------------------------------------------------

def build_latest_update_section(latest_run: dict, total_sequences: int) -> str:
    """
    Compact per-run summary. One header line, one stats line, one family table.
    No stacked tables, no category breakdown walls.
    """
    added     = latest_run.get("sequences_added", 0)
    version   = latest_run.get("version", "—")
    run_date  = latest_run.get("date", "—")
    note      = latest_run.get("note", "")
    breakdown = latest_run.get("breakdown", {})

    by_host  = breakdown.get("by_host", {})
    by_family = breakdown.get("by_family", {})
    qc        = breakdown.get("qc", {})
    tiers     = breakdown.get("labeling_tiers", {})

    human    = by_host.get("human", 0)
    nonhuman = by_host.get("non-human", 0)
    pct      = f"{human / added * 100:.0f}%" if added else "—"

    # ---- Header line --------------------------------------------------------
    header = f"**Latest Update — v{version} · {run_date}**"
    if note:
        header += f" · {note}"

    # ---- Single stats line --------------------------------------------------
    fetched  = qc.get("fetched", 0)
    rejected = qc.get("failed_title", 0) + qc.get("failed_seq", 0)
    gemini   = tiers.get("gemini_annotated", 0)

    stats_parts = [f"**+{added:,} sequences** ({total_sequences:,} total)"]
    if added:
        stats_parts.append(f"{human:,} human ({pct}) · {nonhuman:,} non-human")
    if fetched:
        stats_parts.append(f"{fetched:,} fetched · {rejected:,} QC rejected")
    if gemini:
        stats_parts.append(f"{gemini:,} Gemini-annotated")
    stats_line = " · ".join(stats_parts)

    # ---- Compact two-column family table ------------------------------------
    family_lines = []
    if by_family:
        rows = sorted(by_family.items(), key=lambda x: -x[1])
        # Pair up into two columns
        pairs = []
        for i in range(0, len(rows), 2):
            left = rows[i]
            right = rows[i + 1] if i + 1 < len(rows) else ("", "")
            pairs.append((left, right))

        family_lines = [
            "| Family | Added | Family | Added |",
            "|--------|------:|--------|------:|",
        ]
        for (lf, lc), (rf, rc) in pairs:
            right_str = f"{rf} | {rc:,}" if rf else " | "
            family_lines.append(f"| {lf} | {lc:,} | {right_str} |")

    lines = [LATEST_UPDATE_MARKER, ""]
    lines.append(header)
    lines.append("")
    lines.append(stats_line)
    if family_lines:
        lines.append("")
        lines += family_lines
    lines.append("")
    return "\n".join(lines)


def patch_latest_update(content: str, latest_run: dict, total_sequences: int) -> str:
    """
    Replace or insert the Latest Update section.
    Placement priority:
      1. Replace existing ## Latest Update section if present
      2. Insert after the closing ``` of the citation block
      3. Insert before ## Dataset Summary as fallback
    """
    new_section = build_latest_update_section(latest_run, total_sequences)

    if LATEST_UPDATE_MARKER in content:
        pattern = r"## Latest Update.*?(?=\n## )"
        return re.sub(pattern, new_section.rstrip(), content, flags=re.DOTALL)

    # Insert after citation code block (ends with ```)
    # The citation block is the last ``` before ## Dataset Summary
    cite_end = r"(```\n)(\n*## Dataset Summary)"
    if re.search(cite_end, content):
        return re.sub(cite_end, rf"\1\n{new_section}\n\2", content, count=1)

    # Fallback: before Dataset Summary
    if "## Dataset Summary" in content:
        return content.replace("## Dataset Summary",
                               new_section + "\n---\n\n## Dataset Summary", 1)

    return content.rstrip() + "\n\n" + new_section


# ---------------------------------------------------------------------------
# Update History table
# ---------------------------------------------------------------------------

def build_update_history_section(run_history: list[dict]) -> str:
    """
    Build a compact history table showing all builds, most recent first.
    Includes running total and per-run human/non-human split where available.
    """
    recent = list(reversed(run_history))[:20]

    lines = [
        UPDATE_HISTORY_MARKER,
        "",
        "| Date | Version | Added | Total | Human | Non-human | Note |",
        "|------|---------|-------|-------|-------|-----------|------|",
    ]

    for run in recent:
        added   = run.get("sequences_added", 0)
        total   = run.get("total_after", "—")
        note    = run.get("note", "")
        version = run.get("version", "—")

        breakdown = run.get("breakdown", {})
        by_host   = breakdown.get("by_host", {})
        human     = by_host.get("human", "—") if breakdown else "—"
        nonhuman  = by_host.get("non-human", "—") if breakdown else "—"

        total_str   = f"{total:,}" if isinstance(total, int) else str(total)
        human_str   = f"{human:,}" if isinstance(human, int) else str(human)
        nonhuman_str = f"{nonhuman:,}" if isinstance(nonhuman, int) else str(nonhuman)

        lines.append(
            f"| {run['date']} | v{version} | +{added:,} | {total_str} "
            f"| {human_str} | {nonhuman_str} | {note} |"
        )

    lines.append("")
    return "\n".join(lines)


def patch_update_history(content: str, run_history: list[dict]) -> str:
    new_section = build_update_history_section(run_history)
    if UPDATE_HISTORY_MARKER in content:
        pattern = r"## Update History.*?(?=\n## |\Z)"
        updated = re.sub(pattern, new_section.rstrip(), content, flags=re.DOTALL)
    else:
        updated = content.rstrip() + "\n\n" + new_section
    return updated


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------

def push_readme_to_hub(readme_path: Path, hf_repo: str, hf_token: str) -> None:
    """
    Push the updated README_dataset.md to the HuggingFace dataset repo
    as README.md, replacing the dataset card.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="dataset",
        commit_message="chore: update dataset card [skip ci]",
    )
    logger.info(f"README pushed to HuggingFace: {hf_repo}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def update_readme(state: dict, breakdown: dict = None, path: Path = README_PATH) -> None:
    """
    Apply all patches to the README in one pass.

    Args:
        state:     The current pipeline_state.json dict (post-run)
        breakdown: Per-run stats dict from run_pipeline.py (optional)
        path:      Path to the local README_dataset.md to patch
    """
    if not path.exists():
        logger.warning(f"README not found at {path} — skipping update.")
        return

    content = load_readme(path)

    # Get the most recent run from history (which includes breakdown)
    latest_run = state["run_history"][-1] if state["run_history"] else {}

    # Merge in breakdown if passed directly (in case it wasn't stored yet)
    if breakdown and "breakdown" not in latest_run:
        latest_run = {**latest_run, "breakdown": breakdown}

    content = patch_last_updated(content, state["last_update_date"])
    content = patch_sequence_count(content, state["total_sequences"])
    content = patch_splits_table(content, state["train_sequences"], state["test_sequences"])
    content = patch_latest_update(content, latest_run, state["total_sequences"])
    content = patch_update_history(content, state["run_history"])

    save_readme(content, path)
    logger.info(f"README updated at {path}")

