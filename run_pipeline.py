"""
run_pipeline.py
---------------
Main orchestrator for the monthly Virus-Host-Genomes maintenance pipeline.

Steps:
  1. Load config and current state
  2. Load existing accessions from HuggingFace (for deduplication)
  3. Fetch new sequences from NCBI since last_update_date
  4. Apply quality filters
  5. Run three-tier host labeling
  6. Convert to dataset rows and assign train/test splits
  7. Push new rows to HuggingFace Hub
  8. Update pipeline_state.json (bump patch version)
  9. Update README_dataset.md with new stats
  10. Commit state + README changes back to this GitHub repo

Usage:
  python run_pipeline.py                          # Normal monthly run
  python run_pipeline.py --dry-run                # Fetch + process but don't push
  python run_pipeline.py --bump minor --note "..."  # Force minor version bump
  python run_pipeline.py --since 2026-01-01       # Override the since date
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import date, datetime
from pathlib import Path

import yaml

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.fetch_sequences import run_fetch
from pipeline.quality_filter import filter_records
from pipeline.host_labeler import label_records
from pipeline.dataset_updater import (
    load_existing_accessions,
    labeled_records_to_rows,
    split_new_rows,
    push_new_rows_to_hub,
)
from pipeline.readme_updater import update_readme
from version_manager import record_run, load_state, STATE_FILE

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str, save_log: bool, log_dir: str) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if save_log:
        Path(log_dir).mkdir(exist_ok=True)
        log_file = Path(log_dir) / f"run_{date.today().isoformat()}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args) -> None:
    config = load_config()
    setup_logging(
        config["logging"]["level"],
        config["logging"]["save_run_log"],
        config["logging"]["log_dir"],
    )
    logger = logging.getLogger("run_pipeline")

    logger.info("=" * 60)
    logger.info("Virus-Host-Genomes Pipeline — Monthly Update")
    logger.info(f"Run date: {date.today().isoformat()}")
    logger.info("=" * 60)

    # ---- Step 1: Load state ------------------------------------------------
    state = load_state()
    current_version = state["version"]
    since_date = args.since or state["last_update_date"]
    logger.info(f"Current version: {current_version}")
    logger.info(f"Fetching sequences added since: {since_date}")

    hf_repo = config["dataset"]["huggingface_repo"]
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        logger.error("HF_TOKEN environment variable not set. Aborting.")
        sys.exit(1)

    # ---- Step 2: Load existing accessions ----------------------------------
    try:
        existing_accessions = load_existing_accessions(hf_repo)
    except Exception as e:
        logger.error(f"Failed to load existing accessions: {e}")
        sys.exit(1)

    # ---- Step 3: Fetch new sequences from NCBI -----------------------------
    try:
        raw_records = run_fetch(config, since_date, existing_accessions)
    except Exception as e:
        logger.error(f"NCBI fetch failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    if not raw_records:
        logger.info("No new sequences found. Pipeline complete — no update needed.")
        return

    logger.info(f"Fetched {len(raw_records)} candidate new sequences.")

    # ---- Step 4: Quality filter --------------------------------------------
    passed_records, qc_stats = filter_records(raw_records, config)
    logger.info(f"After QC: {len(passed_records)} sequences passed.")

    if not passed_records:
        logger.info("No sequences passed QC. Pipeline complete — no update.")
        return

    # ---- Step 5: Host labeling ---------------------------------------------
    labeled_results = label_records(passed_records, config)

    # ---- Step 6: Convert to dataset rows + split ---------------------------
    new_rows = labeled_records_to_rows(labeled_results, config)
    new_train, new_test = split_new_rows(new_rows, config["dataset"]["test_ratio"])

    logger.info(
        f"Ready to add: {len(new_rows)} total "
        f"({len(new_train)} train, {len(new_test)} test)"
    )

    # ---- Step 6b: Compute per-run breakdown stats --------------------------
    from collections import Counter
    breakdown = {
        "by_family":        dict(Counter(r["family"] for r in new_rows if r["family"])),
        "by_host":          dict(Counter(r["host"] for r in new_rows)),
        "by_host_category": dict(Counter(r["host_category"] for r in new_rows)),
        "labeling_tiers": {
            "gemini_annotated": sum(1 for r in new_rows if r.get("gemini_annotated")),
            "not_gemini":       sum(1 for r in new_rows if not r.get("gemini_annotated")),
        },
        "qc": {
            "fetched":       qc_stats["total_input"],
            "failed_title":  qc_stats["failed_title"],
            "failed_seq":    qc_stats["failed_sequence"],
            "passed":        qc_stats["passed"],
        },
    }

    # ---- Step 7: Push to HuggingFace ---------------------------------------
    if args.dry_run:
        logger.info("[DRY RUN] Skipping HuggingFace push.")
    else:
        try:
            push_new_rows_to_hub(hf_repo, new_train, new_test, hf_token)
        except Exception as e:
            logger.error(f"HuggingFace push failed: {e}\n{traceback.format_exc()}")
            sys.exit(1)

    # ---- Step 8: Update state + version ------------------------------------
    bump_type = args.bump or config["versioning"]["default_bump"]
    updated_state = record_run(
        sequences_added=len(new_rows),
        train_added=len(new_train),
        test_added=len(new_test),
        bump_type=bump_type,
        note=args.note or "",
        breakdown=breakdown,
    )

    # ---- Step 9: Update local README ---------------------------------------
    readme_path = Path("README_dataset.md")
    update_readme(updated_state, breakdown=breakdown, path=readme_path)

    # ---- Step 10: Push README to HuggingFace Hub ---------------------------
    if not args.dry_run:
        try:
            from pipeline.readme_updater import push_readme_to_hub
            push_readme_to_hub(readme_path, hf_repo, hf_token)
        except Exception as e:
            # Non-fatal — dataset data was already pushed successfully
            logger.warning(f"README push to HuggingFace failed (non-fatal): {e}")

    # ---- Summary -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline run complete.")
    logger.info(f"  New version:       v{updated_state['version']}")
    logger.info(f"  Sequences added:   +{len(new_rows):,}")
    logger.info(f"  Dataset total:     {updated_state['total_sequences']:,}")
    logger.info(f"  Train / Test:      {updated_state['train_sequences']:,} / {updated_state['test_sequences']:,}")
    logger.info(f"  QC rejected:       {qc_stats['failed_title'] + qc_stats['failed_sequence']:,}")
    logger.info(f"  Human / Non-human: {breakdown['by_host'].get('human', 0):,} / {breakdown['by_host'].get('non-human', 0):,}")
    logger.info("=" * 60)

    # Write a machine-readable run summary (useful for GitHub Actions job summary)
    summary = {
        "version": updated_state["version"],
        "date": updated_state["last_update_date"],
        "sequences_added": len(new_rows),
        "total_sequences": updated_state["total_sequences"],
        "qc_rejected": qc_stats["failed_title"] + qc_stats["failed_sequence"],
        "breakdown": breakdown,
        "dry_run": args.dry_run,
    }
    with open("run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("run_summary.json written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the Virus-Host-Genomes monthly maintenance pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and process but do NOT push to HuggingFace or update state."
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Override the since-date for NCBI query (YYYY-MM-DD). "
             "Default: last_update_date from pipeline_state.json"
    )
    parser.add_argument(
        "--bump", choices=["build", "revision"], default=None,
        help="Override the version bump type for this run (default: 'build'). "
             "Use 'revision' when methodology changed this run. "
             "For schema changes, use migrate_schema.py instead."
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="Optional note to attach to this version in the run history."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
