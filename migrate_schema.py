"""
migrate_schema.py
-----------------
Guided migration tool for breaking schema changes (SCHEMA version bump).

Run this script — not run_pipeline.py — whenever you need to:
  - Add a new column to the dataset (e.g. genome annotations)
  - Remove or rename an existing column
  - Change a field's type
  - Restructure the train/test split logic

What this script does that run_pipeline.py cannot:
  1. Loads the ENTIRE existing HuggingFace dataset (all ~58k+ rows)
  2. Applies your transformation function to every row in both splits
  3. Validates the result before touching anything on HuggingFace
  4. Saves a local checkpoint so a failed push doesn't leave you stranded
  5. Pushes the full transformed dataset atomically
  6. Updates the HuggingFace dataset config.json (schema definition)
  7. Bumps the SCHEMA version number and records the migration in state

SCHEMA CHANGE
-----------------------------------------
Step 1 — Write your transform function (see TRANSFORM FUNCTION section below).
Step 2 — Update NEW_FIELDS to declare what columns are being added/removed.
Step 3 — Dry-run to validate locally:
            python migrate_schema.py --dry-run
Step 4 — Run the real migration:
            python migrate_schema.py --note "Added genome_annotation field"
Step 5 — Verify on HuggingFace that the new schema looks correct.
Step 6 — Resume normal monthly pipeline runs.

EXAMPLE: Adding genome annotations
-----------------------------------
If you later want to add a 'genome_annotation' column populated from NCBI:

  NEW_FIELDS = {
      "genome_annotation": {
          "dtype": "string",
          "_type": "Value",
          "description": "Genome annotation from NCBI RefSeq"
      }
  }
  REMOVED_FIELDS = []   # not removing anything

  def transform_row(row: dict, split_name: str) -> dict:
      accession = row["accession"]
      # Fetch annotation for this accession from your annotation source.
      # For backfilling existing rows, use "" or None as placeholder
      # if the annotation isn't available yet.
      row["genome_annotation"] = fetch_annotation(accession) or ""
      return row

For large annotation fetches, set BATCH_SIZE to control how many rows
are processed at once and consider adding rate limiting inside transform_row.
"""

import os
import sys
import json
import logging
import argparse
import traceback
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Callable

import yaml
from datasets import load_dataset, Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).parent))
from version_manager import load_state, save_state, bump_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("migrate_schema")

# ---------------------------------------------------------------------------
# !! CONFIGURE THESE BEFORE RUNNING !!
# ---------------------------------------------------------------------------

# Fields being ADDED in this migration.
# Each entry becomes a new key in config.json's "features" section.
# Use the same format as the existing config.json features.
NEW_FIELDS: dict[str, dict] = {
    # "genome_annotation": {
    #     "dtype": "string",
    #     "_type": "Value",
    #     "description": "Genome annotation from NCBI RefSeq"
    # },
}

# Fields being REMOVED in this migration (list of field name strings).
REMOVED_FIELDS: list[str] = [
    # Example: "old_field_name"
]

# Fields being RENAMED in this migration ({old_name: new_name}).
RENAMED_FIELDS: dict[str, str] = {
    # Example: "old_name": "new_name"
}


def transform_row(row: dict, split_name: str) -> dict:
    """
    This function is called once per row across the entire dataset.
    It receives the existing row dict and must return a modified dict.

    Rules:
      - Add new fields here (with a default value for existing rows)
      - Remove fields here if needed (del row["field"])
      - Rename fields here (row["new"] = row.pop("old"))
      - Return the modified row

    The split_name argument ('train' or 'test') lets you apply
    different logic per split if needed.

    Example for adding genome_annotation:

        row["genome_annotation"] = ""  # Placeholder for existing rows
        return row
    """

    return row


# ---------------------------------------------------------------------------
# Migration pipeline
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("migration_checkpoint")
CONFIG_PATH = Path("config/pipeline_config.yaml")
HF_CONFIG_PATH = Path("config/config__5_.json")   

def load_pipeline_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def validate_transform(sample_row: dict) -> dict:
    """Run the transform on one row and check for obvious errors."""
    try:
        result = transform_row(deepcopy(sample_row), "train")
    except Exception as e:
        raise RuntimeError(f"transform_row raised an exception on sample row: {e}")

    if not isinstance(result, dict):
        raise TypeError(f"transform_row must return a dict, got {type(result)}")

    for field in REMOVED_FIELDS:
        if field in result:
            raise ValueError(
                f"transform_row did not remove '{field}' — add `del row['{field}']` to your transform."
            )

    for old, new in RENAMED_FIELDS.items():
        if old in result:
            raise ValueError(
                f"transform_row did not rename '{old}' → '{new}'. "
                f"Add `row['{new}'] = row.pop('{old}')` to your transform."
            )
        if new not in result:
            raise ValueError(f"Renamed field '{new}' not found in transformed row.")

    for field in NEW_FIELDS:
        if field not in result:
            raise ValueError(
                f"New field '{field}' not found in transformed row. "
                f"Add `row['{field}'] = <default>` to your transform."
            )

    logger.info("transform_row validation passed on sample row.")
    return result


def apply_transform_to_split(dataset: Dataset, split_name: str) -> Dataset:
    """Apply transform_row to every row in a split, returning a new Dataset."""
    logger.info(f"Transforming {split_name} split ({len(dataset):,} rows)...")
    transformed_rows = []
    for i, row in enumerate(dataset):
        try:
            transformed_rows.append(transform_row(deepcopy(row), split_name))
        except Exception as e:
            logger.error(f"transform_row failed on row {i} (accession={row.get('accession', '?')}): {e}")
            raise
        if (i + 1) % 5000 == 0:
            logger.info(f"  ... {i + 1:,} / {len(dataset):,} rows transformed")

    return Dataset.from_list(transformed_rows)


def save_checkpoint(dataset_dict: DatasetDict) -> None:
    """Save the fully transformed dataset locally before pushing."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "transformed_dataset"
    logger.info(f"Saving checkpoint to {checkpoint_path} ...")
    dataset_dict.save_to_disk(str(checkpoint_path))
    logger.info("Checkpoint saved.")


def load_checkpoint() -> DatasetDict:
    """Load a previously saved checkpoint (for retry after push failure)."""
    checkpoint_path = CHECKPOINT_DIR / "transformed_dataset"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run the migration first."
        )
    from datasets import load_from_disk
    logger.info(f"Loading checkpoint from {checkpoint_path} ...")
    return load_from_disk(str(checkpoint_path))


def update_hf_config(new_version: str) -> None:
    """
    Update the local config.json to reflect schema changes.
    This file should be committed alongside the dataset push.
    """
    if not HF_CONFIG_PATH.exists():
        logger.warning(f"HF config not found at {HF_CONFIG_PATH}. Skipping config update.")
        return

    with open(HF_CONFIG_PATH) as f:
        hf_config = json.load(f)

    features = hf_config.get("features", {})

    for field, field_def in NEW_FIELDS.items():
        features[field] = field_def
        logger.info(f"  config.json: added field '{field}'")

    for field in REMOVED_FIELDS:
        if field in features:
            del features[field]
            logger.info(f"  config.json: removed field '{field}'")

    for old, new in RENAMED_FIELDS.items():
        if old in features:
            features[new] = features.pop(old)
            logger.info(f"  config.json: renamed '{old}' → '{new}'")

    hf_config["features"] = features
    hf_config["version"] = new_version

    with open(HF_CONFIG_PATH, "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info(f"config.json updated to version {new_version}.")


def record_schema_migration(note: str) -> dict:
    """Bump the SCHEMA version in pipeline_state.json and record the migration."""
    state = load_state()
    old_version = state["version"]
    new_version = bump_version(old_version, "schema")
    today = date.today().isoformat()

    state["version"] = new_version
    state["last_update_date"] = today
    state["last_run_added"] = 0

    migration_record = {
        "date": today,
        "version": new_version,
        "sequences_added": 0,
        "bump_type": "schema",
        "note": note or "Schema migration",
        "new_fields": list(NEW_FIELDS.keys()),
        "removed_fields": REMOVED_FIELDS,
        "renamed_fields": RENAMED_FIELDS,
    }
    state["run_history"].append(migration_record)
    save_state(state)
    logger.info(f"Schema version bumped: {old_version} → {new_version}")
    return state


def run_migration(args) -> None:
    config = load_pipeline_config()
    hf_repo = config["dataset"]["huggingface_repo"]
    hf_token = os.environ.get("HF_TOKEN", "")

    if not hf_token and not args.dry_run:
        logger.error("HF_TOKEN environment variable not set. Aborting.")
        sys.exit(1)

    if not NEW_FIELDS and not REMOVED_FIELDS and not RENAMED_FIELDS:
        logger.warning(
            "No NEW_FIELDS, REMOVED_FIELDS, or RENAMED_FIELDS defined. "
            "Edit this script to configure your migration before running."
        )
        if not args.force:
            logger.error("Aborting. Use --force to run anyway (e.g. to test transform_row).")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Schema Migration")
    logger.info(f"  New fields:     {list(NEW_FIELDS.keys()) or 'none'}")
    logger.info(f"  Removed fields: {REMOVED_FIELDS or 'none'}")
    logger.info(f"  Renamed fields: {RENAMED_FIELDS or 'none'}")
    logger.info(f"  Dry run:        {args.dry_run}")
    logger.info("=" * 60)

    # ---- Step 1: Load from checkpoint or HuggingFace -----------------------
    if args.from_checkpoint:
        logger.info("Loading from local checkpoint (skipping HuggingFace download)...")
        transformed = load_checkpoint()
    else:
        logger.info(f"Loading full dataset from {hf_repo}...")
        existing = load_dataset(hf_repo)
        logger.info(
            f"Loaded: {len(existing['train']):,} train, {len(existing['test']):,} test rows."
        )

        # ---- Step 2: Validate transform on one sample ----------------------
        sample_row = dict(existing["train"][0])
        validate_transform(sample_row)

        # ---- Step 3: Apply transform to all rows ---------------------------
        transformed_splits = {}
        for split_name in existing:
            transformed_splits[split_name] = apply_transform_to_split(existing[split_name], split_name)
        transformed = DatasetDict(transformed_splits)

        # ---- Step 4: Save checkpoint before pushing ------------------------
        save_checkpoint(transformed)

    # ---- Step 5: Push to HuggingFace ---------------------------------------
    if args.dry_run:
        logger.info("[DRY RUN] Skipping HuggingFace push. Checkpoint saved locally.")
        logger.info(f"  Transformed train: {len(transformed['train']):,} rows")
        logger.info(f"  Transformed test:  {len(transformed['test']):,} rows")
        logger.info(f"  Columns: {transformed['train'].column_names}")
        logger.info("Run without --dry-run to push when ready.")
        return

    logger.info("Pushing transformed dataset to HuggingFace Hub...")
    try:
        transformed.push_to_hub(hf_repo, token=hf_token)
        logger.info("Push complete.")
    except Exception as e:
        logger.error(
            f"Push failed: {e}\n{traceback.format_exc()}\n\n"
            "The local checkpoint is saved. Fix the issue and re-run with --from-checkpoint."
        )
        sys.exit(1)

    # ---- Step 6: Update local config.json ----------------------------------
    state = load_state()
    new_version = bump_version(state["version"], "schema")
    update_hf_config(new_version)

    # ---- Step 7: Record schema bump in state -------------------------------
    updated_state = record_schema_migration(args.note or "")

    logger.info("=" * 60)
    logger.info("Schema migration complete.")
    logger.info(f"  New version:  {updated_state['version']}")
    logger.info(f"  Commit state/pipeline_state.json and config/config.json to git.")
    logger.info(f"  The local checkpoint at {CHECKPOINT_DIR} can be deleted.")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Migrate the Virus-Host-Genomes dataset schema (SCHEMA version bump).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Before running:
  1. Edit NEW_FIELDS, REMOVED_FIELDS, RENAMED_FIELDS at the top of this script
  2. Implement transform_row() with your actual row transformation logic
  3. Test with --dry-run first

Examples:
  python migrate_schema.py --dry-run
  python migrate_schema.py --note "Added genome_annotation field from NCBI RefSeq"
  python migrate_schema.py --from-checkpoint   # retry push after a failure
        """
    )
    parser.add_argument("--dry-run", action="store_true",
        help="Transform all rows locally and save checkpoint, but do NOT push to HuggingFace.")
    parser.add_argument("--from-checkpoint", action="store_true",
        help="Skip download and transform — load the saved checkpoint and push directly. "
             "Use this to retry a failed push without re-processing all rows.")
    parser.add_argument("--note", type=str, default="",
        help="Description of what changed in this migration (recorded in run history).")
    parser.add_argument("--force", action="store_true",
        help="Run even if no schema fields are declared (useful for transform testing).")

    args = parser.parse_args()
    run_migration(args)


if __name__ == "__main__":
    main()
