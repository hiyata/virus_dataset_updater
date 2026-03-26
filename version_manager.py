"""
version_manager.py
------------------
Three-level dataset versioning:  SCHEMA.REVISION.BUILD

  BUILD    X.Y.Z → X.Y.Z+1
           Routine sequence addition. The dataset grew; nothing else changed.
           This is the default for every automated monthly run.
           Example: 1.0.4 → 1.0.5

  REVISION X.Y.Z → X.Y+1.0
           Something meaningful changed in *how* the data was processed,
           but the schema is the same and existing user code still works.
           Use this manually when you:
             - Switch Gemini versions for Tier 3
             - Add a new virus family to target_families
             - Update host_patterns.yml with a significant new group
             - Run a large manual curation batch
           Example: 1.0.5 → 1.1.0

  SCHEMA   X.Y.Z → X+1.0.0
           The dataset structure changed in a breaking way:
           new columns, removed columns, renamed fields, split logic overhaul.
           NEVER triggered by the automated pipeline.
           Always requires running migrate_schema.py first, which calls
           this internally only after the migration succeeds.
           Example: 1.1.3 → 2.0.0

CLI:
  python version_manager.py show
  python version_manager.py bump build
  python version_manager.py bump revision --note "Added bat coronavirus family"
  # For schema bumps: run migrate_schema.py instead — it calls bump schema
  # internally only after a successful migration.
"""

import json
import argparse
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / "state" / "pipeline_state.json"

BUMP_TYPES = ("build", "revision", "schema")


def load_state() -> dict:
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"State saved: version={state['version']}, total={state['total_sequences']}")


def bump_version(current: str, bump_type: str) -> str:
    """
    Increment a three-part version string.

      build    1.2.3 → 1.2.4   (data grew)
      revision 1.2.3 → 1.3.0   (processing changed)
      schema   1.2.3 → 2.0.0   (structure changed)
    """
    if bump_type not in BUMP_TYPES:
        raise ValueError(f"Unknown bump type '{bump_type}'. Must be one of: {', '.join(BUMP_TYPES)}")

    try:
        schema, revision, build = map(int, current.split("."))
    except ValueError:
        raise ValueError(f"Cannot parse version '{current}'. Expected format: X.Y.Z")

    if bump_type == "build":
        build += 1
    elif bump_type == "revision":
        revision += 1
        build = 0
    elif bump_type == "schema":
        schema += 1
        revision = 0
        build = 0

    return f"{schema}.{revision}.{build}"


def record_run(
    sequences_added: int,
    train_added: int,
    test_added: int,
    bump_type: str = "build",
    note: str = "",
    breakdown: dict = None,
) -> dict:
    """
    Update pipeline_state.json after a successful pipeline run.
    Returns the updated state dict.
    """
    if bump_type not in BUMP_TYPES:
        raise ValueError(f"Unknown bump type '{bump_type}'.")

    state = load_state()
    old_version = state["version"]
    new_version = bump_version(old_version, bump_type)
    today = date.today().isoformat()

    state["version"] = new_version
    state["last_update_date"] = today
    state["total_sequences"] += sequences_added
    state["train_sequences"] += train_added
    state["test_sequences"] += test_added
    state["last_run_added"] = sequences_added

    run_record = {
        "date": today,
        "version": new_version,
        "sequences_added": sequences_added,
        "train_added": train_added,
        "test_added": test_added,
        "bump_type": bump_type,
        "total_after": state["total_sequences"],
    }
    if note:
        run_record["note"] = note
    if breakdown:
        run_record["breakdown"] = breakdown

    state["run_history"].append(run_record)
    save_state(state)

    logger.info(f"Version bumped ({bump_type}): {old_version} → {new_version} (+{sequences_added} sequences)")
    return state


def get_current_version() -> str:
    return load_state()["version"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage Virus-Host-Genomes dataset versioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Bump types:
  build      Routine monthly sequence addition — data grew, nothing else changed
  revision   Processing methodology changed — new family, host patterns update, etc.
  schema     Structural/breaking change — DO NOT use directly; run migrate_schema.py

Version format:  SCHEMA.REVISION.BUILD  (e.g. 1.2.5)
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    bump_parser = subparsers.add_parser("bump", help="Manually advance the version")
    bump_parser.add_argument(
        "type",
        choices=["build", "revision"],
        metavar="TYPE",
        help="'build' or 'revision'. For schema bumps, use migrate_schema.py."
    )
    bump_parser.add_argument("--note", default="", help="Note to attach to this history entry")

    subparsers.add_parser("show", help="Print current pipeline state as JSON")

    args = parser.parse_args()

    if args.command == "bump":
        state = load_state()
        old = state["version"]
        new = bump_version(old, args.type)
        state["version"] = new
        state["run_history"].append({
            "date": date.today().isoformat(),
            "version": new,
            "sequences_added": 0,
            "bump_type": args.type,
            "note": args.note,
        })
        save_state(state)
        print(f"Version bumped ({args.type}): {old} → {new}")
        if args.note:
            print(f"Note: {args.note}")

    elif args.command == "show":
        state = load_state()
        print(json.dumps(state, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
