"""
quality_filter.py
-----------------
Applies the same quality-control filters described in the paper (Section 2.2):

  - Excludes sequences with disqualifying title terms (partial, mutant,
    unverified, bac, clone, gene)
  - Enforces standard IUPAC-only nucleotides (A, C, G, T)
  - Enforces minimum genome length
  - Enforces maximum ambiguous character ratio (default: 0 — strict)

Returns only sequences that pass all checks.
"""

import re
import logging
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


def passes_title_check(record: SeqRecord, exclude_terms: list[str]) -> bool:
    """Returns False if the record description contains any excluded term."""
    description_lower = record.description.lower()
    for term in exclude_terms:
        if term.lower() in description_lower:
            logger.debug(f"Excluded '{record.id}' — title contains '{term}'")
            return False
    return True


def passes_sequence_check(
    record: SeqRecord,
    allowed_nucleotides: str,
    min_length: int,
    max_ambiguous_ratio: float,
) -> bool:
    """Returns False if sequence is too short or contains disallowed characters."""
    seq_str = str(record.seq).upper()

    if len(seq_str) < min_length:
        logger.debug(f"Excluded '{record.id}' — too short ({len(seq_str)} bp < {min_length})")
        return False

    allowed_set = set(allowed_nucleotides.upper())
    invalid_chars = [c for c in seq_str if c not in allowed_set]
    ambiguous_ratio = len(invalid_chars) / len(seq_str)

    if ambiguous_ratio > max_ambiguous_ratio:
        logger.debug(
            f"Excluded '{record.id}' — {ambiguous_ratio:.2%} ambiguous chars "
            f"(limit: {max_ambiguous_ratio:.2%})"
        )
        return False

    return True


def filter_records(records: list[SeqRecord], config: dict) -> tuple[list[SeqRecord], dict]:
    """
    Apply all quality filters to a list of SeqRecord objects.

    Returns:
        passed:  List of records that passed all filters
        stats:   Dict with counts of how many failed each check
    """
    qc = config["quality_filter"]
    exclude_terms = qc["exclude_title_terms"]
    allowed_nts = qc["allowed_nucleotides"]
    min_length = qc["min_length"]
    max_ambig = qc["max_ambiguous_ratio"]

    passed = []
    stats = {
        "total_input": len(records),
        "failed_title": 0,
        "failed_sequence": 0,
        "passed": 0,
    }

    for rec in records:
        if not passes_title_check(rec, exclude_terms):
            stats["failed_title"] += 1
            continue
        if not passes_sequence_check(rec, allowed_nts, min_length, max_ambig):
            stats["failed_sequence"] += 1
            continue
        passed.append(rec)

    stats["passed"] = len(passed)
    logger.info(
        f"QC summary: {stats['total_input']} input → "
        f"{stats['failed_title']} failed title check, "
        f"{stats['failed_sequence']} failed sequence check, "
        f"{stats['passed']} passed."
    )
    return passed, stats
