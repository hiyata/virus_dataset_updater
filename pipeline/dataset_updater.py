"""
dataset_updater.py
------------------
Converts labeled SeqRecord objects into HuggingFace dataset rows,
assigns train/test splits, and appends to the existing dataset on
HuggingFace Hub.
  - Never modifies or re-processes existing sequences
  - Deduplicates by accession before any write
  - New sequences get a simple proportional 80/20 train/test assignment
"""

import os
import re
import random
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from Bio.SeqRecord import SeqRecord
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import yaml

logger = logging.getLogger(__name__)

# Schema field order matches the HuggingFace dataset exactly
DATASET_FIELDS = [
    "sequence", "accession", "host", "genus", "isolation_date",
    "strain_name", "location", "virus_name", "isolation_source",
    "lab_culture", "wastewater_sewage", "standardized_host",
    "host_category", "standardized_location", "zoonotic",
    "processing_method", "gemini_annotated", "is_segmented",
    "segment_label", "family",
]

# Families known to have segmented genomes
SEGMENTED_FAMILIES = {
    "Reoviridae", "Orthomyxoviridae", "Bunyaviridae", "Arenaviridae",
    "Nodaviridae", "Birnaviridae",
}


def _extract_genbank_metadata(record: SeqRecord) -> dict:
    """Extract all available metadata fields from a GenBank SeqRecord."""
    meta = {
        "isolation_source": "",
        "strain_name": "",
        "isolation_date": "",
        "location": "",
        "lab_culture": False,
        "wastewater_sewage": False,
    }

    for feature in record.features:
        if feature.type != "source":
            continue
        q = feature.qualifiers
        meta["isolation_source"] = " ".join(q.get("isolation_source", [""])).strip()
        meta["strain_name"] = " ".join(q.get("strain", q.get("isolate", [""]))).strip()
        meta["isolation_date"] = " ".join(q.get("collection_date", [""])).strip()
        meta["location"] = " ".join(q.get("geo_loc_name", q.get("country", [""]))).strip()

        src = meta["isolation_source"].lower()
        meta["lab_culture"] = any(t in src for t in ["cell line", "vero", "culture", "passage"])
        meta["wastewater_sewage"] = any(t in src for t in ["wastewater", "sewage", "effluent"])

    return meta


def _extract_taxonomy(record: SeqRecord) -> tuple[str, str, str]:
    """Returns (family, genus, virus_name) from record taxonomy."""
    taxonomy = record.annotations.get("taxonomy", [])
    organism = record.annotations.get("organism", record.description.split(",")[0])

    family = ""
    genus = ""
    for taxon in reversed(taxonomy):  # Most specific first
        if taxon.endswith("viridae") and not family:
            family = taxon
        if taxon.endswith("virus") and not genus:
            genus = taxon

    return family, genus, organism


def _standardize_location(raw_location: str) -> str:
    """Normalizes 'United Kingdom: Cardiff' → 'United Kingdom'."""
    if ":" in raw_location:
        return raw_location.split(":")[0].strip()
    return raw_location.strip()


@lru_cache(maxsize=1)
def _load_zoonotic_list(patterns_file: str) -> list[str]:
    """Load and cache the known_zoonotic_viruses list from host_patterns.yml."""
    with open(patterns_file) as f:
        raw = yaml.safe_load(f)
    return [v.lower() for v in raw.get("known_zoonotic_viruses", [])]


def _infer_zoonotic(virus_name: str, family: str, genus: str, patterns_file: str) -> bool:
    """
    Check whether any known zoonotic virus name (from host_patterns.yml)
    appears as a substring in the virus_name, family, or genus fields.

    This replaces the previous hardcoded family allowlist.
    """
    known = _load_zoonotic_list(patterns_file)
    searchable = " ".join([virus_name, family, genus]).lower()
    return any(zoonotic in searchable for zoonotic in known)


def labeled_records_to_rows(labeled_results: list[dict], config: dict) -> list[dict]:
    """
    Convert labeled SeqRecord dicts (from host_labeler.py) into
    flat dataset rows matching the HuggingFace schema.
    """
    patterns_file = config["host_labeling"]["host_patterns_file"]
    rows = []
    for item in labeled_results:
        rec: SeqRecord = item["_record"]
        family, genus, virus_name = _extract_taxonomy(rec)
        meta = _extract_genbank_metadata(rec)

        accession = rec.id  # Includes version e.g. "AY446894.2"

        row = {
            "sequence": str(rec.seq).upper(),
            "accession": accession,
            "host": item["host"],
            "genus": genus,
            "isolation_date": meta["isolation_date"],
            "strain_name": meta["strain_name"],
            "location": meta["location"],
            "virus_name": virus_name,
            "isolation_source": meta["isolation_source"],
            "lab_culture": meta["lab_culture"],
            "wastewater_sewage": meta["wastewater_sewage"],
            "standardized_host": item["standardized_host"],
            "host_category": item["host_category"],
            "standardized_location": _standardize_location(meta["location"]),
            "zoonotic": _infer_zoonotic(virus_name, family, genus, patterns_file),
            "processing_method": "Automated pipeline v2",
            "gemini_annotated": item["gemini_annotated"],
            "is_segmented": family in SEGMENTED_FAMILIES,
            "segment_label": "NA",
            "family": family,
        }
        rows.append(row)

    return rows


def split_new_rows(rows: list[dict], test_ratio: float) -> tuple[list[dict], list[dict]]:
    """
    Assign new rows to train/test splits using simple stratified sampling
    by host label (preserves human/non-human class balance across splits).

    Full UMAP+DBSCAN re-splitting is reserved for major changes.
    """
    human_rows = [r for r in rows if r["host"] == "human"]
    nonhuman_rows = [r for r in rows if r["host"] != "human"]

    random.shuffle(human_rows)
    random.shuffle(nonhuman_rows)

    def split(lst):
        n_test = max(1, int(len(lst) * test_ratio)) if lst else 0
        return lst[n_test:], lst[:n_test]

    human_train, human_test = split(human_rows)
    nonhuman_train, nonhuman_test = split(nonhuman_rows)

    train = human_train + nonhuman_train
    test = human_test + nonhuman_test

    logger.info(
        f"Split {len(rows)} new rows → {len(train)} train, {len(test)} test "
        f"({len(human_train)} human train, {len(nonhuman_train)} non-human train)"
    )
    return train, test


def load_existing_accessions(hf_repo: str) -> set[str]:
    """
    Load just the accession column from the existing HF dataset to build
    a deduplication set without loading full sequences into memory.
    """
    logger.info(f"Loading existing accessions from {hf_repo}...")
    dataset = load_dataset(hf_repo, columns=["accession"])
    accessions = set()
    for split_name in dataset:
        for row in dataset[split_name]:
            acc = row["accession"]
            accessions.add(acc)
            accessions.add(acc.split(".")[0])  # Also add version-stripped form
    logger.info(f"Loaded {len(accessions)} existing accessions.")
    return accessions


def push_new_rows_to_hub(
    hf_repo: str,
    new_train_rows: list[dict],
    new_test_rows: list[dict],
    hf_token: str,
) -> None:
    """
    Load the full existing dataset, concatenate new rows, and push back.
    Only runs if there are new rows to add.
    """
    if not new_train_rows and not new_test_rows:
        logger.info("No new rows to push.")
        return

    logger.info(f"Loading full dataset from {hf_repo} for concatenation...")
    existing = load_dataset(hf_repo)

    updates = {}
    for split_name, new_rows in [("train", new_train_rows), ("test", new_test_rows)]:
        if not new_rows:
            updates[split_name] = existing[split_name]
            continue
        new_ds = Dataset.from_list(new_rows)
        combined = concatenate_datasets([existing[split_name], new_ds])
        updates[split_name] = combined
        logger.info(
            f"  {split_name}: {len(existing[split_name])} existing + "
            f"{len(new_rows)} new = {len(combined)} total"
        )

    updated_dataset = DatasetDict(updates)
    logger.info("Pushing updated dataset to HuggingFace Hub...")
    updated_dataset.push_to_hub(hf_repo, token=hf_token)
    logger.info("Push complete.")
