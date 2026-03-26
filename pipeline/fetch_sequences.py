"""
fetch_sequences.py
------------------
Queries NCBI nuccore for complete viral genome sequences added or updated
since the last pipeline run. Only fetches sequences belonging to the 15
target virus families defined in pipeline_config.yaml.

Returns a list of parsed SeqRecord objects ready for quality filtering.
"""

import time
import logging
from datetime import datetime, date
from typing import Optional
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


def build_ncbi_query(families: list[str], since_date: str, until_date: str) -> str:
    """
    Build an NCBI Entrez query string.

    Fetches complete viral genomes from the target families submitted
    between since_date and until_date (both YYYY/MM/DD format).
    """
    family_terms = " OR ".join([f'"{f}"[organism]' for f in families])
    date_filter = f'"{since_date}"[PDAT]:"{until_date}"[PDAT]'
    completeness = '"complete genome"[title]'

    return f"({family_terms}) AND {date_filter} AND {completeness}"


def fetch_accession_ids(
    query: str,
    email: str,
    api_key: Optional[str],
    max_records: int,
    delay: float,
) -> list[str]:
    """
    Run an esearch to get accession IDs matching the query.
    Returns up to max_records accession IDs.
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    logger.info("Running NCBI esearch...")
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records, usehistory="y")
    record = Entrez.read(handle)
    handle.close()

    count = int(record["Count"])
    id_list = record["IdList"]
    logger.info(f"NCBI returned {count} total hits; retrieved {len(id_list)} IDs.")
    time.sleep(delay)
    return id_list


def fetch_sequences_by_ids(
    id_list: list[str],
    existing_accessions: set[str],
    batch_size: int,
    delay: float,
) -> list[SeqRecord]:
    """
    Fetch full GenBank records in batches for the given NCBI IDs.
    Skips any accession already present in existing_accessions.

    Returns a list of BioPython SeqRecord objects.
    """
    new_records = []
    total = len(id_list)

    for start in range(0, total, batch_size):
        batch_ids = id_list[start : start + batch_size]
        batch_num = start // batch_size + 1
        logger.info(f"Fetching batch {batch_num} ({start + 1}–{min(start + batch_size, total)} of {total})...")

        try:
            handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype="gb",
                retmode="text",
            )
            records = list(SeqIO.parse(handle, "genbank"))
            handle.close()
        except Exception as e:
            logger.warning(f"Batch {batch_num} fetch failed: {e}. Skipping.")
            time.sleep(delay * 3)
            continue

        for rec in records:
            # Normalize accession — strip version suffix for dedup check
            base_accession = rec.id.split(".")[0]
            if base_accession in existing_accessions or rec.id in existing_accessions:
                logger.debug(f"Skipping duplicate: {rec.id}")
                continue
            new_records.append(rec)

        logger.info(f"  → {len(records)} fetched, {len(new_records)} new so far.")
        time.sleep(delay)

    logger.info(f"Fetch complete: {len(new_records)} genuinely new sequences.")
    return new_records


def run_fetch(config: dict, last_update_date: str, existing_accessions: set[str]) -> list[SeqRecord]:
    """
    Main entry point for the fetch step.

    Args:
        config:              Loaded pipeline_config.yaml (as dict)
        last_update_date:    ISO date string "YYYY-MM-DD" of last successful run
        existing_accessions: Set of accession IDs already in the HF dataset

    Returns:
        List of new SeqRecord objects not yet in the dataset
    """
    ncbi_cfg = config["ncbi"]
    today_str = date.today().strftime("%Y/%m/%d")

    # NCBI wants YYYY/MM/DD for PDAT filters
    since_str = last_update_date.replace("-", "/")

    query = build_ncbi_query(
        families=ncbi_cfg["target_families"],
        since_date=since_str,
        until_date=today_str,
    )
    logger.info(f"NCBI query: {query}")

    import os
    api_key = os.environ.get(ncbi_cfg["api_key_env"])

    id_list = fetch_accession_ids(
        query=query,
        email=ncbi_cfg["email"],
        api_key=api_key,
        max_records=ncbi_cfg["max_records_per_run"],
        delay=ncbi_cfg["request_delay"],
    )

    if not id_list:
        logger.info("No new sequences found in NCBI for this period.")
        return []

    new_records = fetch_sequences_by_ids(
        id_list=id_list,
        existing_accessions=existing_accessions,
        batch_size=ncbi_cfg["batch_size"],
        delay=ncbi_cfg["request_delay"],
    )

    return new_records
