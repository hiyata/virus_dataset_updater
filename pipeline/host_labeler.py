"""
host_labeler.py
---------------
Implements the three-tier host classification system from the paper (Section 2.3):

  Tier 1: Direct string matching (exact scientific names / common terms)
  Tier 2: Regex-based pattern recognition against a species dictionary
  Tier 3: Gemini AI for contextual inference from metadata

For each sequence, produces:
  - host:              "human" or "non-human"
  - standardized_host: e.g., "Homo sapiens", "Sus scrofa"
  - host_category:     e.g., "Mammal", "Bird", "Insect"
  - gemini_annotated:  bool — True only if Tier 3 was used
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
from Bio.SeqRecord import SeqRecord
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern loader — loads host_patterns.yml once and compiles regexes
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_host_patterns(patterns_file: str) -> dict:
    """
    Load and compile host_patterns.yml.

    Returns a dict with:
      compiled_human:    list of (compiled_regex, standardized_name) for human group
      compiled_nonhuman: list of (compiled_regex, standardized_name, category) for all other groups
      category_mapping:  dict[standardized_name → category_string]
      known_zoonotic:    list of lowercase virus name substrings
    """
    path = Path(patterns_file)
    if not path.exists():
        raise FileNotFoundError(
            f"host_patterns.yml not found at '{patterns_file}'. "
            "Check host_labeling.host_patterns_file in pipeline_config.yaml."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    category_mapping: dict[str, str] = raw.get("category_mapping", {})
    known_zoonotic: list[str] = [v.lower() for v in raw.get("known_zoonotic_viruses", [])]

    compiled_human: list[tuple] = []
    compiled_nonhuman: list[tuple] = []

    for group_name, entries in raw.get("host_patterns", {}).items():
        is_human = (group_name == "human")
        for pattern_str, std_name in entries:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Bad regex in host_patterns.yml [{group_name}] '{pattern_str}': {e}")
                continue

            category = category_mapping.get(std_name, "Unknown")

            if is_human:
                compiled_human.append((compiled, std_name))
            else:
                compiled_nonhuman.append((compiled, std_name, category))

    # Build a scientific-name fallback from the category_mapping keys.
    # GenBank's host qualifier often contains scientific names (e.g. "Sus scrofa")
    # that aren't in the informal-term patterns above. We compile them here so
    # Tier 1 can catch them without needing manual additions to the YAML.
    compiled_scientific: list[tuple] = []
    human_scientific_names = {std for _, std in compiled_human}
    for sci_name, category in category_mapping.items():
        if not sci_name or sci_name in ("Unknown",):
            continue
        try:
            compiled = re.compile(r'\b' + re.escape(sci_name) + r'\b', re.IGNORECASE)
        except re.error:
            continue
        is_human = sci_name in human_scientific_names
        compiled_scientific.append((compiled, sci_name, category, is_human))

    logger.info(
        f"Loaded host patterns: {len(compiled_human)} human regex, "
        f"{len(compiled_nonhuman)} non-human regex, "
        f"{len(compiled_scientific)} scientific name fallbacks, "
        f"{len(known_zoonotic)} zoonotic virus entries."
    )
    return {
        "compiled_human": compiled_human,
        "compiled_nonhuman": compiled_nonhuman,
        "compiled_scientific": compiled_scientific,
        "category_mapping": category_mapping,
        "known_zoonotic": known_zoonotic,
    }


def _extract_host_metadata(record: SeqRecord) -> dict[str, str]:
    """Pull host-relevant fields from a GenBank SeqRecord's features."""
    metadata = {
        "host": "",
        "isolation_source": "",
        "organism": record.annotations.get("organism", ""),
        "description": record.description,
    }
    for feature in record.features:
        if feature.type == "source":
            qualifiers = feature.qualifiers
            metadata["host"] = " ".join(qualifiers.get("host", [""])).strip()
            metadata["isolation_source"] = " ".join(
                qualifiers.get("isolation_source", [""])
            ).strip()
    return metadata

# ---------------------------------------------------------------------------
# Tier 1 — human patterns only, narrowest fields (host + organism)
# ---------------------------------------------------------------------------

def tier1_label(
    metadata: dict[str, str],
    patterns: dict,
) -> Optional[tuple[str, str, str]]:
    """
    Match human regex patterns against the most reliable fields only:
    the explicit 'host' qualifier and the 'organism' annotation.

    Keeping Tier 1 field-narrow means a word like "patient" in a free-text
    description can't accidentally fire here — that's Tier 2's job.

    Returns (host, standardized_host, host_category) or None.
    """
    text = (metadata["host"] + " " + metadata["organism"]).lower()
    text = re.sub(r"[\[\]\(\)\{\}]", " ", text)

    for compiled, std_name in patterns["compiled_human"]:
        if compiled.search(text):
            category = patterns["category_mapping"].get(std_name, "Mammal")
            return "human", std_name, category

    # Also check non-human patterns against these narrow fields, so an
    # explicit host qualifier like "Mus musculus" resolves here rather
    # than being pushed to Tier 2.
    for compiled, std_name, category in patterns["compiled_nonhuman"]:
        if compiled.search(text):
            return "non-human", std_name, category

    # Scientific name fallback: catches GenBank host qualifiers like "Sus scrofa"
    # that aren't covered by informal-term patterns.
    for compiled, std_name, category, is_human in patterns["compiled_scientific"]:
        if compiled.search(text):
            return ("human" if is_human else "non-human"), std_name, category

    return None


# ---------------------------------------------------------------------------
# Tier 2 — all patterns, broader fields (adds isolation_source + description)
# ---------------------------------------------------------------------------

def tier2_label(
    metadata: dict[str, str],
    patterns: dict,
) -> Optional[tuple[str, str, str]]:
    """
    Match all compiled patterns (human first, then non-human) against the
    full combined text: host + isolation_source + description.

    Human patterns are checked before non-human so that a record with
    both "patient" and "pig" in its metadata resolves as human (clinical
    swine-flu sample, for example).

    Returns (host, standardized_host, host_category) or None.
    """
    combined = " ".join([
        metadata["host"],
        metadata["isolation_source"],
        metadata["description"],
    ]).lower()
    combined = re.sub(r"[\[\]\(\)\{\}]", " ", combined)

    # Human patterns first
    for compiled, std_name in patterns["compiled_human"]:
        if compiled.search(combined):
            category = patterns["category_mapping"].get(std_name, "Mammal")
            return "human", std_name, category

    # Non-human patterns
    for compiled, std_name, category in patterns["compiled_nonhuman"]:
        if compiled.search(combined):
            return "non-human", std_name, category

    return None


# ---------------------------------------------------------------------------
# Tier 3 — Gemini
# ---------------------------------------------------------------------------

def tier3_gemini_batch(
    records_metadata: list[tuple[str, dict[str, str]]],
    model_name: str,
    api_key: str,
) -> dict[str, tuple[str, str, str]]:
    """
    Send a batch of unresolved records to Gemini for host inference.
    Returns a dict mapping record.id → (host, standardized_host, host_category).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Format batch prompt
    records_text = "\n\n".join([
        f"Record ID: {rid}\n"
        f"Organism: {meta['organism']}\n"
        f"Host field: {meta['host'] or 'not provided'}\n"
        f"Isolation source: {meta['isolation_source'] or 'not provided'}\n"
        f"Description: {meta['description']}"
        for rid, meta in records_metadata
    ])

    prompt = f"""You are a virology expert. For each viral genome record below, determine the host organism.

For each record, respond with a JSON array where each element has:
- "id": the Record ID
- "host": either "human" or "non-human"
- "standardized_host": the scientific name (e.g., "Homo sapiens", "Sus scrofa", "Aves", "Environmental")
- "host_category": one of: Mammal, Bird, Insect, Fish, Amphibian, Reptile, Environmental, Unknown

Consider: virus type, clinical terminology, isolation source context, known host ranges.
Respond ONLY with valid JSON array, no markdown fences, no other text.

Records:
{records_text}"""

    try:
        response = model.generate_content(prompt)
        results_raw = json.loads(response.text)
        return {
            r["id"]: (r["host"], r["standardized_host"], r["host_category"])
            for r in results_raw
        }
    except Exception as e:
        logger.warning(f"Gemini batch failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main labeling orchestrator
# ---------------------------------------------------------------------------

def label_records(records: list[SeqRecord], config: dict) -> list[dict]:
    """
    Run all three tiers on a list of SeqRecord objects.

    Returns a list of dicts with host labeling fields added, ready to
    be merged into the HuggingFace dataset rows.
    """
    hl_cfg = config["host_labeling"]
    gemini_cfg = hl_cfg["gemini"]

    # Load and compile patterns from host_patterns.yml (cached after first call)
    patterns_file = hl_cfg["host_patterns_file"]
    patterns = _load_host_patterns(patterns_file)

    tier3_queue: list[tuple[str, dict]] = []  # (record.id, metadata)
    results = []

    stats = {"tier1": 0, "tier2": 0, "tier3": 0, "unresolved": 0}

    for rec in records:
        meta = _extract_host_metadata(rec)
        label_result = None

        # Tier 1 — narrow fields, compiled YAML patterns
        label_result = tier1_label(meta, patterns)
        if label_result:
            stats["tier1"] += 1
        else:
            # Tier 2 — broad fields, same compiled patterns
            label_result = tier2_label(meta, patterns)
            if label_result:
                stats["tier2"] += 1
            else:
                # Queue for Tier 3
                tier3_queue.append((rec.id, meta))

        if label_result:
            host, std_host, category = label_result
            results.append({
                "_record": rec,
                "host": host,
                "standardized_host": std_host,
                "host_category": category,
                "gemini_annotated": False,
            })

    # Tier 3 — batch all unresolved records
    if tier3_queue and gemini_cfg["enabled"]:
        api_key = os.environ.get(gemini_cfg["api_key_env"], "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set; Tier 3 skipped.")
        else:
            batch_size = gemini_cfg["batch_size"]
            gemini_results = {}
            for i in range(0, len(tier3_queue), batch_size):
                batch = tier3_queue[i:i + batch_size]
                batch_results = tier3_gemini_batch(batch, gemini_cfg["model"], api_key)
                gemini_results.update(batch_results)
                time.sleep(1)  # Rate limiting

            for rec_id, meta in tier3_queue:
                if rec_id in gemini_results:
                    host, std_host, category = gemini_results[rec_id]
                    stats["tier3"] += 1
                else:
                    host, std_host, category = "non-human", "Unknown", "Unknown"
                    stats["unresolved"] += 1
                    if not gemini_cfg["fallback_on_error"]:
                        raise RuntimeError(f"Tier 3 failed for {rec_id} and fallback is disabled.")

                # Find the original record
                original_rec = next(r for r in records if r.id == rec_id)
                results.append({
                    "_record": original_rec,
                    "host": host,
                    "standardized_host": std_host,
                    "host_category": category,
                    "gemini_annotated": rec_id in gemini_results,
                })
    else:
        # Gemini disabled — label all queued as unresolved
        for rec_id, meta in tier3_queue:
            original_rec = next(r for r in records if r.id == rec_id)
            results.append({
                "_record": original_rec,
                "host": "non-human",
                "standardized_host": "Unknown",
                "host_category": "Unknown",
                "gemini_annotated": False,
            })
            stats["unresolved"] += 1

    logger.info(
        f"Host labeling: {stats['tier1']} Tier-1, {stats['tier2']} Tier-2, "
        f"{stats['tier3']} Tier-3 (Gemini), {stats['unresolved']} unresolved."
    )
    return results
