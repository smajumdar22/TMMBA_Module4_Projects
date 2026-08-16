#!/usr/bin/env python3
"""
Vibe Matcher — a CLI song recommender that only ever returns real songs.

Architecture (see README for the full rationale):
  - BORING 70%: parsing, dataset loading, confidence heuristics, validation,
    deduping, and output formatting. Plain Python, no AI, fully deterministic.
  - AI 30%: exactly one Claude API call that judges which songs in the local
    dataset fit the request. It never invents songs — the validator that
    follows treats its output as untrusted and checks every claim.

Usage:
    python vibe_matcher.py "moody indie rock like Phoebe Bridgers, no sad songs"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
SONGS_PATH = SCRIPT_DIR / "songs.json"
MODEL = os.environ.get("VIBE_MATCHER_MODEL", "claude-haiku-4-5-20251001")

MIN_RESULTS_TO_SHOW = 5
TARGET_RESULTS = 10
CANDIDATES_REQUESTED = 15  # ask the AI for more than we need as a buffer for validation drops

EXPLICIT_CLEAN_PATTERNS = [
    r"\bclean\b", r"\bno explicit\b", r"\bnothing explicit\b",
    r"\bkid.?friendly\b", r"\bfamily.?friendly\b",
    r"\bwork.?safe\b", r"\bworkplace.?safe\b",
    r"\bno swearing\b", r"\bno curs\w*\b",
]

OPPOSITE_MOOD_PAIRS = [
    ("sad", "upbeat"), ("sad", "energetic"), ("sad", "fun"),
    ("calm", "aggressive"), ("calm", "energetic"), ("calm", "intense"),
    ("peaceful", "angry"), ("peaceful", "aggressive"),
    ("melancholy", "upbeat"), ("chill", "aggressive"), ("chill", "intense"),
]

GENERIC_SIGNAL_WORDS = {
    "run", "running", "workout", "gym", "study", "studying", "party", "wedding",
    "breakup", "drive", "driving", "focus", "sleep", "rain", "summer", "winter",
    "roadtrip", "relax", "cry", "crying", "dance", "dancing", "morning", "night",
}


class MissingAPIKeyError(Exception):
    """Raised when ANTHROPIC_API_KEY isn't set. Caught separately by each
    front end (CLI, web UI) so each can present it in its own way."""


# =====================================================================
# BORING 70%: small shared utility
# =====================================================================

def _norm(s):
    """Normalize a string for exact matching: trim, lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


# =====================================================================
# BORING 70%: dataset loading
# =====================================================================

def load_dataset(path=SONGS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(dataset):
    moods = {m for s in dataset for m in s["moods"]}
    genres = {g for s in dataset for g in s["genres"]}
    return moods, genres


# =====================================================================
# BORING 70%: input parsing — exclusions are extracted with regex over the
# dataset's own vocabulary, never guessed, so the validator can always
# check them against ground truth later.
# =====================================================================

def parse_exclusions(query, moods_vocab, genres_vocab):
    text = query.lower()
    excluded_moods, excluded_genres = set(), set()

    exclude_explicit = any(re.search(p, text) for p in EXPLICIT_CLEAN_PATTERNS)

    for vocab, bucket in ((moods_vocab, excluded_moods), (genres_vocab, excluded_genres)):
        for term in vocab:
            pattern = rf"\b(?:no|not|nothing|without|avoid|skip|exclude)\s+{re.escape(term)}\b"
            if re.search(pattern, text):
                bucket.add(term)

    return {
        "excluded_moods": excluded_moods,
        "excluded_genres": excluded_genres,
        "exclude_explicit": exclude_explicit,
    }


def strip_exclusion_language(query):
    """Drop exclusion clauses so the AI's prompt focuses on the positive ask."""
    parts = re.split(r",|;|\band\b", query, flags=re.IGNORECASE)
    exclusion_markers = re.compile(
        r"\b(no|not|nothing|without|avoid|skip|exclude|clean|explicit)\b", re.IGNORECASE
    )
    kept = [p.strip() for p in parts if p.strip() and not exclusion_markers.search(p)]
    cleaned = ", ".join(kept).strip()
    return cleaned if cleaned else query


# =====================================================================
# BORING 70%: confidence heuristics — flag vague or self-contradictory
# requests instead of forcing a confident-looking answer out of them.
# =====================================================================

def detect_contradiction(query, moods_vocab):
    text = query.lower()
    words = set(re.findall(r"[a-z']+", text))
    for a, b in OPPOSITE_MOOD_PAIRS:
        if a not in moods_vocab or b not in moods_vocab:
            continue
        if a in words and b in words:
            negated = re.search(rf"\b(no|not|nothing|without|avoid|skip|exclude)\s+{a}\b", text) or \
                      re.search(rf"\b(no|not|nothing|without|avoid|skip|exclude)\s+{b}\b", text)
            if not negated:
                return (a, b)
    return None


def is_vague(cleaned_query, moods_vocab, genres_vocab, dataset):
    text = cleaned_query.lower().strip()
    if len(text) < 3:
        return True
    words = set(re.findall(r"[a-z']+", text))
    vocab_hit = bool(words & moods_vocab) or bool(words & genres_vocab)
    multiword_genre_hit = any(g in text for g in genres_vocab if " " in g)
    artist_hit = any(_norm(s["artist"]) in text for s in dataset)
    generic_hit = bool(words & GENERIC_SIGNAL_WORDS)
    return not (vocab_hit or multiword_genre_hit or artist_hit or generic_hit)


# =====================================================================
# AI 30%: the single Claude API call. Claude's only job is to judge which
# catalog entries fit the request and say why — it is explicitly told it
# may only use songs from the provided catalog, but that instruction is a
# courtesy, not a guarantee. Everything it returns is re-verified below.
# =====================================================================

def get_ai_candidates(client, query, exclusions, dataset):
    catalog = [
        {"artist": s["artist"], "title": s["title"], "genres": s["genres"],
         "moods": s["moods"], "explicit": s["explicit"]}
        for s in dataset
    ]
    system = (
        "You are a music recommendation judge. You will be given a JSON catalog of "
        "songs (the ONLY songs you are allowed to recommend) and a listener request. "
        "Pick the songs from the catalog that best match the request. You MUST only "
        "use artist/title pairs that appear verbatim in the catalog -- never invent "
        f"or alter a song. Return between 8 and {CANDIDATES_REQUESTED} candidates, "
        "ranked best first. For each, give a one-line reason naming the specific mood "
        "or genre tag(s) from that song's catalog entry that justify the pick. "
        "Respond with ONLY a JSON array, no prose, in this exact shape: "
        '[{"artist": "...", "title": "...", "reason": "..."}]'
    )
    user_payload = {
        "request": query,
        "known_exclusions": {
            "excluded_moods": sorted(exclusions["excluded_moods"]),
            "excluded_genres": sorted(exclusions["excluded_genres"]),
            "exclude_explicit": exclusions["exclude_explicit"],
        },
        "catalog": catalog,
    }
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    return parse_ai_json(raw_text)


def parse_ai_json(raw_text):
    """Pull the JSON array out of the model's reply (tolerating stray code fences
    or prose around it). This is glue, not judgment -- the validator downstream
    still treats every entry as unverified."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict) and "artist" in c and "title" in c]


# =====================================================================
# BORING 70%: the validator -- the accuracy layer. Nothing here trusts
# the AI's output. Every candidate is checked against songs.json directly.
# =====================================================================

def build_lookup(dataset):
    return {(_norm(s["artist"]), _norm(s["title"])): s for s in dataset}


def validate_candidates(candidates, dataset, exclusions):
    lookup = build_lookup(dataset)
    seen = set()
    verified = []
    dropped = {"not_in_dataset": 0, "duplicate": 0, "explicit": 0,
               "excluded_mood": 0, "excluded_genre": 0}

    for cand in candidates:
        key = (_norm(str(cand.get("artist", ""))), _norm(str(cand.get("title", ""))))
        song = lookup.get(key)

        if song is None:
            dropped["not_in_dataset"] += 1
            continue
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if exclusions["exclude_explicit"] and song["explicit"]:
            dropped["explicit"] += 1
            continue
        if exclusions["excluded_moods"] & set(song["moods"]):
            dropped["excluded_mood"] += 1
            continue
        if exclusions["excluded_genres"] & set(song["genres"]):
            dropped["excluded_genre"] += 1
            continue

        seen.add(key)
        verified.append({
            "artist": song["artist"],
            "title": song["title"],
            "reason": str(cand.get("reason", "")).strip() or "Matched the request.",
            "moods": song["moods"],
            "genres": song["genres"],
        })

    return verified, dropped


# =====================================================================
# BORING 70%: output formatting
# =====================================================================

def dropped_detail_string(dropped):
    """'2 not in dataset, 1 explicit' -- shared by the CLI and the web UI
    so there's one place that decides how drop reasons read."""
    return ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in dropped.items() if v)


def format_results(verified, dropped, low_confidence_reason=None):
    lines = []

    if low_confidence_reason:
        lines.append(f"[low confidence] {low_confidence_reason}")
        lines.append("Showing best-effort matches below -- treat these as a starting "
                      "point, not a confident match.\n")

    if not verified:
        lines.append("I couldn't confidently match any songs in my dataset to this request.")
        return "\n".join(lines)

    if len(verified) < MIN_RESULTS_TO_SHOW:
        lines.append(f"I could only confidently match {len(verified)} song(s) -- "
                      f"showing what I have rather than padding the list:\n")
    else:
        lines.append(f"Here are {min(len(verified), TARGET_RESULTS)} matches:\n")

    for i, song in enumerate(verified[:TARGET_RESULTS], 1):
        lines.append(f'{i}. {song["artist"]} - "{song["title"]}"')
        lines.append(f'   why: {song["reason"]}')
        lines.append(f'   tags: {", ".join(song["moods"])} | {", ".join(song["genres"])}')

    total_dropped = sum(dropped.values())
    if total_dropped:
        lines.append(f"\n({total_dropped} AI-suggested song(s) were filtered out: "
                      f"{dropped_detail_string(dropped)})")

    return "\n".join(lines)


# =====================================================================
# Shared orchestration -- ties the boring layer and the one AI call
# together. Both the CLI (below) and the Flask web UI (app.py) call this
# same function, so there is exactly one code path for "how a request
# becomes a validated result."
# =====================================================================

def get_recommendations(query, dataset_path=SONGS_PATH):
    """Returns a dict: {verified, dropped, low_confidence_reason}.
    Raises MissingAPIKeyError or anthropic.APIError on failure -- callers
    decide how to present those."""
    dataset = load_dataset(dataset_path)
    moods_vocab, genres_vocab = build_vocab(dataset)

    exclusions = parse_exclusions(query, moods_vocab, genres_vocab)
    cleaned_query = strip_exclusion_language(query)

    contradiction = detect_contradiction(query, moods_vocab)
    vague = is_vague(cleaned_query, moods_vocab, genres_vocab, dataset)

    low_confidence_reason = None
    if contradiction:
        low_confidence_reason = (
            f'the request asks for both "{contradiction[0]}" and "{contradiction[1]}", '
            "which pull in opposite directions."
        )
    elif vague:
        low_confidence_reason = (
            "the request is too vague to pin down a specific vibe -- "
            "try adding a mood, genre, or reference artist."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    candidates = get_ai_candidates(client, cleaned_query, exclusions, dataset)

    verified, dropped = validate_candidates(candidates, dataset, exclusions)
    return {
        "verified": verified,
        "dropped": dropped,
        "low_confidence_reason": low_confidence_reason,
    }


# =====================================================================
# CLI entry point
# =====================================================================

def run(query, dataset_path=SONGS_PATH):
    try:
        result = get_recommendations(query, dataset_path)
    except MissingAPIKeyError:
        print("Error: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
              file=sys.stderr)
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"Error calling the Claude API: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_results(result["verified"], result["dropped"], result["low_confidence_reason"]))


def main():
    parser = argparse.ArgumentParser(
        description="Vibe Matcher -- recommend real songs from a verified local "
                     "dataset based on mood, genre, or reference artist."
    )
    parser.add_argument(
        "query",
        help='What you want, e.g. "moody indie rock like Phoebe Bridgers, no sad songs"'
    )
    args = parser.parse_args()
    run(args.query)


if __name__ == "__main__":
    main()
