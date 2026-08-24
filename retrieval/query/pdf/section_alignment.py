"""Map corresponding sections between two cycles of the same document.

A comparison question needs the *same* subject from both filings. Ranking each
side independently does not give that: measured on the frozen beta set,
``real_001`` retrieved ten passages per cycle and only one section pair
corresponded, because the two filings renumbered their chapters (Risk
Methodology is section 6 in the 2023-2025 filing and section 5 in the
2026-2028 one) and the later filing split one chapter into two.

The map is derived from the corpus, not written by hand: documents are paired
by the role and cycle already encoded in their filenames, and their section
titles are matched by embedding similarity. A new cycle, or a different corpus,
regenerates without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("config/section_alignment.json")
# Below this the two titles are not the same subject; leaving a section
# unpaired is safer than forcing a wrong pair into a comparison.
DEFAULT_MIN_SIMILARITY = 0.72
# A chapter split across two in a later filing is common; a title matching many
# is usually the map failing, not the document.
MAX_TARGETS_PER_SECTION = 2
# A genuine split leaves both halves near-equally similar to the original title.
# A distant runner-up is a different subject that merely shares vocabulary --
# "Wildfire Mitigations" scored 0.941 against the real counterpart and 0.783
# against a collaboration table. Requiring the second target to stay close to
# the first keeps splits and drops those.
SECOND_TARGET_RATIO = 0.92
# Sections too small to carry a comparison are noise in the map.
MIN_CHUNKS = 5

_FILENAME_RE = re.compile(
    r"^(?P<utility>[^_]+)__(?P<role>.+?)__(?P<cycle>\d{4}-\d{4})__(?P<rest>.+)\.pdf$"
)
_LEADING_NUMBER_RE = re.compile(r"^\s*(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?)\s*")


@dataclass(frozen=True)
class DocumentKey:
    role: str
    cycle: str
    filename: str


def parse_filename(filename: str) -> DocumentKey | None:
    match = _FILENAME_RE.match(filename)
    if match is None:
        return None
    return DocumentKey(match.group("role"), match.group("cycle"), filename)


def pairable_documents(filenames: list[str]) -> list[tuple[DocumentKey, DocumentKey]]:
    """Every (earlier, later) pair sharing a role across two cycles."""
    by_role: dict[str, list[DocumentKey]] = {}
    for filename in filenames:
        key = parse_filename(filename)
        if key is not None:
            by_role.setdefault(key.role, []).append(key)
    pairs = []
    for keys in by_role.values():
        ordered = sorted(keys, key=lambda k: k.cycle)
        for index, earlier in enumerate(ordered):
            for later in ordered[index + 1:]:
                if earlier.cycle != later.cycle:
                    pairs.append((earlier, later))
    return pairs


def top_sections(cur, filename: str) -> list[tuple[str, int]]:
    """Top-level sections of one document, with how many chunks each holds."""
    cur.execute(
        """
        SELECT split_part(c.breadcrumb, ' > ', 2) AS section, count(*)
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.filename = %s AND c.breadcrumb LIKE '%% > %%'
        GROUP BY 1 HAVING count(*) >= %s ORDER BY 2 DESC
        """,
        (filename, MIN_CHUNKS),
    )
    return [(str(section), int(count)) for section, count in cur.fetchall() if section]


def title_only(section: str) -> str:
    """The section title without its number, which is what differs across cycles."""
    return _LEADING_NUMBER_RE.sub("", section).strip()


def align_titles(
    left: list[tuple[str, int]],
    right: list[tuple[str, int]],
    encode,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[dict[str, Any]]:
    """Pair sections by title similarity, allowing a chapter to map to two."""
    if not left or not right:
        return []
    left_vectors = encode([title_only(section) for section, _ in left])
    right_vectors = encode([title_only(section) for section, _ in right])
    alignments = []
    for index, (section, count) in enumerate(left):
        scored = sorted(
            (
                (float(left_vectors[index] @ right_vectors[other]), other)
                for other in range(len(right))
            ),
            reverse=True,
        )
        best = scored[0][0] if scored else 0.0
        targets = [
            {
                "section": right[other][0],
                "chunks": right[other][1],
                "similarity": round(score, 4),
            }
            for rank, (score, other) in enumerate(scored[:MAX_TARGETS_PER_SECTION])
            if score >= min_similarity
            and (rank == 0 or score >= best * SECOND_TARGET_RATIO)
        ]
        alignments.append(
            {
                "section": section,
                "chunks": count,
                "title": title_only(section),
                "targets": targets,
                "best_similarity": round(scored[0][0], 4) if scored else 0.0,
            }
        )
    return alignments
