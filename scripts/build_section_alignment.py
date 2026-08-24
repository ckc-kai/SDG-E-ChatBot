"""Generate the cross-cycle section alignment map from the live corpus.

Run after ingest, or whenever a new filing cycle is added. The output is a
reviewable JSON file: every pair carries its similarity so a wrong pairing can
be spotted and the threshold retuned without touching code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from retrieval.query.pdf.section_alignment import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PATH,
    align_titles,
    pairable_documents,
    top_sections,
)
from retrieval.utils import connect_db, load_config


def _encoder():
    """Encode titles with the corpus's own embedding model, normalised."""
    from sentence_transformers import SentenceTransformer

    name = load_config()["models"]["embedding"]["name"]
    model = SentenceTransformer(name)

    def encode(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args()

    load_dotenv()
    connection = connect_db()
    encode = _encoder()
    pairs_out = []
    with connection, connection.cursor() as cur:
        cur.execute("SELECT filename FROM documents WHERE filename LIKE '%%.pdf'")
        filenames = [row[0] for row in cur.fetchall()]
        for earlier, later in pairable_documents(filenames):
            left = top_sections(cur, earlier.filename)
            right = top_sections(cur, later.filename)
            alignments = align_titles(
                left, right, encode, min_similarity=args.min_similarity
            )
            paired = sum(1 for entry in alignments if entry["targets"])
            pairs_out.append(
                {
                    "role": earlier.role,
                    "earlier": {"cycle": earlier.cycle, "filename": earlier.filename},
                    "later": {"cycle": later.cycle, "filename": later.filename},
                    "sections_earlier": len(left),
                    "sections_later": len(right),
                    "sections_paired": paired,
                    "alignments": alignments,
                }
            )

    payload = {"min_similarity": args.min_similarity, "pairs": pairs_out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for pair in pairs_out:
        print(
            f"\n=== {pair['role']}: {pair['earlier']['cycle']} -> "
            f"{pair['later']['cycle']}  "
            f"({pair['sections_paired']}/{pair['sections_earlier']} paired)"
        )
        if not args.show:
            continue
        for entry in sorted(pair["alignments"], key=lambda e: -e["chunks"]):
            if entry["targets"]:
                for target in entry["targets"]:
                    print(
                        f"  {entry['chunks']:5d}  {entry['section'][:46]:46s}"
                        f" -> {target['section'][:46]:46s} {target['similarity']:.3f}"
                    )
            else:
                print(
                    f"  {entry['chunks']:5d}  {entry['section'][:46]:46s}"
                    f" -> (unpaired, best {entry['best_similarity']:.3f})"
                )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
