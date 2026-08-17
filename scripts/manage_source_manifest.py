"""Audit or apply the additive source-role metadata registry.

The default command is read-only and never renames source files. ``--apply``
upserts only ``source_registry`` rows; it does not touch documents, chunks,
embeddings, Excel facts, or object paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.source_manifest import (
    DEFAULT_MANIFEST_PATH,
    SourceManifest,
    audit_local_files,
    backfill_source_registry,
)
from retrieval.utils import connect_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=Path("resources/wmp/pdf"))
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("excel_cleaning/cleaned_csv_rag_ready"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create/upsert the additive database registry after a clean file audit.",
    )
    args = parser.parse_args()

    manifest = SourceManifest.load(args.manifest)
    report = audit_local_files(
        manifest, pdf_dir=args.pdf_dir, csv_dir=args.csv_dir
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.apply:
        return
    if not report["metadata_only_backfill_sufficient"]:
        raise SystemExit("Refusing database backfill until manifest/file drift is resolved")
    connection = connect_db()
    try:
        count = backfill_source_registry(connection, manifest)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    print(f"Backfilled {count} source metadata records; corpus content was unchanged")


if __name__ == "__main__":
    main()
