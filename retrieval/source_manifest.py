"""Stable source-role manifest, intake validation, and metadata backfill."""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "config/source_manifest.json"
DocumentType = Literal["pdf", "excel", "csv"]

_PDF_NAME_RE = re.compile(
    r"^[a-z0-9-]+__[a-z0-9_-]+__\d{4}-\d{4}__[a-z0-9.-]+__\d{4}-\d{2}-\d{2}\.pdf$"
)
_EXCEL_NAME_RE = re.compile(
    r"^[a-z0-9-]+__[a-z0-9_-]+__[a-z0-9-]+__[a-z0-9.-]+\.(?:xlsx|csv)$"
)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    utility: str
    document_role: str
    document_type: DocumentType
    original_filename: str
    canonical_filename: str
    cycle_start: int | None = None
    cycle_end: int | None = None
    version: str | None = None
    effective_date: str | None = None
    table_role: str | None = None
    sheet_alias: str | None = None
    # As-received name, kept only for provenance once a file has been renamed on
    # disk. ``original_filename`` always tracks the current on-disk name, which is
    # what every consumer resolves against.
    received_filename: str | None = None
    # Human-readable title fed to the embedder as the first line of every
    # contextual embedding. Renaming the files fixed what is on disk; this fixes
    # what is embedded. A filename tokenises badly, and this corpus needs the
    # regulatory cycle and the issuing body legible -- a plan and the guidelines
    # it must comply with are otherwise near-indistinguishable in embedding space.
    display_title: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SourceRecord":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown source-manifest fields: {sorted(unknown)}")
        return cls(**value)


class SourceManifest:
    def __init__(self, records: Iterable[SourceRecord]):
        self.records = tuple(records)
        ids = [record.source_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source_id in source manifest")
        names = [record.original_filename.casefold() for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("duplicate original_filename in source manifest")
        for record in self.records:
            if not all(
                value.strip()
                for value in (
                    record.source_id,
                    record.utility,
                    record.document_role,
                    record.original_filename,
                    record.canonical_filename,
                )
            ):
                raise ValueError("required source-manifest fields must not be empty")
            if record.document_type not in {"pdf", "excel", "csv"}:
                raise ValueError(f"unsupported document_type {record.document_type!r}")

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST_PATH) -> "SourceManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("source manifest must contain a sources array")
        return cls(SourceRecord.from_mapping(row) for row in rows)

    def filenames_for_role(self, role: str) -> tuple[str, ...]:
        return tuple(
            record.original_filename
            for record in self.records
            if record.document_role == role
        )

    def by_source_id(self, source_id: str) -> SourceRecord | None:
        return next(
            (record for record in self.records if record.source_id == source_id), None
        )

    def by_filename(self, filename: str) -> SourceRecord | None:
        """Resolve a record by its current on-disk name, or its as-received one.

        Accepting the received name keeps artefacts written before the rename --
        golden-set source paths, older logs -- resolvable instead of silently
        missing.
        """
        folded = filename.casefold()
        for record in self.records:
            if record.original_filename.casefold() == folded:
                return record
        for record in self.records:
            if (record.received_filename or "").casefold() == folded:
                return record
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "source-manifest-v1",
            "sources": [asdict(row) for row in self.records],
        }


@functools.lru_cache(maxsize=1)
def _default_manifest() -> SourceManifest | None:
    """The checked-in manifest, or None if it cannot be read.

    Ingest must not become unrunnable because a title lookup failed, so callers
    fall back to the filename. Loading once keeps that off the per-document path.
    """
    try:
        return SourceManifest.load()
    except (OSError, ValueError) as exc:
        logger.warning(
            "Source manifest unavailable (%s); falling back to filenames for "
            "embedded document titles",
            exc,
        )
        return None


def title_for_filename(filename: str) -> str:
    """Readable document title for the embedder, falling back to the filename."""
    manifest = _default_manifest()
    record = manifest.by_filename(filename) if manifest else None
    if record is None or not record.display_title:
        logger.debug("No display_title for %s; embedding the filename", filename)
        return filename
    return record.display_title


def validate_intake_filename(filename: str, document_type: str) -> None:
    """Reject new uploads that do not use the canonical intake convention."""
    if document_type == "pdf":
        valid = bool(_PDF_NAME_RE.fullmatch(filename))
    elif document_type in {"excel", "csv"}:
        valid = bool(_EXCEL_NAME_RE.fullmatch(filename))
    else:
        raise ValueError(f"unsupported document type {document_type!r}")
    if not valid:
        raise ValueError(f"filename does not follow the canonical {document_type} convention")


def audit_local_files(
    manifest: SourceManifest,
    *,
    pdf_dir: Path,
    csv_dir: Path,
) -> dict[str, Any]:
    """Compare the manifest with local source files without changing either."""
    expected = {record.original_filename for record in manifest.records}
    found = {
        path.name
        for directory, suffixes in (
            (pdf_dir, {".pdf"}),
            (csv_dir, {".csv", ".xlsx", ".xls"}),
        )
        if directory.exists()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in suffixes
    }
    return {
        "manifest_count": len(manifest.records),
        "found_count": len(found),
        "missing": sorted(expected - found),
        "untracked": sorted(found - expected),
        "metadata_only_backfill_sufficient": not (expected - found or found - expected),
    }


SOURCE_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS source_registry (
    source_id          text PRIMARY KEY,
    utility            text NOT NULL,
    document_role      text NOT NULL,
    document_type      text NOT NULL CHECK (document_type IN ('pdf', 'excel', 'csv')),
    cycle_start        int,
    cycle_end          int,
    version            text,
    effective_date     date,
    original_filename  text NOT NULL UNIQUE,
    canonical_filename text NOT NULL,
    table_role         text,
    sheet_alias        text,
    updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS source_registry_role_idx
    ON source_registry (document_role, table_role);
"""


def ensure_source_registry(conn) -> None:
    """Create only the additive metadata registry; corpus rows are untouched."""
    with conn.cursor() as cur:
        cur.execute(SOURCE_REGISTRY_SQL)
    conn.commit()


def backfill_source_registry(conn, manifest: SourceManifest) -> int:
    """Upsert manifest metadata without changing documents, chunks, or embeddings."""
    ensure_source_registry(conn)
    statement = """
        INSERT INTO source_registry (
            source_id, utility, document_role, document_type, cycle_start,
            cycle_end, version, effective_date, original_filename,
            canonical_filename, table_role, sheet_alias
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_id) DO UPDATE SET
            utility = EXCLUDED.utility,
            document_role = EXCLUDED.document_role,
            document_type = EXCLUDED.document_type,
            cycle_start = EXCLUDED.cycle_start,
            cycle_end = EXCLUDED.cycle_end,
            version = EXCLUDED.version,
            effective_date = EXCLUDED.effective_date,
            original_filename = EXCLUDED.original_filename,
            canonical_filename = EXCLUDED.canonical_filename,
            table_role = EXCLUDED.table_role,
            sheet_alias = EXCLUDED.sheet_alias,
            updated_at = now()
    """
    with conn.cursor() as cur:
        for row in manifest.records:
            cur.execute(
                statement,
                (
                    row.source_id,
                    row.utility,
                    row.document_role,
                    row.document_type,
                    row.cycle_start,
                    row.cycle_end,
                    row.version,
                    row.effective_date,
                    row.original_filename,
                    row.canonical_filename,
                    row.table_role,
                    row.sheet_alias,
                ),
            )
    conn.commit()
    return len(manifest.records)
