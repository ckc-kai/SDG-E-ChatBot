"""What the workbooks actually contain, as standing evidence.

Several questions are answerable only from the corpus's *shape*, and the model
was never shown it:

- "Pull our 2022 actuals" -- the workbooks start at 2023, so the only correct
  answer states that. Shown nothing, the model invented 2022 figures.
- "Undergrounding cost per mile by circuit segment" -- Table 11 records spend
  at initiative level with no segment field, and Table 15 carries segments with
  no cost field, so no key joins them. That is the answer, and it is a fact
  about the schema rather than about any value.
- Picking ``duration_of_X`` when the question meant ``frequency_of_X`` is a
  metric-family confusion that listing the family resolves outright.

All of it is already stored -- ``excel_sources``, ``excel_revisions.
column_inventory``, and one coverage query -- and none of it reached a prompt.
The manifest is derived, cached per revision set, and never invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Enough to name the family a question might mean, without pasting a
# vocabulary the answering model has no use for.
MAX_METRIC_KEYS = 26
MAX_DIMENSIONS = 22
# Above this a column is a vocabulary, not a domain worth enumerating.
MAX_DOMAIN_VALUES = 12
# Dimensions that identify a row rather than describe it. Listing them invites
# the planner to group by an opaque hash, which is what it used to do.
_OPAQUE_DIMENSIONS = {
    "comparison_group_id",
    "tracking_id_normalized",
    "group_tracking_ids",
    "record_id",
    "series_id",
}


@dataclass(frozen=True)
class TableFacts:
    table_number: int
    title: str
    family: str
    row_count: int
    years: tuple[int, ...]
    quarters: tuple[int, ...]
    dimensions: tuple[str, ...]
    metric_keys: tuple[str, ...]
    # Low-cardinality typed columns, with their values. Table 13's `status`
    # holds 'Level 2'/'Level 3' -- the GO 95 Rule 18 priority -- and without
    # that stated the planner read "Level 2" as HFTD Tier 2 and answered a
    # different question with confident wrong counts.
    value_domains: tuple[tuple[str, tuple[str, ...], int], ...] = ()
    # Typed column -> the source column ingest promoted into it. Table 13's
    # ``status`` is the GO 95 Rule 18 priority; without saying so the planner
    # looked for a "go95" field, failed to find one, and declined a question
    # the corpus answers exactly.
    promoted_from: tuple[tuple[str, str], ...] = ()
    description: str = ""
    metric_key_total: int = 0
    dimension_total: int = 0


@dataclass(frozen=True)
class WorkbookManifest:
    tables: tuple[TableFacts, ...] = ()
    years: tuple[int, ...] = ()

    def render(self) -> str:
        if not self.tables:
            return ""
        span = (
            f"{min(self.years)}-{max(self.years)}" if self.years else "unknown"
        )
        lines = [
            "WORKBOOK COVERAGE AND SCHEMA (the quarterly data report tables).",
            "This is the complete inventory. A year, table, or field absent",
            f"here is absent from the corpus. Reported years: {span}.",
            "Fields marked [col] are typed columns; the rest are attributes.",
            "",
        ]
        for table in self.tables:
            years = (
                f"{min(table.years)}-{max(table.years)}" if table.years else "none"
            )
            quarters = (
                ",".join(f"Q{value}" for value in table.quarters)
                if table.quarters
                else "annual"
            )
            lines.append(
                f"Table {table.table_number} - {table.title} "
                f"({table.family}; {table.row_count:,} rows; "
                f"years {years}; {quarters})"
            )
            if table.description:
                lines.append(f"  holds: {table.description}")
            for column, origin in table.promoted_from:
                lines.append(f"  column {column} [col] is the {origin}")
            if table.dimensions:
                more = (
                    f" (+{table.dimension_total - len(table.dimensions)} more)"
                    if table.dimension_total > len(table.dimensions)
                    else ""
                )
                lines.append(f"  fields: {', '.join(table.dimensions)}{more}")
            for column, values, null_count in table.value_domains:
                rendered = ", ".join(repr(value) for value in values)
                blank = f"; {null_count:,} rows blank" if null_count else ""
                lines.append(f"  {column} [col] values: {rendered}{blank}")
            if table.metric_keys:
                more = (
                    f" (+{table.metric_key_total - len(table.metric_keys)} more)"
                    if table.metric_key_total > len(table.metric_keys)
                    else ""
                )
                lines.append(
                    f"  metrics: {', '.join(table.metric_keys)}{more}"
                )
        lines += [
            "",
            "Use this to say plainly what the workbooks cannot support -- a year",
            "not covered, a field a table does not carry, or two tables that",
            "share no key -- instead of estimating or assuming. When a question",
            "needs a field one table lacks, name the table and the missing",
            "field.",
        ]
        return "\n".join(lines)


def _readable(name: str) -> str:
    return name.replace("_", " ").strip()


def _fetch_tables(conn, contracts=None) -> list[TableFacts]:
    if contracts is None:
        from retrieval.ingest.excel.contracts import load_contracts

        contracts = load_contracts()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.table_number, s.title, r.record_count, r.fact_count
            FROM excel_sources s
            JOIN excel_revisions r ON r.id = s.active_revision_id
            ORDER BY s.table_number
            """
        )
        sources = cur.fetchall()

        cur.execute(
            """
            SELECT f.table_number,
                   array_agg(DISTINCT f.reporting_year)
                     FILTER (WHERE f.reporting_year IS NOT NULL),
                   array_agg(DISTINCT f.reporting_quarter)
                     FILTER (WHERE f.reporting_quarter IS NOT NULL)
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state = 'active'
            GROUP BY f.table_number
            """
        )
        fact_periods = {row[0]: (row[1] or [], row[2] or []) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT e.table_number,
                   array_agg(DISTINCT e.reporting_year)
                     FILTER (WHERE e.reporting_year IS NOT NULL),
                   array_agg(DISTINCT e.reporting_quarter)
                     FILTER (WHERE e.reporting_quarter IS NOT NULL)
            FROM excel_records e
            JOIN excel_revisions r ON r.id = e.revision_id AND r.state = 'active'
            GROUP BY e.table_number
            """
        )
        record_periods = {row[0]: (row[1] or [], row[2] or []) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT f.table_number, array_agg(DISTINCT f.semantic_metric_key)
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state = 'active'
            GROUP BY f.table_number
            """
        )
        metric_keys = {row[0]: sorted(row[1] or []) for row in cur.fetchall()}

        # The queryable surface, not the source column list. Several source
        # columns are promoted into typed columns during ingest -- table 13's
        # go95_rule18_priority becomes ``status`` -- so advertising
        # column_inventory would send the planner after jsonb keys that do not
        # exist, and hide the typed column that actually holds the value.
        cur.execute(
            """
            SELECT f.table_number, array_agg(DISTINCT k)
            FROM excel_facts f
            JOIN excel_revisions r ON r.id = f.revision_id AND r.state = 'active',
                 LATERAL jsonb_object_keys(f.dimensions) k
            GROUP BY f.table_number
            """
        )
        json_keys = {row[0]: sorted(row[1] or []) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT e.table_number, array_agg(DISTINCT k)
            FROM excel_records e
            JOIN excel_revisions r ON r.id = e.revision_id AND r.state = 'active',
                 LATERAL jsonb_object_keys(e.attributes) k
            GROUP BY e.table_number
            """
        )
        json_keys.update({row[0]: sorted(row[1] or []) for row in cur.fetchall()})

        # Typed columns carrying a value for this table, so the planner filters
        # and groups on the column rather than hunting for a jsonb key.
        typed_present: dict[int, list[str]] = {}
        for table_source, columns, id_column in (
            (
                "excel_records",
                ("title", "status", "entity_key", "entity_type", "hftd_tier",
                 "line_type", "date_start", "date_due", "date_end"),
                "e",
            ),
            (
                "excel_facts",
                ("metric_name", "measure_name", "unit", "hftd_tier",
                 "line_type", "year_basis", "period_end_date"),
                "f",
            ),
        ):
            checks = ", ".join(
                f"count({id_column}.{column}) AS {column}" for column in columns
            )
            cur.execute(
                f"""
                SELECT {id_column}.table_number, {checks}
                FROM {table_source} {id_column}
                JOIN excel_revisions r
                  ON r.id = {id_column}.revision_id AND r.state = 'active'
                GROUP BY {id_column}.table_number
                """
            )
            for row in cur.fetchall():
                present = [
                    column for column, count in zip(columns, row[1:]) if count
                ]
                typed_present.setdefault(row[0], []).extend(present)

        # Values of the few typed columns small enough to enumerate, plus how
        # many rows leave the column blank -- the blank cohort is part of the
        # answer, not an absence to drop.
        value_domains: dict[int, list[tuple[str, tuple[str, ...], int]]] = {}
        for table_source, alias, columns in (
            ("excel_records", "e", ("status", "entity_type", "hftd_tier", "line_type")),
            ("excel_facts", "f", ("hftd_tier", "line_type", "unit", "year_basis")),
        ):
            for column in columns:
                cur.execute(
                    f"""
                    SELECT {alias}.table_number, {alias}.{column}, count(*)
                    FROM {table_source} {alias}
                    JOIN excel_revisions r
                      ON r.id = {alias}.revision_id AND r.state = 'active'
                    GROUP BY 1, 2
                    """
                )
                grouped: dict[int, list[tuple[str | None, int]]] = {}
                for table_number, value, count in cur.fetchall():
                    grouped.setdefault(table_number, []).append((value, count))
                for table_number, pairs in grouped.items():
                    present = sorted(
                        (value, count) for value, count in pairs if value is not None
                    )
                    if not present or len(present) > MAX_DOMAIN_VALUES:
                        continue
                    blank = sum(
                        count for value, count in pairs if value is None
                    )
                    value_domains.setdefault(table_number, []).append(
                        (column, tuple(value for value, _ in present), blank)
                    )

    tables: list[TableFacts] = []
    for table_number, title, record_count, fact_count in sources:
        try:
            contract = contracts.for_table(table_number)
        except (KeyError, ValueError):
            contract = None
        # Entity columns whose name differs from the source field they hold.
        # The contract names the entity role ('key'); the queryable column is
        # 'entity_key'. Advertise the column, or the planner filters a field
        # that does not exist.
        entity_roles = {
            "key": "entity_key",
            "status": "status",
            "title": "title",
            "date_start": "date_start",
            "date_due": "date_due",
            "date_end": "date_end",
        }
        promoted_from = tuple(
            (entity_roles[role], _readable(origin))
            for role, origin in sorted(
                (getattr(contract, "entity", None) or {}).items()
            )
            if role in entity_roles
            and origin
            and _readable(origin) != entity_roles[role]
        )
        description = " ".join(
            str(getattr(contract, "description", "") or "").split()
        )
        is_records = record_count > 0
        years, quarters = (record_periods if is_records else fact_periods).get(
            table_number, ([], [])
        )
        typed = ["reporting_year", "reporting_quarter"] + sorted(
            dict.fromkeys(typed_present.get(table_number, []))
        )
        attributes = [
            name
            for name in json_keys.get(table_number, [])
            if name not in _OPAQUE_DIMENSIONS
            and not name.startswith("source_")
            and not name.endswith(("_status", "_crosswalk"))
            and name not in {"schema_version", "utility_id", "guideline_url"}
        ]
        dimensions = [f"{name} [col]" for name in typed] + attributes
        keys = metric_keys.get(table_number, [])
        tables.append(
            TableFacts(
                table_number=table_number,
                title=title,
                family="one row per record" if is_records else "one row per metric",
                row_count=record_count if is_records else fact_count,
                years=tuple(sorted(years)),
                quarters=tuple(sorted(quarters)),
                dimensions=tuple(dimensions[:MAX_DIMENSIONS]),
                metric_keys=tuple(keys[:MAX_METRIC_KEYS]),
                metric_key_total=len(keys),
                dimension_total=len(dimensions),
                value_domains=tuple(value_domains.get(table_number, [])),
                promoted_from=promoted_from,
                description=description,
            )
        )
    return tables


_cache: dict[str, WorkbookManifest] = {}


def load_manifest(conn, *, use_cache: bool = True) -> WorkbookManifest:
    """The active revisions' coverage and schema. Empty when Excel is absent."""
    if use_cache and "manifest" in _cache:
        return _cache["manifest"]
    try:
        tables = _fetch_tables(conn)
    except Exception:  # pragma: no cover - diagnostic path only
        logger.warning("Could not build the workbook manifest", exc_info=True)
        return WorkbookManifest()
    years = sorted({year for table in tables for year in table.years})
    manifest = WorkbookManifest(tables=tuple(tables), years=tuple(years))
    if use_cache:
        _cache["manifest"] = manifest
    return manifest


def clear_cache() -> None:
    _cache.clear()
