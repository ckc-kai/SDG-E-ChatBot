"""The manifest must describe what is queryable, and must not invent."""

from __future__ import annotations

import pytest

from retrieval.query.excel.manifest import (
    TableFacts,
    WorkbookManifest,
    clear_cache,
    load_manifest,
)


def _facts(**overrides) -> TableFacts:
    defaults = dict(
        table_number=13,
        title="Open work orders",
        family="one row per record",
        row_count=58029,
        years=(2023, 2024, 2025),
        quarters=(1, 2, 3, 4),
        dimensions=("reporting_year [col]", "status [col]", "equipment_type"),
        metric_keys=(),
        metric_key_total=0,
        dimension_total=3,
        value_domains=(("status", ("Level 2", "Level 3"), 50446),),
    )
    return TableFacts(**{**defaults, **overrides})


class TestRendering:
    def test_states_the_blank_cohort_beside_the_populated_values(self):
        rendered = WorkbookManifest(tables=(_facts(),), years=(2023, 2025)).render()

        assert "'Level 2', 'Level 3'" in rendered
        # The blank count is the half of excel_002's gold answer that the
        # pipeline used to drop silently.
        assert "50,446 rows blank" in rendered

    def test_states_the_covered_year_span(self):
        rendered = WorkbookManifest(tables=(_facts(),), years=(2023, 2025)).render()

        assert "2023-2025" in rendered
        assert "2022" not in rendered

    def test_distinguishes_typed_columns_from_attributes(self):
        rendered = WorkbookManifest(tables=(_facts(),), years=(2023,)).render()

        assert "status [col]" in rendered
        assert "equipment_type" in rendered

    def test_empty_manifest_renders_nothing_rather_than_a_header(self):
        assert WorkbookManifest().render() == ""


class TestLoading:
    def test_a_failing_connection_yields_an_empty_manifest_not_an_error(self):
        """A diagnostic aid must never take down an answer."""

        class Broken:
            def cursor(self):
                raise RuntimeError("no database")

        clear_cache()
        try:
            assert load_manifest(Broken(), use_cache=False).tables == ()
        finally:
            clear_cache()


@pytest.mark.integration
class TestAgainstCorpus:
    """Skipped without a database; asserts the facts the golds depend on."""

    def test_manifest_reports_the_corpus_shape(self):
        psycopg2 = pytest.importorskip("psycopg2")
        from retrieval.utils import connect_db

        try:
            connection = connect_db()
        except Exception:  # pragma: no cover - environment dependent
            pytest.skip("no database available")
        clear_cache()
        try:
            manifest = load_manifest(connection, use_cache=False)
        finally:
            connection.close()
            clear_cache()

        by_number = {table.table_number: table for table in manifest.tables}
        assert 2022 not in manifest.years, "workbooks start at 2023"

        work_orders = by_number[13]
        domains = dict(
            (column, values) for column, values, _ in work_orders.value_domains
        )
        assert domains["status"] == ("Level 2", "Level 3")

        # Table 11 carries no segment field and table 15 no cost field, which
        # is why undergrounding cost per segment cannot be produced.
        assert not any("segment" in name for name in by_number[11].dimensions)
        assert any("segment_id" in name for name in by_number[15].dimensions)


class TestRaggedCoverageIsNotReportedAsUniform:
    """A union span over tables with different years must not read as coverage.

    The metric tables run a year further than the activity records, so the
    union (2023-2026) is not true of any single table. Answers repeated it back
    as the corpus range and were marked down for naming a year the records do
    not hold.
    """

    def _manifest(self, spans):
        from retrieval.query.excel.manifest import TableFacts, WorkbookManifest

        tables = tuple(
            TableFacts(
                table_number=number,
                title=f"Table {number}",
                family="excel_facts",
                row_count=10,
                years=tuple(years),
                quarters=(),
                dimensions=(),
                metric_keys=(),
            )
            for number, years in enumerate(spans, start=1)
        )
        years = tuple(sorted({y for _, s in enumerate(spans) for y in s}))
        return WorkbookManifest(tables=tables, years=years)

    def test_ragged_coverage_is_flagged(self):
        rendered = self._manifest([(2023, 2024, 2025), (2023, 2024, 2025, 2026)]).render()
        assert "Coverage is NOT uniform" in rendered

    def test_uniform_coverage_is_not_flagged(self):
        rendered = self._manifest([(2023, 2024, 2025), (2023, 2024, 2025)]).render()
        assert "Coverage is NOT uniform" not in rendered
