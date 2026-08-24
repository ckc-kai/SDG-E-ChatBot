"""A unit refusal must be repaired into a per-entity breakdown, not lost.

``real_009`` asks for cumulative three-year targets per activity. The planner
correctly reaches table 1 and correctly names ``annual_quant_target``, and the
executor then -- also correctly -- refuses to sum it, because across every
activity that attribute spans 23 units (Poles, Trees, Miles, Findings ...).
Both components are right and the question still got nothing.

Grouping by the activity is the shape the guard already accepts, and it is the
shape the question asked for. The repair is a deterministic rewrite to that
shape; it is never a retry of the refused statement.
"""

from __future__ import annotations

import pytest

from retrieval.query.excel.nl_planner import _regroup_for_units
from retrieval.query.excel.query import (
    FACTS,
    RECORDS,
    ExcelQueryPlan,
    Filter,
    GroupKey,
    PlanError,
    _groups_separate_units,
)

UNIT_REFUSAL = PlanError(
    "refusing to aggregate 'annual_quant_target' across incompatible units: "
    "Arrestors, Base Stations, Capacitors, Circuits, Findings ..."
)


def _plan(**kwargs) -> ExcelQueryPlan:
    base = dict(
        table_number=1,
        source=RECORDS,
        operation="aggregate",
        aggregate="sum",
        value_json_key="annual_quant_target",
        filters=(Filter("reporting_year", "in", [2023, 2024, 2025]),),
        group_by=("reporting_year",),
    )
    base.update(kwargs)
    return ExcelQueryPlan(**base)


class TestRepair:
    def test_a_unit_refusal_regroups_on_the_entity(self):
        repaired = _regroup_for_units(_plan(), UNIT_REFUSAL)
        assert repaired is not None
        assert "entity_key" in repaired.group_by

    def test_the_repaired_plan_satisfies_the_guard_that_refused_it(self):
        """The repair is only worth attempting if it can actually pass."""
        original = _plan()
        assert not _groups_separate_units(original)
        assert _groups_separate_units(_regroup_for_units(original, UNIT_REFUSAL))

    def test_the_repair_keeps_every_original_grouping_and_filter(self):
        repaired = _regroup_for_units(_plan(), UNIT_REFUSAL)
        assert repaired.group_by[: len(_plan().group_by)] == _plan().group_by
        assert repaired.filters == _plan().filters
        assert repaired.value_json_key == "annual_quant_target"

    def test_the_repair_raises_the_limit_to_fit_every_activity(self):
        assert _regroup_for_units(_plan(limit=8), UNIT_REFUSAL).limit >= 50

    def test_an_already_high_limit_is_not_lowered(self):
        assert _regroup_for_units(_plan(limit=200), UNIT_REFUSAL).limit == 200


class TestRefusalsToRepair:
    def test_an_unrelated_plan_error_is_not_repaired(self):
        assert _regroup_for_units(_plan(), PlanError("plan names no reporting period")) is None

    def test_a_fact_table_plan_is_not_repaired(self):
        """Fact tables carry a typed unit column and their own guard."""
        assert _regroup_for_units(_plan(source=FACTS), UNIT_REFUSAL) is None

    def test_a_select_scan_is_not_repaired(self):
        assert _regroup_for_units(_plan(operation="select"), UNIT_REFUSAL) is None

    def test_a_plan_already_grouped_on_every_entity_key_is_not_repaired(self):
        """Nothing left to add means the refusal was about something else."""
        plan = _plan(group_by=("entity_key", "title", "record_id"))
        assert _regroup_for_units(plan, UNIT_REFUSAL) is None

    def test_a_json_group_key_does_not_block_the_entity_repair(self):
        plan = _plan(group_by=(GroupKey("initiative", json_key=True),))
        repaired = _regroup_for_units(plan, UNIT_REFUSAL)
        assert repaired is not None and "entity_key" in repaired.group_by
