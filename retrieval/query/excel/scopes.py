"""Overlapping-scope contracts: which dimension values must never be summed.

Correct arithmetic over the wrong business scope is the failure mode that raw
DDL cannot prevent and that a validating SQL executor will happily certify.
``excel_003`` is the worked example: table 11 reports WMP spend twice, once
scoped to ``Territory`` and once to the ``HFTD`` subset inside it. Both rows
are real, both sum cleanly, and adding them double-counts every dollar in the
HFTD area.

The distinction is per table and per value, which is why it is a contract
rather than a synonym list:

- table 11 uses ``{Territory, HFTD}``  -- hierarchical. ``Territory`` is the
  whole; ``HFTD`` is contained in it. Summing across the column double-counts.
- every other table uses ``{HFTD Tier 2, HFTD Tier 3, Non-HFTD}`` -- a genuine
  partition of the service territory. Summing across *is* the system total,
  and refusing it would break the tier comparisons those tables exist for.

Evidence for the containment rule, re-derived against the active corpus by
``tests/test_excel_scopes.py`` so it cannot rot silently: across all six
(reporting_year, initiative, expense_type) cells of table 11 that carry both
values, the HFTD figure never exceeds the Territory figure, and the 2024
Territory total (667,954) is exactly the gold CAPEX (473,986) plus OPEX
(193,968).
"""

from __future__ import annotations

from dataclasses import dataclass

CONTRACT_VERSION = "2026-08-21.1"


@dataclass(frozen=True)
class OverlappingScope:
    """One column whose values overlap, and the value that is the whole."""

    column: str
    canonical: str
    contained: tuple[str, ...]

    @property
    def values(self) -> tuple[str, ...]:
        return (self.canonical, *self.contained)


OVERLAPPING_SCOPES: dict[int, OverlappingScope] = {
    11: OverlappingScope(
        column="hftd_tier",
        canonical="Territory",
        contained=("HFTD",),
    ),
}


def overlapping_scope(table_number: int | None) -> OverlappingScope | None:
    """The overlap contract for this table, or None when its values partition."""
    return OVERLAPPING_SCOPES.get(table_number) if table_number is not None else None


def scope_is_pinned(scope: OverlappingScope, plan) -> bool:
    """True when the plan already fixes or labels the overlapping column.

    Either is safe. A filter picks one scope; a ``group_by`` labels every row
    with the scope it belongs to, so nothing is silently conflated -- though a
    *total* across those labelled groups is still forbidden, which is enforced
    separately in ``generation.excel_rollup``.
    """
    if any(flt.field == scope.column for flt in plan.filters):
        return True
    return any(
        (entry.field if hasattr(entry, "field") else str(entry)) == scope.column
        for entry in plan.group_by
    )
