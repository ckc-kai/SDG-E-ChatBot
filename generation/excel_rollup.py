"""Deterministic roll-ups over an executed Excel result.

The measured Excel loss splits three ways (see
``logs/progress/2026-08-19-extended.md``): 46% of gold figures are never
fetched, 20% are **fetched as components and never combined**, and 9% are
fetched and dropped in the prose. This module addresses the middle bucket.

A grouped Excel query returns the components of the answer -- spend per
initiative, miles per year, counts per tier -- and the question almost always
also wants the total, each component's share, and often a running total. Before
this module the only deterministic roll-up in the system was
``_verified_entity_history_chunks``, which fires exclusively for table 1 read
with three specific attribute keys. Every other grouped result was handed to
the answering model as bare rows, and the model either did the arithmetic in
its head (the thing this architecture exists to prevent) or omitted the total
entirely.

The rules here are deliberately narrow. A roll-up is emitted only when it is
arithmetically defensible:

- ``sum`` and ``count`` are additive, so their groups total. ``avg``, ``min``
  and ``max`` are not: averaging averages weights the groups wrongly and the
  max of maxima is only correct by accident, so neither is totalled.
- A result the executor flagged as spanning several units is never totalled.
  That guard is the reason ``real_009`` stopped reporting 2,053,923.98 as the
  sum of inspections, poles, trees and miles.
- Shares are emitted only over a strictly positive total whose components are
  all non-negative. A "share" of a set that mixes signs is not a share.
- Nothing here is a percentage *of a target*: that is a different metric with
  a different denominator, and it belongs to the entity-history path.

Every emitted figure carries the arithmetic that produced it, so the answering
model can quote the calculation rather than reproduce it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from retrieval.query.excel.scopes import overlapping_scope

# Additive aggregates only. See the module docstring.
TOTALLABLE_AGGREGATES = {"sum", "count", "count_null", "count_distinct"}

# A roll-up over one row restates that row; over hundreds it is a wall of
# arithmetic nobody asked for. Both ends are excluded.
MIN_ROLLUP_ROWS = 2
MAX_SHARE_ROWS = 25


@dataclass(frozen=True)
class RollUp:
    """One deterministically computed figure and the arithmetic behind it."""

    label: str
    value: Decimal
    calculation: str
    unit: str | None = None

    def render(self) -> str:
        rendered = format(self.value.normalize(), "f")
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.label}={rendered}{unit} ({self.calculation})"


_GENERATED_GROUP_ALIAS = re.compile(r"^group_(\d+)$")


def display_columns(result, plan=None) -> list[str]:
    """Result columns with the generated jsonb group aliases named back.

    ``compile_plan`` aliases a jsonb group key as ``group_0`` on purpose: the
    key name comes from the model, and putting caller text in a SQL identifier
    position is how injection happens. That safety property costs readability
    downstream -- evidence read ``group_0=CAPEX`` where the model needed
    ``expense_type=CAPEX`` -- so the name is restored here, at render time,
    where it is data rather than SQL.
    """
    plan = plan if plan is not None else getattr(result, "plan", None)
    group_by = list(getattr(plan, "group_by", ()) or ())
    named = []
    for column in result.columns:
        match = _GENERATED_GROUP_ALIAS.match(column)
        if match and int(match.group(1)) < len(group_by):
            entry = group_by[int(match.group(1))]
            named.append(getattr(entry, "field", None) or str(entry))
        else:
            named.append(column)
    return named


def _decimal(value: Any) -> Decimal | None:
    """The cell as a Decimal, or None when it is blank or not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _measure_index(result) -> int | None:
    """Index of the aggregated measure column, or None when there is not one.

    The compiler always emits the measure last and names it ``value``. Falling
    back to "the last column, if every populated cell in it parses as a
    number" keeps this working for the ranked shapes that alias it differently,
    without ever mistaking a text column for a measure.
    """
    if not result.columns or not result.rows:
        return None
    last = len(result.columns) - 1
    if result.columns[last] == "value":
        return last
    cells = [row[last] for row in result.rows if row[last] is not None]
    if not cells or any(_decimal(cell) is None for cell in cells):
        return None
    return last


def _totals_across_an_overlapping_scope(result, plan) -> bool:
    """True when the groups are overlapping scopes rather than a partition."""
    scope = overlapping_scope(getattr(plan, "table_number", None))
    if scope is None:
        return False
    if scope.column not in result.columns:
        return False
    index = result.columns.index(scope.column)
    seen = {row[index] for row in result.rows}
    return len(seen & set(scope.values)) > 1


def _is_truncated(result) -> bool:
    """True when the executor said this result is a partial head of a larger set."""
    return any(
        "more exist" in warning for warning in (result.warnings or ())
    )


def _has_mixed_units(result) -> bool:
    """True when the executor said this result spans more than one unit."""
    return any(
        "unit" in warning.casefold() and "several" in warning.casefold()
        for warning in (result.warnings or ())
    )


def _group_label(result, row, measure_index: int, plan=None) -> str:
    """The row's group columns rendered as a single readable key."""
    parts = [
        f"{column}={row[index]}"
        for index, column in enumerate(display_columns(result, plan))
        if index != measure_index
    ]
    return ", ".join(parts) if parts else "all rows"


def roll_up(result, plan=None) -> list[RollUp]:
    """Every defensible deterministic figure derived from one executed result.

    ``plan`` defaults to the one the result carries; callers that hold the
    plan alongside the result (the channel's ``ExcelAnswer`` does) pass it
    explicitly rather than reaching through.

    Returns an empty list -- never a partial or a guess -- whenever the
    arithmetic would not be sound.
    """
    plan = plan if plan is not None else getattr(result, "plan", None)
    if getattr(plan, "operation", "aggregate") == "select":
        # A scan returns rows of attributes, not one comparable measure.
        return []
    if getattr(plan, "aggregate", "sum") not in TOTALLABLE_AGGREGATES:
        return []
    measure_index = _measure_index(result)
    if measure_index is None:
        return []
    if measure_index == 0:
        # No group columns: the result is already a single total.
        return []
    if len(result.rows) < MIN_ROLLUP_ROWS:
        return []
    if _has_mixed_units(result):
        return []
    if _is_truncated(result):
        # Totalling the eight largest groups and calling it "the total" is a
        # wrong number stated with full confidence.
        return []
    if _totals_across_an_overlapping_scope(result, plan):
        # Grouping by an overlapping column is legitimate -- each row is
        # labelled with the scope it belongs to -- but adding those rows up
        # double-counts the contained scope. Table 11's Territory row already
        # includes every HFTD dollar.
        return []

    values = [_decimal(row[measure_index]) for row in result.rows]
    present = [value for value in values if value is not None]
    if len(present) < MIN_ROLLUP_ROWS:
        return []

    unit = result.unit
    total = sum(present, Decimal(0))
    rolled = [
        RollUp(
            label="group_total",
            value=total,
            unit=unit,
            calculation=(
                f"sum of {len(present)} group values"
                + (
                    f" ({len(values) - len(present)} groups blank and excluded)"
                    if len(values) != len(present)
                    else ""
                )
            ),
        )
    ]
    rolled.extend(_shares(result, values, total, measure_index, unit, plan))
    rolled.extend(_year_subtotals(result, values, measure_index, unit))
    return rolled


def _shares(
    result,
    values: list[Decimal | None],
    total: Decimal,
    measure_index: int,
    unit,
    plan=None,
) -> list[RollUp]:
    """Each group's percentage of the total, when that is a real proportion."""
    if total <= 0 or len(result.rows) > MAX_SHARE_ROWS:
        return []
    if any(value < 0 for value in values if value is not None):
        return []
    shares = []
    for row, value in zip(result.rows, values, strict=False):
        if value is None:
            continue
        percent = (value / total * 100).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        label = _group_label(result, row, measure_index, plan)
        shares.append(
            RollUp(
                label=f"share[{label}]",
                value=percent,
                unit="%",
                calculation=(
                    f"{format(value.normalize(), 'f')} / "
                    f"{format(total.normalize(), 'f')} x 100"
                ),
            )
        )
    return shares


def _year_subtotals(
    result, values: list[Decimal | None], measure_index: int, unit
) -> list[RollUp]:
    """Per-year subtotals, when the result is grouped by year and something else.

    A breakdown of spend by initiative *and* year is most often asked as "what
    did each year cost", and that figure is nowhere in the returned rows.
    """
    if "reporting_year" not in result.columns:
        return []
    year_index = result.columns.index("reporting_year")
    if year_index == measure_index:
        return []
    # With year as the only group column each row already is its year's total.
    if len(result.columns) - 1 < 2:
        return []

    by_year: dict[Any, list[Decimal]] = {}
    for row, value in zip(result.rows, values, strict=False):
        if value is None:
            continue
        by_year.setdefault(row[year_index], []).append(value)
    if len(by_year) < 2:
        return []
    return [
        RollUp(
            label=f"subtotal[reporting_year={year}]",
            value=sum(group, Decimal(0)),
            unit=unit,
            calculation=f"sum of {len(group)} groups in {year}",
        )
        for year, group in sorted(by_year.items(), key=lambda item: str(item[0]))
    ]
