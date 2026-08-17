"""Typed, deterministic calculations over provenance-bearing operands."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal, Mapping


Resource = Literal["pdf", "excel"]
Operation = Literal["ratio_percent", "difference", "sum", "change_percent"]


class CalculationError(ValueError):
    """The requested calculation is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True)
class FactRequirement:
    id: str
    source: Resource
    metric: str
    period: str
    unit: str | None = None
    document_role: str | None = None
    table_role: str | None = None


@dataclass(frozen=True)
class CalculationSpec:
    operation: Operation
    left_ref: str
    right_ref: str
    require_same_period: bool | None = None


@dataclass(frozen=True)
class TypedComputationPlan:
    facts: tuple[FactRequirement, ...]
    calculation: CalculationSpec

    def __post_init__(self) -> None:
        ids = [fact.id for fact in self.facts]
        if not ids or len(ids) != len(set(ids)):
            raise CalculationError("fact ids must be non-empty and unique")
        unknown = {
            self.calculation.left_ref,
            self.calculation.right_ref,
        } - set(ids)
        if unknown:
            raise CalculationError(f"calculation references unknown facts: {sorted(unknown)}")


@dataclass(frozen=True)
class EvidenceOperand:
    value: Decimal
    unit: str | None
    period: str
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalculationResult:
    value: Decimal
    unit: str | None
    expression: str
    contributing_sources: tuple[str, ...]


def _operand(
    reference: str,
    operands: Mapping[str, EvidenceOperand],
) -> EvidenceOperand:
    value = operands.get(reference)
    if value is None:
        raise CalculationError(f"missing operand {reference!r}")
    if not isinstance(value.value, Decimal):
        try:
            return EvidenceOperand(
                Decimal(str(value.value)), value.unit, value.period, value.provenance
            )
        except (InvalidOperation, ValueError) as exc:
            raise CalculationError(f"operand {reference!r} is not numeric") from exc
    return value


def _validate_requirement(
    requirement: FactRequirement,
    operand: EvidenceOperand,
) -> None:
    if operand.period != requirement.period:
        raise CalculationError(
            f"operand {requirement.id!r} period {operand.period!r} does not match "
            f"planned period {requirement.period!r}"
        )
    if (
        requirement.unit
        and operand.unit
        and operand.unit.casefold() != requirement.unit.casefold()
    ):
        raise CalculationError(
            f"operand {requirement.id!r} unit {operand.unit!r} does not match "
            f"planned unit {requirement.unit!r}"
        )
    if not operand.provenance:
        raise CalculationError(f"operand {requirement.id!r} has no provenance")


def _validate_alignment(
    spec: CalculationSpec,
    left: EvidenceOperand,
    right: EvidenceOperand,
) -> None:
    same_period = (
        spec.require_same_period
        if spec.require_same_period is not None
        else spec.operation in {"ratio_percent", "sum"}
    )
    if same_period and left.period != right.period:
        raise CalculationError(
            f"operand period mismatch: {left.period!r} != {right.period!r}"
        )
    if spec.operation in {"ratio_percent", "difference", "sum", "change_percent"}:
        if left.unit and right.unit and left.unit.casefold() != right.unit.casefold():
            raise CalculationError(
                f"operand unit mismatch: {left.unit!r} != {right.unit!r}"
            )


def execute_calculation(
    plan: TypedComputationPlan,
    operands: Mapping[str, EvidenceOperand],
) -> CalculationResult:
    """Validate and execute an allow-listed two-operand calculation."""
    spec = plan.calculation
    left = _operand(spec.left_ref, operands)
    right = _operand(spec.right_ref, operands)
    requirements = {fact.id: fact for fact in plan.facts}
    _validate_requirement(requirements[spec.left_ref], left)
    _validate_requirement(requirements[spec.right_ref], right)
    _validate_alignment(spec, left, right)

    if spec.operation == "ratio_percent":
        if right.value == 0:
            raise CalculationError("zero denominator")
        value = (left.value / right.value * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        unit = "percent"
        expression = f"({left.value} / {right.value}) * 100"
    elif spec.operation == "change_percent":
        if right.value == 0:
            raise CalculationError("zero denominator")
        value = ((left.value - right.value) / right.value * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        unit = "percent"
        expression = f"(({left.value} - {right.value}) / {right.value}) * 100"
    elif spec.operation == "difference":
        value = left.value - right.value
        unit = left.unit or right.unit
        expression = f"{left.value} - {right.value}"
    elif spec.operation == "sum":
        value = left.value + right.value
        unit = left.unit or right.unit
        expression = f"{left.value} + {right.value}"
    else:  # Defensive even though the dataclass is statically constrained.
        raise CalculationError(f"unsupported operation {spec.operation!r}")

    sources = tuple(dict.fromkeys((*left.provenance, *right.provenance)))
    return CalculationResult(value, unit, expression, sources)
