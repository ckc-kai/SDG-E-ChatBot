
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from artifact_tool import Workbook, SpreadsheetFile


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

GUIDELINES = {
    2023: (
        "v3.1",
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
        "fileid=53475&shareable=true",
    ),
    2024: (
        "v3.2",
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
        "fileid=56226&shareable=true",
    ),
    2025: (
        "v4.01",
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
        "fileid=58132&shareable=true",
    ),
}

V4_CHANGELOG_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "fileid=57874&shareable=true"
)

UNIFIED_HEADERS = [
    "record_id",
    "comparison_group_id",
    "metric_number",
    "distribution_metric_number_crosswalk",
    "transmission_metric_number_crosswalk",
    "legacy_metric_number_mapped",
    "source_legacy_metric_number_raw",
    "crosswalk_status",
    "metric_type",
    "source_metric_type_raw",
    "metric_name",
    "source_metric_name_raw",
    "line_type",
    "line_type_scope",
    "hftd_tier",
    "source_hftd_tier_raw",
    "unit_raw",
    "unit_canonical",
    "unit_crosswalk_status",
    "actual_value",
    "comments",
    "blank_meaning",
    "utility_id",
    "reporting_year",
    "reporting_quarter",
    "schema_version",
    "source_revision",
    "source_report_quarter",
    "source_file",
    "source_sheet",
    "source_row",
    "source_value_cell",
    "guideline_url",
]

COMPARABLE_HEADERS = [
    "comparison_group_id",
    "metric_type",
    "metric_name",
    "hftd_tier",
    "line_type",
    "actual_value",
    "aggregation_method",
    "component_count",
    "missing_component_count",
    "reporting_year",
    "reporting_quarter",
    "source_files",
]

CROSSWALK_HEADERS = [
    "comparison_group_id",
    "legacy_metric_number",
    "metric_type",
    "legacy_metric_name",
    "metric_name_canonical",
    "hftd_tier",
    "distribution_metric_number",
    "transmission_metric_number",
    "legacy_unit_example",
    "unit_canonical",
    "crosswalk_status",
    "crosswalk_note",
]

UNMAPPED_HEADERS = [
    "source_file",
    "source_sheet",
    "source_cell",
    "value",
    "reason",
]


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        output = " ".join(value.replace("\xa0", " ").split())
        return output or None
    return value


def normalized_text(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None
    return str(value).casefold()


def parse_number(value: Any) -> int | float | None:
    value = clean(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(f"Expected numeric Table 4 value, got {value!r}") from exc
        return int(parsed) if parsed.is_integer() else parsed
    raise TypeError(f"Unsupported numeric value type: {type(value).__name__}")


def column_index(letters: str) -> int:
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def column_letter(zero_based_index: int) -> str:
    value = zero_based_index + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(65 + remainder) + output
    return output


def parse_filename(path: Path) -> dict[str, Any] | None:
    match = re.search(r"SDGE_(\d{4})_Q([1-4])", path.name, re.IGNORECASE)
    if not match:
        return None

    revision_match = re.search(r"(?:_R|_Rev)(\d+)", path.name, re.IGNORECASE)
    return {
        "path": path,
        "name": path.name,
        "year": int(match.group(1)),
        "quarter": int(match.group(2)),
        "revision": int(revision_match.group(1)) if revision_match else 0,
    }


def discover_sources(input_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    candidates = []
    for path in input_dir.glob("*.xlsx"):
        parsed = parse_filename(path)
        if parsed is not None:
            candidates.append(parsed)

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for year in (2023, 2024, 2025):
        quarters = (4,) if year == 2023 else (1, 2, 3, 4)
        for quarter in quarters:
            matches = [
                candidate
                for candidate in candidates
                if candidate["year"] == year
                and candidate["quarter"] == quarter
            ]
            if not matches:
                raise FileNotFoundError(
                    f"No SDG&E {year} Q{quarter} workbook found in {input_dir}"
                )
            selected[(year, quarter)] = max(
                matches,
                key=lambda candidate: candidate["revision"],
            )

    return selected


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[Any]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{NS_MAIN}}}si"):
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{NS_MAIN}}}t")
                    )
                )

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_root = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_map = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationship_root
        }

        worksheet_target = None
        sheets = workbook_root.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            raise ValueError(f"No worksheets found in {path.name}")

        for sheet in sheets:
            if sheet.attrib["name"] == sheet_name:
                relationship_id = sheet.attrib[f"{{{NS_REL}}}id"]
                worksheet_target = relationship_map[relationship_id]
                break

        if worksheet_target is None:
            raise KeyError(f"{sheet_name!r} not found in {path.name}")

        worksheet_path = (
            "xl/" + worksheet_target.replace("../", "").lstrip("/")
        )
        worksheet_root = ET.fromstring(archive.read(worksheet_path))

        dimension = worksheet_root.find(f"{{{NS_MAIN}}}dimension")
        dimension_reference = (
            dimension.attrib.get("ref", "A1")
            if dimension is not None
            else "A1"
        )
        last_cell = dimension_reference.split(":")[-1]
        match = re.match(r"([A-Z]+)(\d+)", last_cell)
        if not match:
            raise ValueError(
                f"Unrecognized worksheet dimension {dimension_reference!r}"
            )

        max_columns = column_index(match.group(1)) + 1
        max_rows = int(match.group(2))
        values = [[None] * max_columns for _ in range(max_rows)]

        sheet_data = worksheet_root.find(f"{{{NS_MAIN}}}sheetData")
        if sheet_data is None:
            return values

        for row in sheet_data:
            row_index = int(row.attrib["r"]) - 1
            for cell in row:
                address_match = re.match(r"([A-Z]+)(\d+)", cell.attrib["r"])
                if not address_match:
                    continue

                column = column_index(address_match.group(1))
                cell_type = cell.attrib.get("t")

                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.iter(f"{{{NS_MAIN}}}t")
                    )
                else:
                    value_node = cell.find(f"{{{NS_MAIN}}}v")
                    if value_node is None:
                        value = None
                    else:
                        raw = value_node.text
                        if cell_type == "s":
                            value = shared_strings[int(raw)]
                        elif cell_type == "b":
                            value = raw == "1"
                        elif cell_type in {"str", "e"}:
                            value = raw
                        else:
                            try:
                                numeric = float(raw)
                                value = (
                                    int(numeric)
                                    if numeric.is_integer()
                                    else numeric
                                )
                            except (TypeError, ValueError):
                                value = raw

                values[row_index][column] = value

        return values


def normalize_metric_type(value: Any) -> Any:
    return clean(value)


def normalize_hftd_tier(value: Any) -> Any:
    value = clean(value)
    if value is None:
        return None

    normalized = value.casefold()
    if normalized in {"hftd tier 2", "hftd tier 2"}:
        return "HFTD Tier 2"
    if normalized in {"hftd tier 3", "hftd tier 3"}:
        return "HFTD Tier 3"
    if normalized in {"non-hftd", "non- hftd"}:
        return "Non-HFTD"

    raise ValueError(f"Unrecognized HFTD tier: {value!r}")


def parse_legacy_metric_name(metric_name: str) -> tuple[str, str | None]:
    metric_name = clean(metric_name)
    if metric_name is None:
        raise ValueError("Legacy metric name is blank")

    patterns = (
        (r"\s*-\s*HFTD tier 2$", "HFTD Tier 2"),
        (r"\s*-\s*HFTD tier 3$", "HFTD Tier 3"),
        (r"\s*-\s*non-HFTD$", "Non-HFTD"),
    )
    for pattern, hftd in patterns:
        if re.search(pattern, metric_name, flags=re.IGNORECASE):
            base = re.sub(
                pattern,
                "",
                metric_name,
                flags=re.IGNORECASE,
            )
            return clean(base), hftd

    # The final v3 row is an optional "Other" placeholder without an HFTD suffix.
    if metric_name.startswith("Other relevant weather pattern metrics tracked"):
        return metric_name, None

    raise ValueError(
        f"Unable to parse HFTD tier from legacy metric name {metric_name!r}"
    )


def comparison_group_id(
    metric_type: str,
    metric_name: str,
    hftd_tier: str | None,
) -> str:
    payload = json.dumps(
        [
            clean(metric_type),
            clean(metric_name),
            clean(hftd_tier),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "T4-" + hashlib.sha1(payload).hexdigest()[:14]


def record_id(
    comparison_id: str,
    line_type: str,
    year: int,
    quarter: int,
    source_file: str,
    source_row: int,
) -> str:
    payload = json.dumps(
        [
            comparison_id,
            line_type,
            year,
            quarter,
            source_file,
            source_row,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "T4R-" + hashlib.sha1(payload).hexdigest()[:16]


def parse_legacy_template(
    values: list[list[Any]],
) -> list[dict[str, Any]]:
    records = []
    metric_type = None

    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        if clean(row[2]):
            metric_type = clean(row[2])

        legacy_metric_number = clean(row[3])
        source_metric_name = clean(row[4])
        if source_metric_name is None:
            continue

        metric_name, hftd_tier = parse_legacy_metric_name(
            source_metric_name
        )
        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "source_metric_type_raw": metric_type,
                "source_legacy_metric_number_raw": legacy_metric_number,
                "source_metric_name_raw": source_metric_name,
                "metric_name_legacy_base": metric_name,
                "hftd_tier": hftd_tier,
                "unit_raw": clean(row[-3]),
                "comments": clean(row[-2]),
                "blank_meaning": clean(row[-1]),
            }
        )

    if len(records) != 13:
        raise AssertionError(
            f"Expected 13 v3 Table 4 rows, found {len(records)}"
        )
    return records


def locate_2023_actual_columns(
    values: list[list[Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for column in range(len(values[8])):
        year = clean(values[8][column])
        quarter = clean(values[7][column])
        if year == 2023 and quarter in {1, 2, 3, 4}:
            result[int(quarter)] = column

    if sorted(result) != [1, 2, 3, 4]:
        raise AssertionError(
            f"Expected 2023 Q1-Q4 actual columns; found {result}"
        )
    return result


def locate_2024_actual_column(
    values: list[list[Any]],
    expected_quarter: int,
) -> int:
    matching_columns = [
        column
        for column, value in enumerate(values[8])
        if clean(value) == 2024
    ]
    if len(matching_columns) != 1:
        raise AssertionError(
            f"Expected one 2024 actual column; found {matching_columns}"
        )

    column = matching_columns[0]
    quarter_header = clean(values[7][column])
    if quarter_header != f"Q{expected_quarter}":
        raise AssertionError(
            f"Expected Q{expected_quarter}, found {quarter_header!r}"
        )
    return column


def parse_v4_template(
    values: list[list[Any]],
    source_file: str,
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    expected_headers = [
        "METRIC NUMBER",
        "METRIC TYPE",
        "METRIC NAME",
        "LINE TYPE",
        "HFTD TIER",
        "UNIT(S)",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "REPORTING QUARTER",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in values[0][:12]]
    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Table 4 header does not match Data Guidelines v4.01.\n"
            f"Expected: {expected_headers}\n"
            f"Found: {actual_headers}"
        )

    records = []
    unmapped_cells: list[list[Any]] = []

    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        metric_name = clean(row[2])
        if metric_name is None:
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_number": int(row[0]),
                "metric_type": clean(row[1]),
                "metric_name": metric_name,
                "line_type": clean(row[3]),
                "hftd_tier": normalize_hftd_tier(row[4]),
                "unit_raw": clean(row[5]),
                "comments": clean(row[6]),
                "blank_meaning": clean(row[7]),
                "utility_id": clean(row[8]),
                "reporting_year": int(row[9]),
                "reporting_quarter": int(row[10]),
                "actual_value": parse_number(row[11]),
            }
        )

        # Preserve any populated cells outside the official 12-column schema.
        for zero_based_column, value in enumerate(row[12:], start=12):
            if clean(value) is not None:
                unmapped_cells.append(
                    [
                        source_file,
                        "Table 4",
                        (
                            f"{column_letter(zero_based_column)}"
                            f"{zero_based_row + 1}"
                        ),
                        value,
                        "Populated cell outside the official v4.01 Table 4 schema",
                    ]
                )

    if len(records) != 24:
        raise AssertionError(
            f"Expected 24 v4.01 Table 4 rows, found {len(records)}"
        )

    return records, unmapped_cells


def build_crosswalk(
    legacy_records: list[dict[str, Any]],
    v4_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    v4_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in v4_records:
        key = (
            normalized_text(record["metric_name"]),
            record["hftd_tier"],
        )
        v4_groups[key].append(record)

    crosswalk = []
    by_template_index = {}

    for legacy in legacy_records:
        legacy_base = legacy["metric_name_legacy_base"]
        hftd_tier = legacy["hftd_tier"]

        if hftd_tier is None:
            item = {
                "template_index": legacy["template_index"],
                "comparison_group_id": comparison_group_id(
                    legacy["source_metric_type_raw"],
                    legacy_base,
                    None,
                ),
                "legacy_metric_number": legacy[
                    "source_legacy_metric_number_raw"
                ],
                "metric_type": normalize_metric_type(
                    legacy["source_metric_type_raw"]
                ),
                "legacy_metric_name": legacy[
                    "source_metric_name_raw"
                ],
                "metric_name_canonical": legacy_base,
                "hftd_tier": None,
                "distribution_metric_number": None,
                "transmission_metric_number": None,
                "legacy_unit_example": legacy["unit_raw"],
                "unit_canonical": legacy["unit_raw"],
                "crosswalk_status": "legacy_only_other_placeholder",
                "crosswalk_note": (
                    "The optional legacy Other row has no prepopulated "
                    "v4.01 counterpart."
                ),
            }
            crosswalk.append(item)
            by_template_index[legacy["template_index"]] = item
            continue

        # Legacy names include an HFTD suffix; v4.01 moves HFTD into a field.
        key = (normalized_text(legacy_base), hftd_tier)
        matches = v4_groups.get(key, [])

        if len(matches) != 2:
            raise AssertionError(
                f"Expected Distribution and Transmission matches for "
                f"{legacy_base!r}, {hftd_tier}; found {len(matches)}"
            )

        by_line_type = {
            record["line_type"]: record
            for record in matches
        }
        if set(by_line_type) != {"Distribution", "Transmission"}:
            raise AssertionError(
                f"Unexpected line types for {legacy_base!r}, {hftd_tier}: "
                f"{sorted(by_line_type)}"
            )

        distribution = by_line_type["Distribution"]
        transmission = by_line_type["Transmission"]

        if (
            normalized_text(distribution["metric_type"])
            != normalized_text(transmission["metric_type"])
        ):
            raise AssertionError("Distribution/Transmission metric types differ")

        if distribution["unit_raw"] != transmission["unit_raw"]:
            raise AssertionError("Distribution/Transmission units differ")

        if (
            normalized_text(legacy["source_metric_type_raw"])
            != normalized_text(distribution["metric_type"])
        ):
            raise AssertionError(
                "Legacy/v4.01 metric type mismatch for "
                f"{legacy_base!r}, {hftd_tier}"
            )

        item = {
            "template_index": legacy["template_index"],
            "comparison_group_id": comparison_group_id(
                distribution["metric_type"],
                distribution["metric_name"],
                hftd_tier,
            ),
            "legacy_metric_number": legacy[
                "source_legacy_metric_number_raw"
            ],
            "metric_type": distribution["metric_type"],
            "legacy_metric_name": legacy["source_metric_name_raw"],
            "metric_name_canonical": distribution["metric_name"],
            "hftd_tier": hftd_tier,
            "distribution_metric_number": distribution["metric_number"],
            "transmission_metric_number": transmission["metric_number"],
            "legacy_unit_example": legacy["unit_raw"],
            "unit_canonical": distribution["unit_raw"],
            "crosswalk_status": "legacy_all_line_types_to_v4_line_type_pair",
            "crosswalk_note": (
                "v4.0 added LINE TYPE and doubled Table 4 to report "
                "Distribution and Transmission separately. The legacy "
                "reported value is preserved as an all-line-types aggregate "
                "and is not copied into either line type."
            ),
        }
        crosswalk.append(item)
        by_template_index[legacy["template_index"]] = item

    if len(crosswalk) != 13:
        raise AssertionError("Expected 13 Table 4 crosswalk rows")

    return crosswalk, by_template_index


def validate_quarterly_templates(
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[Any]],
]:
    legacy_2023 = parse_legacy_template(loaded[(2023, 4)])
    reference_2024 = parse_legacy_template(loaded[(2024, 4)])

    legacy_key_2023 = [
        (
            normalized_text(record["source_metric_type_raw"]),
            normalized_text(record["metric_name_legacy_base"]),
            record["hftd_tier"],
        )
        for record in legacy_2023
    ]
    legacy_key_2024 = [
        (
            normalized_text(record["source_metric_type_raw"]),
            normalized_text(record["metric_name_legacy_base"]),
            record["hftd_tier"],
        )
        for record in reference_2024
    ]
    if legacy_key_2023 != legacy_key_2024:
        raise AssertionError(
            "The 2023 v3.1 and 2024 v3.2 Table 4 metric structures differ."
        )

    for quarter in (1, 2, 3, 4):
        current = parse_legacy_template(loaded[(2024, quarter)])
        current_key = [
            (
                normalized_text(record["source_metric_type_raw"]),
                normalized_text(record["metric_name_legacy_base"]),
                record["hftd_tier"],
            )
            for record in current
        ]
        if current_key != legacy_key_2024:
            raise AssertionError(
                f"2024 Q{quarter} Table 4 schema differs from Q4."
            )

    v4_reference, reference_unmapped = parse_v4_template(
        loaded[(2025, 4)],
        "2025 Q4 reference",
    )
    stable_fields = (
        "metric_number",
        "metric_type",
        "metric_name",
        "line_type",
        "hftd_tier",
        "unit_raw",
        "utility_id",
    )
    v4_key = [
        tuple(record[field] for field in stable_fields)
        for record in v4_reference
    ]

    all_unmapped = list(reference_unmapped)
    for quarter in (1, 2, 3, 4):
        current, unmapped = parse_v4_template(
            loaded[(2025, quarter)],
            f"2025 Q{quarter}",
        )
        current_key = [
            tuple(record[field] for field in stable_fields)
            for record in current
        ]
        if current_key != v4_key:
            raise AssertionError(
                f"2025 Q{quarter} Table 4 schema differs from Q4."
            )
        if quarter != 4:
            all_unmapped.extend(unmapped)

    return legacy_2023, v4_reference, all_unmapped


def build_unified_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], list[list[Any]]],
    crosswalk_by_index: dict[int, dict[str, Any]],
) -> list[list[Any]]:
    output = []

    # 2023: Q4 workbook contains Q1-Q4 actual columns.
    source = selected[(2023, 4)]
    values = loaded[(2023, 4)]
    records = parse_legacy_template(values)
    actual_columns = locate_2023_actual_columns(values)

    for record in records:
        mapping = crosswalk_by_index[record["template_index"]]
        for quarter, value_column in sorted(actual_columns.items()):
            actual_value = parse_number(
                values[record["source_row"] - 1][value_column]
            )
            line_type = "All line types (legacy aggregate)"
            comparison_id = mapping["comparison_group_id"]
            output.append(
                [
                    record_id(
                        comparison_id,
                        line_type,
                        2023,
                        quarter,
                        source["name"],
                        record["source_row"],
                    ),
                    comparison_id,
                    None,
                    mapping["distribution_metric_number"],
                    mapping["transmission_metric_number"],
                    mapping["legacy_metric_number"],
                    record["source_legacy_metric_number_raw"],
                    mapping["crosswalk_status"],
                    mapping["metric_type"],
                    record["source_metric_type_raw"],
                    mapping["metric_name_canonical"],
                    record["source_metric_name_raw"],
                    line_type,
                    "reported_all_line_types",
                    mapping["hftd_tier"],
                    mapping["hftd_tier"],
                    record["unit_raw"],
                    mapping["unit_canonical"],
                    (
                        "same_measurement_definition_simplified_label"
                        if mapping["hftd_tier"] is not None
                        else "legacy_only"
                    ),
                    actual_value,
                    record["comments"],
                    record["blank_meaning"],
                    "SDG&E",
                    2023,
                    quarter,
                    GUIDELINES[2023][0],
                    source["revision"],
                    4,
                    source["name"],
                    "Table 4",
                    record["source_row"],
                    (
                        f"{column_letter(value_column)}"
                        f"{record['source_row']}"
                    ),
                    GUIDELINES[2023][1],
                ]
            )

    # 2024: each workbook contributes its subject quarter.
    for quarter in (1, 2, 3, 4):
        source = selected[(2024, quarter)]
        values = loaded[(2024, quarter)]
        records = parse_legacy_template(values)
        value_column = locate_2024_actual_column(values, quarter)

        for record in records:
            mapping = crosswalk_by_index[record["template_index"]]
            actual_value = parse_number(
                values[record["source_row"] - 1][value_column]
            )
            line_type = "All line types (legacy aggregate)"
            comparison_id = mapping["comparison_group_id"]
            output.append(
                [
                    record_id(
                        comparison_id,
                        line_type,
                        2024,
                        quarter,
                        source["name"],
                        record["source_row"],
                    ),
                    comparison_id,
                    None,
                    mapping["distribution_metric_number"],
                    mapping["transmission_metric_number"],
                    mapping["legacy_metric_number"],
                    record["source_legacy_metric_number_raw"],
                    mapping["crosswalk_status"],
                    mapping["metric_type"],
                    record["source_metric_type_raw"],
                    mapping["metric_name_canonical"],
                    record["source_metric_name_raw"],
                    line_type,
                    "reported_all_line_types",
                    mapping["hftd_tier"],
                    mapping["hftd_tier"],
                    record["unit_raw"],
                    mapping["unit_canonical"],
                    (
                        "same_measurement_definition_simplified_label"
                        if mapping["hftd_tier"] is not None
                        else "legacy_only"
                    ),
                    actual_value,
                    record["comments"],
                    record["blank_meaning"],
                    "SDG&E",
                    2024,
                    quarter,
                    GUIDELINES[2024][0],
                    source["revision"],
                    quarter,
                    source["name"],
                    "Table 4",
                    record["source_row"],
                    (
                        f"{column_letter(value_column)}"
                        f"{record['source_row']}"
                    ),
                    GUIDELINES[2024][1],
                ]
            )

    # Map v4 records to the legacy comparison groups by metric/HFTD.
    mapping_by_v4_metric_number = {}
    for mapping in crosswalk_by_index.values():
        if mapping["distribution_metric_number"] is not None:
            mapping_by_v4_metric_number[
                mapping["distribution_metric_number"]
            ] = mapping
            mapping_by_v4_metric_number[
                mapping["transmission_metric_number"]
            ] = mapping

    # 2025: explicit Distribution/Transmission rows.
    for quarter in (1, 2, 3, 4):
        source = selected[(2025, quarter)]
        records, _ = parse_v4_template(
            loaded[(2025, quarter)],
            source["name"],
        )

        for record in records:
            if record["reporting_year"] != 2025:
                raise AssertionError(
                    f"Unexpected reporting year in {source['name']} "
                    f"row {record['source_row']}"
                )
            if record["reporting_quarter"] != quarter:
                raise AssertionError(
                    f"Unexpected reporting quarter in {source['name']} "
                    f"row {record['source_row']}"
                )

            mapping = mapping_by_v4_metric_number[record["metric_number"]]
            comparison_id = mapping["comparison_group_id"]
            output.append(
                [
                    record_id(
                        comparison_id,
                        record["line_type"],
                        2025,
                        quarter,
                        source["name"],
                        record["source_row"],
                    ),
                    comparison_id,
                    record["metric_number"],
                    mapping["distribution_metric_number"],
                    mapping["transmission_metric_number"],
                    mapping["legacy_metric_number"],
                    None,
                    "v4_line_type_specific",
                    record["metric_type"],
                    record["metric_type"],
                    record["metric_name"],
                    record["metric_name"],
                    record["line_type"],
                    "line_type_specific",
                    record["hftd_tier"],
                    record["hftd_tier"],
                    record["unit_raw"],
                    record["unit_raw"],
                    "v4_native",
                    record["actual_value"],
                    record["comments"],
                    record["blank_meaning"],
                    record["utility_id"],
                    2025,
                    quarter,
                    GUIDELINES[2025][0],
                    source["revision"],
                    quarter,
                    source["name"],
                    "Table 4",
                    record["source_row"],
                    f"L{record['source_row']}",
                    GUIDELINES[2025][1],
                ]
            )

    expected_rows = (13 * 4) + (13 * 4) + (24 * 4)
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified rows; got {len(output)}"
        )

    return output


def build_comparable_rows(
    unified_rows: list[list[Any]],
) -> list[list[Any]]:
    index = {name: position for position, name in enumerate(UNIFIED_HEADERS)}
    comparable = []

    # Legacy rows are already reported across all line types.
    for row in unified_rows:
        if row[index["line_type_scope"]] != "reported_all_line_types":
            continue
        if row[index["hftd_tier"]] is None:
            # Exclude the optional Other placeholder from comparable metrics.
            continue

        comparable.append(
            [
                row[index["comparison_group_id"]],
                row[index["metric_type"]],
                row[index["metric_name"]],
                row[index["hftd_tier"]],
                "All line types",
                row[index["actual_value"]],
                "reported_legacy_all_line_types",
                1,
                0 if row[index["actual_value"]] is not None else 1,
                row[index["reporting_year"]],
                row[index["reporting_quarter"]],
                row[index["source_file"]],
            ]
        )

    # Sum the two v4.01 line types for a comparable all-line-types view.
    groups: dict[
        tuple[str, int, int],
        list[list[Any]],
    ] = defaultdict(list)
    for row in unified_rows:
        if row[index["line_type_scope"]] != "line_type_specific":
            continue
        key = (
            row[index["comparison_group_id"]],
            row[index["reporting_year"]],
            row[index["reporting_quarter"]],
        )
        groups[key].append(row)

    for (comparison_id, year, quarter), rows in sorted(groups.items()):
        if len(rows) != 2:
            raise AssertionError(
                f"Expected two 2025 line-type components for "
                f"{comparison_id}, {year} Q{quarter}; got {len(rows)}"
            )

        line_types = {row[index["line_type"]] for row in rows}
        if line_types != {"Distribution", "Transmission"}:
            raise AssertionError(
                f"Unexpected line-type components: {line_types}"
            )

        values = [row[index["actual_value"]] for row in rows]
        missing = sum(value is None for value in values)
        aggregate_value = (
            sum(value for value in values if value is not None)
            if missing == 0
            else None
        )
        source_files = "; ".join(
            sorted({row[index["source_file"]] for row in rows})
        )

        comparable.append(
            [
                comparison_id,
                rows[0][index["metric_type"]],
                rows[0][index["metric_name"]],
                rows[0][index["hftd_tier"]],
                "All line types",
                aggregate_value,
                "derived_sum_distribution_plus_transmission",
                2,
                missing,
                year,
                quarter,
                source_files,
            ]
        )

    expected_rows = 12 * 4 * 3
    if len(comparable) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} comparable rows; got {len(comparable)}"
        )

    comparable.sort(
        key=lambda row: (
            row[9],
            row[10],
            row[1],
            row[3],
        )
    )
    return comparable


def build_crosswalk_rows(
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    return [
        [
            item["comparison_group_id"],
            item["legacy_metric_number"],
            item["metric_type"],
            item["legacy_metric_name"],
            item["metric_name_canonical"],
            item["hftd_tier"],
            item["distribution_metric_number"],
            item["transmission_metric_number"],
            item["legacy_unit_example"],
            item["unit_canonical"],
            item["crosswalk_status"],
            item["crosswalk_note"],
        ]
        for item in crosswalk
    ]


def write_csv(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_rows(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    *,
    chunk_size: int = 300,
) -> None:
    sheet.get_range_by_indexes(0, 0, 1, len(headers)).values = [headers]
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        sheet.get_range_by_indexes(
            start + 1,
            0,
            len(chunk),
            len(headers),
        ).values = chunk


def format_sheet(
    sheet: Any,
    headers: list[str],
    row_count: int,
    *,
    freeze_columns: int,
) -> None:
    last_column = column_letter(len(headers) - 1)
    last_row = row_count + 1

    sheet.get_range(f"A1:{last_column}1").format = {
        "fill": "#0F766E",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 34,
    }
    sheet.freeze_panes.freeze_rows(1)
    sheet.freeze_panes.freeze_columns(freeze_columns)

    # Moderate widths, with larger text/source fields.
    for column in range(len(headers)):
        letter = column_letter(column)
        header = headers[column]
        width = 18
        if header in {
            "metric_type",
            "source_metric_type_raw",
            "metric_name",
            "source_metric_name_raw",
            "unit_raw",
            "unit_canonical",
            "comments",
            "blank_meaning",
            "crosswalk_note",
            "source_files",
            "guideline_url",
        }:
            width = 42
        elif header in {
            "source_file",
            "crosswalk_status",
            "unit_crosswalk_status",
            "aggregation_method",
        }:
            width = 34
        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width

    sheet.get_range(
        f"A1:{last_column}{last_row}"
    ).format.wrap_text = True


def build_workbook(
    output_path: Path,
    unified_rows: list[list[Any]],
    comparable_rows: list[list[Any]],
    crosswalk_rows: list[list[Any]],
    unmapped_rows: list[list[Any]],
    selected: dict[tuple[int, int], dict[str, Any]],
) -> None:
    workbook = Workbook.create()

    readme = workbook.worksheets.add("README")
    unified_sheet = workbook.worksheets.add("Unified Actuals")
    comparable_sheet = workbook.worksheets.add("Comparable Aggregates")
    crosswalk_sheet = workbook.worksheets.add("Metric Crosswalk")
    unmapped_sheet = workbook.worksheets.add("Unmapped Cells")

    readme_rows = [
        ["SDG&E Table 4 Unified Dataset, 2023–2025", "", "", ""],
        ["Unified observations", len(unified_rows), "Comparable rows", len(comparable_rows)],
        [
            "Projection treatment",
            "No projection values are included. Table 4 quarterly observations "
            "are actuals; unused future placeholder columns are ignored.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "All 2023 Q1–Q4 values are read from the selected Q4 v3.1 workbook.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "Each highest-revision v3.2 workbook contributes its subject quarter.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Each highest-revision v4.01 workbook contributes its explicit "
            "Distribution and Transmission observations.",
            "",
            "",
        ],
        [
            "Critical schema change",
            "v4.0 added LINE TYPE and doubled Table 4 to report Distribution "
            "and Transmission separately.",
            "Legacy handling",
            "2023–2024 values remain all-line-types aggregates; they are not "
            "duplicated into Distribution and Transmission.",
        ],
        [
            "Comparable view",
            "Comparable Aggregates preserves legacy values and derives 2025 "
            "all-line-types totals by summing Distribution + Transmission.",
            "",
            "",
        ],
        [
            "Unit change",
            "v4.0 shortened UNIT(S) text and moved calculations to the glossary. "
            "The measurement remains circuit mile days; no numeric conversion is applied.",
            "",
            "",
        ],
        [
            "Optional Other row",
            "The legacy 'Other' placeholder is retained in Unified Actuals but "
            "excluded from Comparable Aggregates because v4.01 has no prepopulated counterpart.",
            "",
            "",
        ],
        [
            "Unmapped source cells",
            len(unmapped_rows),
            "Treatment",
            "Preserved on the Unmapped Cells sheet rather than silently discarded.",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified point"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 4 requires circuit mile days split by HFTD Tier 2, Tier 3, and Non-HFTD.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Table 4 retains the same required weather metrics and HFTD breakdown.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Quarterly Table 4 uses explicit reporting year, quarter, and actual value fields.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "LINE TYPE added; workbook doubled. UNIT(S) labels simplified.",
        ],
        ["", "", "", ""],
        ["Selected source file", "Year", "Quarter", "Revision"],
    ]
    for _, source in sorted(selected.items()):
        readme_rows.append(
            [
                source["name"],
                source["year"],
                source["quarter"],
                source["revision"],
            ]
        )

    readme.get_range_by_indexes(
        0, 0, len(readme_rows), 4
    ).values = readme_rows
    readme.merge_cells("A1:D1")
    readme.get_range("A1:D1").format = {
        "fill": "#0F766E",
        "font": {
            "bold": True,
            "color": "#FFFFFF",
            "font_size": 15,
        },
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "row_height": 30,
    }
    for row_number, row in enumerate(readme_rows, start=1):
        if row[0] in {"Official source", "Selected source file"}:
            readme.get_range(
                f"A{row_number}:D{row_number}"
            ).format = {
                "fill": "#1D4ED8",
                "font": {"bold": True, "color": "#FFFFFF"},
                "horizontal_alignment": "center",
                "vertical_alignment": "center",
                "wrap_text": True,
            }

    readme.get_range(
        f"A1:D{len(readme_rows)}"
    ).format.wrap_text = True
    for column, width in zip(("A", "B", "C", "D"), (34, 64, 68, 58)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(unified_sheet, UNIFIED_HEADERS, unified_rows)
    format_sheet(
        unified_sheet,
        UNIFIED_HEADERS,
        len(unified_rows),
        freeze_columns=8,
    )
    actual_column = column_letter(UNIFIED_HEADERS.index("actual_value"))
    unified_sheet.get_range(
        f"{actual_column}2:{actual_column}{len(unified_rows) + 1}"
    ).format.number_format = "0.########"

    write_rows(
        comparable_sheet,
        COMPARABLE_HEADERS,
        comparable_rows,
    )
    format_sheet(
        comparable_sheet,
        COMPARABLE_HEADERS,
        len(comparable_rows),
        freeze_columns=5,
    )
    comparable_value_column = column_letter(
        COMPARABLE_HEADERS.index("actual_value")
    )
    comparable_sheet.get_range(
        f"{comparable_value_column}2:"
        f"{comparable_value_column}{len(comparable_rows) + 1}"
    ).format.number_format = "0.########"

    write_rows(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        crosswalk_rows,
    )
    format_sheet(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        len(crosswalk_rows),
        freeze_columns=5,
    )

    write_rows(
        unmapped_sheet,
        UNMAPPED_HEADERS,
        unmapped_rows,
    )
    format_sheet(
        unmapped_sheet,
        UNMAPPED_HEADERS,
        len(unmapped_rows),
        freeze_columns=3,
    )
    if unmapped_rows:
        unmapped_sheet.get_range(
            f"A2:E{len(unmapped_rows) + 1}"
        ).format = {
            "fill": "#FEF3C7",
            "font": {"color": "#92400E"},
            "wrap_text": True,
        }

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 4 actuals for 2023-2025. "
            "The converter preserves the v4.0 LINE TYPE schema change "
            "and creates a separate cross-year comparable aggregate view."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing the source SDG&E XLSX files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table4_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 4")
        for key, source in selected.items()
    }

    legacy_reference, v4_reference, unmapped_rows = (
        validate_quarterly_templates(loaded)
    )
    crosswalk, crosswalk_by_index = build_crosswalk(
        legacy_reference,
        v4_reference,
    )

    unified_rows = build_unified_rows(
        selected,
        loaded,
        crosswalk_by_index,
    )
    comparable_rows = build_comparable_rows(unified_rows)
    metric_crosswalk_rows = build_crosswalk_rows(crosswalk)

    unified_csv = (
        output_dir / "sdge_table4_2023_2025_unified_actuals.csv"
    )
    comparable_csv = (
        output_dir / "sdge_table4_2023_2025_comparable_aggregates.csv"
    )
    crosswalk_csv = (
        output_dir / "sdge_table4_metric_crosswalk.csv"
    )
    unmapped_csv = (
        output_dir / "sdge_table4_unmapped_cells.csv"
    )
    workbook_path = (
        output_dir / "sdge_table4_2023_2025_unified.xlsx"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(
        comparable_csv,
        COMPARABLE_HEADERS,
        comparable_rows,
    )
    write_csv(
        crosswalk_csv,
        CROSSWALK_HEADERS,
        metric_crosswalk_rows,
    )
    write_csv(
        unmapped_csv,
        UNMAPPED_HEADERS,
        unmapped_rows,
    )

    summary = {
        "unified_rows": len(unified_rows),
        "comparable_rows": len(comparable_rows),
        "crosswalk_rows": len(metric_crosswalk_rows),
        "unmapped_cells": len(unmapped_rows),
        "crosswalk_status_counts": dict(
            Counter(
                item["crosswalk_status"]
                for item in crosswalk
            )
        ),
        "unexpected_schema_changes": 0,
        "sources": [
            {
                "name": source["name"],
                "year": source["year"],
                "quarter": source["quarter"],
                "revision": source["revision"],
            }
            for _, source in sorted(selected.items())
        ],
    }
    validation_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        comparable_rows,
        metric_crosswalk_rows,
        unmapped_rows,
        selected,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {comparable_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {unmapped_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(f"Comparable rows: {len(comparable_rows)}")
    print(f"Unmapped source cells preserved: {len(unmapped_rows)}")


if __name__ == "__main__":
    main()
