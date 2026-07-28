
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
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

V3_2_CHANGELOG_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=Data+Guidelines&fileid=56230&shareable=true"
)
V4_CHANGELOG_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "fileid=57874&shareable=true"
)

UNIFIED_HEADERS = [
    "record_id",
    "metric_number",
    "row_aligned_v4_metric_number",
    "legacy_metric_number_mapped",
    "source_legacy_metric_number_raw",
    "legacy_metric_number_status",
    "semantic_crosswalk_status",
    "metric_type",
    "source_metric_type_raw",
    "metric_name",
    "source_metric_name_raw",
    "wind_warning_status",
    "source_wind_warning_status_raw",
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

DIRECT_COMPARABLE_HEADERS = [
    "metric_number",
    "metric_type",
    "metric_name",
    "wind_warning_status",
    "unit_canonical",
    "actual_value",
    "reporting_year",
    "reporting_quarter",
    "source_file",
    "source_row",
    "source_value_cell",
]

CROSSWALK_HEADERS = [
    "template_index",
    "direct_metric_number",
    "row_aligned_v4_metric_number",
    "legacy_metric_number_v3_1",
    "legacy_metric_number_v3_2",
    "legacy_metric_number_status",
    "metric_type",
    "metric_name",
    "wind_warning_status_legacy",
    "wind_warning_status_v4",
    "wind_warning_status_canonical",
    "semantic_crosswalk_status",
    "semantic_crosswalk_note",
    "unit_legacy",
    "unit_v4",
    "unit_crosswalk_status",
]

RECONCILIATION_HEADERS = [
    "metric_name",
    "reporting_year",
    "reporting_quarter",
    "reported_total",
    "rfw_only",
    "hww_only",
    "hww_and_rfw",
    "reported_neither",
    "sum_warning_statuses",
    "derived_total_v4",
    "legacy_total_minus_warning_components",
    "reconciliation_status",
    "source_file",
    "source_cells",
]

ISSUE_HEADERS = [
    "issue_type",
    "metric_name",
    "reporting_year",
    "reporting_quarter",
    "reported_total",
    "sum_warning_statuses",
    "difference",
    "source_file",
    "source_cells",
    "note",
]

SCHEMA_CHANGE_HEADERS = [
    "change",
    "2023 v3.1",
    "2024 v3.2",
    "2025 v4.01",
    "converter_action",
]


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.replace("\xa0", " ").split())
        return normalized or None
    return value


def parse_number(value: Any) -> int | float | None:
    value = clean(value)
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        if value < 0:
            raise ValueError(
                f"Table 10 actual values must be nonnegative or blank: {value}"
            )
        return value

    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric Table 10 value, found {value!r}"
            ) from exc

        if parsed < 0:
            raise ValueError(
                f"Table 10 actual values must be nonnegative or blank: {value}"
            )
        return int(parsed) if parsed.is_integer() else parsed

    raise TypeError(f"Unsupported Table 10 value type: {type(value).__name__}")


def column_index(letters: str) -> int:
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def column_letter(zero_based_index: int) -> str:
    number = zero_based_index + 1
    output = ""
    while number:
        number, remainder = divmod(number - 1, 26)
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
    """Read computed worksheet values without altering the source workbook."""
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
        dimension_match = re.match(r"([A-Z]+)(\d+)", last_cell)
        if not dimension_match:
            raise ValueError(
                f"Unrecognized worksheet dimension {dimension_reference!r}"
            )

        max_columns = column_index(dimension_match.group(1)) + 1
        max_rows = int(dimension_match.group(2))
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


def parse_legacy_template(values: list[list[Any]]) -> list[dict[str, Any]]:
    headers = [clean(value) for value in values[8]]
    required = {
        "Metric type",
        "#",
        "Wind Warning Status",
        "Metric name",
        "Unit(s)",
        "Comments",
        "Blank Meaning",
    }
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Missing legacy Table 10 headers: {sorted(missing)}")

    unit_column = headers.index("Unit(s)")
    comments_column = headers.index("Comments")
    blank_meaning_column = headers.index("Blank Meaning")

    records = []
    metric_type = None
    legacy_metric_number_mapped = None

    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]

        if clean(row[2]) is not None:
            metric_type = clean(row[2])
        source_number = clean(row[3])
        if source_number is not None:
            legacy_metric_number_mapped = source_number

        metric_name = clean(row[5])
        if metric_name is None:
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_type_raw": metric_type,
                "legacy_metric_number_raw": source_number,
                "legacy_metric_number_mapped": legacy_metric_number_mapped,
                "wind_warning_status_raw": clean(row[4]),
                "metric_name_raw": metric_name,
                "unit_raw": clean(row[unit_column]),
                "comments": clean(row[comments_column]),
                "blank_meaning": clean(row[blank_meaning_column]),
            }
        )

    if len(records) != 44:
        raise AssertionError(
            f"Expected 44 legacy Table 10 rows; found {len(records)}"
        )

    return records


def locate_2023_actual_columns(
    values: list[list[Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}

    for column in range(len(values[8])):
        reporting_year = clean(values[8][column])
        reporting_quarter = clean(values[7][column])

        if (
            reporting_year == 2023
            and reporting_quarter in {"Q1", "Q2", "Q3", "Q4"}
        ):
            result[int(str(reporting_quarter)[1])] = column

    if sorted(result) != [1, 2, 3, 4]:
        raise AssertionError(
            f"Expected 2023 Q1-Q4 actual columns; found {result}"
        )

    return result


def locate_2024_actual_column(
    values: list[list[Any]],
    expected_quarter: int,
) -> int:
    matching_columns = []

    for column in range(len(values[8])):
        reporting_year = clean(values[8][column])
        reporting_quarter = clean(values[7][column])

        if (
            reporting_year == 2024
            and reporting_quarter == f"Q{expected_quarter}"
        ):
            matching_columns.append(column)

    if len(matching_columns) != 1:
        raise AssertionError(
            f"Expected one 2024 Q{expected_quarter} actual column; "
            f"found {matching_columns}"
        )

    return matching_columns[0]


def parse_v4_template(values: list[list[Any]]) -> list[dict[str, Any]]:
    expected_headers = [
        "METRIC NUMBER",
        "METRIC TYPE",
        "METRIC NAME",
        "WIND WARNING STATUS",
        "UNIT(S)",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "REPORTING QUARTER",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in values[0][:11]]

    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Table 10 header does not match Data Guidelines v4.01.\n"
            f"Expected: {expected_headers}\n"
            f"Found: {actual_headers}"
        )

    records = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[2]) is None:
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_number": int(row[0]),
                "metric_type": clean(row[1]),
                "metric_name": clean(row[2]),
                "wind_warning_status": clean(row[3]),
                "unit_raw": clean(row[4]),
                "comments": clean(row[5]),
                "blank_meaning": clean(row[6]),
                "utility_id": clean(row[7]),
                "reporting_year": int(row[8]),
                "reporting_quarter": int(row[9]),
                "actual_value": parse_number(row[10]),
            }
        )

    if len(records) != 44:
        raise AssertionError(
            f"Expected 44 v4.01 Table 10 rows; found {len(records)}"
        )

    return records


def semantic_mapping(
    legacy_status: Any,
    v4_status: Any,
) -> tuple[str | None, str, str]:
    legacy_status = clean(legacy_status)
    v4_status = clean(v4_status)

    if legacy_status == v4_status:
        return (
            v4_status,
            "exact_wind_status",
            "Legacy and v4.01 wind-warning statuses match.",
        )

    if legacy_status == "None" and v4_status == "Neither":
        return (
            "Neither",
            "normalized_none_to_neither",
            "Legacy value 'None' and v4.01 value 'Neither' represent no "
            "RFW or HWW.",
        )

    if legacy_status == "N/A" and v4_status is None:
        return (
            None,
            "normalized_na_to_null",
            "Legacy WIND WARNING STATUS is literal N/A; v4.01 stores blank. "
            "Both normalize to null.",
        )

    if (
        legacy_status == "All (regardless of RFW/HWW status)"
        and v4_status == "Neither"
    ):
        return (
            "All statuses (legacy total)",
            "legacy_total_not_equivalent_to_v4_neither",
            "The legacy row is a total across all warning statuses. The "
            "row-aligned v4.01 row is the mutually exclusive Neither category, "
            "so the rows must not be merged.",
        )

    raise AssertionError(
        f"Unexpected Table 10 wind-status transition: "
        f"{legacy_status!r} -> {v4_status!r}"
    )


def build_crosswalk(
    v3_1_records: list[dict[str, Any]],
    v3_2_records: list[dict[str, Any]],
    v4_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    if not (
        len(v3_1_records)
        == len(v3_2_records)
        == len(v4_records)
        == 44
    ):
        raise AssertionError("All Table 10 schema versions must contain 44 rows")

    expected_numbers = list(range(1100000000, 1100000044))
    actual_numbers = sorted(record["metric_number"] for record in v4_records)
    if actual_numbers != expected_numbers:
        raise AssertionError(
            "The 2025 Table 10 metric numbers are not the expected contiguous "
            "range 1100000000-1100000043"
        )

    crosswalk = []
    by_template_index: dict[int, dict[str, Any]] = {}
    by_metric_number: dict[int, dict[str, Any]] = {}

    for old_2023, old_2024, new_2025 in zip(
        v3_1_records,
        v3_2_records,
        v4_records,
    ):
        if old_2023["metric_type_raw"] != old_2024["metric_type_raw"]:
            raise AssertionError("v3.1/v3.2 metric-type mismatch")
        if old_2023["metric_name_raw"] != old_2024["metric_name_raw"]:
            raise AssertionError("v3.1/v3.2 metric-name mismatch")
        if old_2023["wind_warning_status_raw"] != old_2024[
            "wind_warning_status_raw"
        ]:
            raise AssertionError("v3.1/v3.2 wind-status mismatch")
        if old_2023["unit_raw"] != old_2024["unit_raw"]:
            raise AssertionError("v3.1/v3.2 unit mismatch")

        if old_2024["metric_type_raw"] != new_2025["metric_type"]:
            raise AssertionError(
                f"Legacy/v4.01 metric-type mismatch at template index "
                f"{old_2024['template_index']}"
            )
        if old_2024["metric_name_raw"] != new_2025["metric_name"]:
            raise AssertionError(
                f"Legacy/v4.01 metric-name mismatch at template index "
                f"{old_2024['template_index']}"
            )

        (
            canonical_wind_status,
            semantic_status,
            semantic_note,
        ) = semantic_mapping(
            old_2024["wind_warning_status_raw"],
            new_2025["wind_warning_status"],
        )

        unit_status = (
            "exact"
            if old_2024["unit_raw"] == new_2025["unit_raw"]
            else "simplified_label_same_measurement"
        )
        legacy_number_status = (
            "stable"
            if old_2023["legacy_metric_number_mapped"]
            == old_2024["legacy_metric_number_mapped"]
            else "corrected_in_v3_2"
        )
        direct_metric_number = (
            None
            if semantic_status
            == "legacy_total_not_equivalent_to_v4_neither"
            else new_2025["metric_number"]
        )

        item = {
            "template_index": old_2024["template_index"],
            "direct_metric_number": direct_metric_number,
            "row_aligned_v4_metric_number": new_2025["metric_number"],
            "legacy_metric_number_v3_1": old_2023[
                "legacy_metric_number_mapped"
            ],
            "legacy_metric_number_v3_2": old_2024[
                "legacy_metric_number_mapped"
            ],
            "legacy_metric_number_status": legacy_number_status,
            "metric_type": new_2025["metric_type"],
            "metric_name": new_2025["metric_name"],
            "wind_warning_status_legacy": old_2024[
                "wind_warning_status_raw"
            ],
            "wind_warning_status_v4": new_2025[
                "wind_warning_status"
            ],
            "wind_warning_status_canonical": canonical_wind_status,
            "semantic_crosswalk_status": semantic_status,
            "semantic_crosswalk_note": semantic_note,
            "unit_legacy": old_2024["unit_raw"],
            "unit_v4": new_2025["unit_raw"],
            "unit_crosswalk_status": unit_status,
        }

        crosswalk.append(item)
        by_template_index[item["template_index"]] = item
        by_metric_number[new_2025["metric_number"]] = item

    semantic_counts = Counter(
        item["semantic_crosswalk_status"]
        for item in crosswalk
    )
    expected_semantic_counts = Counter(
        {
            "exact_wind_status": 15,
            "normalized_none_to_neither": 2,
            "normalized_na_to_null": 24,
            "legacy_total_not_equivalent_to_v4_neither": 3,
        }
    )
    if semantic_counts != expected_semantic_counts:
        raise AssertionError(
            f"Unexpected semantic crosswalk results: {semantic_counts}"
        )

    unit_counts = Counter(
        item["unit_crosswalk_status"]
        for item in crosswalk
    )
    if unit_counts != Counter(
        {"exact": 17, "simplified_label_same_measurement": 27}
    ):
        raise AssertionError(
            f"Unexpected unit crosswalk results: {unit_counts}"
        )

    number_counts = Counter(
        item["legacy_metric_number_status"]
        for item in crosswalk
    )
    if number_counts != Counter({"stable": 43, "corrected_in_v3_2": 1}):
        raise AssertionError(
            f"Unexpected legacy-number changes: {number_counts}"
        )

    corrected = [
        item
        for item in crosswalk
        if item["legacy_metric_number_status"] == "corrected_in_v3_2"
    ]
    if not (
        len(corrected) == 1
        and corrected[0]["legacy_metric_number_v3_1"] == "4.a."
        and corrected[0]["legacy_metric_number_v3_2"] == "5.d."
    ):
        raise AssertionError(
            "Expected the single legacy identifier correction 4.a. -> 5.d."
        )

    return crosswalk, by_template_index, by_metric_number


def validate_quarterly_schemas(
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> None:
    reference_2024 = parse_legacy_template(loaded[(2024, 4)])
    stable_legacy_fields = (
        "metric_type_raw",
        "legacy_metric_number_mapped",
        "wind_warning_status_raw",
        "metric_name_raw",
        "unit_raw",
    )
    reference_2024_key = [
        tuple(record[field] for field in stable_legacy_fields)
        for record in reference_2024
    ]

    for quarter in (1, 2, 3, 4):
        current = parse_legacy_template(loaded[(2024, quarter)])
        current_key = [
            tuple(record[field] for field in stable_legacy_fields)
            for record in current
        ]
        if current_key != reference_2024_key:
            raise AssertionError(
                f"2024 Q{quarter} Table 10 schema differs from Q4"
            )

    reference_2025 = parse_v4_template(loaded[(2025, 4)])
    stable_v4_fields = (
        "metric_number",
        "metric_type",
        "metric_name",
        "wind_warning_status",
        "unit_raw",
        "utility_id",
    )
    reference_2025_key = [
        tuple(record[field] for field in stable_v4_fields)
        for record in reference_2025
    ]

    for quarter in (1, 2, 3, 4):
        current = parse_v4_template(loaded[(2025, quarter)])
        current_key = [
            tuple(record[field] for field in stable_v4_fields)
            for record in current
        ]
        if current_key != reference_2025_key:
            raise AssertionError(
                f"2025 Q{quarter} Table 10 schema differs from Q4"
            )


def make_record_id(
    semantic_identifier: Any,
    year: int,
    quarter: int,
    source_file: str,
    source_row: int,
) -> str:
    payload = json.dumps(
        [
            semantic_identifier,
            year,
            quarter,
            source_file,
            source_row,
        ],
        separators=(",", ":"),
    ).encode("utf-8")

    return "T10R-" + hashlib.sha1(payload).hexdigest()[:16]


def build_legacy_output_row(
    *,
    source_record: dict[str, Any],
    mapping: dict[str, Any],
    actual_value: Any,
    reporting_year: int,
    reporting_quarter: int,
    schema_version: str,
    guideline_url: str,
    source: dict[str, Any],
    source_report_quarter: int,
    source_value_column: int,
) -> list[Any]:
    semantic_identifier = (
        mapping["direct_metric_number"]
        if mapping["direct_metric_number"] is not None
        else (
            "legacy_total",
            mapping["metric_name"],
        )
    )

    return [
        make_record_id(
            semantic_identifier,
            reporting_year,
            reporting_quarter,
            source["name"],
            source_record["source_row"],
        ),
        mapping["direct_metric_number"],
        mapping["row_aligned_v4_metric_number"],
        mapping["legacy_metric_number_v3_2"],
        source_record["legacy_metric_number_raw"],
        mapping["legacy_metric_number_status"],
        mapping["semantic_crosswalk_status"],
        mapping["metric_type"],
        source_record["metric_type_raw"],
        mapping["metric_name"],
        source_record["metric_name_raw"],
        mapping["wind_warning_status_canonical"],
        source_record["wind_warning_status_raw"],
        source_record["unit_raw"],
        mapping["unit_v4"],
        mapping["unit_crosswalk_status"],
        parse_number(actual_value),
        source_record["comments"],
        source_record["blank_meaning"],
        "SDG&E",
        reporting_year,
        reporting_quarter,
        schema_version,
        source["revision"],
        source_report_quarter,
        source["name"],
        "Table 10",
        source_record["source_row"],
        (
            f"{column_letter(source_value_column)}"
            f"{source_record['source_row']}"
        ),
        guideline_url,
    ]


def build_v4_output_row(
    *,
    source_record: dict[str, Any],
    mapping: dict[str, Any],
    source: dict[str, Any],
) -> list[Any]:
    semantic_status = (
        "v4_neither_no_legacy_equivalent"
        if mapping["semantic_crosswalk_status"]
        == "legacy_total_not_equivalent_to_v4_neither"
        else "v4_native"
    )

    return [
        make_record_id(
            source_record["metric_number"],
            source_record["reporting_year"],
            source_record["reporting_quarter"],
            source["name"],
            source_record["source_row"],
        ),
        source_record["metric_number"],
        source_record["metric_number"],
        mapping["legacy_metric_number_v3_2"],
        None,
        mapping["legacy_metric_number_status"],
        semantic_status,
        source_record["metric_type"],
        source_record["metric_type"],
        source_record["metric_name"],
        source_record["metric_name"],
        source_record["wind_warning_status"],
        source_record["wind_warning_status"],
        source_record["unit_raw"],
        source_record["unit_raw"],
        "v4_native",
        source_record["actual_value"],
        source_record["comments"],
        source_record["blank_meaning"],
        source_record["utility_id"],
        source_record["reporting_year"],
        source_record["reporting_quarter"],
        GUIDELINES[2025][0],
        source["revision"],
        source_record["reporting_quarter"],
        source["name"],
        "Table 10",
        source_record["source_row"],
        f"K{source_record['source_row']}",
        GUIDELINES[2025][1],
    ]


def build_unified_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], list[list[Any]]],
    by_template_index: dict[int, dict[str, Any]],
    by_metric_number: dict[int, dict[str, Any]],
) -> list[list[Any]]:
    output: list[list[Any]] = []

    # 2023 Q4 contains separate actual columns for all four 2023 quarters.
    source = selected[(2023, 4)]
    values = loaded[(2023, 4)]
    records = parse_legacy_template(values)
    actual_columns = locate_2023_actual_columns(values)

    for source_record in records:
        mapping = by_template_index[source_record["template_index"]]

        for quarter, value_column in sorted(actual_columns.items()):
            output.append(
                build_legacy_output_row(
                    source_record=source_record,
                    mapping=mapping,
                    actual_value=values[
                        source_record["source_row"] - 1
                    ][value_column],
                    reporting_year=2023,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2023][0],
                    guideline_url=GUIDELINES[2023][1],
                    source=source,
                    source_report_quarter=4,
                    source_value_column=value_column,
                )
            )

    # Each 2024 v3.2 workbook contributes its subject-quarter actual.
    for quarter in (1, 2, 3, 4):
        source = selected[(2024, quarter)]
        values = loaded[(2024, quarter)]
        records = parse_legacy_template(values)
        value_column = locate_2024_actual_column(values, quarter)

        for source_record in records:
            mapping = by_template_index[source_record["template_index"]]
            output.append(
                build_legacy_output_row(
                    source_record=source_record,
                    mapping=mapping,
                    actual_value=values[
                        source_record["source_row"] - 1
                    ][value_column],
                    reporting_year=2024,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2024][0],
                    guideline_url=GUIDELINES[2024][1],
                    source=source,
                    source_report_quarter=quarter,
                    source_value_column=value_column,
                )
            )

    # The 2025 v4.01 quarterly workbook is already long-form.
    for quarter in (1, 2, 3, 4):
        source = selected[(2025, quarter)]
        records = parse_v4_template(loaded[(2025, quarter)])

        for source_record in records:
            if source_record["reporting_year"] != 2025:
                raise AssertionError(
                    f"Unexpected reporting year in {source['name']} "
                    f"row {source_record['source_row']}"
                )
            if source_record["reporting_quarter"] != quarter:
                raise AssertionError(
                    f"Unexpected reporting quarter in {source['name']} "
                    f"row {source_record['source_row']}"
                )

            mapping = by_metric_number[source_record["metric_number"]]
            output.append(
                build_v4_output_row(
                    source_record=source_record,
                    mapping=mapping,
                    source=source,
                )
            )

    expected_rows = 44 * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified Table 10 rows; "
            f"found {len(output)}"
        )

    return output


def build_direct_comparable_rows(
    unified_rows: list[list[Any]],
) -> list[list[Any]]:
    index = {
        header: position
        for position, header in enumerate(UNIFIED_HEADERS)
    }

    excluded_statuses = {
        "legacy_total_not_equivalent_to_v4_neither",
        "v4_neither_no_legacy_equivalent",
    }
    output = []

    for row in unified_rows:
        if row[index["metric_number"]] is None:
            continue
        if row[index["semantic_crosswalk_status"]] in excluded_statuses:
            continue

        output.append(
            [
                row[index["metric_number"]],
                row[index["metric_type"]],
                row[index["metric_name"]],
                row[index["wind_warning_status"]],
                row[index["unit_canonical"]],
                row[index["actual_value"]],
                row[index["reporting_year"]],
                row[index["reporting_quarter"]],
                row[index["source_file"]],
                row[index["source_row"]],
                row[index["source_value_cell"]],
            ]
        )

    expected_rows = 41 * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} directly comparable records; "
            f"found {len(output)}"
        )

    return output


def build_reconciliation_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> tuple[list[list[Any]], list[list[Any]]]:
    rows = []
    issues = []
    metric_starts = (0, 4, 8)

    # Legacy years: compare the reported total with the three warning categories.
    legacy_periods = [
        (2023, quarter, selected[(2023, 4)], loaded[(2023, 4)])
        for quarter in (1, 2, 3, 4)
    ] + [
        (2024, quarter, selected[(2024, quarter)], loaded[(2024, quarter)])
        for quarter in (1, 2, 3, 4)
    ]

    for year, quarter, source, values in legacy_periods:
        records = parse_legacy_template(values)
        value_column = (
            locate_2023_actual_columns(values)[quarter]
            if year == 2023
            else locate_2024_actual_column(values, quarter)
        )

        for start in metric_starts:
            metric_name = records[start]["metric_name_raw"]
            values_by_status = [
                parse_number(
                    values[records[start + offset]["source_row"] - 1][
                        value_column
                    ]
                )
                for offset in range(4)
            ]
            rfw, hww, both, total = values_by_status
            component_sum = (
                rfw + hww + both
                if all(value is not None for value in (rfw, hww, both))
                else None
            )
            residual = (
                total - component_sum
                if total is not None and component_sum is not None
                else None
            )

            if residual is None:
                status = "incomplete_legacy_values"
            elif residual < -1e-9:
                status = "reported_total_less_than_warning_components"
            elif abs(residual) <= 1e-9:
                status = "warning_components_reconcile_to_reported_total"
            else:
                status = "positive_unallocated_residual"

            source_cells = "; ".join(
                f"{column_letter(value_column)}"
                f"{records[start + offset]['source_row']}"
                for offset in range(4)
            )
            rows.append(
                [
                    metric_name,
                    year,
                    quarter,
                    total,
                    rfw,
                    hww,
                    both,
                    None,
                    component_sum,
                    None,
                    residual,
                    status,
                    source["name"],
                    source_cells,
                ]
            )

            if status == "reported_total_less_than_warning_components":
                issues.append(
                    [
                        "legacy_total_less_than_components",
                        metric_name,
                        year,
                        quarter,
                        total,
                        component_sum,
                        residual,
                        source["name"],
                        source_cells,
                        "The reported all-status total is smaller than the "
                        "sum of RFW-only, HWW-only, and HWW&RFW values. The "
                        "converter preserves the source and does not derive "
                        "a Neither value.",
                    ]
                )

    # v4.01: calculate a total from the four mutually exclusive statuses.
    for quarter in (1, 2, 3, 4):
        source = selected[(2025, quarter)]
        records = parse_v4_template(loaded[(2025, quarter)])

        for start in metric_starts:
            metric_name = records[start]["metric_name"]
            rfw = records[start]["actual_value"]
            hww = records[start + 1]["actual_value"]
            both = records[start + 2]["actual_value"]
            neither = records[start + 3]["actual_value"]
            warning_sum = (
                rfw + hww + both
                if all(value is not None for value in (rfw, hww, both))
                else None
            )
            derived_total = (
                rfw + hww + both + neither
                if all(
                    value is not None
                    for value in (rfw, hww, both, neither)
                )
                else None
            )
            source_cells = "; ".join(
                f"K{records[start + offset]['source_row']}"
                for offset in range(4)
            )

            rows.append(
                [
                    metric_name,
                    2025,
                    quarter,
                    None,
                    rfw,
                    hww,
                    both,
                    neither,
                    warning_sum,
                    derived_total,
                    None,
                    "derived_total_from_four_v4_statuses",
                    source["name"],
                    source_cells,
                ]
            )

    expected_rows = 3 * 4 * 3
    if len(rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} reconciliation rows; found {len(rows)}"
        )

    return rows, issues


def build_crosswalk_rows(
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    return [
        [
            item["template_index"],
            item["direct_metric_number"],
            item["row_aligned_v4_metric_number"],
            item["legacy_metric_number_v3_1"],
            item["legacy_metric_number_v3_2"],
            item["legacy_metric_number_status"],
            item["metric_type"],
            item["metric_name"],
            item["wind_warning_status_legacy"],
            item["wind_warning_status_v4"],
            item["wind_warning_status_canonical"],
            item["semantic_crosswalk_status"],
            item["semantic_crosswalk_note"],
            item["unit_legacy"],
            item["unit_v4"],
            item["unit_crosswalk_status"],
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


def excel_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("="):
        return "'" + value
    return value


def excel_safe_rows(rows: list[list[Any]]) -> list[list[Any]]:
    return [[excel_safe(value) for value in row] for row in rows]


def write_rows(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    *,
    chunk_size: int = 300,
) -> None:
    sheet.get_range_by_indexes(0, 0, 1, len(headers)).values = [headers]
    safe_rows = excel_safe_rows(rows)

    for start in range(0, len(safe_rows), chunk_size):
        chunk = safe_rows[start : start + chunk_size]
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

    large_text = {
        "metric_type",
        "source_metric_type_raw",
        "metric_name",
        "source_metric_name_raw",
        "unit_raw",
        "unit_canonical",
        "comments",
        "blank_meaning",
        "semantic_crosswalk_note",
        "note",
        "guideline_url",
    }
    medium_text = {
        "semantic_crosswalk_status",
        "unit_crosswalk_status",
        "source_file",
        "reconciliation_status",
        "source_cells",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 18

        if header in large_text:
            width = 42
        elif header in medium_text:
            width = 32

        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width

    sheet.get_range(
        f"A1:{last_column}{last_row}"
    ).format.wrap_text = True


def build_workbook(
    output_path: Path,
    unified_rows: list[list[Any]],
    direct_rows: list[list[Any]],
    reconciliation_rows: list[list[Any]],
    issue_rows: list[list[Any]],
    crosswalk_rows: list[list[Any]],
    selected: dict[tuple[int, int], dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()

    readme = workbook.worksheets.add("README")
    actuals = workbook.worksheets.add("Unified Actuals")
    direct = workbook.worksheets.add("Direct Comparable")
    reconciliation = workbook.worksheets.add("Status Reconciliation")
    issues = workbook.worksheets.add("Validation Issues")
    crosswalk_sheet = workbook.worksheets.add("Metric Crosswalk")
    changes = workbook.worksheets.add("Schema Changes")

    readme_rows = [
        ["SDG&E Table 10 Unified Dataset, 2023–2025", "", "", ""],
        [
            "Unified source observations",
            validation["unified_rows"],
            "Rows per quarter",
            44,
        ],
        [
            "Directly comparable observations",
            validation["direct_comparable_rows"],
            "Common semantic series",
            41,
        ],
        [
            "Reporting periods",
            "2023 Q1–Q4, 2024 Q1–Q4, 2025 Q1–Q4",
            "",
            "",
        ],
        [
            "Projection treatment",
            "All projection columns are excluded.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "The selected Q4 v3.1 workbook supplies the four 2023 "
            "quarterly actual columns.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "Each highest-revision v3.2 workbook supplies its "
            "subject-quarter actual.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Each highest-revision v4.01 workbook is already long-form.",
            "",
            "",
        ],
        [
            "Critical semantic break",
            "Three legacy rows are all-status totals, while their row-aligned "
            "2025 rows are the mutually exclusive Neither category. They are "
            "kept separate and excluded from Direct Comparable.",
            "",
            "",
        ],
        [
            "Wind-status normalization",
            "Two legacy 'None' values map to Neither; 24 legacy N/A values "
            "map to blank/null.",
            "",
            "",
        ],
        [
            "Unit treatment",
            "27 UNIT(S) labels were simplified in v4.01 and 17 were unchanged. "
            "No numeric conversion is applied.",
            "",
            "",
        ],
        [
            "Legacy identifier correction",
            "One 2023 row uses legacy # 4.a.; the corresponding v3.2 row uses "
            "5.d. Both source and mapped identifiers are retained.",
            "",
            "",
        ],
        [
            "Status reconciliation issues",
            validation["validation_issue_rows"],
            "Treatment",
            "Source values are preserved and the issue is documented; no "
            "Neither value is inferred from an inconsistent total.",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Quarterly actuals are required for all Table 10 metrics and "
            "values must be numeric, nonnegative, or blank.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Table 10 retains the same PSPS metric purpose and value constraint.",
        ],
        [
            "v3.2 Template Changelog",
            "2024 transition",
            V3_2_CHANGELOG_URL,
            "Table 10 year/quarter headers were changed to cover-sheet "
            "references. Direct workbook comparison also found one legacy "
            "identifier change, 4.a. to 5.d.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Quarterly Table 10 contains subject-period actuals; projections "
            "are reported through the Annual-WMP workbook.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "Most Table 10 UNIT(S) descriptions were simplified.",
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
        0,
        0,
        len(readme_rows),
        4,
    ).values = excel_safe_rows(readme_rows)
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
    for column, width in zip(("A", "B", "C", "D"), (34, 68, 68, 62)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(actuals, UNIFIED_HEADERS, unified_rows)
    format_sheet(
        actuals,
        UNIFIED_HEADERS,
        len(unified_rows),
        freeze_columns=7,
    )
    value_column = column_letter(UNIFIED_HEADERS.index("actual_value"))
    actuals.get_range(
        f"{value_column}2:{value_column}{len(unified_rows) + 1}"
    ).format.number_format = "0.########"

    semantic_column = column_letter(
        UNIFIED_HEADERS.index("semantic_crosswalk_status")
    )
    actuals.get_range(
        f"{semantic_column}2:"
        f"{semantic_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=OR(${semantic_column}2="legacy_total_not_equivalent_to_v4_neither",'
        f'${semantic_column}2="v4_neither_no_legacy_equivalent")',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    write_rows(direct, DIRECT_COMPARABLE_HEADERS, direct_rows)
    format_sheet(
        direct,
        DIRECT_COMPARABLE_HEADERS,
        len(direct_rows),
        freeze_columns=5,
    )
    direct_value_column = column_letter(
        DIRECT_COMPARABLE_HEADERS.index("actual_value")
    )
    direct.get_range(
        f"{direct_value_column}2:"
        f"{direct_value_column}{len(direct_rows) + 1}"
    ).format.number_format = "0.########"

    write_rows(
        reconciliation,
        RECONCILIATION_HEADERS,
        reconciliation_rows,
    )
    format_sheet(
        reconciliation,
        RECONCILIATION_HEADERS,
        len(reconciliation_rows),
        freeze_columns=3,
    )

    write_rows(issues, ISSUE_HEADERS, issue_rows)
    format_sheet(
        issues,
        ISSUE_HEADERS,
        len(issue_rows),
        freeze_columns=4,
    )
    if issue_rows:
        issues.get_range(
            f"A2:J{len(issue_rows) + 1}"
        ).format = {
            "fill": "#FEE2E2",
            "font": {"color": "#991B1B"},
            "wrap_text": True,
        }

    write_rows(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        crosswalk_rows,
    )
    format_sheet(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        len(crosswalk_rows),
        freeze_columns=6,
    )

    schema_change_rows = [
        [
            "Actual-value layout",
            "Wide workbook containing historical quarterly actuals and "
            "future annual projections.",
            "Only the subject-quarter actual remains, with projection columns "
            "still present.",
            "Long-form quarterly records with explicit utility, year, quarter, "
            "metric number, and actual value.",
            "Unpivot 2023; append each 2024/2025 quarter; exclude projections.",
        ],
        [
            "Projection location",
            "Four specified metrics have projections in quarterly Table 10.",
            "Same projection requirement.",
            "Projections are moved to the Annual-WMP workbook.",
            "Drop all projection values as requested.",
        ],
        [
            "Wind-warning categories",
            "The first three PSPS metrics contain RFW only, HWW only, "
            "HWW&RFW, and an all-status total.",
            "Same.",
            "The fourth category is Neither rather than an all-status total.",
            "Do not merge the three legacy totals with the v4.01 Neither rows. "
            "Keep both source observations and exclude them from the direct "
            "cross-year comparison.",
        ],
        [
            "No-warning label",
            "Fast-trip metrics use literal None; non-warning-applicable rows use N/A.",
            "Same.",
            "Neither is used for no warning; blank is used where warning status "
            "does not apply.",
            "Normalize None to Neither and N/A/blank to null.",
        ],
        [
            "Metric identifier",
            "Legacy # values; one row is labeled 4.a.",
            "Legacy # values; that row is labeled 5.d.",
            "Standard metric numbers 1100000000–1100000043.",
            "Use v4.01 metric numbers for direct mappings and retain both "
            "legacy identifiers for lineage.",
        ],
        [
            "UNIT(S)",
            "Long calculation and measurement descriptions.",
            "Same descriptions.",
            "Most descriptions are simplified; calculations move to the glossary.",
            "Use v4.01 unit labels as canonical. Preserve raw labels and apply "
            "no numeric conversion.",
        ],
        [
            "Value constraint",
            "Numeric ≥ 0 or blank.",
            "Numeric ≥ 0 or blank.",
            "Numeric ≥ 0 or blank.",
            "Reject negative or nonnumeric populated actual values.",
        ],
    ]

    write_rows(changes, SCHEMA_CHANGE_HEADERS, schema_change_rows)
    format_sheet(
        changes,
        SCHEMA_CHANGE_HEADERS,
        len(schema_change_rows),
        freeze_columns=1,
    )

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 10 actual values for 2023-2025 while "
            "preserving the semantic break between legacy all-status totals "
            "and v4.01 Neither rows. Projection values are excluded."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing the source SDG&E XLSX files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table10_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 10")
        for key, source in selected.items()
    }

    validate_quarterly_schemas(loaded)

    v3_1_records = parse_legacy_template(loaded[(2023, 4)])
    v3_2_records = parse_legacy_template(loaded[(2024, 4)])
    v4_records = parse_v4_template(loaded[(2025, 4)])

    crosswalk, by_template_index, by_metric_number = build_crosswalk(
        v3_1_records,
        v3_2_records,
        v4_records,
    )

    unified_rows = build_unified_rows(
        selected,
        loaded,
        by_template_index,
        by_metric_number,
    )
    direct_rows = build_direct_comparable_rows(unified_rows)
    reconciliation_rows, issue_rows = build_reconciliation_rows(
        selected,
        loaded,
    )
    metric_crosswalk_rows = build_crosswalk_rows(crosswalk)

    workbook_path = (
        output_dir / "sdge_table10_2023_2025_unified.xlsx"
    )
    unified_csv = (
        output_dir / "sdge_table10_2023_2025_unified_actuals.csv"
    )
    direct_csv = (
        output_dir / "sdge_table10_direct_comparable_actuals.csv"
    )
    reconciliation_csv = (
        output_dir / "sdge_table10_status_reconciliation.csv"
    )
    issues_csv = (
        output_dir / "sdge_table10_validation_issues.csv"
    )
    crosswalk_csv = output_dir / "sdge_table10_metric_crosswalk.csv"
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(
        direct_csv,
        DIRECT_COMPARABLE_HEADERS,
        direct_rows,
    )
    write_csv(
        reconciliation_csv,
        RECONCILIATION_HEADERS,
        reconciliation_rows,
    )
    write_csv(issues_csv, ISSUE_HEADERS, issue_rows)
    write_csv(
        crosswalk_csv,
        CROSSWALK_HEADERS,
        metric_crosswalk_rows,
    )

    validation = {
        "unified_rows": len(unified_rows),
        "direct_comparable_rows": len(direct_rows),
        "reconciliation_rows": len(reconciliation_rows),
        "validation_issue_rows": len(issue_rows),
        "metrics_per_quarter": 44,
        "reporting_periods": 12,
        "crosswalk_rows": len(metric_crosswalk_rows),
        "semantic_crosswalk_status_counts": dict(
            Counter(
                item["semantic_crosswalk_status"]
                for item in crosswalk
            )
        ),
        "unit_crosswalk_status_counts": dict(
            Counter(
                item["unit_crosswalk_status"]
                for item in crosswalk
            )
        ),
        "legacy_metric_number_status_counts": dict(
            Counter(
                item["legacy_metric_number_status"]
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
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        direct_rows,
        reconciliation_rows,
        issue_rows,
        metric_crosswalk_rows,
        selected,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {direct_csv}")
    print(f"Created: {reconciliation_csv}")
    print(f"Created: {issues_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(f"Direct comparable rows: {len(direct_rows)}")
    print(f"Validation issues: {len(issue_rows)}")
    print(
        "Semantic crosswalk:",
        validation["semantic_crosswalk_status_counts"],
    )
    print(
        "Unit crosswalk:",
        validation["unit_crosswalk_status_counts"],
    )


if __name__ == "__main__":
    main()
