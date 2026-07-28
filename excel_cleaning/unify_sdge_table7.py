
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
    "fileid=56073&shareable=true"
)

UNIFIED_HEADERS = [
    "record_id",
    "metric_number",
    "legacy_metric_number_mapped",
    "source_legacy_metric_number_raw",
    "crosswalk_status",
    "metric_type",
    "source_metric_type_raw",
    "line_type",
    "source_line_type_raw",
    "hftd_tier",
    "source_hftd_tier_raw",
    "area_type",
    "source_area_type_raw",
    "wui_status",
    "source_wui_status_raw",
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

CROSSWALK_HEADERS = [
    "metric_number",
    "legacy_metric_number",
    "metric_type",
    "line_type_canonical",
    "line_type_legacy",
    "hftd_tier",
    "area_type",
    "wui_status",
    "unit_canonical",
    "crosswalk_status",
    "crosswalk_note",
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
                f"Table 7 actual values must be nonnegative or blank: {value}"
            )
        return value

    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric Table 7 value, found {value!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(
                f"Table 7 actual values must be nonnegative or blank: {value}"
            )
        return int(parsed) if parsed.is_integer() else parsed

    raise TypeError(f"Unsupported Table 7 value type: {type(value).__name__}")


def normalize_line_type(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None

    if isinstance(value, str) and value.casefold() in {
        "n/a",
        "na",
        "not applicable",
    }:
        return None

    if value not in {"Distribution", "Transmission"}:
        raise ValueError(f"Unexpected Table 7 line type: {value!r}")
    return value


def normalize_hftd_tier(value: Any) -> str:
    value = clean(value)
    mapping = {
        "Non-HFTD": "Non-HFTD",
        "Non- HFTD": "Non-HFTD",
        "HFTD Tier 2": "HFTD Tier 2",
        "HFTD Tier 3": "HFTD Tier 3",
    }
    if value not in mapping:
        raise ValueError(f"Unexpected Table 7 HFTD tier: {value!r}")
    return mapping[value]


def normalize_area_type(value: Any) -> str:
    value = clean(value)
    allowed = {"Urban", "Rural", "Highly rural"}
    if value not in allowed:
        raise ValueError(f"Unexpected Table 7 area type: {value!r}")
    return value


def normalize_wui_status(value: Any) -> str:
    value = clean(value)
    allowed = {"WUI", "Non-WUI"}
    if value not in allowed:
        raise ValueError(f"Unexpected Table 7 WUI status: {value!r}")
    return value


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
    """Read computed values from an XLSX worksheet without modifying it."""
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
        "Line Type",
        "HFTD Tier",
        "Area Type",
        "WUI Status",
        "Unit(s)",
        "Comments",
        "Blank Meaning",
    }
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Missing legacy Table 7 headers: {sorted(missing)}")

    unit_column = headers.index("Unit(s)")
    comments_column = headers.index("Comments")
    blank_meaning_column = headers.index("Blank Meaning")

    records = []
    metric_type = None

    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        if clean(row[2]) is not None:
            metric_type = clean(row[2])

        dimensions = [clean(row[column]) for column in range(3, 8)]
        if all(value is None for value in dimensions):
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_type_raw": metric_type,
                "legacy_metric_number_raw": dimensions[0],
                "line_type_raw": dimensions[1],
                "hftd_tier_raw": dimensions[2],
                "area_type_raw": dimensions[3],
                "wui_status_raw": dimensions[4],
                "unit_raw": clean(row[unit_column]),
                "comments": clean(row[comments_column]),
                "blank_meaning": clean(row[blank_meaning_column]),
            }
        )

    if len(records) != 180:
        raise AssertionError(
            f"Expected 180 legacy Table 7 rows; found {len(records)}"
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
        "LINE TYPE",
        "HFTD TIER",
        "AREA TYPE",
        "WUI STATUS",
        "UNIT(S)",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "REPORTING QUARTER",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in values[0][:13]]

    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Table 7 header does not match Data Guidelines v4.01.\n"
            f"Expected: {expected_headers}\n"
            f"Found: {actual_headers}"
        )

    records = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[1]) is None:
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_number": int(row[0]),
                "metric_type": clean(row[1]),
                "line_type": normalize_line_type(row[2]),
                "line_type_raw": clean(row[2]),
                "hftd_tier": normalize_hftd_tier(row[3]),
                "area_type": normalize_area_type(row[4]),
                "wui_status": normalize_wui_status(row[5]),
                "unit_raw": clean(row[6]),
                "comments": clean(row[7]),
                "blank_meaning": clean(row[8]),
                "utility_id": clean(row[9]),
                "reporting_year": int(row[10]),
                "reporting_quarter": int(row[11]),
                "actual_value": parse_number(row[12]),
            }
        )

    if len(records) != 180:
        raise AssertionError(
            f"Expected 180 v4.01 Table 7 rows; found {len(records)}"
        )

    return records


def semantic_key_from_legacy(
    record: dict[str, Any],
) -> tuple[str, str | None, str, str, str]:
    return (
        clean(record["metric_type_raw"]),
        normalize_line_type(record["line_type_raw"]),
        normalize_hftd_tier(record["hftd_tier_raw"]),
        normalize_area_type(record["area_type_raw"]),
        normalize_wui_status(record["wui_status_raw"]),
    )


def semantic_key_from_v4(
    record: dict[str, Any],
) -> tuple[str, str | None, str, str, str]:
    return (
        record["metric_type"],
        record["line_type"],
        record["hftd_tier"],
        record["area_type"],
        record["wui_status"],
    )


def build_crosswalk(
    v3_1_records: list[dict[str, Any]],
    v3_2_records: list[dict[str, Any]],
    v4_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    v3_1_by_legacy = {
        record["legacy_metric_number_raw"]: record
        for record in v3_1_records
    }
    v3_2_by_legacy = {
        record["legacy_metric_number_raw"]: record
        for record in v3_2_records
    }

    if len(v3_1_by_legacy) != 180 or len(v3_2_by_legacy) != 180:
        raise AssertionError("Legacy Table 7 identifiers are not unique")

    if set(v3_1_by_legacy) != set(v3_2_by_legacy):
        raise AssertionError(
            "The v3.1 and v3.2 Table 7 legacy identifier sets differ"
        )

    # Confirm that v3.1 and v3.2 retained the same semantic dimensions and units.
    for legacy_number, record_2023 in v3_1_by_legacy.items():
        record_2024 = v3_2_by_legacy[legacy_number]
        if semantic_key_from_legacy(record_2023) != semantic_key_from_legacy(
            record_2024
        ):
            raise AssertionError(
                f"Unexpected v3.1-to-v3.2 semantic change for {legacy_number}"
            )
        if record_2023["unit_raw"] != record_2024["unit_raw"]:
            raise AssertionError(
                f"Unexpected v3.1-to-v3.2 unit change for {legacy_number}"
            )

    v4_by_key = {
        semantic_key_from_v4(record): record
        for record in v4_records
    }
    if len(v4_by_key) != 180:
        raise AssertionError(
            "The 2025 Table 7 semantic dimension combinations are not unique"
        )

    expected_numbers = list(range(1070000000, 1070000180))
    actual_numbers = sorted(record["metric_number"] for record in v4_records)
    if actual_numbers != expected_numbers:
        raise AssertionError(
            "The 2025 Table 7 metric numbers are not the expected contiguous "
            "range 1070000000-1070000179"
        )

    crosswalk = []
    by_legacy_number = {}
    by_metric_number = {}

    for legacy_record in v3_2_records:
        key = semantic_key_from_legacy(legacy_record)
        if key not in v4_by_key:
            raise AssertionError(
                f"No v4.01 Table 7 match for legacy metric "
                f"{legacy_record['legacy_metric_number_raw']}: {key}"
            )

        v4_record = v4_by_key[key]
        if legacy_record["unit_raw"] != v4_record["unit_raw"]:
            raise AssertionError(
                f"Unexpected unit change for "
                f"{legacy_record['legacy_metric_number_raw']}"
            )

        line_type_changed_representation = (
            clean(legacy_record["line_type_raw"]) == "N/A"
            and v4_record["line_type_raw"] is None
        )
        status = (
            "normalized_legacy_na_to_null"
            if line_type_changed_representation
            else "exact_dimensions_and_unit"
        )
        note = (
            "Legacy LINE TYPE is the literal text N/A; v4.01 stores the "
            "not-applicable value as a blank. Both normalize to null."
            if line_type_changed_representation
            else None
        )

        item = {
            "metric_number": v4_record["metric_number"],
            "legacy_metric_number": legacy_record[
                "legacy_metric_number_raw"
            ],
            "metric_type": v4_record["metric_type"],
            "line_type_canonical": v4_record["line_type"],
            "line_type_legacy": legacy_record["line_type_raw"],
            "hftd_tier": v4_record["hftd_tier"],
            "area_type": v4_record["area_type"],
            "wui_status": v4_record["wui_status"],
            "unit_canonical": v4_record["unit_raw"],
            "crosswalk_status": status,
            "crosswalk_note": note,
        }

        crosswalk.append(item)
        by_legacy_number[item["legacy_metric_number"]] = item
        by_metric_number[item["metric_number"]] = item

    status_counts = Counter(
        item["crosswalk_status"] for item in crosswalk
    )
    expected_status_counts = Counter(
        {
            "exact_dimensions_and_unit": 72,
            "normalized_legacy_na_to_null": 108,
        }
    )
    if status_counts != expected_status_counts:
        raise AssertionError(
            f"Unexpected Table 7 crosswalk results: {status_counts}"
        )

    return crosswalk, by_legacy_number, by_metric_number


def validate_quarterly_schemas(
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> None:
    reference_2024 = parse_legacy_template(loaded[(2024, 4)])
    stable_legacy_fields = (
        "metric_type_raw",
        "legacy_metric_number_raw",
        "line_type_raw",
        "hftd_tier_raw",
        "area_type_raw",
        "wui_status_raw",
        "unit_raw",
    )
    reference_2024_key = [
        tuple(clean(record[field]) for field in stable_legacy_fields)
        for record in reference_2024
    ]

    for quarter in (1, 2, 3, 4):
        current = parse_legacy_template(loaded[(2024, quarter)])
        current_key = [
            tuple(clean(record[field]) for field in stable_legacy_fields)
            for record in current
        ]
        if current_key != reference_2024_key:
            raise AssertionError(
                f"2024 Q{quarter} Table 7 schema differs from Q4"
            )

    reference_2025 = parse_v4_template(loaded[(2025, 4)])
    stable_v4_fields = (
        "metric_number",
        "metric_type",
        "line_type",
        "hftd_tier",
        "area_type",
        "wui_status",
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
                f"2025 Q{quarter} Table 7 schema differs from Q4"
            )


def make_record_id(
    metric_number: int,
    year: int,
    quarter: int,
    source_file: str,
    source_row: int,
) -> str:
    payload = json.dumps(
        [
            metric_number,
            year,
            quarter,
            source_file,
            source_row,
        ],
        separators=(",", ":"),
    ).encode("utf-8")

    return "T7R-" + hashlib.sha1(payload).hexdigest()[:16]


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
    return [
        make_record_id(
            mapping["metric_number"],
            reporting_year,
            reporting_quarter,
            source["name"],
            source_record["source_row"],
        ),
        mapping["metric_number"],
        mapping["legacy_metric_number"],
        source_record["legacy_metric_number_raw"],
        mapping["crosswalk_status"],
        mapping["metric_type"],
        source_record["metric_type_raw"],
        mapping["line_type_canonical"],
        source_record["line_type_raw"],
        mapping["hftd_tier"],
        source_record["hftd_tier_raw"],
        mapping["area_type"],
        source_record["area_type_raw"],
        mapping["wui_status"],
        source_record["wui_status_raw"],
        source_record["unit_raw"],
        mapping["unit_canonical"],
        "exact",
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
        "Table 7",
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
    return [
        make_record_id(
            source_record["metric_number"],
            source_record["reporting_year"],
            source_record["reporting_quarter"],
            source["name"],
            source_record["source_row"],
        ),
        source_record["metric_number"],
        mapping["legacy_metric_number"],
        None,
        "v4_native",
        source_record["metric_type"],
        source_record["metric_type"],
        source_record["line_type"],
        source_record["line_type_raw"],
        source_record["hftd_tier"],
        source_record["hftd_tier"],
        source_record["area_type"],
        source_record["area_type"],
        source_record["wui_status"],
        source_record["wui_status"],
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
        "Table 7",
        source_record["source_row"],
        f"M{source_record['source_row']}",
        GUIDELINES[2025][1],
    ]


def build_unified_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], list[list[Any]]],
    by_legacy_number: dict[str, dict[str, Any]],
    by_metric_number: dict[int, dict[str, Any]],
) -> list[list[Any]]:
    output: list[list[Any]] = []

    # 2023: the Q4 v3.1 workbook contains separate Q1-Q4 2023 actuals.
    source = selected[(2023, 4)]
    values = loaded[(2023, 4)]
    records = parse_legacy_template(values)
    actual_columns = locate_2023_actual_columns(values)

    for source_record in records:
        mapping = by_legacy_number[
            source_record["legacy_metric_number_raw"]
        ]

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

    # 2024: each highest-revision v3.2 workbook supplies its subject quarter.
    for quarter in (1, 2, 3, 4):
        source = selected[(2024, quarter)]
        values = loaded[(2024, quarter)]
        records = parse_legacy_template(values)
        value_column = locate_2024_actual_column(values, quarter)

        for source_record in records:
            mapping = by_legacy_number[
                source_record["legacy_metric_number_raw"]
            ]
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

    # 2025: v4.01 is already long-form.
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
            if semantic_key_from_v4(source_record) != (
                mapping["metric_type"],
                mapping["line_type_canonical"],
                mapping["hftd_tier"],
                mapping["area_type"],
                mapping["wui_status"],
            ):
                raise AssertionError(
                    f"v4.01 crosswalk mismatch in {source['name']} "
                    f"row {source_record['source_row']}"
                )

            output.append(
                build_v4_output_row(
                    source_record=source_record,
                    mapping=mapping,
                    source=source,
                )
            )

    expected_rows = 180 * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified Table 7 records; "
            f"found {len(output)}"
        )

    return output


def build_crosswalk_rows(
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    return [
        [
            item["metric_number"],
            item["legacy_metric_number"],
            item["metric_type"],
            item["line_type_canonical"],
            item["line_type_legacy"],
            item["hftd_tier"],
            item["area_type"],
            item["wui_status"],
            item["unit_canonical"],
            item["crosswalk_status"],
            item["crosswalk_note"],
        ]
        for item in sorted(
            crosswalk,
            key=lambda row: row["metric_number"],
        )
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

    large_text = {
        "metric_type",
        "source_metric_type_raw",
        "comments",
        "blank_meaning",
        "crosswalk_note",
        "guideline_url",
    }
    medium_text = {
        "crosswalk_status",
        "source_file",
        "unit_raw",
        "unit_canonical",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 18

        if header in large_text:
            width = 42
        elif header in medium_text:
            width = 30

        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width

    sheet.get_range(
        f"A1:{last_column}{last_row}"
    ).format.wrap_text = True


def build_workbook(
    output_path: Path,
    unified_rows: list[list[Any]],
    crosswalk_rows: list[list[Any]],
    selected: dict[tuple[int, int], dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()

    readme = workbook.worksheets.add("README")
    actuals = workbook.worksheets.add("Unified Actuals")
    crosswalk_sheet = workbook.worksheets.add("Metric Crosswalk")
    changes = workbook.worksheets.add("Schema Changes")

    readme_rows = [
        ["SDG&E Table 7 Unified Dataset, 2023–2025", "", "", ""],
        [
            "Unified actual observations",
            validation["unified_rows"],
            "Metrics per quarter",
            180,
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
            "Exact semantic crosswalks",
            validation["crosswalk_status_counts"][
                "exact_dimensions_and_unit"
            ],
            "Legacy N/A normalized to null",
            validation["crosswalk_status_counts"][
                "normalized_legacy_na_to_null"
            ],
        ],
        [
            "Unit treatment",
            "All 180 metric/dimension combinations retain the same units "
            "across the three schema periods; no numeric conversion is applied.",
            "",
            "",
        ],
        [
            "LINE TYPE treatment",
            "For critical facilities, customer counts, substations, and "
            "weather stations, the legacy templates store the literal text "
            "'N/A' while v4.01 stores a blank. The unified line_type is null, "
            "and source_line_type_raw preserves the original representation.",
            "",
            "",
        ],
        [
            "Crosswalk validation",
            "All 180 legacy semantic combinations match exactly one v4.01 "
            "metric number.",
            "Unexpected changes",
            0,
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 7 values must be numeric, nonnegative, or blank and are "
            "broken down by HFTD, area type, and WUI status.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Table 7 retains the same metric purpose and value constraints.",
        ],
        [
            "v3.2 Change Documentation",
            "2024 transition",
            V3_2_CHANGELOG_URL,
            "Previously reported actual columns J:X were removed and the "
            "remaining actual year/quarter headers are populated from the "
            "cover-sheet reporting period.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Quarterly Table 7 contains subject-period actuals with explicit "
            "utility, year, quarter, and metric-number fields; projected "
            "annual values are reported in the Annual-WMP workbook.",
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
    for column, width in zip(("A", "B", "C", "D"), (34, 66, 68, 58)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(actuals, UNIFIED_HEADERS, unified_rows)
    format_sheet(
        actuals,
        UNIFIED_HEADERS,
        len(unified_rows),
        freeze_columns=5,
    )

    actual_value_column = column_letter(
        UNIFIED_HEADERS.index("actual_value")
    )
    actuals.get_range(
        f"{actual_value_column}2:"
        f"{actual_value_column}{len(unified_rows) + 1}"
    ).format.number_format = "0.########"

    status_column = column_letter(
        UNIFIED_HEADERS.index("crosswalk_status")
    )
    actuals.get_range(
        f"{status_column}2:"
        f"{status_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${status_column}2="normalized_legacy_na_to_null"',
        {"fill": "#DBEAFE", "font": {"color": "#1E3A8A"}},
    )

    write_rows(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        crosswalk_rows,
    )
    format_sheet(
        crosswalk_sheet,
        CROSSWALK_HEADERS,
        len(crosswalk_rows),
        freeze_columns=4,
    )

    schema_change_rows = [
        [
            "Actual-value layout",
            "Wide workbook with historical actual and future projection columns; "
            "the Q4 workbook contains all 2023 quarterly actuals.",
            "Only the subject-quarter actual remains, with projection columns "
            "still present.",
            "Long-form quarterly records with explicit metric number, utility, "
            "year, quarter, and actual value.",
            "Unpivot 2023; append each 2024/2025 quarter; exclude projections.",
        ],
        [
            "Metric identifier",
            "Legacy # identifier.",
            "Same legacy # identifier.",
            "Standard METRIC NUMBER values 1070000000–1070000179.",
            "Use v4.01 metric_number and retain the legacy # for lineage.",
        ],
        [
            "LINE TYPE not applicable",
            "Literal text N/A for 108 rows covering non-line equipment/customer metrics.",
            "Literal text N/A for the same 108 rows.",
            "Blank LINE TYPE for those rows.",
            "Normalize N/A and blank to null; preserve source_line_type_raw.",
        ],
        [
            "Semantic dimensions",
            "Metric type, line type, HFTD tier, area type, and WUI status.",
            "Same 180 combinations.",
            "Same 180 combinations.",
            "Require a one-to-one semantic crosswalk; fail on any mismatch.",
        ],
        [
            "Units",
            "Circuit miles or category-specific counts.",
            "Same units.",
            "Same units.",
            "No numeric conversion.",
        ],
        [
            "Projections",
            "Projected annual values are included in the wide workbook.",
            "Projected annual values remain in quarterly workbooks.",
            "Projected annual values are reported in Annual-WMP workbooks.",
            "Drop all projections as requested.",
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
            "Combine SDG&E Table 7 actual values for 2023-2025 into a "
            "v4.01-style long dataset. Projection values are excluded."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing the source SDG&E XLSX files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table7_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 7")
        for key, source in selected.items()
    }

    validate_quarterly_schemas(loaded)

    v3_1_records = parse_legacy_template(loaded[(2023, 4)])
    v3_2_records = parse_legacy_template(loaded[(2024, 4)])
    v4_records = parse_v4_template(loaded[(2025, 4)])

    crosswalk, by_legacy_number, by_metric_number = build_crosswalk(
        v3_1_records,
        v3_2_records,
        v4_records,
    )

    unified_rows = build_unified_rows(
        selected,
        loaded,
        by_legacy_number,
        by_metric_number,
    )
    metric_crosswalk_rows = build_crosswalk_rows(crosswalk)

    unified_csv = (
        output_dir / "sdge_table7_2023_2025_unified_actuals.csv"
    )
    crosswalk_csv = output_dir / "sdge_table7_metric_crosswalk.csv"
    workbook_path = (
        output_dir / "sdge_table7_2023_2025_unified.xlsx"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(
        crosswalk_csv,
        CROSSWALK_HEADERS,
        metric_crosswalk_rows,
    )

    validation = {
        "unified_rows": len(unified_rows),
        "metrics_per_quarter": 180,
        "reporting_periods": 12,
        "crosswalk_rows": len(metric_crosswalk_rows),
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
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        metric_crosswalk_rows,
        selected,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(
        "Crosswalk status:",
        validation["crosswalk_status_counts"],
    )


if __name__ == "__main__":
    main()
