
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
V4_CHANGELOG_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "fileid=57874&shareable=true"
)

UNIFIED_HEADERS = [
    "record_id",
    "metric_number",
    "legacy_metric_number_mapped",
    "source_legacy_metric_number_raw",
    "v3_1_to_v3_2_domain_status",
    "v3_2_to_v4_label_status",
    "metric_type",
    "source_metric_type_raw",
    "ignition_driver",
    "source_ignition_driver_raw",
    "line_type",
    "source_line_type_raw",
    "hftd_tier",
    "source_hftd_tier_raw",
    "unit_raw",
    "unit_canonical",
    "unit_crosswalk_status",
    "ignition_driver_tracked",
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
    "metric_type_v3_1",
    "ignition_driver_v3_1",
    "metric_type_v3_2",
    "ignition_driver_v3_2",
    "v3_1_to_v3_2_domain_status",
    "v3_1_to_v3_2_change_notes",
    "metric_type_v4",
    "ignition_driver_v4",
    "v3_2_to_v4_label_status",
    "v3_2_to_v4_change_notes",
    "line_type",
    "hftd_tier",
    "unit_v3",
    "unit_v4",
    "unit_crosswalk_status",
    "ignition_driver_tracked_2023",
    "ignition_driver_tracked_2024_source",
    "ignition_driver_tracked_2025",
    "tracked_status_change_2023_to_2025",
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
            raise ValueError(f"Table 6 actual value cannot be negative: {value}")
        return value

    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric Table 6 value, found {value!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(f"Table 6 actual value cannot be negative: {value}")
        return int(parsed) if parsed.is_integer() else parsed

    raise TypeError(f"Unsupported Table 6 value type: {type(value).__name__}")


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
    """Read computed values from one worksheet without altering the workbook."""
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


def normalize_line_type(value: Any) -> str:
    value = clean(value)
    if value not in {"Distribution", "Transmission"}:
        raise ValueError(f"Unexpected Table 6 line type: {value!r}")
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
        raise ValueError(f"Unexpected Table 6 HFTD tier: {value!r}")
    return mapping[value]


def normalize_tracked(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None

    normalized = str(value).casefold()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"

    raise ValueError(
        f"IGNITION DRIVER TRACKED must be Yes, No, or blank; found {value!r}"
    )


V3_1_METRIC_TYPE_CORRECTIONS = {
    "4. Equipment/facility failure or damage":
        "4. Equipment / facility failure or damage",
}

V3_1_DRIVER_CORRECTIONS = {
    "Insulator and brushing": "Insulator and bushing",
    "Lightning arrestor": "Lightning arrester",
}

V4_METRIC_TYPE_UPDATES = {
    "8. Vandalism/theft": "8. Vandalism / theft",
}

V4_DIRECT_DRIVER_UPDATES = {
    "Anchor/guy": "Anchor / guy",
    "Voltage regulator/booster": "Voltage regulator / booster",
    "Vandalism/theft": "Vandalism / theft",
}


def canonicalize_v3_1_dimensions(
    metric_type: Any,
    ignition_driver: Any,
    line_type: Any,
    hftd_tier: Any,
) -> tuple[str, str, str, str]:
    metric_type = clean(metric_type)
    ignition_driver = clean(ignition_driver)

    return (
        V3_1_METRIC_TYPE_CORRECTIONS.get(metric_type, metric_type),
        V3_1_DRIVER_CORRECTIONS.get(
            ignition_driver,
            ignition_driver,
        ),
        normalize_line_type(line_type),
        normalize_hftd_tier(hftd_tier),
    )


def canonicalize_v3_2_dimensions(
    metric_type: Any,
    ignition_driver: Any,
    line_type: Any,
    hftd_tier: Any,
) -> tuple[str, str, str, str]:
    metric_type = clean(metric_type)
    ignition_driver = clean(ignition_driver)

    metric_type = V4_METRIC_TYPE_UPDATES.get(
        metric_type,
        metric_type,
    )
    ignition_driver = V4_DIRECT_DRIVER_UPDATES.get(
        ignition_driver,
        ignition_driver,
    )

    if ignition_driver == "Unknown":
        if metric_type == "2. Contact from object":
            ignition_driver = "Unknown (contact from object)"
        elif metric_type == "4. Equipment / facility failure or damage":
            ignition_driver = "Unknown (equipment failure)"
        elif metric_type == "10. Unknown":
            ignition_driver = "Unknown (unknown)"
        else:
            raise ValueError(
                f"Unknown driver has unexpected metric type {metric_type!r}"
            )

    if ignition_driver == "Other":
        if metric_type == "2. Contact from object":
            ignition_driver = "Other (contact from object)"
        elif metric_type == "4. Equipment / facility failure or damage":
            ignition_driver = "Other (equipment failure)"
        elif metric_type == "11. Other":
            ignition_driver = "Other (other)"
        else:
            raise ValueError(
                f"Other driver has unexpected metric type {metric_type!r}"
            )

    return (
        metric_type,
        ignition_driver,
        normalize_line_type(line_type),
        normalize_hftd_tier(hftd_tier),
    )


def parse_legacy_template(values: list[list[Any]]) -> list[dict[str, Any]]:
    header = [clean(value) for value in values[8]]

    required_headers = {
        "Metric type",
        "#",
        "Ignition driver",
        "Line Type",
        "HFTD Tier",
        "Are ignitions tracked for ignition driver? (yes / no)",
        "Unit(s)",
        "Comments",
        "Blank Meaning",
    }
    missing = required_headers - set(header)
    if missing:
        raise ValueError(f"Missing legacy Table 6 headers: {sorted(missing)}")

    unit_column = header.index("Unit(s)")
    comments_column = header.index("Comments")
    blank_meaning_column = header.index("Blank Meaning")

    records = []
    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]

        dimensions = [clean(row[column]) for column in range(2, 8)]
        if all(value is None for value in dimensions):
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_type_raw": dimensions[0],
                "legacy_metric_number_raw": dimensions[1],
                "ignition_driver_raw": dimensions[2],
                "line_type_raw": dimensions[3],
                "hftd_tier_raw": dimensions[4],
                "tracked_raw": normalize_tracked(dimensions[5]),
                "unit_raw": clean(row[unit_column]),
                "comments": clean(row[comments_column]),
                "blank_meaning": clean(row[blank_meaning_column]),
            }
        )

    if len(records) != 222:
        raise AssertionError(
            f"Expected 222 legacy Table 6 rows; found {len(records)}"
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
        "IGNITION DRIVER",
        "LINE TYPE",
        "HFTD TIER",
        "UNIT(S)",
        "IGNITION DRIVER TRACKED",
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
            "The 2025 Table 6 header does not match Data Guidelines v4.01.\n"
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
                "ignition_driver": clean(row[2]),
                "line_type": normalize_line_type(row[3]),
                "hftd_tier": normalize_hftd_tier(row[4]),
                "unit_raw": clean(row[5]),
                "tracked": normalize_tracked(row[6]),
                "comments": clean(row[7]),
                "blank_meaning": clean(row[8]),
                "utility_id": clean(row[9]),
                "reporting_year": int(row[10]),
                "reporting_quarter": int(row[11]),
                "actual_value": parse_number(row[12]),
            }
        )

    if len(records) != 222:
        raise AssertionError(
            f"Expected 222 v4.01 Table 6 rows; found {len(records)}"
        )

    return records


def dimensions_from_legacy(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(record["metric_type_raw"]),
        clean(record["ignition_driver_raw"]),
        normalize_line_type(record["line_type_raw"]),
        normalize_hftd_tier(record["hftd_tier_raw"]),
    )


def dimensions_from_v4(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        record["metric_type"],
        record["ignition_driver"],
        record["line_type"],
        record["hftd_tier"],
    )


def change_notes(
    before: tuple[str, str, str, str],
    after: tuple[str, str, str, str],
) -> str | None:
    labels = (
        "metric type",
        "ignition driver",
        "line type",
        "HFTD tier",
    )
    notes = [
        f"{label}: {old!r} -> {new!r}"
        for label, old, new in zip(labels, before, after)
        if old != new
    ]
    return "; ".join(notes) if notes else None


def build_crosswalk(
    v3_1_records: list[dict[str, Any]],
    v3_2_records: list[dict[str, Any]],
    v4_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    v3_1_by_number = {
        record["legacy_metric_number_raw"]: record
        for record in v3_1_records
    }
    v3_2_by_number = {
        record["legacy_metric_number_raw"]: record
        for record in v3_2_records
    }

    if len(v3_1_by_number) != 222 or len(v3_2_by_number) != 222:
        raise AssertionError("Legacy Table 6 identifiers are not unique")

    if set(v3_1_by_number) != set(v3_2_by_number):
        raise AssertionError(
            "v3.1 and v3.2 Table 6 legacy identifier sets differ"
        )

    v4_by_dimensions = {
        dimensions_from_v4(record): record
        for record in v4_records
    }
    if len(v4_by_dimensions) != 222:
        raise AssertionError("v4.01 Table 6 dimension combinations are not unique")

    crosswalk = []
    by_legacy_number: dict[str, dict[str, Any]] = {}
    by_metric_number: dict[int, dict[str, Any]] = {}

    for legacy_number in sorted(
        v3_2_by_number,
        key=lambda value: v3_2_records.index(v3_2_by_number[value]),
    ):
        old_2023 = v3_1_by_number[legacy_number]
        old_2024 = v3_2_by_number[legacy_number]

        raw_2023_dimensions = dimensions_from_legacy(old_2023)
        corrected_2023_dimensions = canonicalize_v3_1_dimensions(
            old_2023["metric_type_raw"],
            old_2023["ignition_driver_raw"],
            old_2023["line_type_raw"],
            old_2023["hftd_tier_raw"],
        )
        raw_2024_dimensions = dimensions_from_legacy(old_2024)

        if corrected_2023_dimensions != raw_2024_dimensions:
            raise AssertionError(
                f"v3.1-to-v3.2 domain crosswalk failed for {legacy_number}:\n"
                f"{raw_2023_dimensions}\n-> {corrected_2023_dimensions}\n"
                f"expected {raw_2024_dimensions}"
            )

        canonical_dimensions = canonicalize_v3_2_dimensions(
            old_2024["metric_type_raw"],
            old_2024["ignition_driver_raw"],
            old_2024["line_type_raw"],
            old_2024["hftd_tier_raw"],
        )

        if canonical_dimensions not in v4_by_dimensions:
            raise AssertionError(
                f"No v4.01 Table 6 match for {legacy_number}: "
                f"{canonical_dimensions}"
            )

        new_2025 = v4_by_dimensions[canonical_dimensions]

        if old_2023["unit_raw"] != "# ignitions":
            raise AssertionError(
                f"Unexpected 2023 Table 6 unit for {legacy_number}"
            )
        if old_2024["unit_raw"] != "# ignitions":
            raise AssertionError(
                f"Unexpected 2024 Table 6 unit for {legacy_number}"
            )
        if new_2025["unit_raw"] != "# ignitions":
            raise AssertionError(
                f"Unexpected 2025 Table 6 unit for {legacy_number}"
            )

        v3_1_status = (
            "exact"
            if raw_2023_dimensions == corrected_2023_dimensions
            else "corrected_in_v3_2"
        )
        v4_status = (
            "exact"
            if raw_2024_dimensions == canonical_dimensions
            else "canonical_label_update"
        )
        tracked_change = (
            "same"
            if old_2023["tracked_raw"] == new_2025["tracked"]
            else "changed"
        )

        item = {
            "metric_number": new_2025["metric_number"],
            "legacy_metric_number": legacy_number,
            "metric_type_v3_1": old_2023["metric_type_raw"],
            "ignition_driver_v3_1": old_2023["ignition_driver_raw"],
            "metric_type_v3_2": old_2024["metric_type_raw"],
            "ignition_driver_v3_2": old_2024["ignition_driver_raw"],
            "v3_1_to_v3_2_domain_status": v3_1_status,
            "v3_1_to_v3_2_change_notes": change_notes(
                raw_2023_dimensions,
                corrected_2023_dimensions,
            ),
            "metric_type_v4": new_2025["metric_type"],
            "ignition_driver_v4": new_2025["ignition_driver"],
            "v3_2_to_v4_label_status": v4_status,
            "v3_2_to_v4_change_notes": change_notes(
                raw_2024_dimensions,
                canonical_dimensions,
            ),
            "line_type": new_2025["line_type"],
            "hftd_tier": new_2025["hftd_tier"],
            "unit_v3": old_2024["unit_raw"],
            "unit_v4": new_2025["unit_raw"],
            "unit_crosswalk_status": "exact",
            "ignition_driver_tracked_2023": old_2023["tracked_raw"],
            "ignition_driver_tracked_2024_source": old_2024["tracked_raw"],
            "ignition_driver_tracked_2025": new_2025["tracked"],
            "tracked_status_change_2023_to_2025": tracked_change,
        }

        crosswalk.append(item)
        by_legacy_number[legacy_number] = item
        by_metric_number[item["metric_number"]] = item

    if len(crosswalk) != 222:
        raise AssertionError("Expected 222 Table 6 crosswalk records")
    if len(by_metric_number) != 222:
        raise AssertionError("v4.01 Table 6 metric numbers are not unique")

    expected_v3_2_status = Counter(
        {"exact": 209, "corrected_in_v3_2": 13}
    )
    actual_v3_2_status = Counter(
        item["v3_1_to_v3_2_domain_status"]
        for item in crosswalk
    )
    if actual_v3_2_status != expected_v3_2_status:
        raise AssertionError(
            f"Unexpected v3.1-to-v3.2 corrections: {actual_v3_2_status}"
        )

    expected_v4_status = Counter(
        {"exact": 168, "canonical_label_update": 54}
    )
    actual_v4_status = Counter(
        item["v3_2_to_v4_label_status"]
        for item in crosswalk
    )
    if actual_v4_status != expected_v4_status:
        raise AssertionError(
            f"Unexpected v3.2-to-v4 label updates: {actual_v4_status}"
        )

    expected_tracking_status = Counter({"same": 216, "changed": 6})
    actual_tracking_status = Counter(
        item["tracked_status_change_2023_to_2025"]
        for item in crosswalk
    )
    if actual_tracking_status != expected_tracking_status:
        raise AssertionError(
            f"Unexpected tracking-status changes: {actual_tracking_status}"
        )

    return crosswalk, by_legacy_number, by_metric_number


def validate_quarterly_schemas(
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> None:
    reference_2024 = parse_legacy_template(loaded[(2024, 4)])
    stable_legacy_fields = (
        "metric_type_raw",
        "legacy_metric_number_raw",
        "ignition_driver_raw",
        "line_type_raw",
        "hftd_tier_raw",
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
                f"2024 Q{quarter} Table 6 schema differs from Q4"
            )

    reference_2025 = parse_v4_template(loaded[(2025, 4)])
    stable_v4_fields = (
        "metric_number",
        "metric_type",
        "ignition_driver",
        "line_type",
        "hftd_tier",
        "unit_raw",
        "tracked",
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
                f"2025 Q{quarter} Table 6 schema differs from Q4"
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

    return "T6R-" + hashlib.sha1(payload).hexdigest()[:16]


def build_legacy_output_row(
    *,
    source_record: dict[str, Any],
    mapping: dict[str, Any],
    actual_value: Any,
    utility_id: str,
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
        mapping["v3_1_to_v3_2_domain_status"],
        mapping["v3_2_to_v4_label_status"],
        mapping["metric_type_v4"],
        source_record["metric_type_raw"],
        mapping["ignition_driver_v4"],
        source_record["ignition_driver_raw"],
        mapping["line_type"],
        source_record["line_type_raw"],
        mapping["hftd_tier"],
        source_record["hftd_tier_raw"],
        source_record["unit_raw"],
        mapping["unit_v4"],
        mapping["unit_crosswalk_status"],
        source_record["tracked_raw"],
        parse_number(actual_value),
        source_record["comments"],
        source_record["blank_meaning"],
        utility_id,
        reporting_year,
        reporting_quarter,
        schema_version,
        source["revision"],
        source_report_quarter,
        source["name"],
        "Table 6",
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
            mapping["metric_number"],
            source_record["reporting_year"],
            source_record["reporting_quarter"],
            source["name"],
            source_record["source_row"],
        ),
        mapping["metric_number"],
        mapping["legacy_metric_number"],
        None,
        mapping["v3_1_to_v3_2_domain_status"],
        mapping["v3_2_to_v4_label_status"],
        source_record["metric_type"],
        source_record["metric_type"],
        source_record["ignition_driver"],
        source_record["ignition_driver"],
        source_record["line_type"],
        source_record["line_type"],
        source_record["hftd_tier"],
        source_record["hftd_tier"],
        source_record["unit_raw"],
        source_record["unit_raw"],
        "v4_native",
        source_record["tracked"],
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
        "Table 6",
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

    # 2023: all four subject-quarter actuals are in the Q4 v3.1 workbook.
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
                    utility_id="SDG&E",
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
                    utility_id="SDG&E",
                    reporting_year=2024,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2024][0],
                    guideline_url=GUIDELINES[2024][1],
                    source=source,
                    source_report_quarter=quarter,
                    source_value_column=value_column,
                )
            )

    # 2025: v4.01 rows already contain their reporting period.
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

            if dimensions_from_v4(source_record) != (
                mapping["metric_type_v4"],
                mapping["ignition_driver_v4"],
                mapping["line_type"],
                mapping["hftd_tier"],
            ):
                raise AssertionError(
                    f"v4.01 crosswalk mismatch at "
                    f"{source['name']} row {source_record['source_row']}"
                )

            output.append(
                build_v4_output_row(
                    source_record=source_record,
                    mapping=mapping,
                    source=source,
                )
            )

    expected_rows = 222 * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified records; found {len(output)}"
        )

    return output


def build_crosswalk_rows(
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    return [
        [
            item["metric_number"],
            item["legacy_metric_number"],
            item["metric_type_v3_1"],
            item["ignition_driver_v3_1"],
            item["metric_type_v3_2"],
            item["ignition_driver_v3_2"],
            item["v3_1_to_v3_2_domain_status"],
            item["v3_1_to_v3_2_change_notes"],
            item["metric_type_v4"],
            item["ignition_driver_v4"],
            item["v3_2_to_v4_label_status"],
            item["v3_2_to_v4_change_notes"],
            item["line_type"],
            item["hftd_tier"],
            item["unit_v3"],
            item["unit_v4"],
            item["unit_crosswalk_status"],
            item["ignition_driver_tracked_2023"],
            item["ignition_driver_tracked_2024_source"],
            item["ignition_driver_tracked_2025"],
            item["tracked_status_change_2023_to_2025"],
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
        "ignition_driver",
        "source_ignition_driver_raw",
        "comments",
        "blank_meaning",
        "v3_1_to_v3_2_change_notes",
        "v3_2_to_v4_change_notes",
        "guideline_url",
    }
    medium_text = {
        "v3_1_to_v3_2_domain_status",
        "v3_2_to_v4_label_status",
        "unit_crosswalk_status",
        "source_file",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 18

        if header in large_text:
            width = 40
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
        ["SDG&E Table 6 Unified Dataset, 2023–2025", "", "", ""],
        [
            "Unified actual observations",
            validation["unified_rows"],
            "Metrics per quarter",
            222,
        ],
        [
            "Reporting periods",
            "2023 Q1–Q4, 2024 Q1–Q4, 2025 Q1–Q4",
            "",
            "",
        ],
        [
            "Projection treatment",
            "All annual projection columns are excluded.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "The selected Q4 v3.1 workbook supplies separate Q1–Q4 "
            "2023 actual columns.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "Each selected highest-revision v3.2 workbook supplies its "
            "subject-quarter actual.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Each selected highest-revision v4.01 workbook is already "
            "long-form.",
            "",
            "",
        ],
        [
            "v3.1 → v3.2 domain corrections",
            validation["v3_1_to_v3_2_status_counts"][
                "corrected_in_v3_2"
            ],
            "Unchanged rows",
            validation["v3_1_to_v3_2_status_counts"]["exact"],
        ],
        [
            "v3.2 → v4.01 label updates",
            validation["v3_2_to_v4_status_counts"][
                "canonical_label_update"
            ],
            "Exact labels",
            validation["v3_2_to_v4_status_counts"]["exact"],
        ],
        [
            "Tracking-status changes",
            validation["tracking_change_counts"]["changed"],
            "Unchanged tracking status",
            validation["tracking_change_counts"]["same"],
        ],
        [
            "2024 tracking values",
            "SDG&E leaves the tracking field blank in all four 2024 "
            "workbooks. The converter preserves the blanks and does not "
            "backfill from another year.",
            "",
            "",
        ],
        [
            "Unit treatment",
            "All three schema periods use '# ignitions'; no numeric "
            "conversion is applied.",
            "",
            "",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 6 actual values must be numeric and nonnegative or "
            "blank, and must be consistent with spatial ignition data.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Table 6 retains the same ignition-reporting requirements.",
        ],
        [
            "v3.2 Change Log",
            "2024 transition",
            V3_2_CHANGELOG_URL,
            "Older actual-period columns were removed and Table 6 domain "
            "values were revised to align with like values in other tables.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Quarterly Table 6 reports actuals with explicit utility, year, "
            "quarter, and metric-number fields; projections are reported "
            "through the Annual-WMP workbook.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "The tracking column was renamed to IGNITION DRIVER TRACKED.",
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
        freeze_columns=6,
    )

    actual_value_column = column_letter(
        UNIFIED_HEADERS.index("actual_value")
    )
    actuals.get_range(
        f"{actual_value_column}2:"
        f"{actual_value_column}{len(unified_rows) + 1}"
    ).format.number_format = "0.########"

    v3_status_column = column_letter(
        UNIFIED_HEADERS.index("v3_1_to_v3_2_domain_status")
    )
    actuals.get_range(
        f"{v3_status_column}2:"
        f"{v3_status_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${v3_status_column}2="corrected_in_v3_2"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    v4_status_column = column_letter(
        UNIFIED_HEADERS.index("v3_2_to_v4_label_status")
    )
    actuals.get_range(
        f"{v4_status_column}2:"
        f"{v4_status_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${v4_status_column}2="canonical_label_update"',
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
            "Wide layout with multiple historical quarter columns, "
            "including 2023 Q1–Q4.",
            "Only the subject-quarter actual remains in each quarterly file.",
            "Long-form rows with explicit REPORTING YEAR, REPORTING QUARTER, "
            "and ACTUAL VALUE.",
            "Unpivot 2023 and append each 2024/2025 quarter.",
        ],
        [
            "Domain corrections",
            "Contains legacy forms such as Equipment/facility, Insulator and "
            "brushing, and Lightning arrestor.",
            "Domain values revised to Equipment / facility, Insulator and "
            "bushing, and Lightning arrester.",
            "Uses the corrected forms.",
            "Preserve raw labels and map the 13 affected rows to the corrected "
            "v3.2 dimensions.",
        ],
        [
            "Canonical labels",
            "Legacy driver naming.",
            "Legacy forms such as Anchor/guy, Voltage regulator/booster, "
            "Vandalism/theft, and context-free Unknown/Other.",
            "Standardized punctuation and context-qualified Unknown/Other labels.",
            "Use v4.01 canonical labels while preserving source labels.",
        ],
        [
            "Metric identifier",
            "Legacy # values.",
            "Same legacy # values.",
            "Standard METRIC NUMBER values 1060000000–1060000221.",
            "Use v4.01 metric_number and retain the legacy number for lineage.",
        ],
        [
            "Tracking column",
            "ARE IGNITIONS TRACKED FOR IGNITION DRIVER? populated Yes/No.",
            "Same legacy column, but SDG&E leaves it blank.",
            "IGNITION DRIVER TRACKED populated Yes/No.",
            "Preserve the source value; do not infer the 2024 status.",
        ],
        [
            "Tracking practice",
            "Six insulator/brushing combinations are marked No.",
            "Blank in SDG&E's files.",
            "The corresponding insulator/bushing combinations are marked Yes.",
            "Flag the six cross-year changes in the crosswalk; do not rewrite "
            "historical values.",
        ],
        [
            "Projection location",
            "Annual projection columns are present.",
            "Annual projection columns are present.",
            "Projections are submitted using the Annual-WMP workbook.",
            "Exclude all projection values as requested.",
        ],
        [
            "Unit",
            "# ignitions",
            "# ignitions",
            "# ignitions",
            "No conversion.",
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
            "Combine SDG&E Table 6 actual values for 2023-2025 into a "
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
        default="/mnt/data/table6_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 6")
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
        output_dir / "sdge_table6_2023_2025_unified_actuals.csv"
    )
    crosswalk_csv = output_dir / "sdge_table6_metric_crosswalk.csv"
    workbook_path = (
        output_dir / "sdge_table6_2023_2025_unified.xlsx"
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
        "metrics_per_quarter": 222,
        "reporting_periods": 12,
        "crosswalk_rows": len(metric_crosswalk_rows),
        "v3_1_to_v3_2_status_counts": dict(
            Counter(
                item["v3_1_to_v3_2_domain_status"]
                for item in crosswalk
            )
        ),
        "v3_2_to_v4_status_counts": dict(
            Counter(
                item["v3_2_to_v4_label_status"]
                for item in crosswalk
            )
        ),
        "tracking_change_counts": dict(
            Counter(
                item["tracked_status_change_2023_to_2025"]
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
        "v3.1 to v3.2:",
        validation["v3_1_to_v3_2_status_counts"],
    )
    print(
        "v3.2 to v4.01:",
        validation["v3_2_to_v4_status_counts"],
    )
    print(
        "Tracking changes:",
        validation["tracking_change_counts"],
    )


if __name__ == "__main__":
    main()
