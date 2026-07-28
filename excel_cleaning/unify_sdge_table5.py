
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
    "legacy_metric_number_status",
    "label_crosswalk_status",
    "metric_type",
    "source_metric_type_raw",
    "metric_name",
    "source_metric_name_raw",
    "risk_event_driver",
    "source_risk_event_driver_raw",
    "line_type",
    "source_line_type_raw",
    "hftd_tier",
    "source_hftd_tier_raw",
    "unit_raw",
    "unit_canonical",
    "unit_crosswalk_status",
    "risk_event_driver_tracked",
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
    "legacy_metric_number_v3_1",
    "legacy_metric_number_v3_2",
    "legacy_metric_number_status",
    "metric_type_canonical",
    "metric_type_legacy",
    "metric_name_canonical",
    "risk_event_category_legacy",
    "risk_event_driver_canonical",
    "risk_event_driver_legacy",
    "line_type",
    "hftd_tier",
    "label_crosswalk_status",
    "label_change_notes",
    "unit_legacy",
    "unit_v4",
    "unit_crosswalk_status",
    "risk_event_driver_tracked_2023",
    "risk_event_driver_tracked_2025",
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
        if value < 0:
            raise ValueError(f"Table 5 actual value cannot be negative: {value}")
        return value
    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric Table 5 value, found {value!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(f"Table 5 actual value cannot be negative: {value}")
        return int(parsed) if parsed.is_integer() else parsed
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


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


def normalize_line_type(value: Any) -> str:
    value = clean(value)
    if value not in {"Distribution", "Transmission"}:
        raise ValueError(f"Unexpected Table 5 line type: {value!r}")
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
        raise ValueError(f"Unexpected Table 5 HFTD tier: {value!r}")
    return mapping[value]


def normalize_tracked(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None

    normalized = str(value).strip().casefold()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"

    raise ValueError(
        f"RISK EVENT DRIVER TRACKED must be Yes, No, or blank; found {value!r}"
    )


METRIC_TYPE_MAP = {
    "7. Vandalism/theft": "7. Vandalism / theft",
    "13. Object contact": "13. Contact from object",
    "18. Vandalism/theft": "18. Vandalism / theft",
}

DIRECT_DRIVER_MAP = {
    "Other contact from object": "Other (contact from object)",
    "Anchor/guy": "Anchor / guy",
    "Connector device": "Connection device",
    "Voltage regulator/booster": "Voltage regulator / booster",
    "Vandalism/theft": "Vandalism / theft",
    "All Other": "Other (other)",
    "Lightning arrestor": "Lightning arrester",
}


def canonical_metric_type(value: Any) -> str:
    value = clean(value)
    if value is None:
        raise ValueError("Table 5 metric type cannot be blank")
    return METRIC_TYPE_MAP.get(value, value)


def canonical_driver(driver: Any, metric_type: Any) -> str:
    driver = clean(driver)
    metric_type = canonical_metric_type(metric_type)

    if driver is None:
        raise ValueError("Table 5 risk event driver cannot be blank")

    if driver in DIRECT_DRIVER_MAP:
        return DIRECT_DRIVER_MAP[driver]

    if driver == "Unknown":
        if metric_type in {
            "2. Contact from object",
            "13. Contact from object",
        }:
            return "Unknown (contact from object)"
        if metric_type in {
            "4. Equipment / facility failure or damage",
            "15. Equipment / facility failure or damage",
        }:
            return "Unknown (equipment failure)"
        if metric_type in {"9. Unknown", "23. Unknown"}:
            return "Unknown (unknown)"

    if driver == "Other":
        if metric_type in {
            "2. Contact from object",
            "13. Contact from object",
        }:
            return "Other (contact from object)"
        if metric_type in {
            "4. Equipment / facility failure or damage",
            "15. Equipment / facility failure or damage",
        }:
            return "Other (equipment failure)"

    return driver


def parse_legacy_template(values: list[list[Any]]) -> list[dict[str, Any]]:
    records = []

    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        metric_name = clean(row[1])
        metric_type = clean(row[2])
        legacy_metric_number = clean(row[3])
        risk_event_driver = clean(row[4])
        line_type = clean(row[5])
        hftd_tier = clean(row[6])

        if all(
            value is None
            for value in (
                metric_name,
                metric_type,
                legacy_metric_number,
                risk_event_driver,
                line_type,
                hftd_tier,
            )
        ):
            continue

        records.append(
            {
                "template_index": len(records),
                "source_row": zero_based_row + 1,
                "metric_name_raw": metric_name,
                "metric_type_raw": metric_type,
                "legacy_metric_number_raw": legacy_metric_number,
                "risk_event_driver_raw": risk_event_driver,
                "line_type_raw": line_type,
                "hftd_tier_raw": hftd_tier,
                "tracked_raw": normalize_tracked(row[7]),
                "unit_raw": clean(row[-3]),
                "comments": clean(row[-2]),
                "blank_meaning": clean(row[-1]),
            }
        )

    if len(records) != 456:
        raise AssertionError(
            f"Expected 456 legacy Table 5 rows; found {len(records)}"
        )
    return records


def locate_2023_actual_columns(
    values: list[list[Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}

    for column in range(len(values[8])):
        year = clean(values[8][column])
        quarter = clean(values[7][column])
        if year == 2023 and quarter in {"Q1", "Q2", "Q3", "Q4"}:
            result[int(str(quarter)[1])] = column

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
        year = clean(values[8][column])
        quarter = clean(values[7][column])
        if year == 2024 and quarter == f"Q{expected_quarter}":
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
        "RISK EVENT DRIVER",
        "LINE TYPE",
        "HFTD TIER",
        "UNIT(S)",
        "RISK EVENT DRIVER TRACKED",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "REPORTING QUARTER",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in values[0][:14]]

    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Table 5 header does not match Data Guidelines v4.01.\n"
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
                "risk_event_driver": clean(row[3]),
                "line_type": normalize_line_type(row[4]),
                "hftd_tier": normalize_hftd_tier(row[5]),
                "unit_raw": clean(row[6]),
                "tracked": normalize_tracked(row[7]),
                "comments": clean(row[8]),
                "blank_meaning": clean(row[9]),
                "utility_id": clean(row[10]),
                "reporting_year": int(row[11]),
                "reporting_quarter": int(row[12]),
                "actual_value": parse_number(row[13]),
            }
        )

    if len(records) != 456:
        raise AssertionError(
            f"Expected 456 v4.01 Table 5 rows; found {len(records)}"
        )
    return records


def label_change_notes(
    legacy: dict[str, Any],
    v4: dict[str, Any],
) -> str | None:
    notes = []

    if clean(legacy["metric_type_raw"]) != clean(v4["metric_type"]):
        notes.append(
            f"metric type: {legacy['metric_type_raw']!r} -> "
            f"{v4['metric_type']!r}"
        )

    if clean(legacy["risk_event_driver_raw"]) != clean(
        v4["risk_event_driver"]
    ):
        notes.append(
            f"risk event driver: {legacy['risk_event_driver_raw']!r} -> "
            f"{v4['risk_event_driver']!r}"
        )

    return "; ".join(notes) if notes else None


def build_crosswalk(
    v3_1_records: list[dict[str, Any]],
    v3_2_records: list[dict[str, Any]],
    v4_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not (
        len(v3_1_records)
        == len(v3_2_records)
        == len(v4_records)
        == 456
    ):
        raise AssertionError("All Table 5 schema versions must contain 456 rows")

    crosswalk = []
    by_template_index: dict[int, dict[str, Any]] = {}

    for old_2023, old_2024, new_2025 in zip(
        v3_1_records,
        v3_2_records,
        v4_records,
    ):
        if clean(old_2023["metric_name_raw"]) != clean(
            old_2024["metric_name_raw"]
        ):
            raise AssertionError(
                "The v3.1 and v3.2 risk event categories differ at "
                f"template row {old_2023['template_index']}"
            )

        if clean(old_2023["metric_type_raw"]) != clean(
            old_2024["metric_type_raw"]
        ):
            raise AssertionError(
                "The v3.1 and v3.2 metric types differ at "
                f"template row {old_2023['template_index']}"
            )

        driver_2023 = clean(old_2023["risk_event_driver_raw"])
        driver_2024 = clean(old_2024["risk_event_driver_raw"])
        if driver_2023 != driver_2024:
            if (driver_2023, driver_2024) != (
                "Lightning arrestor",
                "Lightning arrester",
            ):
                raise AssertionError(
                    "The v3.1 and v3.2 risk event drivers differ at "
                    f"template row {old_2023['template_index']}: "
                    f"{driver_2023!r} -> {driver_2024!r}"
                )

        if normalize_line_type(old_2024["line_type_raw"]) != new_2025[
            "line_type"
        ]:
            raise AssertionError("Legacy/v4.01 line type mismatch")

        if normalize_hftd_tier(old_2024["hftd_tier_raw"]) != new_2025[
            "hftd_tier"
        ]:
            raise AssertionError("Legacy/v4.01 HFTD tier mismatch")

        if clean(old_2024["metric_name_raw"]) != clean(
            new_2025["metric_name"]
        ):
            raise AssertionError(
                "Legacy Risk event category does not match v4.01 METRIC NAME"
            )

        canonical_type = canonical_metric_type(
            old_2024["metric_type_raw"]
        )
        canonical_risk_driver = canonical_driver(
            old_2024["risk_event_driver_raw"],
            old_2024["metric_type_raw"],
        )

        if canonical_type != new_2025["metric_type"]:
            raise AssertionError(
                f"Metric type crosswalk failed: "
                f"{old_2024['metric_type_raw']!r} -> {canonical_type!r}, "
                f"expected {new_2025['metric_type']!r}"
            )

        if canonical_risk_driver != new_2025["risk_event_driver"]:
            raise AssertionError(
                f"Risk event driver crosswalk failed: "
                f"{old_2024['risk_event_driver_raw']!r} -> "
                f"{canonical_risk_driver!r}, expected "
                f"{new_2025['risk_event_driver']!r}"
            )

        if old_2023["unit_raw"] != "# risk events (excluding ignitions)":
            raise AssertionError("Unexpected v3.1 Table 5 unit")
        if old_2024["unit_raw"] != "# risk events (excluding ignitions)":
            raise AssertionError("Unexpected v3.2 Table 5 unit")
        if new_2025["unit_raw"] != "# risk events":
            raise AssertionError("Unexpected v4.01 Table 5 unit")

        number_status = (
            "stable"
            if old_2023["legacy_metric_number_raw"]
            == old_2024["legacy_metric_number_raw"]
            else "corrected_in_v3_2"
        )
        label_status = (
            "exact_label_match"
            if (
                old_2024["metric_type_raw"] == new_2025["metric_type"]
                and old_2024["risk_event_driver_raw"]
                == new_2025["risk_event_driver"]
            )
            else "canonical_label_update"
        )

        tracked_2023 = old_2023["tracked_raw"]
        tracked_2025 = new_2025["tracked"]
        tracked_change = (
            "same"
            if tracked_2023 == tracked_2025
            else "changed"
        )

        item = {
            "template_index": old_2023["template_index"],
            "metric_number": new_2025["metric_number"],
            "legacy_metric_number_v3_1": old_2023[
                "legacy_metric_number_raw"
            ],
            "legacy_metric_number_v3_2": old_2024[
                "legacy_metric_number_raw"
            ],
            "legacy_metric_number_status": number_status,
            "metric_type_canonical": new_2025["metric_type"],
            "metric_type_legacy": old_2024["metric_type_raw"],
            "metric_name_canonical": new_2025["metric_name"],
            "risk_event_category_legacy": old_2024["metric_name_raw"],
            "risk_event_driver_canonical": new_2025[
                "risk_event_driver"
            ],
            "risk_event_driver_legacy": old_2024[
                "risk_event_driver_raw"
            ],
            "line_type": new_2025["line_type"],
            "hftd_tier": new_2025["hftd_tier"],
            "label_crosswalk_status": label_status,
            "label_change_notes": label_change_notes(
                old_2024,
                new_2025,
            ),
            "unit_legacy": old_2024["unit_raw"],
            "unit_v4": new_2025["unit_raw"],
            "unit_crosswalk_status": (
                "simplified_label_same_excludes_ignitions_definition"
            ),
            "risk_event_driver_tracked_2023": tracked_2023,
            "risk_event_driver_tracked_2025": tracked_2025,
            "tracked_status_change_2023_to_2025": tracked_change,
        }

        crosswalk.append(item)
        by_template_index[item["template_index"]] = item

    if len({item["metric_number"] for item in crosswalk}) != 456:
        raise AssertionError("v4.01 Table 5 metric numbers are not unique")

    number_counts = Counter(
        item["legacy_metric_number_status"]
        for item in crosswalk
    )
    if number_counts != Counter({"stable": 444, "corrected_in_v3_2": 12}):
        raise AssertionError(
            f"Unexpected v3.2 metric-number corrections: {number_counts}"
        )

    return crosswalk, by_template_index


def validate_quarterly_schemas(
    loaded: dict[tuple[int, int], list[list[Any]]],
) -> None:
    reference_2024 = parse_legacy_template(loaded[(2024, 4)])
    stable_legacy_fields = (
        "metric_name_raw",
        "metric_type_raw",
        "legacy_metric_number_raw",
        "risk_event_driver_raw",
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
                f"2024 Q{quarter} Table 5 schema differs from Q4"
            )

    reference_2025 = parse_v4_template(loaded[(2025, 4)])
    stable_v4_fields = (
        "metric_number",
        "metric_type",
        "metric_name",
        "risk_event_driver",
        "line_type",
        "hftd_tier",
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
                f"2025 Q{quarter} Table 5 schema differs from Q4"
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
    return "T5R-" + hashlib.sha1(payload).hexdigest()[:16]


def create_unified_row(
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
    v4_native: bool,
) -> list[Any]:
    if v4_native:
        source_metric_type = source_record["metric_type"]
        source_metric_name = source_record["metric_name"]
        source_driver = source_record["risk_event_driver"]
        source_line_type = source_record["line_type"]
        source_hftd = source_record["hftd_tier"]
        source_unit = source_record["unit_raw"]
        source_legacy_number = None
        tracked = source_record["tracked"]
        comments = source_record["comments"]
        blank_meaning = source_record["blank_meaning"]
    else:
        source_metric_type = source_record["metric_type_raw"]
        source_metric_name = source_record["metric_name_raw"]
        source_driver = source_record["risk_event_driver_raw"]
        source_line_type = source_record["line_type_raw"]
        source_hftd = source_record["hftd_tier_raw"]
        source_unit = source_record["unit_raw"]
        source_legacy_number = source_record[
            "legacy_metric_number_raw"
        ]
        tracked = source_record["tracked_raw"]
        comments = source_record["comments"]
        blank_meaning = source_record["blank_meaning"]

    return [
        make_record_id(
            mapping["metric_number"],
            reporting_year,
            reporting_quarter,
            source["name"],
            source_record["source_row"],
        ),
        mapping["metric_number"],
        mapping["legacy_metric_number_v3_2"],
        source_legacy_number,
        mapping["legacy_metric_number_status"],
        mapping["label_crosswalk_status"],
        mapping["metric_type_canonical"],
        source_metric_type,
        mapping["metric_name_canonical"],
        source_metric_name,
        mapping["risk_event_driver_canonical"],
        source_driver,
        mapping["line_type"],
        source_line_type,
        mapping["hftd_tier"],
        source_hftd,
        source_unit,
        mapping["unit_v4"],
        mapping["unit_crosswalk_status"],
        tracked,
        parse_number(actual_value),
        comments,
        blank_meaning,
        utility_id,
        reporting_year,
        reporting_quarter,
        schema_version,
        source["revision"],
        source_report_quarter,
        source["name"],
        "Table 5",
        source_record["source_row"],
        (
            f"{column_letter(source_value_column)}"
            f"{source_record['source_row']}"
        ),
        guideline_url,
    ]


def build_unified_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], list[list[Any]]],
    crosswalk_by_index: dict[int, dict[str, Any]],
) -> list[list[Any]]:
    output: list[list[Any]] = []

    # 2023 v3.1: Q4 workbook contains separate 2023 Q1-Q4 actual columns.
    source = selected[(2023, 4)]
    values = loaded[(2023, 4)]
    records = parse_legacy_template(values)
    actual_columns = locate_2023_actual_columns(values)

    for source_record in records:
        mapping = crosswalk_by_index[source_record["template_index"]]
        for quarter, value_column in sorted(actual_columns.items()):
            output.append(
                create_unified_row(
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
                    v4_native=False,
                )
            )

    # 2024 v3.2: each workbook contains its subject-quarter actual.
    for quarter in (1, 2, 3, 4):
        source = selected[(2024, quarter)]
        values = loaded[(2024, quarter)]
        records = parse_legacy_template(values)
        value_column = locate_2024_actual_column(values, quarter)

        for source_record in records:
            mapping = crosswalk_by_index[
                source_record["template_index"]
            ]
            output.append(
                create_unified_row(
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
                    v4_native=False,
                )
            )

    # 2025 v4.01: each row already identifies the reporting period.
    mapping_by_metric_number = {
        mapping["metric_number"]: mapping
        for mapping in crosswalk_by_index.values()
    }

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

            mapping = mapping_by_metric_number[
                source_record["metric_number"]
            ]
            output.append(
                create_unified_row(
                    source_record=source_record,
                    mapping=mapping,
                    actual_value=source_record["actual_value"],
                    utility_id=source_record["utility_id"],
                    reporting_year=2025,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2025][0],
                    guideline_url=GUIDELINES[2025][1],
                    source=source,
                    source_report_quarter=quarter,
                    source_value_column=13,
                    v4_native=True,
                )
            )

    expected_rows = 456 * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified rows; got {len(output)}"
        )

    return output


def build_crosswalk_rows(
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    return [
        [
            item["metric_number"],
            item["legacy_metric_number_v3_1"],
            item["legacy_metric_number_v3_2"],
            item["legacy_metric_number_status"],
            item["metric_type_canonical"],
            item["metric_type_legacy"],
            item["metric_name_canonical"],
            item["risk_event_category_legacy"],
            item["risk_event_driver_canonical"],
            item["risk_event_driver_legacy"],
            item["line_type"],
            item["hftd_tier"],
            item["label_crosswalk_status"],
            item["label_change_notes"],
            item["unit_legacy"],
            item["unit_v4"],
            item["unit_crosswalk_status"],
            item["risk_event_driver_tracked_2023"],
            item["risk_event_driver_tracked_2025"],
            item["tracked_status_change_2023_to_2025"],
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
    chunk_size: int = 250,
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

    text_heavy = {
        "metric_type",
        "source_metric_type_raw",
        "metric_name",
        "source_metric_name_raw",
        "risk_event_driver",
        "source_risk_event_driver_raw",
        "comments",
        "blank_meaning",
        "label_change_notes",
        "guideline_url",
    }
    medium = {
        "legacy_metric_number_status",
        "label_crosswalk_status",
        "unit_crosswalk_status",
        "source_file",
    }

    for column_index_, header in enumerate(headers):
        letter = column_letter(column_index_)
        width = 18
        if header in text_heavy:
            width = 40
        elif header in medium:
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
        ["SDG&E Table 5 Unified Dataset, 2023–2025", "", "", ""],
        [
            "Unified actual observations",
            validation["unified_rows"],
            "Metrics per quarter",
            456,
        ],
        [
            "Reporting periods",
            "2023 Q1–Q4, 2024 Q1–Q4, 2025 Q1–Q4",
            "",
            "",
        ],
        [
            "Projection treatment",
            "Projection columns are intentionally excluded.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "All four 2023 actual quarters are read from the selected "
            "Q4 v3.1 workbook.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "Each selected v3.2 workbook contributes its subject-quarter "
            "actual value.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Each selected v4.01 workbook is already long-form.",
            "",
            "",
        ],
        [
            "v3.2 legacy-number corrections",
            validation["legacy_metric_number_status_counts"][
                "corrected_in_v3_2"
            ],
            "Stable legacy numbers",
            validation["legacy_metric_number_status_counts"]["stable"],
        ],
        [
            "Canonical label updates",
            validation["label_crosswalk_status_counts"][
                "canonical_label_update"
            ],
            "Exact labels",
            validation["label_crosswalk_status_counts"][
                "exact_label_match"
            ],
        ],
        [
            "Tracking-field treatment",
            "The source Yes/No value is preserved. The 2024 SDG&E files "
            "leave the field blank; the converter does not backfill it.",
            "",
            "",
        ],
        [
            "Unit treatment",
            "The legacy label '# risk events (excluding ignitions)' is "
            "mapped to '# risk events'. No numeric conversion is applied "
            "because the guidelines continue to define Table 5 risk events "
            "as excluding ignitions.",
            "",
            "",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 5 covers wire-down and outage risk events; all reported "
            "risk events exclude ignitions.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Same Table 5 risk-event scope as v3.1.",
        ],
        [
            "v3.2 Template Change Log",
            "2024 transition",
            V3_2_CHANGELOG_URL,
            "Older actual columns were removed, reporting-period headers "
            "were populated from the cover sheet, and 12 legacy # values "
            "were corrected to reflect HFTD tier.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Quarterly Table 5 uses explicit UTILITY ID, REPORTING YEAR, "
            "REPORTING QUARTER, and ACTUAL VALUE fields; projections are "
            "reported in the Annual-WMP workbook.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "ARE RISK EVENTS TRACKED FOR IGNITION DRIVER? was renamed "
            "RISK EVENT DRIVER TRACKED.",
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
    for column, width in zip(("A", "B", "C", "D"), (34, 64, 68, 58)):
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
    actual_value_column = column_letter(
        UNIFIED_HEADERS.index("actual_value")
    )
    actuals.get_range(
        f"{actual_value_column}2:"
        f"{actual_value_column}{len(unified_rows) + 1}"
    ).format.number_format = "0.########"

    label_status_column = column_letter(
        UNIFIED_HEADERS.index("label_crosswalk_status")
    )
    actuals.get_range(
        f"{label_status_column}2:"
        f"{label_status_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${label_status_column}2="canonical_label_update"',
        {"fill": "#DBEAFE", "font": {"color": "#1E3A8A"}},
    )

    number_status_column = column_letter(
        UNIFIED_HEADERS.index("legacy_metric_number_status")
    )
    actuals.get_range(
        f"{number_status_column}2:"
        f"{number_status_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${number_status_column}2="corrected_in_v3_2"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
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
            "Historical columns include 2023 Q1–Q4 actuals.",
            "Only the subject reporting quarter remains in each workbook.",
            "One record per metric and reporting quarter.",
            "Unpivot 2023; append each 2024 and 2025 quarter.",
        ],
        [
            "Metric identifier",
            "Legacy # column; 12 rows contain duplicated incorrect HFTD suffixes.",
            "Legacy # column corrected for those 12 rows.",
            "Standard METRIC NUMBER values 1050000000–1050000455.",
            "Use v4.01 METRIC NUMBER; retain both v3.1 and v3.2 legacy identifiers.",
        ],
        [
            "Risk-event category",
            "RISK EVENT CATEGORY",
            "RISK EVENT CATEGORY",
            "METRIC NAME",
            "Map the legacy category to canonical metric_name.",
        ],
        [
            "Tracking field",
            "ARE RISK EVENTS TRACKED FOR IGNITION DRIVER? populated Yes/No.",
            "Same legacy column exists, but SDG&E leaves it blank in all four files.",
            "RISK EVENT DRIVER TRACKED, populated Yes/No.",
            "Preserve the source value; do not backfill 2024.",
        ],
        [
            "Driver terminology",
            "Some legacy labels use forms such as Connector device, "
            "Vandalism/theft, Unknown, Other, and Lightning arrestor.",
            "Same legacy terminology.",
            "Standardized forms such as Connection device, "
            "Vandalism / theft, context-qualified Unknown/Other, and "
            "Lightning arrester.",
            "Use v4.01 canonical labels while retaining source raw labels.",
        ],
        [
            "Unit label",
            "# risk events (excluding ignitions)",
            "# risk events (excluding ignitions)",
            "# risk events",
            "No numeric conversion. Guidelines continue to state that "
            "Table 5 risk events exclude ignitions.",
        ],
        [
            "Projections",
            "Annual projection columns are present.",
            "Annual projection columns are present.",
            "Projections are moved to the Annual-WMP workbook.",
            "Drop all projection values as requested.",
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
            "Combine SDG&E Table 5 actual values for 2023-2025 into a "
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
        default="/mnt/data/table5_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 5")
        for key, source in selected.items()
    }

    validate_quarterly_schemas(loaded)

    v3_1_records = parse_legacy_template(loaded[(2023, 4)])
    v3_2_records = parse_legacy_template(loaded[(2024, 4)])
    v4_records = parse_v4_template(loaded[(2025, 4)])

    crosswalk, crosswalk_by_index = build_crosswalk(
        v3_1_records,
        v3_2_records,
        v4_records,
    )
    unified_rows = build_unified_rows(
        selected,
        loaded,
        crosswalk_by_index,
    )
    metric_crosswalk_rows = build_crosswalk_rows(crosswalk)

    unified_csv = (
        output_dir / "sdge_table5_2023_2025_unified_actuals.csv"
    )
    crosswalk_csv = output_dir / "sdge_table5_metric_crosswalk.csv"
    workbook_path = (
        output_dir / "sdge_table5_2023_2025_unified.xlsx"
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
        "metrics_per_quarter": 456,
        "reporting_periods": 12,
        "crosswalk_rows": len(metric_crosswalk_rows),
        "legacy_metric_number_status_counts": dict(
            Counter(
                item["legacy_metric_number_status"]
                for item in crosswalk
            )
        ),
        "label_crosswalk_status_counts": dict(
            Counter(
                item["label_crosswalk_status"]
                for item in crosswalk
            )
        ),
        "tracked_status_change_counts": dict(
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
        "Legacy metric number status:",
        validation["legacy_metric_number_status_counts"],
    )
    print(
        "Label crosswalk status:",
        validation["label_crosswalk_status_counts"],
    )


if __name__ == "__main__":
    main()
