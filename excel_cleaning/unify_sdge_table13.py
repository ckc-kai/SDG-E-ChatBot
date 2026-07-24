
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta
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
SDGE_2023_SOURCE_PAGE = "https://www.sdge.com/2023-wildfire-mitigation-plan"

OFFICIAL_2023_DOWNLOADS = {
    1: (
        "SDGE_2023_Q1_Tables1-15_downloaded.xlsx",
        "https://www.sdge.com/sites/default/files/regulatory/"
        "SDGE%20Quarterly%20Data%20Report%20on%20Non-Spatial%20Data"
        "%205-01-2023.xlsx",
    ),
    2: (
        "SDGE_2023_Q2_Tables1-15_downloaded.xlsx",
        "https://www.sdge.com/sites/default/files/regulatory/"
        "SDGE%20Quarterly%20Data%20Report%20on%20Non-Spatial%20Data"
        "%208-01-2023.xlsx",
    ),
    3: (
        "SDGE_2023_Q3_Tables1-15_downloaded.xlsx",
        "https://www.sdge.com/sites/default/files/regulatory/"
        "SDGE%20Quarterly%20Data%20Report%20on%20Non-Spatial%20Data"
        "%2011-01-2023.xlsx",
    ),
}

LEGACY_HEADERS = [
    "a. Work order number",
    "b. Equipment Type",
    "c. HFTD Tier",
    "d. Line type",
    "e. Date the work order was originally opened",
    "f. Due date of the original work order",
    "g. GO 95 rule 18 priority level of the original work order",
    "h. Optional utility-specific repair priority",
    "i. Date(s) the work order was reinspected or modified (if applicable)",
    "j. Due date of the work order after it was reinspected or modified (if applicable)",
    "k. Priority of the work order after it was reinspected or modified (if applicable)",
    "l. Reason for reinspection (if applicable)",
]

V4_HEADERS = [
    "METRIC NUMBER",
    "WORK ORDER NUMBER",
    "EQUIPMENT TYPE",
    "HFTD TIER",
    "LINE TYPE",
    "DATE OPENED",
    "DUE DATE",
    "GO 95 RULE 18 PRIORITY",
    "UTILITY SPECIFIC REPAIR PRIORITY",
    "DATE REINSPECTED OR MODIFIED",
    "DUE DATE AFTER REINSPECTED OR MODIFIED",
    "GO 95 RULE 18 PRIORITY AFTER REINSPECTED OR MODIFIED",
    "REASON FOR REINSPECTION",
    "COMMENTS",
    "BLANK MEANING",
    "UTILITY ID",
    "REPORTING YEAR",
    "REPORTING QUARTER",
]

UNIFIED_HEADERS = [
    "record_id",
    "work_order_key",
    "work_order_key_method",
    "source_metric_number",
    "metric_number_scope",
    "work_order_number",
    "source_work_order_number_raw",
    "equipment_type",
    "hftd_tier",
    "source_hftd_tier_raw",
    "hftd_crosswalk_status",
    "line_type",
    "date_opened",
    "source_date_opened_raw",
    "due_date",
    "source_due_date_raw",
    "go95_rule18_priority",
    "source_go95_rule18_priority_raw",
    "go95_priority_crosswalk_status",
    "utility_specific_repair_priority",
    "source_utility_specific_repair_priority_raw",
    "utility_priority_crosswalk_status",
    "date_reinspected_or_modified",
    "due_date_after_reinspected_or_modified",
    "go95_priority_after_reinspected_or_modified",
    "reason_for_reinspection",
    "comments",
    "blank_meaning",
    "utility_id",
    "reporting_year",
    "reporting_quarter",
    "period_end_date",
    "schema_version",
    "source_revision",
    "source_file",
    "source_sheet",
    "source_row",
    "source_metric_cell",
    "source_url",
    "guideline_url",
    "exact_duplicate_group_key",
    "exact_duplicate_count",
    "exact_duplicate_index",
    "work_order_key_occurrence_count_in_period",
    "prior_period_presence_status",
    "first_seen_period",
    "last_seen_period",
    "quarters_observed",
]

LIFECYCLE_HEADERS = [
    "work_order_key",
    "work_order_key_method",
    "work_order_number",
    "equipment_type",
    "date_opened",
    "line_type",
    "first_seen_period",
    "last_seen_period",
    "quarters_observed",
    "snapshot_rows",
    "max_rows_in_single_snapshot",
    "ambiguous_multiple_rows_in_snapshot",
    "exact_duplicate_seen",
    "distinct_hftd_tiers",
    "distinct_due_dates",
    "distinct_go95_priorities",
    "present_in_2025_q4",
    "latest_hftd_tier",
    "latest_due_date",
    "latest_go95_priority",
]

QUARTER_SUMMARY_HEADERS = [
    "reporting_year",
    "reporting_quarter",
    "schema_version",
    "source_revision",
    "source_file",
    "source_rows",
    "unique_work_order_keys",
    "exact_duplicate_groups",
    "extra_exact_duplicate_rows",
    "newly_seen",
    "continued_unchanged",
    "continued_with_attribute_change",
    "reappeared_after_gap",
    "first_available_snapshot",
    "distribution_rows",
    "transmission_rows",
    "non_hftd_rows",
    "hftd_tier_2_rows",
    "hftd_tier_3_rows",
    "go95_priority_blank_rows",
    "go95_priority_level_2_rows",
    "go95_priority_level_3_rows",
]

SCHEMA_HEADERS = [
    "canonical_field",
    "v3_1_field",
    "v3_2_field",
    "v4_01_field",
    "change_status",
    "converter_action",
]

ISSUE_HEADERS = [
    "issue_type",
    "severity",
    "reporting_year",
    "reporting_quarter",
    "work_order_key",
    "work_order_number",
    "source_file",
    "source_rows",
    "field_name",
    "raw_value",
    "note",
]


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.replace("\xa0", " ").split())
        return normalized or None
    return value


def normalize_header(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None
    return str(value).strip().rstrip()


def normalize_identifier(value: Any) -> str:
    value = clean(value)
    if value is None:
        raise ValueError("WORK ORDER NUMBER cannot be blank")
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("WORK ORDER NUMBER cannot be NaN")
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value)


def parse_excel_date(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid date: {value!r}")

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        converted = datetime(1899, 12, 30) + timedelta(days=float(value))
        return converted.date().isoformat()

    if isinstance(value, datetime):
        return value.date().isoformat()

    text = str(value).strip()
    if text in {"#", "N/A", "NA", "Not Applicable"}:
        return None

    formats = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date value: {value!r}")


def normalize_hftd(value: Any) -> tuple[str, str]:
    raw = clean(value)
    mapping = {
        "Tier 2": ("HFTD Tier 2", "normalized_legacy_tier_prefix"),
        "Tier 3": ("HFTD Tier 3", "normalized_legacy_tier_prefix"),
        "HFTD Tier 2": ("HFTD Tier 2", "v4_native"),
        "HFTD Tier 3": ("HFTD Tier 3", "v4_native"),
        "Non-HFTD": ("Non-HFTD", "exact"),
    }
    if raw not in mapping:
        raise ValueError(f"Unexpected Table 13 HFTD value: {raw!r}")
    return mapping[raw]


def normalize_line_type(value: Any) -> str:
    raw = clean(value)
    if raw not in {"Distribution", "Transmission"}:
        raise ValueError(f"Unexpected Table 13 LINE TYPE: {raw!r}")
    return raw


def normalize_go95_priority(value: Any) -> tuple[str | None, str]:
    raw = clean(value)
    if raw is None:
        return None, "source_blank"

    text = str(raw).strip()
    if text in {"#", "N/A", "NA", "Not Applicable"}:
        if text == "#":
            return None, "normalized_legacy_hash_placeholder_to_null"
        return None, "normalized_legacy_na_to_null"

    if text in {"Level 1", "Level 2", "Level 3"}:
        return text, "preserved_priority"

    raise ValueError(f"Unexpected GO 95 priority value: {raw!r}")


def normalize_optional_text(value: Any) -> tuple[str | None, str]:
    raw = clean(value)
    if raw is None:
        return None, "source_blank"

    text = str(raw).strip()
    if text in {"N/A", "NA", "Not Applicable"}:
        return None, "normalized_na_to_null"

    return text, "preserved_text"


def period_end_date(year: int, quarter: int) -> str:
    return {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31",
    }[quarter]


def period_label(year: int, quarter: int) -> str:
    return f"{year}-Q{quarter}"


def stable_hash(prefix: str, payload: Any, length: int = 16) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return prefix + hashlib.sha1(encoded).hexdigest()[:length]


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

    revision_match = re.search(
        r"(?:_R|_Rev)(\d+)",
        path.name,
        re.IGNORECASE,
    )
    return {
        "path": path,
        "name": path.name,
        "year": int(match.group(1)),
        "quarter": int(match.group(2)),
        "revision_number": (
            int(revision_match.group(1))
            if revision_match
            else 0
        ),
        "revision": (
            f"R{int(revision_match.group(1))}"
            if revision_match
            else "R0"
        ),
    }


def ensure_official_2023_sources(input_dir: Path) -> None:
    for quarter, (filename, url) in OFFICIAL_2023_DOWNLOADS.items():
        existing = [
            path
            for path in input_dir.glob("*.xlsx")
            if (
                (parsed := parse_filename(path)) is not None
                and parsed["year"] == 2023
                and parsed["quarter"] == quarter
            )
        ]
        if existing:
            continue

        destination = input_dir / filename
        print(f"Downloading official 2023 Q{quarter} source: {url}")
        urllib.request.urlretrieve(url, destination)


def discover_sources(input_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    ensure_official_2023_sources(input_dir)

    candidates = []
    for path in input_dir.glob("*.xlsx"):
        parsed = parse_filename(path)
        if parsed is not None:
            candidates.append(parsed)

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for year in (2023, 2024, 2025):
        for quarter in (1, 2, 3, 4):
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
            source = max(
                matches,
                key=lambda item: item["revision_number"],
            )
            source["source_url"] = (
                OFFICIAL_2023_DOWNLOADS[quarter][1]
                if year == 2023 and quarter in OFFICIAL_2023_DOWNLOADS
                else None
            )
            selected[(year, quarter)] = source

    return selected


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[Any]]:
    """Read cached worksheet values without modifying the source XLSX."""
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


def core_business_fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["work_order_number"],
        record["equipment_type"],
        record["hftd_tier"],
        record["line_type"],
        record["date_opened"],
        record["due_date"],
        record["go95_rule18_priority"],
        record["utility_specific_repair_priority"],
        record["date_reinspected_or_modified"],
        record["due_date_after_reinspected_or_modified"],
        record["go95_priority_after_reinspected_or_modified"],
        record["reason_for_reinspection"],
    )


def parse_legacy(
    values: list[list[Any]],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    actual_headers = [
        normalize_header(value)
        for value in values[8][3:15]
    ]
    if actual_headers != LEGACY_HEADERS:
        raise ValueError(
            f"Legacy Table 13 schema mismatch in {source['name']}.\n"
            f"Expected: {LEGACY_HEADERS}\n"
            f"Found: {actual_headers}"
        )

    utility_raw = clean(values[3][4])
    if utility_raw not in {"SDG&E", "SDGE"}:
        raise AssertionError(
            f"Unexpected utility value in {source['name']}: {utility_raw!r}"
        )

    records = []
    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        if clean(row[3]) is None:
            continue

        hftd, hftd_status = normalize_hftd(row[5])
        go95, go95_status = normalize_go95_priority(row[9])
        utility_priority, utility_priority_status = normalize_optional_text(
            row[10]
        )
        priority_after, _ = normalize_optional_text(row[13])

        work_order_number = normalize_identifier(row[3])
        equipment_type = str(clean(row[4]))
        line_type = normalize_line_type(row[6])
        opened = parse_excel_date(row[7])
        due = parse_excel_date(row[8])
        reinspected = parse_excel_date(row[11])
        due_after = parse_excel_date(row[12])

        record = {
            "source_metric_number": None,
            "metric_number_scope": "not_present_in_v3",
            "work_order_number": work_order_number,
            "source_work_order_number_raw": clean(row[3]),
            "equipment_type": equipment_type,
            "hftd_tier": hftd,
            "source_hftd_tier_raw": clean(row[5]),
            "hftd_crosswalk_status": hftd_status,
            "line_type": line_type,
            "date_opened": opened,
            "source_date_opened_raw": clean(row[7]),
            "due_date": due,
            "source_due_date_raw": clean(row[8]),
            "go95_rule18_priority": go95,
            "source_go95_rule18_priority_raw": clean(row[9]),
            "go95_priority_crosswalk_status": go95_status,
            "utility_specific_repair_priority": utility_priority,
            "source_utility_specific_repair_priority_raw": clean(row[10]),
            "utility_priority_crosswalk_status": utility_priority_status,
            "date_reinspected_or_modified": reinspected,
            "due_date_after_reinspected_or_modified": due_after,
            "go95_priority_after_reinspected_or_modified": priority_after,
            "reason_for_reinspection": clean(row[14]),
            "comments": None,
            "blank_meaning": None,
            "utility_id": "SDG&E",
            "reporting_year": source["year"],
            "reporting_quarter": source["quarter"],
            "period_end_date": period_end_date(
                source["year"],
                source["quarter"],
            ),
            "schema_version": GUIDELINES[source["year"]][0],
            "source_revision": source["revision"],
            "source_file": source["name"],
            "source_sheet": "Table 13",
            "source_row": zero_based_row + 1,
            "source_metric_cell": None,
            "source_url": source.get("source_url"),
            "guideline_url": GUIDELINES[source["year"]][1],
        }

        record["work_order_key"] = stable_hash(
            "T13WO-",
            [
                record["work_order_number"],
                record["equipment_type"],
                record["date_opened"],
                record["line_type"],
            ],
            16,
        )
        record["work_order_key_method"] = (
            "sha1(work_order_number|equipment_type|date_opened|line_type)"
        )
        record["business_fingerprint"] = core_business_fingerprint(record)
        records.append(record)

    if not records:
        raise AssertionError(f"No Table 13 records found in {source['name']}")
    return records


def parse_v4(
    values: list[list[Any]],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    actual_headers = [
        normalize_header(value)
        for value in values[0][:18]
    ]
    if actual_headers != V4_HEADERS:
        raise ValueError(
            f"v4.01 Table 13 schema mismatch in {source['name']}.\n"
            f"Expected: {V4_HEADERS}\n"
            f"Found: {actual_headers}"
        )

    records = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[1]) is None:
            continue

        metric_number = int(row[0])
        hftd, hftd_status = normalize_hftd(row[3])
        go95, go95_status = normalize_go95_priority(row[7])
        utility_priority, utility_priority_status = normalize_optional_text(
            row[8]
        )
        priority_after, _ = normalize_optional_text(row[11])

        record = {
            "source_metric_number": metric_number,
            "metric_number_scope": (
                "quarter_scoped_sequential_record_identifier"
            ),
            "work_order_number": normalize_identifier(row[1]),
            "source_work_order_number_raw": clean(row[1]),
            "equipment_type": str(clean(row[2])),
            "hftd_tier": hftd,
            "source_hftd_tier_raw": clean(row[3]),
            "hftd_crosswalk_status": hftd_status,
            "line_type": normalize_line_type(row[4]),
            "date_opened": parse_excel_date(row[5]),
            "source_date_opened_raw": clean(row[5]),
            "due_date": parse_excel_date(row[6]),
            "source_due_date_raw": clean(row[6]),
            "go95_rule18_priority": go95,
            "source_go95_rule18_priority_raw": clean(row[7]),
            "go95_priority_crosswalk_status": go95_status,
            "utility_specific_repair_priority": utility_priority,
            "source_utility_specific_repair_priority_raw": clean(row[8]),
            "utility_priority_crosswalk_status": utility_priority_status,
            "date_reinspected_or_modified": parse_excel_date(row[9]),
            "due_date_after_reinspected_or_modified": parse_excel_date(row[10]),
            "go95_priority_after_reinspected_or_modified": priority_after,
            "reason_for_reinspection": clean(row[12]),
            "comments": clean(row[13]),
            "blank_meaning": clean(row[14]),
            "utility_id": clean(row[15]),
            "reporting_year": int(row[16]),
            "reporting_quarter": int(row[17]),
            "period_end_date": period_end_date(int(row[16]), int(row[17])),
            "schema_version": GUIDELINES[2025][0],
            "source_revision": source["revision"],
            "source_file": source["name"],
            "source_sheet": "Table 13",
            "source_row": zero_based_row + 1,
            "source_metric_cell": f"A{zero_based_row + 1}",
            "source_url": None,
            "guideline_url": GUIDELINES[2025][1],
        }

        if record["utility_id"] != "SDG&E":
            raise AssertionError(
                f"Unexpected utility ID in {source['name']} row "
                f"{record['source_row']}: {record['utility_id']!r}"
            )
        if record["reporting_year"] != source["year"]:
            raise AssertionError(
                f"Reporting year mismatch in {source['name']} row "
                f"{record['source_row']}"
            )
        if record["reporting_quarter"] != source["quarter"]:
            raise AssertionError(
                f"Reporting quarter mismatch in {source['name']} row "
                f"{record['source_row']}"
            )

        record["work_order_key"] = stable_hash(
            "T13WO-",
            [
                record["work_order_number"],
                record["equipment_type"],
                record["date_opened"],
                record["line_type"],
            ],
            16,
        )
        record["work_order_key_method"] = (
            "sha1(work_order_number|equipment_type|date_opened|line_type)"
        )
        record["business_fingerprint"] = core_business_fingerprint(record)
        records.append(record)

    expected_numbers = list(
        range(1130000000, 1130000000 + len(records))
    )
    actual_numbers = [
        record["source_metric_number"]
        for record in records
    ]
    if actual_numbers != expected_numbers:
        raise AssertionError(
            f"Metric numbers in {source['name']} are not sequential from "
            "1130000000"
        )

    return records


def validate_record(
    record: dict[str, Any],
    issues: list[list[Any]],
) -> None:
    required = (
        "work_order_number",
        "equipment_type",
        "hftd_tier",
        "line_type",
        "date_opened",
    )
    for field in required:
        if record[field] is None:
            issues.append(
                [
                    "missing_required_field",
                    "error",
                    record["reporting_year"],
                    record["reporting_quarter"],
                    record["work_order_key"],
                    record["work_order_number"],
                    record["source_file"],
                    record["source_row"],
                    field,
                    None,
                    "Required Table 13 field is blank.",
                ]
            )

    if (
        record["due_date"] is not None
        and record["date_opened"] is not None
        and record["due_date"] < record["date_opened"]
    ):
        issues.append(
            [
                "due_date_before_date_opened",
                "error",
                record["reporting_year"],
                record["reporting_quarter"],
                record["work_order_key"],
                record["work_order_number"],
                record["source_file"],
                record["source_row"],
                "due_date",
                record["due_date"],
                "Due date precedes the date the work order was opened.",
            ]
        )

    modified_fields = (
        record["due_date_after_reinspected_or_modified"],
        record["go95_priority_after_reinspected_or_modified"],
        record["reason_for_reinspection"],
    )
    if (
        record["date_reinspected_or_modified"] is None
        and any(value is not None for value in modified_fields)
    ):
        issues.append(
            [
                "modified_detail_without_modified_date",
                "warning",
                record["reporting_year"],
                record["reporting_quarter"],
                record["work_order_key"],
                record["work_order_number"],
                record["source_file"],
                record["source_row"],
                "date_reinspected_or_modified",
                None,
                "A post-reinspection field is populated without a "
                "reinspection/modification date.",
            ]
        )

    if (
        record["reporting_year"] == 2025
        and record["due_date"] is None
        and record["go95_rule18_priority"] is None
        and record["blank_meaning"] is None
    ):
        issues.append(
            [
                "blank_required_fields_without_blank_meaning",
                "warning",
                record["reporting_year"],
                record["reporting_quarter"],
                record["work_order_key"],
                record["work_order_number"],
                record["source_file"],
                record["source_row"],
                "blank_meaning",
                None,
                "DUE DATE and GO 95 priority are blank but BLANK MEANING "
                "is also blank.",
            ]
        )


def assign_duplicate_metadata(
    period_records: list[dict[str, Any]],
    issues: list[list[Any]],
) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in period_records:
        groups[record["business_fingerprint"]].append(record)

    for fingerprint, records in groups.items():
        group_key = stable_hash("T13DUP-", list(fingerprint), 14)
        count = len(records)
        for index, record in enumerate(records, start=1):
            record["exact_duplicate_group_key"] = group_key
            record["exact_duplicate_count"] = count
            record["exact_duplicate_index"] = index

        if count > 1:
            example = records[0]
            issues.append(
                [
                    "potential_exact_duplicate_source_rows",
                    "warning",
                    example["reporting_year"],
                    example["reporting_quarter"],
                    example["work_order_key"],
                    example["work_order_number"],
                    example["source_file"],
                    ";".join(
                        str(record["source_row"])
                        for record in records
                    ),
                    "core_work_order_fields",
                    None,
                    "Multiple source rows have identical normalized Table 13 "
                    "business fields. They are preserved because the source "
                    "does not provide an asset-level identifier that proves "
                    "they are accidental duplicates.",
                ]
            )


def assign_lifecycle_metadata(
    period_records: dict[tuple[int, int], list[dict[str, Any]]],
) -> None:
    ordered_periods = sorted(period_records)

    all_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    period_keys: dict[tuple[int, int], set[str]] = {}
    period_fingerprints: dict[
        tuple[int, int],
        dict[str, set[tuple[Any, ...]]],
    ] = {}

    for period in ordered_periods:
        key_counter = Counter(
            record["work_order_key"]
            for record in period_records[period]
        )
        period_keys[period] = set(key_counter)
        fingerprint_map: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

        for record in period_records[period]:
            record["work_order_key_occurrence_count_in_period"] = (
                key_counter[record["work_order_key"]]
            )
            all_by_key[record["work_order_key"]].append(record)
            fingerprint_map[record["work_order_key"]].add(
                record["business_fingerprint"]
            )

        period_fingerprints[period] = fingerprint_map

    history_periods_by_key: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for period in ordered_periods:
        for key in period_keys[period]:
            history_periods_by_key[key].append(period)

    for period_index, period in enumerate(ordered_periods):
        previous_period = (
            ordered_periods[period_index - 1]
            if period_index > 0
            else None
        )

        for record in period_records[period]:
            key = record["work_order_key"]
            observed_periods = history_periods_by_key[key]
            record["first_seen_period"] = period_label(*observed_periods[0])
            record["last_seen_period"] = period_label(*observed_periods[-1])
            record["quarters_observed"] = len(observed_periods)

            if previous_period is None:
                status = "first_available_snapshot"
            elif key in period_keys[previous_period]:
                if (
                    record["business_fingerprint"]
                    in period_fingerprints[previous_period].get(key, set())
                ):
                    status = "continued_unchanged"
                else:
                    status = "continued_with_attribute_change"
            else:
                earlier_periods = [
                    candidate
                    for candidate in observed_periods
                    if candidate < period
                ]
                status = (
                    "reappeared_after_gap"
                    if earlier_periods
                    else "newly_seen"
                )
            record["prior_period_presence_status"] = status


def build_lifecycle_rows(
    period_records: dict[tuple[int, int], list[dict[str, Any]]],
) -> list[list[Any]]:
    all_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts_by_key_period: Counter[tuple[str, int, int]] = Counter()

    for period, records in period_records.items():
        for record in records:
            all_by_key[record["work_order_key"]].append(record)
            counts_by_key_period[
                (
                    record["work_order_key"],
                    period[0],
                    period[1],
                )
            ] += 1

    rows = []
    for key, records in sorted(all_by_key.items()):
        records = sorted(
            records,
            key=lambda record: (
                record["reporting_year"],
                record["reporting_quarter"],
                record["source_row"],
            ),
        )
        periods = sorted(
            {
                (
                    record["reporting_year"],
                    record["reporting_quarter"],
                )
                for record in records
            }
        )
        latest = records[-1]
        max_rows = max(
            counts_by_key_period[(key, year, quarter)]
            for year, quarter in periods
        )

        rows.append(
            [
                key,
                records[0]["work_order_key_method"],
                records[0]["work_order_number"],
                records[0]["equipment_type"],
                records[0]["date_opened"],
                records[0]["line_type"],
                period_label(*periods[0]),
                period_label(*periods[-1]),
                len(periods),
                len(records),
                max_rows,
                max_rows > 1,
                any(
                    record["exact_duplicate_count"] > 1
                    for record in records
                ),
                ";".join(
                    sorted(
                        {
                            record["hftd_tier"]
                            for record in records
                            if record["hftd_tier"] is not None
                        }
                    )
                )
                or None,
                ";".join(
                    sorted(
                        {
                            record["due_date"]
                            for record in records
                            if record["due_date"] is not None
                        }
                    )
                )
                or None,
                ";".join(
                    sorted(
                        {
                            record["go95_rule18_priority"]
                            for record in records
                            if record["go95_rule18_priority"] is not None
                        }
                    )
                )
                or None,
                (2025, 4) in periods,
                latest["hftd_tier"],
                latest["due_date"],
                latest["go95_rule18_priority"],
            ]
        )

    return rows


def record_to_row(record: dict[str, Any]) -> list[Any]:
    record_id = stable_hash(
        "T13R-",
        [
            record["source_file"],
            record["source_row"],
            record["reporting_year"],
            record["reporting_quarter"],
        ],
        16,
    )

    return [
        record_id,
        record["work_order_key"],
        record["work_order_key_method"],
        record["source_metric_number"],
        record["metric_number_scope"],
        record["work_order_number"],
        record["source_work_order_number_raw"],
        record["equipment_type"],
        record["hftd_tier"],
        record["source_hftd_tier_raw"],
        record["hftd_crosswalk_status"],
        record["line_type"],
        record["date_opened"],
        record["source_date_opened_raw"],
        record["due_date"],
        record["source_due_date_raw"],
        record["go95_rule18_priority"],
        record["source_go95_rule18_priority_raw"],
        record["go95_priority_crosswalk_status"],
        record["utility_specific_repair_priority"],
        record["source_utility_specific_repair_priority_raw"],
        record["utility_priority_crosswalk_status"],
        record["date_reinspected_or_modified"],
        record["due_date_after_reinspected_or_modified"],
        record["go95_priority_after_reinspected_or_modified"],
        record["reason_for_reinspection"],
        record["comments"],
        record["blank_meaning"],
        record["utility_id"],
        record["reporting_year"],
        record["reporting_quarter"],
        record["period_end_date"],
        record["schema_version"],
        record["source_revision"],
        record["source_file"],
        record["source_sheet"],
        record["source_row"],
        record["source_metric_cell"],
        record["source_url"],
        record["guideline_url"],
        record["exact_duplicate_group_key"],
        record["exact_duplicate_count"],
        record["exact_duplicate_index"],
        record["work_order_key_occurrence_count_in_period"],
        record["prior_period_presence_status"],
        record["first_seen_period"],
        record["last_seen_period"],
        record["quarters_observed"],
    ]


def build_quarter_summary(
    period_records: dict[tuple[int, int], list[dict[str, Any]]],
    sources: dict[tuple[int, int], dict[str, Any]],
) -> list[list[Any]]:
    rows = []
    for (year, quarter), records in sorted(period_records.items()):
        duplicate_groups = {
            record["exact_duplicate_group_key"]
            for record in records
            if record["exact_duplicate_count"] > 1
        }
        statuses = Counter(
            record["prior_period_presence_status"]
            for record in records
        )
        hftd = Counter(record["hftd_tier"] for record in records)
        line = Counter(record["line_type"] for record in records)
        priority = Counter(
            record["go95_rule18_priority"]
            for record in records
        )

        rows.append(
            [
                year,
                quarter,
                GUIDELINES[year][0],
                sources[(year, quarter)]["revision"],
                sources[(year, quarter)]["name"],
                len(records),
                len(
                    {
                        record["work_order_key"]
                        for record in records
                    }
                ),
                len(duplicate_groups),
                sum(
                    record["exact_duplicate_count"] - 1
                    for record in records
                    if record["exact_duplicate_index"] == 1
                    and record["exact_duplicate_count"] > 1
                ),
                statuses["newly_seen"],
                statuses["continued_unchanged"],
                statuses["continued_with_attribute_change"],
                statuses["reappeared_after_gap"],
                statuses["first_available_snapshot"],
                line["Distribution"],
                line["Transmission"],
                hftd["Non-HFTD"],
                hftd["HFTD Tier 2"],
                hftd["HFTD Tier 3"],
                priority[None],
                priority["Level 2"],
                priority["Level 3"],
            ]
        )
    return rows


def build_schema_rows() -> list[list[Any]]:
    mappings = [
        (
            "source_metric_number",
            None,
            None,
            "METRIC NUMBER",
            "new_in_v4_01",
            "Preserve as a source-quarter row identifier. Do not use it "
            "to join work orders across quarters because numbering restarts "
            "at 1130000000 in every quarterly workbook.",
        ),
        (
            "work_order_number",
            LEGACY_HEADERS[0],
            LEGACY_HEADERS[0],
            "WORK ORDER NUMBER",
            "renamed",
            "Convert numeric-looking values to text without decimal suffixes.",
        ),
        (
            "equipment_type",
            LEGACY_HEADERS[1],
            LEGACY_HEADERS[1],
            "EQUIPMENT TYPE",
            "capitalization_standardized",
            "Preserve the source equipment description.",
        ),
        (
            "hftd_tier",
            LEGACY_HEADERS[2],
            LEGACY_HEADERS[2],
            "HFTD TIER",
            "labels_standardized",
            "Normalize legacy Tier 2/Tier 3 to HFTD Tier 2/HFTD Tier 3.",
        ),
        (
            "line_type",
            LEGACY_HEADERS[3],
            LEGACY_HEADERS[3],
            "LINE TYPE",
            "capitalization_standardized",
            "Require Distribution or Transmission.",
        ),
        (
            "date_opened",
            LEGACY_HEADERS[4],
            LEGACY_HEADERS[4],
            "DATE OPENED",
            "renamed",
            "Convert Excel serial dates to ISO YYYY-MM-DD.",
        ),
        (
            "due_date",
            LEGACY_HEADERS[5],
            LEGACY_HEADERS[5],
            "DUE DATE",
            "renamed",
            "Convert Excel serial dates to ISO YYYY-MM-DD.",
        ),
        (
            "go95_rule18_priority",
            LEGACY_HEADERS[6],
            LEGACY_HEADERS[6],
            "GO 95 RULE 18 PRIORITY",
            "renamed_and_blank_representation_standardized",
            "Preserve Level 1/2/3. Normalize legacy #, N/A, and NA "
            "placeholders to null while preserving the raw source value.",
        ),
        (
            "utility_specific_repair_priority",
            LEGACY_HEADERS[7],
            LEGACY_HEADERS[7],
            "UTILITY SPECIFIC REPAIR PRIORITY",
            "renamed",
            "Normalize legacy N/A/NA to null; preserve other text.",
        ),
        (
            "date_reinspected_or_modified",
            LEGACY_HEADERS[8],
            LEGACY_HEADERS[8],
            "DATE REINSPECTED OR MODIFIED",
            "renamed",
            "Convert populated dates to ISO YYYY-MM-DD.",
        ),
        (
            "due_date_after_reinspected_or_modified",
            LEGACY_HEADERS[9],
            LEGACY_HEADERS[9],
            "DUE DATE AFTER REINSPECTED OR MODIFIED",
            "renamed",
            "Convert populated dates to ISO YYYY-MM-DD.",
        ),
        (
            "go95_priority_after_reinspected_or_modified",
            LEGACY_HEADERS[10],
            LEGACY_HEADERS[10],
            "GO 95 RULE 18 PRIORITY AFTER REINSPECTED OR MODIFIED",
            "renamed_and_clarified",
            "Preserve populated text; normalize N/A/NA to null.",
        ),
        (
            "reason_for_reinspection",
            LEGACY_HEADERS[11],
            LEGACY_HEADERS[11],
            "REASON FOR REINSPECTION",
            "renamed",
            "Preserve source text.",
        ),
        (
            "comments",
            None,
            None,
            "COMMENTS",
            "new_in_v4_01",
            "Preserve source comments.",
        ),
        (
            "blank_meaning",
            None,
            None,
            "BLANK MEANING",
            "new_in_v4_01",
            "Preserve the v4.01 explanation for intentionally blank fields.",
        ),
        (
            "utility_id",
            "Utility workbook metadata",
            "Utility workbook metadata",
            "UTILITY ID",
            "moved_into_each_row",
            "Normalize SDGE and SDG&E to SDG&E.",
        ),
        (
            "reporting_year",
            "Derived from source quarter",
            "Derived from source quarter",
            "REPORTING YEAR",
            "new_row_metadata_in_v4_01",
            "Validate against the source filename.",
        ),
        (
            "reporting_quarter",
            "Derived from source quarter",
            "Derived from source quarter",
            "REPORTING QUARTER",
            "new_row_metadata_in_v4_01",
            "Validate against the source filename.",
        ),
    ]
    return [list(item) for item in mappings]


def csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("="):
        return "'" + value
    return value


def write_csv(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([csv_safe(value) for value in row])


def excel_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("="):
        return "'" + value
    return value


def write_rows(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    *,
    chunk_size: int = 1000,
) -> None:
    sheet.get_range_by_indexes(0, 0, 1, len(headers)).values = [headers]

    for start in range(0, len(rows), chunk_size):
        chunk = [
            [excel_safe(value) for value in row]
            for row in rows[start : start + chunk_size]
        ]
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
        "row_height": 36,
    }
    sheet.freeze_panes.freeze_rows(1)
    sheet.freeze_panes.freeze_columns(freeze_columns)

    large_text = {
        "work_order_key_method",
        "equipment_type",
        "reason_for_reinspection",
        "comments",
        "blank_meaning",
        "source_file",
        "source_url",
        "guideline_url",
        "converter_action",
        "note",
    }
    medium_text = {
        "metric_number_scope",
        "hftd_crosswalk_status",
        "go95_priority_crosswalk_status",
        "utility_priority_crosswalk_status",
        "prior_period_presence_status",
        "source_rows",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 17
        if header in large_text:
            width = 38
        elif header in medium_text:
            width = 28
        elif "date" in header or "period" in header:
            width = 16

        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width

    sheet.get_range(
        f"A1:{last_column}{last_row}"
    ).format.wrap_text = True


def build_workbook(
    output_path: Path,
    unified_rows: list[list[Any]],
    lifecycle_rows: list[list[Any]],
    quarter_summary_rows: list[list[Any]],
    schema_rows: list[list[Any]],
    issue_rows: list[list[Any]],
    sources: dict[tuple[int, int], dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()
    readme = workbook.worksheets.add("README")
    summary = workbook.worksheets.add("Quarter Summary")
    snapshots = workbook.worksheets.add("Unified Preview")
    lifecycle = workbook.worksheets.add("Lifecycle Summary")
    schema = workbook.worksheets.add("Schema Crosswalk")
    issues = workbook.worksheets.add("Validation Issues")

    readme_rows = [
        ["SDG&E Table 13 Unified Open-Work-Order Snapshots, 2023–2025", "", "", ""],
        [
            "Unified snapshot rows in full CSV",
            validation["unified_rows"],
            "Reporting periods",
            12,
        ],
        [
            "Workbook snapshot preview rows",
            min(validation["unified_rows"], 500),
            "Full dataset location",
            "sdge_table13_2023_2025_unified_snapshots.csv",
        ],
        [
            "Distinct composite work-order keys",
            validation["distinct_work_order_keys"],
            "Lifecycle summary rows",
            validation["lifecycle_rows"],
        ],
        [
            "Projection handling",
            "Table 13 contains no projection fields; no projections are included.",
            "",
            "",
        ],
        [
            "Snapshot interpretation",
            "Each quarterly file is a complete list of work orders open at "
            "quarter end. The same work order can therefore appear in multiple "
            "quarters and should not be deduplicated across periods.",
            "",
            "",
        ],
        [
            "Metric-number warning",
            "The v4.01 METRIC NUMBER restarts at 1130000000 each quarter and "
            "identifies a row only within its source submission. It is not a "
            "stable work-order identifier.",
            "",
            "",
        ],
        [
            "Composite work-order key",
            "A stable analytical key is derived from work order number, "
            "equipment type, date opened, and line type. The source does not "
            "provide an asset-level identifier, so multiple rows may still "
            "share the same key.",
            "",
            "",
        ],
        [
            "Exact duplicate groups",
            validation["exact_duplicate_groups"],
            "Extra duplicate rows preserved",
            validation["extra_exact_duplicate_rows"],
        ],
        [
            "Legacy priority placeholders",
            validation["go95_priority_normalization_counts"].get(
                "normalized_legacy_hash_placeholder_to_null",
                0,
            ),
            "Legacy N/A/NA normalized",
            validation["go95_priority_normalization_counts"].get(
                "normalized_legacy_na_to_null",
                0,
            ),
        ],
        [
            "Date validation issues",
            validation["date_validation_issues"],
            "Missing-field issues",
            validation["missing_required_field_issues"],
        ],
        [
            "2023 source completion",
            "Official SDG&E Q1–Q3 workbooks were downloaded when absent so "
            "the output includes all four 2023 quarterly snapshots.",
            "",
            "",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified requirement/change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Report every work order open at quarter end, with equipment, "
            "HFTD/line type, dates, priorities, and reinspection details.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Table 13 purpose and the 12 legacy fields remain unchanged.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Table 13 remains quarterly; v4.01 adds metric and submission "
            "metadata and requires consistency with the Base WMP/WMP Update.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 schema transition",
            V4_CHANGELOG_URL,
            "The eight long legacy date/priority/reinspection column names "
            "were replaced with shorter standardized names.",
        ],
        [
            "SDG&E 2023 WMP page",
            "2023 source files",
            SDGE_2023_SOURCE_PAGE,
            "Official public source for the Q1–Q3 non-spatial workbooks.",
        ],
        ["", "", "", ""],
        ["Selected source file", "Year", "Quarter", "Revision"],
    ]

    for (year, quarter), source in sorted(sources.items()):
        readme_rows.append(
            [
                source["name"],
                year,
                quarter,
                source["revision"],
            ]
        )

    readme.get_range_by_indexes(
        0,
        0,
        len(readme_rows),
        4,
    ).values = [
        [excel_safe(value) for value in row]
        for row in readme_rows
    ]
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
        "row_height": 32,
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
    for column, width in zip(("A", "B", "C", "D"), (34, 70, 68, 58)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(summary, QUARTER_SUMMARY_HEADERS, quarter_summary_rows)
    format_sheet(
        summary,
        QUARTER_SUMMARY_HEADERS,
        len(quarter_summary_rows),
        freeze_columns=5,
    )

    preview_rows = unified_rows[:500]
    write_rows(
        snapshots,
        UNIFIED_HEADERS,
        preview_rows,
        chunk_size=1000,
    )
    format_sheet(
        snapshots,
        UNIFIED_HEADERS,
        len(preview_rows),
        freeze_columns=6,
    )

    status_column = column_letter(
        UNIFIED_HEADERS.index("prior_period_presence_status")
    )
    snapshots.get_range(
        f"{status_column}2:"
        f"{status_column}{len(preview_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${status_column}2="newly_seen"',
        {"fill": "#DCFCE7", "font": {"color": "#166534"}},
    )
    snapshots.get_range(
        f"{status_column}2:"
        f"{status_column}{len(preview_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${status_column}2="continued_with_attribute_change"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    duplicate_count_column = column_letter(
        UNIFIED_HEADERS.index("exact_duplicate_count")
    )
    snapshots.get_range(
        f"{duplicate_count_column}2:"
        f"{duplicate_count_column}{len(preview_rows) + 1}"
    ).conditional_formats.add_custom(
        f"=${duplicate_count_column}2>1",
        {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
    )

    lifecycle_preview_rows = lifecycle_rows[:1000]
    write_rows(
        lifecycle,
        LIFECYCLE_HEADERS,
        lifecycle_preview_rows,
        chunk_size=1000,
    )
    format_sheet(
        lifecycle,
        LIFECYCLE_HEADERS,
        len(lifecycle_preview_rows),
        freeze_columns=6,
    )

    write_rows(schema, SCHEMA_HEADERS, schema_rows)
    format_sheet(
        schema,
        SCHEMA_HEADERS,
        len(schema_rows),
        freeze_columns=1,
    )

    write_rows(issues, ISSUE_HEADERS, issue_rows, chunk_size=1000)
    format_sheet(
        issues,
        ISSUE_HEADERS,
        len(issue_rows),
        freeze_columns=4,
    )
    if issue_rows:
        severity_column = column_letter(ISSUE_HEADERS.index("severity"))
        issues.get_range(
            f"A2:K{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_column}2="error"',
            {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
        )
        issues.get_range(
            f"A2:K{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_column}2="warning"',
            {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
        )

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 13 open-work-order snapshots for every "
            "quarter in 2023-2025. The converter verifies v3.1, v3.2, and "
            "v4.01 schemas, normalizes documented label changes, preserves "
            "source duplicates, and creates a lifecycle summary."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing source SDG&E workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table13_output",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(input_dir)
    period_records: dict[tuple[int, int], list[dict[str, Any]]] = {}
    issues: list[list[Any]] = []

    for period, source in sorted(sources.items()):
        values = read_xlsx_sheet(source["path"], "Table 13")
        records = (
            parse_v4(values, source)
            if source["year"] == 2025
            else parse_legacy(values, source)
        )

        for record in records:
            validate_record(record, issues)

        assign_duplicate_metadata(records, issues)
        period_records[period] = records

    assign_lifecycle_metadata(period_records)

    unified_rows = [
        record_to_row(record)
        for period in sorted(period_records)
        for record in period_records[period]
    ]
    lifecycle_rows = build_lifecycle_rows(period_records)
    quarter_summary_rows = build_quarter_summary(
        period_records,
        sources,
    )
    schema_rows = build_schema_rows()

    workbook_path = (
        output_dir / "sdge_table13_schema_validation.xlsx"
    )
    unified_csv = (
        output_dir / "sdge_table13_2023_2025_unified_snapshots.csv"
    )
    lifecycle_csv = (
        output_dir / "sdge_table13_work_order_lifecycle_summary.csv"
    )
    quarter_summary_csv = (
        output_dir / "sdge_table13_quarter_summary.csv"
    )
    schema_csv = (
        output_dir / "sdge_table13_schema_crosswalk.csv"
    )
    issues_csv = (
        output_dir / "sdge_table13_validation_issues.csv"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(lifecycle_csv, LIFECYCLE_HEADERS, lifecycle_rows)
    write_csv(
        quarter_summary_csv,
        QUARTER_SUMMARY_HEADERS,
        quarter_summary_rows,
    )
    write_csv(schema_csv, SCHEMA_HEADERS, schema_rows)
    write_csv(issues_csv, ISSUE_HEADERS, issues)

    all_records = [
        record
        for records in period_records.values()
        for record in records
    ]
    duplicate_issue_rows = [
        row
        for row in issues
        if row[0] == "potential_exact_duplicate_source_rows"
    ]

    validation = {
        "unified_rows": len(unified_rows),
        "reporting_periods": 12,
        "lifecycle_rows": len(lifecycle_rows),
        "distinct_work_order_keys": len(
            {
                record["work_order_key"]
                for record in all_records
            }
        ),
        "source_rows_by_period": {
            period_label(*period): len(records)
            for period, records in sorted(period_records.items())
        },
        "exact_duplicate_groups": len(duplicate_issue_rows),
        "extra_exact_duplicate_rows": sum(
            max(record["exact_duplicate_count"] - 1, 0)
            for record in all_records
            if record["exact_duplicate_index"] == 1
        ),
        "go95_priority_normalization_counts": dict(
            Counter(
                record["go95_priority_crosswalk_status"]
                for record in all_records
            )
        ),
        "hftd_normalization_counts": dict(
            Counter(
                record["hftd_crosswalk_status"]
                for record in all_records
            )
        ),
        "date_validation_issues": sum(
            row[0] in {
                "due_date_before_date_opened",
                "modified_detail_without_modified_date",
            }
            for row in issues
        ),
        "missing_required_field_issues": sum(
            row[0] == "missing_required_field"
            for row in issues
        ),
        "blank_meaning_issues": sum(
            row[0] == "blank_required_fields_without_blank_meaning"
            for row in issues
        ),
        "validation_issue_rows": len(issues),
        "source_metric_number_restarts_each_2025_quarter": True,
        "sources": [
            {
                "name": source["name"],
                "year": source["year"],
                "quarter": source["quarter"],
                "revision": source["revision"],
                "source_url": source.get("source_url"),
            }
            for _, source in sorted(sources.items())
        ],
    }
    validation_path.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        lifecycle_rows,
        quarter_summary_rows,
        schema_rows,
        issues,
        sources,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {lifecycle_csv}")
    print(f"Created: {quarter_summary_csv}")
    print(f"Created: {schema_csv}")
    print(f"Created: {issues_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(f"Distinct work-order keys: {validation['distinct_work_order_keys']}")
    print(f"Exact duplicate groups: {validation['exact_duplicate_groups']}")
    print(f"Validation issue rows: {len(issues)}")


if __name__ == "__main__":
    main()
