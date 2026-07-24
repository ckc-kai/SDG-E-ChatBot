
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
        "v4.01 Annual-EOY",
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
PETITION_DOCKET_URL = (
    "https://efiling.energysafety.ca.gov/Lists/DocketLog.aspx?"
    "docketnumber=2024-QDR"
)

UNIFIED_HEADERS = [
    "record_id",
    "comparison_group_id",
    "metric_number",
    "mapped_v4_metric_number",
    "initiative_mapping_status",
    "tracking_id_normalized",
    "source_tracking_id_raw",
    "group_tracking_ids",
    "wmp_category",
    "wmp_initiative",
    "wmp_activity",
    "source_wmp_category_raw",
    "source_wmp_initiative_activity_raw",
    "expense_type",
    "hftd_tier",
    "unit_raw",
    "unit_canonical",
    "actual_value_raw",
    "actual_value_canonical",
    "primary_driver_targeted",
    "secondary_drivers_targeted",
    "year_initiated",
    "most_recent_proceeding",
    "memorandum_account",
    "current_compliance_status",
    "associated_rules",
    "other_spend_category",
    "comments",
    "blank_meaning",
    "utility_id",
    "reporting_year",
    "schema_version",
    "source_revision",
    "source_file",
    "source_sheet",
    "source_row",
    "source_value_cell",
    "guideline_url",
]

COMPARABLE_HEADERS = [
    "comparison_group_id",
    "mapped_v4_metric_number",
    "wmp_category",
    "wmp_initiative",
    "wmp_activity",
    "expense_type",
    "hftd_tier",
    "unit_canonical",
    "actual_value",
    "aggregation_method",
    "component_count",
    "tracking_id_coverage_status",
    "group_tracking_ids",
    "matched_tracking_ids",
    "missing_current_tracking_ids",
    "reporting_year",
    "source_files",
]

CROSSWALK_HEADERS = [
    "comparison_group_id",
    "wmp_category",
    "wmp_initiative",
    "wmp_activity",
    "group_tracking_ids",
    "capex_territory_metric_number",
    "capex_hftd_metric_number",
    "opex_territory_metric_number",
    "opex_hftd_metric_number",
    "2023_matched_tracking_ids",
    "2023_missing_current_tracking_ids",
    "2023_coverage_status",
    "2024_matched_tracking_ids",
    "2024_missing_current_tracking_ids",
    "2024_coverage_status",
]

UNMAPPED_HEADERS = [
    "reporting_year",
    "tracking_id_normalized",
    "source_tracking_id_raw",
    "wmp_category_raw",
    "wmp_initiative_activity_raw",
    "expense_type",
    "hftd_tier",
    "actual_value",
    "source_file",
    "source_row",
    "source_value_cell",
    "mapping_reason",
]

REVISION_HEADERS = [
    "revision_change_type",
    "tracking_id_normalized",
    "expense_type",
    "hftd_tier",
    "prior_value",
    "amended_value",
    "prior_source_rows",
    "amended_source_rows",
    "prior_metadata",
    "amended_metadata",
    "note",
]

ISSUE_HEADERS = [
    "issue_type",
    "reporting_year",
    "comparison_group_id",
    "tracking_id",
    "expense_type",
    "territory_value",
    "hftd_value",
    "source_file",
    "source_rows",
    "note",
]

SCHEMA_CHANGE_HEADERS = [
    "change",
    "2023 v3.1",
    "2024 v3.2 / amended filing",
    "2025 v4.01 Annual-EOY",
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
        return value

    if isinstance(value, str):
        normalized = value.replace(",", "").replace("$", "")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric Table 11 value, found {value!r}"
            ) from exc

        return int(parsed) if parsed.is_integer() else parsed

    raise TypeError(f"Unsupported Table 11 value type: {type(value).__name__}")


def normalize_tracking_id(value: Any) -> str | None:
    value = clean(value)
    if value is None:
        return None

    value = str(value).strip()
    match = re.fullmatch(r"WMP\.(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"WMP.{int(match.group(1))}"

    return value


def split_tracking_ids(value: Any) -> tuple[str, ...]:
    value = clean(value)
    if value is None:
        return ()

    identifiers = {
        normalize_tracking_id(part)
        for part in str(value).split(";")
        if normalize_tracking_id(part) is not None
    }
    return tuple(sorted(identifiers))


def normalize_expense_type(value: Any) -> str:
    value = clean(value)
    if value is None:
        raise ValueError("Table 11 EXPENSE TYPE cannot be blank")

    normalized = str(value).upper()
    if normalized not in {"CAPEX", "OPEX"}:
        raise ValueError(f"Unexpected Table 11 expense type: {value!r}")
    return normalized


def normalize_hftd_tier(value: Any) -> str:
    value = clean(value)
    if value not in {"Territory", "HFTD"}:
        raise ValueError(f"Unexpected Table 11 HFTD tier: {value!r}")
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


def workbook_sheet_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook_root.find(f"{{{NS_MAIN}}}sheets")
        return [sheet.attrib["name"] for sheet in sheets] if sheets is not None else []


def discover_sources(input_dir: Path) -> dict[str, Any]:
    quarterly_candidates = []
    for path in input_dir.glob("*.xlsx"):
        parsed = parse_filename(path)
        if parsed is not None:
            quarterly_candidates.append(parsed)

    source_2023 = max(
        [
            item
            for item in quarterly_candidates
            if item["year"] == 2023 and item["quarter"] == 4
        ],
        key=lambda item: item["revision"],
    )

    quarterly_2024 = {}
    for quarter in (1, 2, 3, 4):
        matches = [
            item
            for item in quarterly_candidates
            if item["year"] == 2024 and item["quarter"] == quarter
        ]
        if not matches:
            raise FileNotFoundError(f"No 2024 Q{quarter} SDG&E workbook found")
        quarterly_2024[quarter] = max(
            matches,
            key=lambda item: item["revision"],
        )

    amendment_candidates = [
        path
        for path in input_dir.glob("*.xlsx")
        if "petition" in path.name.casefold()
        and "amend" in path.name.casefold()
        and "Table 11" in workbook_sheet_names(path)
    ]
    amendment = max(
        amendment_candidates,
        key=lambda path: path.stat().st_mtime,
    ) if amendment_candidates else None

    eoy_candidates = [
        path
        for path in input_dir.glob("*.xlsx")
        if "eoy" in path.name.casefold()
        and any(
            name.replace(" ", "").casefold() == "table11"
            for name in workbook_sheet_names(path)
        )
    ]
    if not eoy_candidates:
        raise FileNotFoundError(
            "No 2025 Annual-EOY workbook containing Table11 was found"
        )
    source_2025 = max(eoy_candidates, key=lambda path: path.stat().st_mtime)

    selected_2024 = amendment or quarterly_2024[4]["path"]

    return {
        "2023": {
            **source_2023,
            "sheet": "Table 11",
            "source_type": "quarterly_q4",
        },
        "2024_quarters": quarterly_2024,
        "2024": {
            "path": selected_2024,
            "name": selected_2024.name,
            "year": 2024,
            "quarter": 4,
            "revision": (
                "Petition amendment R0"
                if amendment is not None
                else quarterly_2024[4]["revision"]
            ),
            "sheet": "Table 11",
            "source_type": (
                "petition_amendment"
                if amendment is not None
                else "quarterly_q4"
            ),
        },
        "2025": {
            "path": source_2025,
            "name": source_2025.name,
            "year": 2025,
            "revision": "Annual-EOY",
            "sheet": next(
                name
                for name in workbook_sheet_names(source_2025)
                if name.replace(" ", "").casefold() == "table11"
            ),
            "source_type": "annual_eoy",
        },
    }


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[Any]]:
    """Read cached/computed worksheet values without modifying the source file."""
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


def locate_legacy_actual_columns(
    values: list[list[Any]],
    reporting_year: int,
) -> dict[tuple[str, str], int]:
    section_row = [clean(value) for value in values[4]]
    year_row = [clean(value) for value in values[8]]
    expense_row = [clean(value) for value in values[6]]
    tier_row = [clean(value) for value in values[7]]

    actual_starts = [
        index
        for index, value in enumerate(section_row)
        if value in {"Actual", "Actuals"}
    ]
    if len(actual_starts) != 1:
        raise AssertionError(
            f"Expected one legacy Actual section, found {actual_starts}"
        )
    actual_start = actual_starts[0]

    projected_starts = [
        index
        for index, value in enumerate(section_row)
        if value == "Projected"
    ]
    actual_end = (
        projected_starts[0]
        if projected_starts
        else len(section_row)
    )

    year_starts = [
        index
        for index in range(actual_start, actual_end)
        if year_row[index] == reporting_year
    ]
    if len(year_starts) != 1:
        raise AssertionError(
            f"Expected one actual {reporting_year} block; found {year_starts}"
        )
    start = year_starts[0]
    columns = range(start, start + 4)

    mapping: dict[tuple[str, str], int] = {}
    current_expense = None
    for column in columns:
        if expense_row[column] is not None:
            current_expense = normalize_expense_type(
                str(expense_row[column]).split()[0]
            )
        tier = normalize_hftd_tier(tier_row[column])
        mapping[(current_expense, tier)] = column

    expected = {
        ("CAPEX", "Territory"),
        ("CAPEX", "HFTD"),
        ("OPEX", "Territory"),
        ("OPEX", "HFTD"),
    }
    if set(mapping) != expected:
        raise AssertionError(
            f"Unexpected legacy Table 11 actual columns: {mapping}"
        )

    return mapping


def parse_legacy_rows(
    values: list[list[Any]],
    reporting_year: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    headers = [clean(value) for value in values[8]]
    required = {
        "WMPInitiativeCategory",
        "WMPInitiativeActivity",
        "UtilityInitiativeTrackingID",
        "Comments",
        "Blank Meaning",
    }
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Missing legacy Table 11 headers: {sorted(missing)}")

    blank_column = headers.index("Blank Meaning")
    actual_columns = locate_legacy_actual_columns(values, reporting_year)

    rows = []
    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        source_tracking_id = clean(row[3])
        if source_tracking_id is None:
            continue

        actuals = {
            combination: parse_number(row[column])
            for combination, column in actual_columns.items()
        }

        rows.append(
            {
                "source_row": zero_based_row + 1,
                "wmp_category_raw": clean(row[1]),
                "wmp_initiative_activity_raw": clean(row[2]),
                "tracking_id_raw": source_tracking_id,
                "tracking_id_normalized": normalize_tracking_id(
                    source_tracking_id
                ),
                "primary_driver_targeted": clean(row[4]),
                "secondary_drivers_targeted": clean(row[5]),
                "year_initiated": clean(row[6]),
                "most_recent_proceeding": clean(row[7]),
                "memorandum_account": clean(row[8]),
                "current_compliance_status": clean(row[9]),
                "associated_rules": clean(row[10]),
                "other_spend_category": clean(row[11]),
                "comments": clean(row[12]),
                "blank_meaning": clean(row[blank_column]),
                "actuals": actuals,
            }
        )

    if reporting_year in {2023, 2024} and not rows:
        raise AssertionError(f"No {reporting_year} Table 11 rows were parsed")

    return rows, actual_columns


def parse_v4_rows(values: list[list[Any]]) -> list[dict[str, Any]]:
    expected_headers = [
        "METRIC NUMBER",
        "WMP CATEGORY",
        "WMP INITIATIVE",
        "WMP ACTIVITY (IF BLANK THIS REFERS TO ALL ACTIVITIES UNDER THE INITIATIVE)",
        "EXPENSE TYPE",
        "HFTD TIER",
        "UNIT(S)",
        "UTILITY MITIGATION ACTIVITY TRACKING IDS",
        "PRIMARY DRIVER TARGETED",
        "SECONDARY DRIVERS TARGETED",
        "YEAR INITIATED",
        "MOST RECENT PROCEEDING",
        "MEMORANDUM ACCOUNT",
        "CURRENT COMPLIANCE STATUS",
        "ASSOCIATED RULES",
        "OTHER SPEND CATEGORY",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in values[0][:21]]
    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Annual-EOY Table 11 header does not match v4.01.\n"
            f"Expected: {expected_headers}\nFound: {actual_headers}"
        )

    rows = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[0]) is None:
            continue

        record = {
            "source_row": zero_based_row + 1,
            "metric_number": int(row[0]),
            "wmp_category": clean(row[1]),
            "wmp_initiative": clean(row[2]),
            "wmp_activity": clean(row[3]),
            "expense_type": normalize_expense_type(row[4]),
            "hftd_tier": normalize_hftd_tier(row[5]),
            "unit_raw": clean(row[6]),
            "tracking_ids_raw": clean(row[7]),
            "tracking_ids": split_tracking_ids(row[7]),
            "primary_driver_targeted": clean(row[8]),
            "secondary_drivers_targeted": clean(row[9]),
            "year_initiated": clean(row[10]),
            "most_recent_proceeding": clean(row[11]),
            "memorandum_account": clean(row[12]),
            "current_compliance_status": clean(row[13]),
            "associated_rules": clean(row[14]),
            "other_spend_category": clean(row[15]),
            "comments": clean(row[16]),
            "blank_meaning": clean(row[17]),
            "utility_id": clean(row[18]),
            "reporting_year": int(row[19]),
            "actual_value": parse_number(row[20]),
        }

        if record["unit_raw"] != "$ Thousands":
            raise AssertionError(
                f"Unexpected v4.01 Table 11 unit in row {record['source_row']}"
            )
        if record["utility_id"] != "SDG&E":
            raise AssertionError(
                f"Unexpected utility ID in row {record['source_row']}"
            )
        if record["reporting_year"] != 2025:
            raise AssertionError(
                f"Unexpected reporting year in row {record['source_row']}"
            )

        rows.append(record)

    if len(rows) != 180:
        raise AssertionError(
            f"Expected 180 v4.01 Table 11 rows; found {len(rows)}"
        )

    expected_numbers = list(range(3110000000, 3110000180))
    actual_numbers = sorted(row["metric_number"] for row in rows)
    if actual_numbers != expected_numbers:
        raise AssertionError(
            "The v4.01 metric numbers are not the expected range "
            "3110000000-3110000179"
        )

    return rows


def comparison_group_id(
    category: Any,
    initiative: Any,
    activity: Any,
) -> str:
    payload = json.dumps(
        [clean(category), clean(initiative), clean(activity)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "T11-" + hashlib.sha1(payload).hexdigest()[:14]


def build_v4_groups(
    v4_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    order = []

    for row in v4_rows:
        key = (
            row["wmp_category"],
            row["wmp_initiative"],
            row["wmp_activity"],
        )
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)

    if len(grouped) != 45:
        raise AssertionError(
            f"Expected 45 v4.01 Table 11 initiative/activity groups; "
            f"found {len(grouped)}"
        )

    groups = []
    tracking_id_to_group = {}
    metric_number_to_group = {}

    expected_combinations = {
        ("CAPEX", "Territory"),
        ("CAPEX", "HFTD"),
        ("OPEX", "Territory"),
        ("OPEX", "HFTD"),
    }

    for key in order:
        rows = grouped[key]
        combinations = {
            (row["expense_type"], row["hftd_tier"]): row
            for row in rows
        }
        if set(combinations) != expected_combinations:
            raise AssertionError(
                f"Unexpected v4.01 combinations for group {key}"
            )

        tracking_sets = {row["tracking_ids"] for row in rows}
        if len(tracking_sets) != 1:
            raise AssertionError(
                f"Tracking IDs differ within v4.01 group {key}"
            )
        tracking_ids = next(iter(tracking_sets))

        group = {
            "comparison_group_id": comparison_group_id(*key),
            "wmp_category": key[0],
            "wmp_initiative": key[1],
            "wmp_activity": key[2],
            "tracking_ids": tracking_ids,
            "tracking_ids_text": ";".join(tracking_ids) or None,
            "combinations": combinations,
        }
        groups.append(group)

        for tracking_id in tracking_ids:
            if tracking_id in tracking_id_to_group:
                raise AssertionError(
                    f"Tracking ID {tracking_id} appears in multiple v4 groups"
                )
            tracking_id_to_group[tracking_id] = group

        for row in rows:
            metric_number_to_group[row["metric_number"]] = group

    return groups, tracking_id_to_group, metric_number_to_group


def make_record_id(
    identifier: Any,
    reporting_year: int,
    source_file: str,
    source_row: int,
    expense_type: str,
    hftd_tier: str,
) -> str:
    payload = json.dumps(
        [
            identifier,
            reporting_year,
            source_file,
            source_row,
            expense_type,
            hftd_tier,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "T11R-" + hashlib.sha1(payload).hexdigest()[:16]


def build_unified_rows(
    legacy_by_year: dict[int, list[dict[str, Any]]],
    v4_rows: list[dict[str, Any]],
    tracking_id_to_group: dict[str, dict[str, Any]],
    sources: dict[str, Any],
    actual_columns: dict[int, dict[tuple[str, str], int]],
) -> tuple[list[list[Any]], list[list[Any]]]:
    output = []
    unmapped = []

    for year in (2023, 2024):
        source = sources[str(year)]
        for legacy in legacy_by_year[year]:
            tracking_id = legacy["tracking_id_normalized"]
            group = tracking_id_to_group.get(tracking_id)

            for (expense_type, hftd_tier), actual_value in legacy["actuals"].items():
                source_column = actual_columns[year][
                    (expense_type, hftd_tier)
                ]

                if group is None:
                    comparison_id = (
                        "T11-LEGACY-"
                        + hashlib.sha1(
                            str(tracking_id).encode("utf-8")
                        ).hexdigest()[:12]
                    )
                    mapped_metric_number = None
                    mapping_status = "legacy_tracking_id_not_in_v4_group"
                    category = legacy["wmp_category_raw"]
                    initiative = legacy["wmp_initiative_activity_raw"]
                    activity = None
                    group_tracking_ids = None
                else:
                    comparison_id = group["comparison_group_id"]
                    mapped_metric_number = group["combinations"][
                        (expense_type, hftd_tier)
                    ]["metric_number"]
                    mapping_status = "component_of_v4_tracking_id_group"
                    category = group["wmp_category"]
                    initiative = group["wmp_initiative"]
                    activity = group["wmp_activity"]
                    group_tracking_ids = group["tracking_ids_text"]

                output.append(
                    [
                        make_record_id(
                            mapped_metric_number or tracking_id,
                            year,
                            source["name"],
                            legacy["source_row"],
                            expense_type,
                            hftd_tier,
                        ),
                        comparison_id,
                        None,
                        mapped_metric_number,
                        mapping_status,
                        tracking_id,
                        legacy["tracking_id_raw"],
                        group_tracking_ids,
                        category,
                        initiative,
                        activity,
                        legacy["wmp_category_raw"],
                        legacy["wmp_initiative_activity_raw"],
                        expense_type,
                        hftd_tier,
                        (
                            f"{expense_type} ($ thousands)"
                        ),
                        "$ Thousands",
                        actual_value,
                        actual_value,
                        legacy["primary_driver_targeted"],
                        legacy["secondary_drivers_targeted"],
                        legacy["year_initiated"],
                        legacy["most_recent_proceeding"],
                        legacy["memorandum_account"],
                        legacy["current_compliance_status"],
                        legacy["associated_rules"],
                        legacy["other_spend_category"],
                        legacy["comments"],
                        legacy["blank_meaning"],
                        "SDG&E",
                        year,
                        GUIDELINES[year][0],
                        source["revision"],
                        source["name"],
                        source["sheet"],
                        legacy["source_row"],
                        (
                            f"{column_letter(source_column)}"
                            f"{legacy['source_row']}"
                        ),
                        GUIDELINES[year][1],
                    ]
                )

                if group is None:
                    unmapped.append(
                        [
                            year,
                            tracking_id,
                            legacy["tracking_id_raw"],
                            legacy["wmp_category_raw"],
                            legacy["wmp_initiative_activity_raw"],
                            expense_type,
                            hftd_tier,
                            actual_value,
                            source["name"],
                            legacy["source_row"],
                            (
                                f"{column_letter(source_column)}"
                                f"{legacy['source_row']}"
                            ),
                            "Tracking ID is not listed in any 2025 v4.01 "
                            "Table 11 initiative/activity group.",
                        ]
                    )

    source_2025 = sources["2025"]
    for row in v4_rows:
        group = tracking_id_to_group.get(
            row["tracking_ids"][0]
        ) if row["tracking_ids"] else None

        if group is None:
            # Groups without tracking IDs are still valid v4-native groups.
            matching_group_id = comparison_group_id(
                row["wmp_category"],
                row["wmp_initiative"],
                row["wmp_activity"],
            )
            group_tracking_ids = None
        else:
            matching_group_id = group["comparison_group_id"]
            group_tracking_ids = group["tracking_ids_text"]

        output.append(
            [
                make_record_id(
                    row["metric_number"],
                    2025,
                    source_2025["name"],
                    row["source_row"],
                    row["expense_type"],
                    row["hftd_tier"],
                ),
                matching_group_id,
                row["metric_number"],
                row["metric_number"],
                "v4_native",
                None,
                row["tracking_ids_raw"],
                group_tracking_ids,
                row["wmp_category"],
                row["wmp_initiative"],
                row["wmp_activity"],
                row["wmp_category"],
                (
                    row["wmp_activity"]
                    if row["wmp_activity"] is not None
                    else row["wmp_initiative"]
                ),
                row["expense_type"],
                row["hftd_tier"],
                row["unit_raw"],
                row["unit_raw"],
                row["actual_value"],
                row["actual_value"],
                row["primary_driver_targeted"],
                row["secondary_drivers_targeted"],
                row["year_initiated"],
                row["most_recent_proceeding"],
                row["memorandum_account"],
                row["current_compliance_status"],
                row["associated_rules"],
                row["other_spend_category"],
                row["comments"],
                row["blank_meaning"],
                row["utility_id"],
                2025,
                GUIDELINES[2025][0],
                source_2025["revision"],
                source_2025["name"],
                source_2025["sheet"],
                row["source_row"],
                f"U{row['source_row']}",
                GUIDELINES[2025][1],
            ]
        )

    expected_rows = (
        len(legacy_by_year[2023]) * 4
        + len(legacy_by_year[2024]) * 4
        + len(v4_rows)
    )
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} unified rows; found {len(output)}"
        )

    return output, unmapped


def coverage_for_year(
    group: dict[str, Any],
    legacy_rows: list[dict[str, Any]],
) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    source_ids = {
        row["tracking_id_normalized"]
        for row in legacy_rows
    }
    group_ids = set(group["tracking_ids"])
    matched = tuple(sorted(group_ids & source_ids))
    missing = tuple(sorted(group_ids - source_ids))

    component_count = sum(
        row["tracking_id_normalized"] in group_ids
        for row in legacy_rows
    )

    if not group_ids:
        status = "v4_group_has_no_tracking_ids"
    elif not matched:
        status = "no_legacy_tracking_id_coverage"
    elif missing:
        status = "partial_tracking_id_coverage"
    else:
        status = "complete_tracking_id_coverage"

    return status, matched, missing, component_count


def build_comparable_rows(
    groups: list[dict[str, Any]],
    legacy_by_year: dict[int, list[dict[str, Any]]],
    sources: dict[str, Any],
) -> list[list[Any]]:
    output = []

    for year in (2023, 2024):
        legacy_rows = legacy_by_year[year]
        source = sources[str(year)]

        for group in groups:
            (
                coverage_status,
                matched_ids,
                missing_ids,
                component_count,
            ) = coverage_for_year(group, legacy_rows)

            group_ids = set(group["tracking_ids"])
            components = [
                row
                for row in legacy_rows
                if row["tracking_id_normalized"] in group_ids
            ]

            for combination, v4_row in sorted(
                group["combinations"].items()
            ):
                expense_type, hftd_tier = combination
                values = [
                    component["actuals"][combination]
                    for component in components
                ]

                actual_value = (
                    sum(value for value in values if value is not None)
                    if components and all(value is not None for value in values)
                    else None
                )

                output.append(
                    [
                        group["comparison_group_id"],
                        v4_row["metric_number"],
                        group["wmp_category"],
                        group["wmp_initiative"],
                        group["wmp_activity"],
                        expense_type,
                        hftd_tier,
                        "$ Thousands",
                        actual_value,
                        "sum_legacy_tracking_id_components",
                        component_count,
                        coverage_status,
                        group["tracking_ids_text"],
                        ";".join(matched_ids) or None,
                        ";".join(missing_ids) or None,
                        year,
                        source["name"],
                    ]
                )

    source_2025 = sources["2025"]
    for group in groups:
        for combination, v4_row in sorted(
            group["combinations"].items()
        ):
            output.append(
                [
                    group["comparison_group_id"],
                    v4_row["metric_number"],
                    group["wmp_category"],
                    group["wmp_initiative"],
                    group["wmp_activity"],
                    combination[0],
                    combination[1],
                    "$ Thousands",
                    v4_row["actual_value"],
                    "reported_v4_annual_eoy",
                    1,
                    "v4_native",
                    group["tracking_ids_text"],
                    group["tracking_ids_text"],
                    None,
                    2025,
                    source_2025["name"],
                ]
            )

    expected_rows = len(groups) * 4 * 3
    if len(output) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} comparable rows; found {len(output)}"
        )

    return output


def build_crosswalk_rows(
    groups: list[dict[str, Any]],
    legacy_by_year: dict[int, list[dict[str, Any]]],
) -> list[list[Any]]:
    rows = []

    for group in groups:
        status_2023, matched_2023, missing_2023, _ = coverage_for_year(
            group,
            legacy_by_year[2023],
        )
        status_2024, matched_2024, missing_2024, _ = coverage_for_year(
            group,
            legacy_by_year[2024],
        )

        rows.append(
            [
                group["comparison_group_id"],
                group["wmp_category"],
                group["wmp_initiative"],
                group["wmp_activity"],
                group["tracking_ids_text"],
                group["combinations"][("CAPEX", "Territory")][
                    "metric_number"
                ],
                group["combinations"][("CAPEX", "HFTD")][
                    "metric_number"
                ],
                group["combinations"][("OPEX", "Territory")][
                    "metric_number"
                ],
                group["combinations"][("OPEX", "HFTD")][
                    "metric_number"
                ],
                ";".join(matched_2023) or None,
                ";".join(missing_2023) or None,
                status_2023,
                ";".join(matched_2024) or None,
                ";".join(missing_2024) or None,
                status_2024,
            ]
        )

    return rows


def aggregate_legacy_by_id(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        for combination, value in row["actuals"].items():
            key = (
                row["tracking_id_normalized"],
                combination[0],
                combination[1],
            )
            if key not in output:
                output[key] = {
                    "value": 0 if value is not None else None,
                    "source_rows": [],
                    "metadata": [],
                }
            elif output[key]["value"] is None and value is not None:
                output[key]["value"] = 0

            if value is not None:
                output[key]["value"] += value

            output[key]["source_rows"].append(row["source_row"])
            output[key]["metadata"].append(
                {
                    "category": row["wmp_category_raw"],
                    "activity": row["wmp_initiative_activity_raw"],
                    "primary_driver": row["primary_driver_targeted"],
                    "comments": row["comments"],
                }
            )

    return output


def build_revision_comparison(
    q4_rows: list[dict[str, Any]],
    amended_rows: list[dict[str, Any]],
) -> list[list[Any]]:
    prior = aggregate_legacy_by_id(q4_rows)
    amended = aggregate_legacy_by_id(amended_rows)
    output = []

    for key in sorted(set(prior) | set(amended)):
        prior_item = prior.get(key)
        amended_item = amended.get(key)
        tracking_id, expense_type, hftd_tier = key

        prior_value = prior_item["value"] if prior_item else None
        amended_value = amended_item["value"] if amended_item else None
        prior_rows = prior_item["source_rows"] if prior_item else []
        amended_rows_list = (
            amended_item["source_rows"] if amended_item else []
        )

        if prior_value != amended_value:
            output.append(
                [
                    "actual_value_changed",
                    tracking_id,
                    expense_type,
                    hftd_tier,
                    prior_value,
                    amended_value,
                    ";".join(map(str, prior_rows)),
                    ";".join(map(str, amended_rows_list)),
                    None,
                    None,
                    "The amended filing changed the selected 2024 actual.",
                ]
            )

        prior_metadata = (
            json.dumps(prior_item["metadata"], ensure_ascii=False)
            if prior_item
            else None
        )
        amended_metadata = (
            json.dumps(amended_item["metadata"], ensure_ascii=False)
            if amended_item
            else None
        )

        if prior_metadata != amended_metadata and expense_type == "CAPEX" and hftd_tier == "Territory":
            output.append(
                [
                    "metadata_or_duplicate_structure_changed",
                    tracking_id,
                    None,
                    None,
                    None,
                    None,
                    ";".join(map(str, prior_rows)),
                    ";".join(map(str, amended_rows_list)),
                    prior_metadata,
                    amended_metadata,
                    "Metadata or duplicate-row structure changed; actual "
                    "values are compared separately.",
                ]
            )

    return output


def validate_financial_values(
    unified_rows: list[list[Any]],
) -> list[list[Any]]:
    index = {
        header: position
        for position, header in enumerate(UNIFIED_HEADERS)
    }
    grouped = defaultdict(dict)
    issues = []

    for row in unified_rows:
        actual_value = row[index["actual_value_canonical"]]
        if actual_value is not None and actual_value < 0:
            issues.append(
                [
                    "negative_expenditure",
                    row[index["reporting_year"]],
                    row[index["comparison_group_id"]],
                    row[index["tracking_id_normalized"]],
                    row[index["expense_type"]],
                    (
                        actual_value
                        if row[index["hftd_tier"]] == "Territory"
                        else None
                    ),
                    (
                        actual_value
                        if row[index["hftd_tier"]] == "HFTD"
                        else None
                    ),
                    row[index["source_file"]],
                    row[index["source_row"]],
                    "Source value is negative. It is preserved as reported "
                    "and flagged because the nominal schema constraint is "
                    "nonnegative; the value may represent a credit or true-up.",
                ]
            )

        key = (
            row[index["reporting_year"]],
            row[index["comparison_group_id"]],
            row[index["tracking_id_normalized"]],
            row[index["expense_type"]],
            row[index["source_file"]],
            row[index["source_row"]],
        )
        grouped[key][row[index["hftd_tier"]]] = row

    for key, rows in grouped.items():
        territory = rows.get("Territory")
        hftd = rows.get("HFTD")
        if territory is None or hftd is None:
            continue

        territory_value = territory[index["actual_value_canonical"]]
        hftd_value = hftd[index["actual_value_canonical"]]

        if (
            territory_value is not None
            and hftd_value is not None
            and hftd_value > territory_value + 1e-9
        ):
            issues.append(
                [
                    "hftd_exceeds_territory",
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    territory_value,
                    hftd_value,
                    key[4],
                    key[5],
                    "HFTD expenditure must be a subset of Territory expenditure.",
                ]
            )

    return issues


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
    chunk_size: int = 250,
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
        "wmp_category",
        "wmp_initiative",
        "wmp_activity",
        "source_wmp_category_raw",
        "source_wmp_initiative_activity_raw",
        "primary_driver_targeted",
        "secondary_drivers_targeted",
        "associated_rules",
        "comments",
        "blank_meaning",
        "group_tracking_ids",
        "prior_metadata",
        "amended_metadata",
        "note",
        "guideline_url",
    }
    medium_text = {
        "initiative_mapping_status",
        "tracking_id_coverage_status",
        "source_file",
        "mapping_reason",
        "revision_change_type",
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
    comparable_rows: list[list[Any]],
    crosswalk_rows: list[list[Any]],
    unmapped_rows: list[list[Any]],
    revision_rows: list[list[Any]],
    issue_rows: list[list[Any]],
    sources: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()

    readme = workbook.worksheets.add("README")
    actuals = workbook.worksheets.add("Unified Actuals")
    comparable = workbook.worksheets.add("Comparable Aggregates")
    crosswalk = workbook.worksheets.add("Group Crosswalk")
    unmapped = workbook.worksheets.add("Unmapped Legacy")
    revision = workbook.worksheets.add("Revision Comparison")
    issues = workbook.worksheets.add("Validation Issues")
    changes = workbook.worksheets.add("Schema Changes")

    readme_rows = [
        ["SDG&E Table 11 Unified Dataset, 2023–2025", "", "", ""],
        [
            "Unified source observations",
            validation["unified_rows"],
            "Comparable aggregate rows",
            validation["comparable_rows"],
        ],
        [
            "2023 source rows",
            validation["legacy_source_rows"]["2023"],
            "2024 source rows",
            validation["legacy_source_rows"]["2024"],
        ],
        [
            "2025 v4.01 metric rows",
            validation["v4_metric_rows"],
            "2025 initiative/activity groups",
            validation["v4_groups"],
        ],
        [
            "Projection treatment",
            "Projection columns and Annual-WMP projected values are excluded.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "Only the 2023 actual CAPEX/OPEX × Territory/HFTD block is "
            "extracted from the selected Q4 v3.1 workbook.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "The later Petition-to-Amend workbook is selected when present. "
            "Only its 2024 actual block is extracted.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Actuals are read from the separate v4.01 Annual-EOY Table11 "
            "workbook; Table 11 is not part of the 2025 quarterly workbook.",
            "",
            "",
        ],
        [
            "Crosswalk method",
            "Legacy rows map to 2025 initiative/activity groups only when "
            "their normalized WMP tracking ID appears in that v4.01 group. "
            "Multiple legacy components are summed in Comparable Aggregates.",
            "",
            "",
        ],
        [
            "Unmapped legacy observations",
            validation["unmapped_legacy_rows"],
            "Treatment",
            "Preserved in Unified Actuals and Unmapped Legacy; not forced "
            "into a 2025 group.",
        ],
        [
            "2025 groups without tracking IDs",
            validation["v4_groups_without_tracking_ids"],
            "Treatment",
            "Retained as native 2025 rows; legacy comparison remains blank.",
        ],
        [
            "2024 amendment actual changes",
            validation["2024_revision_actual_value_changes"],
            "Metadata/duplicate changes",
            validation["2024_revision_metadata_changes"],
        ],
        [
            "HFTD subset validation issues",
            validation["hftd_subset_issues"],
            "Requirement",
            "HFTD must not exceed Territory for the same initiative, "
            "expense type, and reporting year.",
        ],
        [
            "Negative expenditure observations",
            validation["negative_expenditure_issues"],
            "Treatment",
            "Preserved as reported and flagged as potential credits or "
            "accounting true-ups.",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 11 reports annual actual/projected initiative costs in "
            "thousands of dollars, split by CAPEX/OPEX and Territory/HFTD.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Same annual financial meaning; multiple tracking IDs may use "
            "semicolon delimiters.",
        ],
        [
            "2024 QDR docket / petition amendment",
            "2024 amended filing",
            PETITION_DOCKET_URL,
            "The later filing revises Tables 1, 11, and 12. The converter "
            "compares its 2024 actuals with the local Q4 R1 workbook.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Actual expenditures move to the Annual-EOY workbook; projected "
            "expenditures move to Annual-WMP. Reporting is at activity level "
            "where defined and otherwise at initiative level.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "Used with direct workbook/schema comparison to validate the "
            "annual-template restructuring.",
        ],
        ["", "", "", ""],
        ["Selected source", "Year", "Source type", "Revision"],
        [
            sources["2023"]["name"],
            2023,
            sources["2023"]["source_type"],
            sources["2023"]["revision"],
        ],
        [
            sources["2024"]["name"],
            2024,
            sources["2024"]["source_type"],
            sources["2024"]["revision"],
        ],
        [
            sources["2025"]["name"],
            2025,
            sources["2025"]["source_type"],
            sources["2025"]["revision"],
        ],
    ]

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
        if row[0] in {"Official source", "Selected source"}:
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
    for column, width in zip(("A", "B", "C", "D"), (34, 68, 48, 60)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(actuals, UNIFIED_HEADERS, unified_rows)
    format_sheet(
        actuals,
        UNIFIED_HEADERS,
        len(unified_rows),
        freeze_columns=8,
    )
    for header in ("actual_value_raw", "actual_value_canonical"):
        letter = column_letter(UNIFIED_HEADERS.index(header))
        actuals.get_range(
            f"{letter}2:{letter}{len(unified_rows) + 1}"
        ).format.number_format = "0.########"

    mapping_column = column_letter(
        UNIFIED_HEADERS.index("initiative_mapping_status")
    )
    actuals.get_range(
        f"{mapping_column}2:"
        f"{mapping_column}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${mapping_column}2="legacy_tracking_id_not_in_v4_group"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    write_rows(comparable, COMPARABLE_HEADERS, comparable_rows)
    format_sheet(
        comparable,
        COMPARABLE_HEADERS,
        len(comparable_rows),
        freeze_columns=8,
    )
    comparable_value = column_letter(
        COMPARABLE_HEADERS.index("actual_value")
    )
    comparable.get_range(
        f"{comparable_value}2:"
        f"{comparable_value}{len(comparable_rows) + 1}"
    ).format.number_format = "0.########"

    write_rows(crosswalk, CROSSWALK_HEADERS, crosswalk_rows)
    format_sheet(
        crosswalk,
        CROSSWALK_HEADERS,
        len(crosswalk_rows),
        freeze_columns=5,
    )

    write_rows(unmapped, UNMAPPED_HEADERS, unmapped_rows)
    format_sheet(
        unmapped,
        UNMAPPED_HEADERS,
        len(unmapped_rows),
        freeze_columns=5,
    )
    if unmapped_rows:
        unmapped.get_range(
            f"A2:L{len(unmapped_rows) + 1}"
        ).format = {
            "fill": "#FEF3C7",
            "font": {"color": "#92400E"},
            "wrap_text": True,
        }

    write_rows(revision, REVISION_HEADERS, revision_rows)
    format_sheet(
        revision,
        REVISION_HEADERS,
        len(revision_rows),
        freeze_columns=4,
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

    schema_change_rows = [
        [
            "Reporting frequency and workbook",
            "Annual financial columns are carried in quarterly QDR Table 11.",
            "Same wide annual financial table, with later amendment available.",
            "Actuals are reported once through Annual-EOY; projections are "
            "reported separately through Annual-WMP.",
            "Extract one annual actual block for 2023 and 2024 and the "
            "Annual-EOY ACTUAL VALUE for 2025.",
        ],
        [
            "Row granularity",
            "One row per utility initiative tracking ID/activity, with four "
            "financial value columns for each year.",
            "Same legacy granularity.",
            "Activity-level where defined by Energy Safety and otherwise "
            "initiative-level; one row per CAPEX/OPEX × Territory/HFTD metric.",
            "Preserve source rows and aggregate legacy tracking-ID components "
            "only for the separate comparable view.",
        ],
        [
            "Metric identifier",
            "No standardized metric number; UtilityInitiativeTrackingID is "
            "the operational identifier.",
            "Same.",
            "Standard METRIC NUMBER values 3110000000–3110000179 plus "
            "semicolon-delimited mitigation tracking IDs.",
            "Use metric_number only for native/mapped v4 combinations and "
            "retain raw and normalized tracking IDs.",
        ],
        [
            "Category structure",
            "WMPInitiativeCategory and WMPInitiativeActivity.",
            "Same legacy fields.",
            "WMP CATEGORY, WMP INITIATIVE, and optional WMP ACTIVITY.",
            "Use canonical v4 fields only when tracking-ID membership supports "
            "the crosswalk; otherwise retain a legacy-only group.",
        ],
        [
            "Financial dimensions",
            "CAPEX/OPEX and Territory/HFTD in separate wide columns.",
            "Same.",
            "EXPENSE TYPE and HFTD TIER are row dimensions.",
            "Unpivot legacy columns into the v4-style dimensions.",
        ],
        [
            "Units",
            "Thousands of dollars.",
            "Thousands of dollars.",
            "$ Thousands.",
            "No numeric conversion; use $ Thousands as canonical unit.",
        ],
        [
            "Territory/HFTD relationship",
            "HFTD is a subset of Territory.",
            "Same.",
            "Same.",
            "Validate HFTD ≤ Territory for every comparable pair.",
        ],
        [
            "Projections",
            "Future annual projection columns are present.",
            "Future annual projection columns are present and may be amended.",
            "Projected expenditures are in Annual-WMP, not Annual-EOY.",
            "Exclude all projections as requested.",
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
            "Combine SDG&E Table 11 annual actual expenditures for 2023-2025. "
            "Legacy rows are unpivoted and crosswalked to v4.01 Annual-EOY "
            "initiative/activity groups using normalized mitigation tracking IDs. "
            "Projection values are excluded."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing the SDG&E XLSX source files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table11_output",
        help="Directory for generated CSV/XLSX outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(input_dir)

    values_2023 = read_xlsx_sheet(
        sources["2023"]["path"],
        sources["2023"]["sheet"],
    )
    values_2024 = read_xlsx_sheet(
        sources["2024"]["path"],
        sources["2024"]["sheet"],
    )
    values_2025 = read_xlsx_sheet(
        sources["2025"]["path"],
        sources["2025"]["sheet"],
    )

    rows_2023, actual_columns_2023 = parse_legacy_rows(
        values_2023,
        2023,
    )
    rows_2024, actual_columns_2024 = parse_legacy_rows(
        values_2024,
        2024,
    )
    v4_rows = parse_v4_rows(values_2025)

    # Validate the 2024 quarterly templates and compare Q4 R1 with the
    # later petition-amendment workbook when available.
    quarterly_2024_rows = {}
    for quarter, source in sources["2024_quarters"].items():
        values = read_xlsx_sheet(source["path"], "Table 11")
        parsed, _ = parse_legacy_rows(values, 2024)
        quarterly_2024_rows[quarter] = parsed

    unique_id_sets = [
        {
            row["tracking_id_normalized"]
            for row in quarterly_2024_rows[quarter]
        }
        for quarter in (1, 2, 3, 4)
    ]
    if not all(current == unique_id_sets[0] for current in unique_id_sets):
        raise AssertionError(
            "The 2024 quarterly Table 11 tracking-ID sets differ"
        )

    groups, tracking_id_to_group, _ = build_v4_groups(v4_rows)

    legacy_by_year = {
        2023: rows_2023,
        2024: rows_2024,
    }
    actual_columns = {
        2023: actual_columns_2023,
        2024: actual_columns_2024,
    }

    unified_rows, unmapped_rows = build_unified_rows(
        legacy_by_year,
        v4_rows,
        tracking_id_to_group,
        sources,
        actual_columns,
    )
    comparable_rows = build_comparable_rows(
        groups,
        legacy_by_year,
        sources,
    )
    crosswalk_rows = build_crosswalk_rows(
        groups,
        legacy_by_year,
    )

    q4_rows = quarterly_2024_rows[4]
    revision_rows = build_revision_comparison(
        q4_rows,
        rows_2024,
    ) if sources["2024"]["source_type"] == "petition_amendment" else []

    issue_rows = validate_financial_values(unified_rows)

    workbook_path = (
        output_dir / "sdge_table11_2023_2025_unified.xlsx"
    )
    unified_csv = (
        output_dir / "sdge_table11_2023_2025_unified_actuals.csv"
    )
    comparable_csv = (
        output_dir / "sdge_table11_comparable_group_aggregates.csv"
    )
    crosswalk_csv = (
        output_dir / "sdge_table11_group_crosswalk.csv"
    )
    unmapped_csv = (
        output_dir / "sdge_table11_unmapped_legacy_components.csv"
    )
    revision_csv = (
        output_dir / "sdge_table11_revision_comparison.csv"
    )
    issues_csv = (
        output_dir / "sdge_table11_validation_issues.csv"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(comparable_csv, COMPARABLE_HEADERS, comparable_rows)
    write_csv(crosswalk_csv, CROSSWALK_HEADERS, crosswalk_rows)
    write_csv(unmapped_csv, UNMAPPED_HEADERS, unmapped_rows)
    write_csv(revision_csv, REVISION_HEADERS, revision_rows)
    write_csv(issues_csv, ISSUE_HEADERS, issue_rows)

    revision_actual_changes = sum(
        row[0] == "actual_value_changed"
        for row in revision_rows
    )
    revision_metadata_changes = sum(
        row[0] == "metadata_or_duplicate_structure_changed"
        for row in revision_rows
    )

    validation = {
        "unified_rows": len(unified_rows),
        "comparable_rows": len(comparable_rows),
        "legacy_source_rows": {
            "2023": len(rows_2023),
            "2024": len(rows_2024),
        },
        "v4_metric_rows": len(v4_rows),
        "v4_groups": len(groups),
        "v4_groups_without_tracking_ids": sum(
            not group["tracking_ids"]
            for group in groups
        ),
        "unmapped_legacy_rows": len(unmapped_rows),
        "unmapped_legacy_tracking_ids": sorted(
            {
                row[1]
                for row in unmapped_rows
            }
        ),
        "2024_revision_actual_value_changes": revision_actual_changes,
        "2024_revision_metadata_changes": revision_metadata_changes,
        "hftd_subset_issues": sum(
            row[0] == "hftd_exceeds_territory"
            for row in issue_rows
        ),
        "negative_expenditure_issues": sum(
            row[0] == "negative_expenditure"
            for row in issue_rows
        ),
        "validation_issue_rows": len(issue_rows),
        "coverage_status_counts": {
            "2023": dict(
                Counter(
                    coverage_for_year(group, rows_2023)[0]
                    for group in groups
                )
            ),
            "2024": dict(
                Counter(
                    coverage_for_year(group, rows_2024)[0]
                    for group in groups
                )
            ),
        },
        "sources": {
            "2023": sources["2023"]["name"],
            "2024": sources["2024"]["name"],
            "2025": sources["2025"]["name"],
        },
    }
    validation_path.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        comparable_rows,
        crosswalk_rows,
        unmapped_rows,
        revision_rows,
        issue_rows,
        sources,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {comparable_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {unmapped_csv}")
    print(f"Created: {revision_csv}")
    print(f"Created: {issues_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(f"Comparable rows: {len(comparable_rows)}")
    print(f"Unmapped legacy observations: {len(unmapped_rows)}")
    print(f"2024 revision actual changes: {revision_actual_changes}")
    print(f"Validation issues: {len(issue_rows)}")


if __name__ == "__main__":
    main()
