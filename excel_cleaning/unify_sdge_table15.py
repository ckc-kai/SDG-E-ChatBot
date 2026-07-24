
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
SDGE_WMP_PAGE_URL = (
    "https://www.sdge.com/2026-2028-wildfire-mitigation-plan"
)
R1_COVER_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2025+Data+Submissions&fileid=58976&shareable=true"
)
R1_DATA_URL = (
    "https://www.sdge.com/sites/default/files/regulatory/"
    "SDGE_2026-WMP_R1.xlsx"
)
R0_DATA_URL = (
    "https://www.sdge.com/sites/default/files/regulatory/"
    "SDGE_2026-WMP_R0_Tabular%20Wildfire%20Mitigation%20"
    "Annual-WMP%20Data.xlsx"
)
Q4_2024_R2_DOCKET_URL = (
    "https://efiling.energysafety.ca.gov/Lists/DocketLog.aspx?"
    "docketnumber=2025+Data+Submissions"
)

LEGACY_HEADERS = [
    "Top- Risk Circuit/ Segment/ Span ID",
    "Risk Granularity",
    "Line Class",
    "Circuit/ Segment/ Span Length (mi)",
    "Inclusion Reason",
    "HFTD Area",
    "Overall Utility Risk",
    "Ignition Risk",
    "PSPS Risk",
    "Ignition Likelihood",
    "Equipment Likelihood of Ignition",
    "Contact from Vegetation Likelihood of Ignition",
    "Contact from Object Likelihood of Ignition",
    "Burn Probability",
    "PSPS Likelihood",
    "Wildfire Consequence",
    "Wildfire Hazard Intensity",
    "Wildfire Exposure Potential",
    "Wildfire Vulnerability",
    "PSPS Consequence",
    "PSPS Exposure Potential",
    "Vulnerability of Community to PSPS",
]

V4_HEADERS = [
    "METRIC NUMBER",
    "TOP-RISK CIRCUIT / SEGMENT / SPAN ID",
    "RISK GRANULARITY",
    "LINE TYPE",
    "CIRCUIT / SEGMENT / SPAN LENGTH (MI)",
    "INCLUSION REASON",
    "HFTD TIER",
    "OVERALL UTILITY RISK",
    "WILDFIRE RISK",
    "OUTAGE PROGRAM RISK",
    "WILDFIRE LIKELIHOOD",
    "IGNITION LIKELIHOOD",
    "EQUIPMENT CAUSED LIKELIHOOD OF IGNITION",
    "CONTACT FROM VEGETATION LIKELIHOOD OF IGNITION",
    "CONTACT FROM OBJECT LIKELIHOOD OF IGNITION",
    "BURN LIKELIHOOD",
    "WILDFIRE CONSEQUENCE",
    "WILDFIRE HAZARD INTENSITY",
    "WILDFIRE EXPOSURE POTENTIAL",
    "PSPS RISK",
    "PSPS LIKELIHOOD",
    "PSPS CONSEQUENCE",
    "PSPS EXPOSURE POTENTIAL",
    "PSPS VULNERABILITY",
    "PEDS RISK",
    "PEDS LIKELIHOOD",
    "PEDS CONSEQUENCE",
    "PEDS EXPOSURE POTENTIAL",
    "PEDS VULNERABILITY",
    "COMMENTS",
    "BLANK MEANING",
    "UTILITY ID",
    "REPORTING YEAR",
]

RISK_METRICS = [
    "overall_utility_risk",
    "wildfire_risk",
    "outage_program_risk",
    "wildfire_likelihood",
    "ignition_likelihood",
    "equipment_caused_likelihood_of_ignition",
    "contact_from_vegetation_likelihood_of_ignition",
    "contact_from_object_likelihood_of_ignition",
    "burn_likelihood",
    "wildfire_consequence",
    "wildfire_hazard_intensity",
    "wildfire_exposure_potential",
    "wildfire_vulnerability",
    "psps_risk",
    "psps_likelihood",
    "psps_consequence",
    "psps_exposure_potential",
    "psps_vulnerability",
    "peds_risk",
    "peds_likelihood",
    "peds_consequence",
    "peds_exposure_potential",
    "peds_vulnerability",
]

UNIFIED_HEADERS = [
    "record_id",
    "source_vintage_year",
    "year_basis",
    "reporting_year",
    "year_alignment_status",
    "metric_number",
    "metric_number_status",
    "segment_id",
    "risk_granularity",
    "source_risk_granularity_raw",
    "risk_granularity_status",
    "line_type",
    "source_line_type_or_class_raw",
    "line_type_status",
    "segment_length_mi",
    "inclusion_reason",
    "hftd_tier",
    "source_hftd_raw",
    "hftd_status",
    "source_row_order",
    "overall_risk_rank",
    *RISK_METRICS,
    "comments",
    "blank_meaning",
    "utility_id",
    "schema_version",
    "source_revision",
    "revision_selection_status",
    "source_file",
    "source_sheet",
    "source_row",
    "source_url",
    "guideline_url",
]

LONG_HEADERS = [
    "record_id",
    "source_vintage_year",
    "reporting_year",
    "year_alignment_status",
    "metric_number",
    "segment_id",
    "risk_granularity",
    "line_type",
    "hftd_tier",
    "overall_risk_rank",
    "canonical_metric",
    "source_metric_name",
    "value",
    "availability_status",
    "crosswalk_status",
    "comparability_status",
    "source_file",
    "source_row",
]

CROSSWALK_HEADERS = [
    "segment_id",
    "present_2023",
    "rank_2023",
    "length_mi_2023",
    "overall_risk_2023",
    "present_2024",
    "rank_2024",
    "length_mi_2024",
    "overall_risk_2024",
    "present_2025_vintage",
    "reporting_year_2025_vintage",
    "rank_2025_vintage",
    "length_mi_2025_vintage",
    "overall_risk_2025_vintage",
    "source_vintages_present",
    "hftd_tiers_observed",
    "line_types_observed",
    "inclusion_reasons_observed",
]

REVISION_HEADERS = [
    "segment_id",
    "revision_status",
    "r0_metric_number",
    "r1_metric_number",
    "r0_source_row",
    "r1_source_row",
    "r0_rank",
    "r1_rank",
    "r0_length_mi",
    "r1_length_mi",
    "length_delta_mi",
    "r0_overall_utility_risk",
    "r1_overall_utility_risk",
    "overall_risk_delta",
    "r0_wildfire_risk",
    "r1_wildfire_risk",
    "r0_outage_program_risk",
    "r1_outage_program_risk",
    "r0_psps_risk",
    "r1_psps_risk",
    "r0_peds_risk",
    "r1_peds_risk",
    "changed_fields",
]

REVISION_SUMMARY_HEADERS = [
    "revision_check",
    "count_or_value",
    "status",
    "note",
]

SCHEMA_HEADERS = [
    "canonical_field",
    "v3_1_field",
    "v3_2_field",
    "v4_01_field",
    "schema_change",
    "converter_action",
]

RECON_HEADERS = [
    "source_vintage_year",
    "reporting_year",
    "segment_id",
    "check_name",
    "reported_value",
    "calculated_value",
    "difference",
    "status",
    "source_file",
    "source_row",
]

ISSUE_HEADERS = [
    "issue_type",
    "severity",
    "source_vintage_year",
    "reporting_year",
    "source_file",
    "field_name",
    "raw_value",
    "affected_rows",
    "note",
]

SOURCE_SUMMARY_HEADERS = [
    "source_vintage_year",
    "reporting_year",
    "schema_version",
    "source_revision",
    "source_file",
    "rows",
    "distinct_segment_ids",
    "risk_granularity_values",
    "line_type_or_class_values",
    "hftd_values",
    "inclusion_reason_values",
    "overall_risk_min",
    "overall_risk_max",
    "formula_mismatches",
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
    return str(value).strip() if value is not None else None


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
    text = str(value).replace(",", "").replace("$", "")
    number = float(text)
    return int(number) if number.is_integer() else number


def relative_close(
    a: Any,
    b: Any,
    tolerance: float = 1e-8,
) -> bool:
    if a is None or b is None:
        return False
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) <= tolerance * scale


def column_index(letters: str) -> int:
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def column_letter(index: int) -> str:
    number = index + 1
    output = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        output = chr(65 + remainder) + output
    return output


def parse_revision(path: Path) -> int:
    match = re.search(r"(?:_R|_Rev)(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def stable_hash(prefix: str, payload: Any, length: int = 16) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return prefix + hashlib.sha1(encoded).hexdigest()[:length]


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[Any]]:
    """Read cached cell values directly from an XLSX package."""
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
        reference = (
            dimension.attrib.get("ref", "A1")
            if dimension is not None
            else "A1"
        )
        last_cell = reference.split(":")[-1]
        match = re.match(r"([A-Z]+)(\d+)", last_cell)
        if not match:
            raise ValueError(
                f"Unrecognized worksheet dimension {reference!r}"
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
                address = re.match(r"([A-Z]+)(\d+)", cell.attrib["r"])
                if not address:
                    continue
                column = column_index(address.group(1))
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


def discover_sources(
    input_dir: Path,
    annual_wmp_r1: Path,
    annual_wmp_r0: Path,
) -> dict[str, dict[str, Any]]:
    source_2023 = input_dir / "SDGE_2023_Q4_Tables1-15.xlsx"
    if not source_2023.exists():
        raise FileNotFoundError(source_2023)

    candidates_2024 = list(
        input_dir.glob("SDGE_2024_Q4_Tables1-15*.xlsx")
    )
    if not candidates_2024:
        raise FileNotFoundError("No 2024 Q4 Tables 1-15 workbook found")
    source_2024 = max(candidates_2024, key=parse_revision)

    if not annual_wmp_r1.exists():
        raise FileNotFoundError(annual_wmp_r1)
    if not annual_wmp_r0.exists():
        raise FileNotFoundError(annual_wmp_r0)

    return {
        "2023": {
            "path": source_2023,
            "source_vintage_year": 2023,
            "reporting_year": 2023,
            "schema_version": "v3.1",
            "revision": f"R{parse_revision(source_2023)}",
            "revision_selection_status": "required_annual_q4_snapshot",
            "source_url": None,
            "guideline_url": GUIDELINES[2023][1],
        },
        "2024": {
            "path": source_2024,
            "source_vintage_year": 2024,
            "reporting_year": 2024,
            "schema_version": "v3.2",
            "revision": f"R{parse_revision(source_2024)}",
            "revision_selection_status": (
                "highest_local_q4_revision; official_q4_r2_is_docketed_"
                "but_not_available_in_input_directory"
            ),
            "source_url": Q4_2024_R2_DOCKET_URL,
            "guideline_url": GUIDELINES[2024][1],
        },
        "2025": {
            "path": annual_wmp_r1,
            "source_vintage_year": 2025,
            "reporting_year": None,
            "schema_version": "v4.01",
            "revision": "R1",
            "revision_selection_status": (
                "highest_identified_annual_wmp_tabular_revision"
            ),
            "source_url": R1_DATA_URL,
            "guideline_url": GUIDELINES[2025][1],
        },
        "2025_r0": {
            "path": annual_wmp_r0,
            "source_vintage_year": 2025,
            "reporting_year": None,
            "schema_version": "v4.01",
            "revision": "R0",
            "revision_selection_status": "revision_baseline_only",
            "source_url": R0_DATA_URL,
            "guideline_url": GUIDELINES[2025][1],
        },
    }


def normalize_risk_granularity(value: Any) -> tuple[str, str]:
    raw = clean(value)
    mapping = {
        "Segment Level": (
            "Segment",
            "normalized_legacy_segment_level_to_segment",
        ),
        "Segment-level": (
            "Segment",
            "normalized_v4_segment_level_to_segment",
        ),
        "Segment": ("Segment", "schema_native"),
        "Circuit": ("Circuit", "schema_native"),
        "Span": ("Span", "schema_native"),
    }
    if raw not in mapping:
        raise ValueError(f"Unexpected risk granularity: {raw!r}")
    return mapping[raw]


def normalize_legacy_line_class(value: Any) -> tuple[str, str]:
    raw = clean(value)
    if raw == "Primary Overhead Conductor":
        return (
            "Distribution",
            "inferred_distribution_from_nonconforming_legacy_line_class",
        )
    if raw in {"Distribution", "Transmission"}:
        return raw, "legacy_schema_native"
    raise ValueError(f"Unexpected legacy Line Class: {raw!r}")


def normalize_line_type(value: Any) -> tuple[str, str]:
    raw = clean(value)
    if raw in {"Distribution", "Transmission"}:
        return raw, "v4_native"
    if raw == "Transmision":
        return "Transmission", "corrected_source_typo_transmision"
    raise ValueError(f"Unexpected v4 LINE TYPE: {raw!r}")


def normalize_hftd_legacy(value: Any) -> tuple[str, str]:
    raw = clean(value)
    mapping = {
        2: ("HFTD Tier 2", "normalized_numeric_tier"),
        3: ("HFTD Tier 3", "normalized_numeric_tier"),
        "2": ("HFTD Tier 2", "normalized_numeric_tier"),
        "3": ("HFTD Tier 3", "normalized_numeric_tier"),
        "Non-HFTD": ("Non-HFTD", "schema_native"),
        "HFTD Tier 2": ("HFTD Tier 2", "schema_native"),
        "HFTD Tier 3": ("HFTD Tier 3", "schema_native"),
    }
    if raw not in mapping:
        raise ValueError(f"Unexpected legacy HFTD Area: {raw!r}")
    return mapping[raw]


def normalize_hftd_v4(value: Any) -> tuple[str, str]:
    raw = clean(value)
    mapping = {
        "Non-HFTD": ("Non-HFTD", "v4_native"),
        "HFTD TIER 2": (
            "HFTD Tier 2",
            "normalized_v4_capitalization",
        ),
        "HFTD TIER 3": (
            "HFTD Tier 3",
            "normalized_v4_capitalization",
        ),
        "HFTD Tier 2": ("HFTD Tier 2", "v4_native"),
        "HFTD Tier 3": ("HFTD Tier 3", "v4_native"),
    }
    if raw not in mapping:
        raise ValueError(f"Unexpected v4 HFTD TIER: {raw!r}")
    return mapping[raw]


def validate_inclusion_reason(value: Any) -> str:
    raw = clean(value)
    allowed = {
        ">1% contribution",
        "Top 5% highest risk",
        "Both >1% and Top 5%",
    }
    if raw not in allowed:
        raise ValueError(f"Unexpected inclusion reason: {raw!r}")
    return raw


def assign_ranks(records: list[dict[str, Any]]) -> None:
    ordered = sorted(
        records,
        key=lambda record: (
            -float(record["overall_utility_risk"]),
            record["segment_id"],
        ),
    )
    for rank, record in enumerate(ordered, start=1):
        record["overall_risk_rank"] = rank


def parse_legacy(
    values: list[list[Any]],
    source: dict[str, Any],
    issues: list[list[Any]],
) -> list[dict[str, Any]]:
    headers = [normalize_header(value) for value in values[8][:22]]
    if headers != LEGACY_HEADERS:
        raise ValueError(
            f"Legacy Table 15 schema mismatch in {source['path'].name}.\n"
            f"Expected: {LEGACY_HEADERS}\nFound: {headers}"
        )

    utility = clean(values[1][1])
    if utility not in {"SDG&E", "SDGE"}:
        raise AssertionError(
            f"Unexpected utility in {source['path'].name}: {utility!r}"
        )

    records: list[dict[str, Any]] = []
    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        if clean(row[0]) is None:
            continue

        granularity, granularity_status = normalize_risk_granularity(row[1])
        line_type, line_status = normalize_legacy_line_class(row[2])
        hftd_tier, hftd_status = normalize_hftd_legacy(row[5])

        record = {
            "source_vintage_year": source["source_vintage_year"],
            "year_basis": "q4_annual_update",
            "reporting_year": source["reporting_year"],
            "year_alignment_status": "source_vintage_equals_reporting_year",
            "metric_number": None,
            "metric_number_status": "not_present_in_v3_1_or_v3_2_schema",
            "segment_id": str(clean(row[0])),
            "risk_granularity": granularity,
            "source_risk_granularity_raw": clean(row[1]),
            "risk_granularity_status": granularity_status,
            "line_type": line_type,
            "source_line_type_or_class_raw": clean(row[2]),
            "line_type_status": line_status,
            "segment_length_mi": parse_number(row[3]),
            "inclusion_reason": validate_inclusion_reason(row[4]),
            "hftd_tier": hftd_tier,
            "source_hftd_raw": clean(row[5]),
            "hftd_status": hftd_status,
            "source_row_order": zero_based_row - 8,
            "overall_risk_rank": None,
            "overall_utility_risk": parse_number(row[6]),
            "wildfire_risk": parse_number(row[7]),
            "outage_program_risk": None,
            "wildfire_likelihood": None,
            "ignition_likelihood": parse_number(row[9]),
            "equipment_caused_likelihood_of_ignition": parse_number(row[10]),
            "contact_from_vegetation_likelihood_of_ignition": parse_number(row[11]),
            "contact_from_object_likelihood_of_ignition": parse_number(row[12]),
            "burn_likelihood": parse_number(row[13]),
            "wildfire_consequence": parse_number(row[15]),
            "wildfire_hazard_intensity": parse_number(row[16]),
            "wildfire_exposure_potential": parse_number(row[17]),
            "wildfire_vulnerability": parse_number(row[18]),
            "psps_risk": parse_number(row[8]),
            "psps_likelihood": parse_number(row[14]),
            "psps_consequence": parse_number(row[19]),
            "psps_exposure_potential": parse_number(row[20]),
            "psps_vulnerability": parse_number(row[21]),
            "peds_risk": None,
            "peds_likelihood": None,
            "peds_consequence": None,
            "peds_exposure_potential": None,
            "peds_vulnerability": None,
            "comments": None,
            "blank_meaning": None,
            "utility_id": "SDG&E",
            "schema_version": source["schema_version"],
            "source_revision": source["revision"],
            "revision_selection_status": source["revision_selection_status"],
            "source_file": source["path"].name,
            "source_sheet": "Table 15",
            "source_row": zero_based_row + 1,
            "source_url": source["source_url"],
            "guideline_url": source["guideline_url"],
        }
        records.append(record)

    if len(records) != 28:
        raise AssertionError(
            f"Expected 28 legacy Table 15 rows in {source['path'].name}; "
            f"found {len(records)}"
        )

    if len({record["segment_id"] for record in records}) != len(records):
        raise AssertionError(
            f"Duplicate segment IDs in {source['path'].name}"
        )

    assign_ranks(records)

    issues.append(
        [
            "legacy_line_class_schema_nonconformance",
            "warning",
            source["source_vintage_year"],
            source["reporting_year"],
            source["path"].name,
            "Line Class",
            "Primary Overhead Conductor",
            len(records),
            "The v3.1/v3.2 schema restricts this field to Distribution or "
            "Transmission, but SDG&E reported Primary Overhead Conductor. "
            "The analytical line_type is inferred as Distribution and the raw "
            "source value is retained.",
        ]
    )
    issues.append(
        [
            "risk_granularity_label_normalized",
            "info",
            source["source_vintage_year"],
            source["reporting_year"],
            source["path"].name,
            "Risk Granularity",
            "Segment Level",
            len(records),
            "The restricted schema value is Segment. The source label "
            "Segment Level is normalized to Segment.",
        ]
    )

    return records


def parse_v4(
    values: list[list[Any]],
    source: dict[str, Any],
    *,
    expected_metric_prefix: int,
    metric_status: str,
    issues: list[list[Any]] | None = None,
) -> list[dict[str, Any]]:
    headers = [normalize_header(value) for value in values[0][:33]]
    if headers != V4_HEADERS:
        raise ValueError(
            f"v4.01 Table 15 schema mismatch in {source['path'].name}.\n"
            f"Expected: {V4_HEADERS}\nFound: {headers}"
        )

    records: list[dict[str, Any]] = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[0]) is None:
            continue

        granularity, granularity_status = normalize_risk_granularity(row[2])
        line_type, line_status = normalize_line_type(row[3])
        hftd_tier, hftd_status = normalize_hftd_v4(row[6])
        reporting_year = int(row[32])

        record = {
            "source_vintage_year": source["source_vintage_year"],
            "year_basis": "annual_wmp_submission_vintage",
            "reporting_year": reporting_year,
            "year_alignment_status": (
                "2025_submission_vintage_reports_2026_risk_year"
                if source["source_vintage_year"] == 2025
                and reporting_year == 2026
                else "source_vintage_differs_from_reporting_year"
            ),
            "metric_number": int(row[0]),
            "metric_number_status": metric_status,
            "segment_id": str(clean(row[1])),
            "risk_granularity": granularity,
            "source_risk_granularity_raw": clean(row[2]),
            "risk_granularity_status": granularity_status,
            "line_type": line_type,
            "source_line_type_or_class_raw": clean(row[3]),
            "line_type_status": line_status,
            "segment_length_mi": parse_number(row[4]),
            "inclusion_reason": validate_inclusion_reason(row[5]),
            "hftd_tier": hftd_tier,
            "source_hftd_raw": clean(row[6]),
            "hftd_status": hftd_status,
            "source_row_order": zero_based_row,
            "overall_risk_rank": None,
            "overall_utility_risk": parse_number(row[7]),
            "wildfire_risk": parse_number(row[8]),
            "outage_program_risk": parse_number(row[9]),
            "wildfire_likelihood": parse_number(row[10]),
            "ignition_likelihood": parse_number(row[11]),
            "equipment_caused_likelihood_of_ignition": parse_number(row[12]),
            "contact_from_vegetation_likelihood_of_ignition": parse_number(row[13]),
            "contact_from_object_likelihood_of_ignition": parse_number(row[14]),
            "burn_likelihood": parse_number(row[15]),
            "wildfire_consequence": parse_number(row[16]),
            "wildfire_hazard_intensity": parse_number(row[17]),
            "wildfire_exposure_potential": parse_number(row[18]),
            "wildfire_vulnerability": None,
            "psps_risk": parse_number(row[19]),
            "psps_likelihood": parse_number(row[20]),
            "psps_consequence": parse_number(row[21]),
            "psps_exposure_potential": parse_number(row[22]),
            "psps_vulnerability": parse_number(row[23]),
            "peds_risk": parse_number(row[24]),
            "peds_likelihood": parse_number(row[25]),
            "peds_consequence": parse_number(row[26]),
            "peds_exposure_potential": parse_number(row[27]),
            "peds_vulnerability": parse_number(row[28]),
            "comments": clean(row[29]),
            "blank_meaning": clean(row[30]),
            "utility_id": clean(row[31]),
            "schema_version": source["schema_version"],
            "source_revision": source["revision"],
            "revision_selection_status": source["revision_selection_status"],
            "source_file": source["path"].name,
            "source_sheet": "Table 15",
            "source_row": zero_based_row + 1,
            "source_url": source["source_url"],
            "guideline_url": source["guideline_url"],
        }
        records.append(record)

    if len(records) != 261:
        raise AssertionError(
            f"Expected 261 v4 Table 15 rows in {source['path'].name}; "
            f"found {len(records)}"
        )
    if len({record["segment_id"] for record in records}) != len(records):
        raise AssertionError(
            f"Duplicate segment IDs in {source['path'].name}"
        )

    expected_numbers = list(
        range(expected_metric_prefix, expected_metric_prefix + len(records))
    )
    actual_numbers = [record["metric_number"] for record in records]
    if actual_numbers != expected_numbers:
        raise AssertionError(
            f"Metric numbers in {source['path'].name} do not match the "
            f"expected sequence beginning {expected_metric_prefix}"
        )

    assign_ranks(records)

    if issues is not None:
        issues.append(
            [
                "risk_granularity_label_normalized",
                "info",
                source["source_vintage_year"],
                records[0]["reporting_year"],
                source["path"].name,
                "RISK GRANULARITY",
                "Segment-level",
                len(records),
                "The restricted schema value is Segment. The source label "
                "Segment-level is normalized to Segment.",
            ]
        )
        issues.append(
            [
                "source_vintage_reporting_year_difference",
                "info",
                source["source_vintage_year"],
                records[0]["reporting_year"],
                source["path"].name,
                "REPORTING YEAR",
                records[0]["reporting_year"],
                len(records),
                "The Annual-WMP workbook was filed in 2025 for the 2026-2028 "
                "Base WMP and explicitly reports 2026. The converter preserves "
                "source_vintage_year=2025 and reporting_year=2026.",
            ]
        )

    return records


def record_id(record: dict[str, Any]) -> str:
    return stable_hash(
        "T15R-",
        [
            record["source_vintage_year"],
            record["reporting_year"],
            record["segment_id"],
            record["source_file"],
            record["source_row"],
        ],
    )


def record_to_unified_row(record: dict[str, Any]) -> list[Any]:
    return [
        record_id(record),
        record["source_vintage_year"],
        record["year_basis"],
        record["reporting_year"],
        record["year_alignment_status"],
        record["metric_number"],
        record["metric_number_status"],
        record["segment_id"],
        record["risk_granularity"],
        record["source_risk_granularity_raw"],
        record["risk_granularity_status"],
        record["line_type"],
        record["source_line_type_or_class_raw"],
        record["line_type_status"],
        record["segment_length_mi"],
        record["inclusion_reason"],
        record["hftd_tier"],
        record["source_hftd_raw"],
        record["hftd_status"],
        record["source_row_order"],
        record["overall_risk_rank"],
        *[record[metric] for metric in RISK_METRICS],
        record["comments"],
        record["blank_meaning"],
        record["utility_id"],
        record["schema_version"],
        record["source_revision"],
        record["revision_selection_status"],
        record["source_file"],
        record["source_sheet"],
        record["source_row"],
        record["source_url"],
        record["guideline_url"],
    ]


def source_metric_name(record: dict[str, Any], metric: str) -> str | None:
    if record["schema_version"] in {"v3.1", "v3.2"}:
        mapping = {
            "overall_utility_risk": "Overall Utility Risk",
            "wildfire_risk": "Ignition Risk",
            "ignition_likelihood": "Ignition Likelihood",
            "equipment_caused_likelihood_of_ignition": (
                "Equipment Likelihood of Ignition"
            ),
            "contact_from_vegetation_likelihood_of_ignition": (
                "Contact from Vegetation Likelihood of Ignition"
            ),
            "contact_from_object_likelihood_of_ignition": (
                "Contact from Object Likelihood of Ignition"
            ),
            "burn_likelihood": "Burn Probability",
            "wildfire_consequence": "Wildfire Consequence",
            "wildfire_hazard_intensity": "Wildfire Hazard Intensity",
            "wildfire_exposure_potential": "Wildfire Exposure Potential",
            "wildfire_vulnerability": "Wildfire Vulnerability",
            "psps_risk": "PSPS Risk",
            "psps_likelihood": "PSPS Likelihood",
            "psps_consequence": "PSPS Consequence",
            "psps_exposure_potential": "PSPS Exposure Potential",
            "psps_vulnerability": "Vulnerability of Community to PSPS",
        }
        return mapping.get(metric)

    mapping = {
        metric_name: metric_name.replace("_", " ").upper()
        for metric_name in RISK_METRICS
    }
    return mapping.get(metric)


def metric_status(
    record: dict[str, Any],
    metric: str,
) -> tuple[str, str, str]:
    legacy = record["schema_version"] in {"v3.1", "v3.2"}
    value = record[metric]

    if legacy and metric in {
        "outage_program_risk",
        "wildfire_likelihood",
        "peds_risk",
        "peds_likelihood",
        "peds_consequence",
        "peds_exposure_potential",
        "peds_vulnerability",
    }:
        return (
            "not_in_legacy_schema",
            "new_in_v4_01",
            "not_comparable_missing_legacy_field",
        )

    if not legacy and metric == "wildfire_vulnerability":
        return (
            "not_in_v4_01_template",
            "legacy_field_not_separately_reported_in_v4_01",
            "not_comparable_missing_v4_field",
        )

    availability = "reported" if value is not None else "source_blank"

    if legacy and metric == "wildfire_risk":
        crosswalk = "legacy_ignition_risk_mapped_to_wildfire_risk"
    elif legacy and metric == "burn_likelihood":
        crosswalk = "legacy_burn_probability_mapped_to_burn_likelihood"
    elif legacy and metric == "psps_vulnerability":
        crosswalk = (
            "legacy_vulnerability_of_community_to_psps_mapped_to_"
            "psps_vulnerability"
        )
    else:
        crosswalk = "same_or_standardized_metric"

    comparability = (
        "schema_aligned_but_risk_model_scale_changes_between_vintages"
    )
    if metric in {"overall_utility_risk", "outage_program_risk"}:
        comparability = (
            "definition_expanded_in_v4_to_include_peds_and_outage_program"
        )

    return availability, crosswalk, comparability


def build_long_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in records:
        for metric in RISK_METRICS:
            availability, crosswalk, comparability = metric_status(
                record,
                metric,
            )
            rows.append(
                [
                    record_id(record),
                    record["source_vintage_year"],
                    record["reporting_year"],
                    record["year_alignment_status"],
                    record["metric_number"],
                    record["segment_id"],
                    record["risk_granularity"],
                    record["line_type"],
                    record["hftd_tier"],
                    record["overall_risk_rank"],
                    metric,
                    source_metric_name(record, metric),
                    record[metric],
                    availability,
                    crosswalk,
                    comparability,
                    record["source_file"],
                    record["source_row"],
                ]
            )
    return rows


def build_segment_crosswalk(
    records: list[dict[str, Any]],
) -> list[list[Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_segment[record["segment_id"]].append(record)

    rows: list[list[Any]] = []
    for segment_id, segment_records in sorted(by_segment.items()):
        by_vintage = {
            record["source_vintage_year"]: record
            for record in segment_records
        }

        record_2023 = by_vintage.get(2023)
        record_2024 = by_vintage.get(2024)
        record_2025 = by_vintage.get(2025)

        rows.append(
            [
                segment_id,
                record_2023 is not None,
                (
                    record_2023["overall_risk_rank"]
                    if record_2023
                    else None
                ),
                (
                    record_2023["segment_length_mi"]
                    if record_2023
                    else None
                ),
                (
                    record_2023["overall_utility_risk"]
                    if record_2023
                    else None
                ),
                record_2024 is not None,
                (
                    record_2024["overall_risk_rank"]
                    if record_2024
                    else None
                ),
                (
                    record_2024["segment_length_mi"]
                    if record_2024
                    else None
                ),
                (
                    record_2024["overall_utility_risk"]
                    if record_2024
                    else None
                ),
                record_2025 is not None,
                (
                    record_2025["reporting_year"]
                    if record_2025
                    else None
                ),
                (
                    record_2025["overall_risk_rank"]
                    if record_2025
                    else None
                ),
                (
                    record_2025["segment_length_mi"]
                    if record_2025
                    else None
                ),
                (
                    record_2025["overall_utility_risk"]
                    if record_2025
                    else None
                ),
                ";".join(
                    str(year)
                    for year in sorted(by_vintage)
                ),
                ";".join(
                    sorted(
                        {
                            record["hftd_tier"]
                            for record in segment_records
                        }
                    )
                ),
                ";".join(
                    sorted(
                        {
                            record["line_type"]
                            for record in segment_records
                        }
                    )
                ),
                ";".join(
                    sorted(
                        {
                            record["inclusion_reason"]
                            for record in segment_records
                        }
                    )
                ),
            ]
        )

    return rows


def build_revision_comparison(
    r0_records: list[dict[str, Any]],
    r1_records: list[dict[str, Any]],
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    r0_by_id = {record["segment_id"]: record for record in r0_records}
    r1_by_id = {record["segment_id"]: record for record in r1_records}
    all_ids = sorted(set(r0_by_id) | set(r1_by_id))

    comparison_rows: list[list[Any]] = []
    field_change_counts: Counter[str] = Counter()

    compared_fields = [
        "metric_number",
        "segment_length_mi",
        "overall_utility_risk",
        "wildfire_risk",
        "outage_program_risk",
        "wildfire_consequence",
        "psps_risk",
        "psps_consequence",
        "peds_risk",
        "peds_consequence",
        "comments",
    ]

    for segment_id in all_ids:
        r0 = r0_by_id.get(segment_id)
        r1 = r1_by_id.get(segment_id)

        if r0 is None:
            status = "added_in_r1"
            changed_fields = "new segment"
        elif r1 is None:
            status = "removed_in_r1"
            changed_fields = "removed segment"
        else:
            changed = [
                field
                for field in compared_fields
                if r0[field] != r1[field]
            ]
            for field in changed:
                field_change_counts[field] += 1
            status = "common_segment_changed" if changed else "unchanged"
            changed_fields = ";".join(changed) if changed else None

        def get(record: dict[str, Any] | None, field: str) -> Any:
            return record[field] if record is not None else None

        length_delta = (
            float(r1["segment_length_mi"]) - float(r0["segment_length_mi"])
            if r0 is not None and r1 is not None
            else None
        )
        risk_delta = (
            float(r1["overall_utility_risk"])
            - float(r0["overall_utility_risk"])
            if r0 is not None and r1 is not None
            else None
        )

        comparison_rows.append(
            [
                segment_id,
                status,
                get(r0, "metric_number"),
                get(r1, "metric_number"),
                get(r0, "source_row"),
                get(r1, "source_row"),
                get(r0, "overall_risk_rank"),
                get(r1, "overall_risk_rank"),
                get(r0, "segment_length_mi"),
                get(r1, "segment_length_mi"),
                length_delta,
                get(r0, "overall_utility_risk"),
                get(r1, "overall_utility_risk"),
                risk_delta,
                get(r0, "wildfire_risk"),
                get(r1, "wildfire_risk"),
                get(r0, "outage_program_risk"),
                get(r1, "outage_program_risk"),
                get(r0, "psps_risk"),
                get(r1, "psps_risk"),
                get(r0, "peds_risk"),
                get(r1, "peds_risk"),
                changed_fields,
            ]
        )

    removed = sorted(set(r0_by_id) - set(r1_by_id))
    added = sorted(set(r1_by_id) - set(r0_by_id))
    common = set(r0_by_id) & set(r1_by_id)

    summary_rows = [
        [
            "R0 metric-number prefix",
            "1150000000–1150000260",
            "superseded",
            "The v4 template changelog corrected Table 15 metric numbers "
            "from a 115... prefix to 215....",
        ],
        [
            "R1 metric-number prefix",
            "2150000000–2150000260",
            "validated",
            "The selected R1 workbook uses the corrected prefix and a "
            "continuous sequence.",
        ],
        [
            "Rows in R0",
            len(r0_records),
            "validated",
            "R0 is retained only for revision comparison.",
        ],
        [
            "Rows in R1",
            len(r1_records),
            "selected",
            "R1 is used in the unified dataset.",
        ],
        [
            "Common segment IDs",
            len(common),
            "compared",
            "Common records were matched by TOP-RISK CIRCUIT / SEGMENT / "
            "SPAN ID.",
        ],
        [
            "Segments removed in R1",
            len(removed),
            "revised",
            "; ".join(removed),
        ],
        [
            "Segments added in R1",
            len(added),
            "revised",
            "; ".join(added),
        ],
    ]

    for field, count in sorted(field_change_counts.items()):
        summary_rows.append(
            [
                f"Common segments changed: {field}",
                count,
                "revised",
                "Count of the 259 common segment IDs whose value changed "
                "from R0 to R1.",
            ]
        )

    stats = {
        "r0_rows": len(r0_records),
        "r1_rows": len(r1_records),
        "common_segment_ids": len(common),
        "removed_segment_ids": removed,
        "added_segment_ids": added,
        "field_change_counts": dict(field_change_counts),
    }
    return comparison_rows, summary_rows, stats


def build_reconciliation(
    records: list[dict[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[list[Any]], int]:
    rows: list[list[Any]] = []
    mismatch_count = 0

    def add_check(
        record: dict[str, Any],
        check_name: str,
        reported: Any,
        calculated: Any,
    ) -> None:
        nonlocal mismatch_count

        if reported is None or calculated is None:
            difference = None
            status = "not_testable_due_to_blank"
        else:
            difference = float(reported) - float(calculated)
            if relative_close(reported, calculated, tolerance):
                status = "reconciled"
            else:
                status = "mismatch"
                mismatch_count += 1

        rows.append(
            [
                record["source_vintage_year"],
                record["reporting_year"],
                record["segment_id"],
                check_name,
                reported,
                calculated,
                difference,
                status,
                record["source_file"],
                record["source_row"],
            ]
        )

    for record in records:
        legacy = record["schema_version"] in {"v3.1", "v3.2"}

        if legacy:
            overall_sum = (
                record["wildfire_risk"] + record["psps_risk"]
                if record["wildfire_risk"] is not None
                and record["psps_risk"] is not None
                else None
            )
            add_check(
                record,
                "legacy_overall_equals_ignition_plus_psps_risk",
                record["overall_utility_risk"],
                overall_sum,
            )

            wildfire_product = (
                record["ignition_likelihood"]
                * record["wildfire_consequence"]
                if record["ignition_likelihood"] is not None
                and record["wildfire_consequence"] is not None
                else None
            )
            add_check(
                record,
                "legacy_ignition_risk_equals_ignition_likelihood_times_"
                "wildfire_consequence",
                record["wildfire_risk"],
                wildfire_product,
            )
        else:
            overall_sum = (
                record["wildfire_risk"] + record["outage_program_risk"]
                if record["wildfire_risk"] is not None
                and record["outage_program_risk"] is not None
                else None
            )
            add_check(
                record,
                "v4_overall_equals_wildfire_plus_outage_program_risk",
                record["overall_utility_risk"],
                overall_sum,
            )

            outage_sum = (
                record["psps_risk"] + record["peds_risk"]
                if record["psps_risk"] is not None
                and record["peds_risk"] is not None
                else None
            )
            add_check(
                record,
                "v4_outage_program_equals_psps_plus_peds_risk",
                record["outage_program_risk"],
                outage_sum,
            )

            wildfire_product = (
                record["ignition_likelihood"]
                * record["wildfire_consequence"]
                if record["ignition_likelihood"] is not None
                and record["wildfire_consequence"] is not None
                else None
            )
            add_check(
                record,
                "v4_wildfire_risk_equals_ignition_likelihood_times_"
                "wildfire_consequence",
                record["wildfire_risk"],
                wildfire_product,
            )

            psps_product = (
                record["psps_likelihood"] * record["psps_consequence"]
                if record["psps_likelihood"] is not None
                and record["psps_consequence"] is not None
                else None
            )
            add_check(
                record,
                "v4_psps_risk_equals_likelihood_times_consequence",
                record["psps_risk"],
                psps_product,
            )

            peds_product = (
                record["peds_likelihood"] * record["peds_consequence"]
                if record["peds_likelihood"] is not None
                and record["peds_consequence"] is not None
                else None
            )
            add_check(
                record,
                "v4_peds_risk_equals_likelihood_times_consequence",
                record["peds_risk"],
                peds_product,
            )

    return rows, mismatch_count


def count_formula_mismatches(
    records: list[dict[str, Any]],
    tolerance: float,
) -> int:
    _, mismatches = build_reconciliation(
        records,
        tolerance=tolerance,
    )
    return mismatches


def validate_records(
    records: list[dict[str, Any]],
    issues: list[list[Any]],
) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_file"]].append(record)

        if record["segment_length_mi"] is None:
            issues.append(
                [
                    "missing_segment_length",
                    "error",
                    record["source_vintage_year"],
                    record["reporting_year"],
                    record["source_file"],
                    "segment_length_mi",
                    None,
                    1,
                    f"Segment {record['segment_id']} has no length.",
                ]
            )
        elif float(record["segment_length_mi"]) < 0:
            issues.append(
                [
                    "negative_segment_length",
                    "error",
                    record["source_vintage_year"],
                    record["reporting_year"],
                    record["source_file"],
                    "segment_length_mi",
                    record["segment_length_mi"],
                    1,
                    f"Segment {record['segment_id']} has a negative length.",
                ]
            )

        for metric in RISK_METRICS:
            value = record[metric]
            if value is not None and float(value) < 0:
                issues.append(
                    [
                        "negative_risk_value",
                        "error",
                        record["source_vintage_year"],
                        record["reporting_year"],
                        record["source_file"],
                        metric,
                        value,
                        1,
                        f"Segment {record['segment_id']} has a negative risk "
                        "component.",
                    ]
                )

    for source_file, source_records in by_source.items():
        if len({record["segment_id"] for record in source_records}) != len(
            source_records
        ):
            issues.append(
                [
                    "duplicate_segment_id_within_source",
                    "error",
                    source_records[0]["source_vintage_year"],
                    source_records[0]["reporting_year"],
                    source_file,
                    "segment_id",
                    None,
                    len(source_records),
                    "One or more segment IDs repeat within the source.",
                ]
            )


def build_schema_rows() -> list[list[Any]]:
    return [
        [
            "metric_number",
            None,
            None,
            "METRIC NUMBER",
            "new_in_v4_01_and_prefix_corrected",
            "Legacy rows remain blank. Validate R1 as 2150000000–2150000260. "
            "R0's 115... prefix is retained only in the revision comparison.",
        ],
        [
            "segment_id",
            "Top- Risk Circuit/ Segment/ Span ID",
            "Top- Risk Circuit/ Segment/ Span ID",
            "TOP-RISK CIRCUIT / SEGMENT / SPAN ID",
            "capitalization_and_punctuation_standardized",
            "Preserve as text and use it to compare R0 with R1 and link "
            "segments across source vintages.",
        ],
        [
            "risk_granularity",
            "Risk Granularity",
            "Risk Granularity",
            "RISK GRANULARITY",
            "restricted_labels_standardized",
            "Normalize Segment Level and Segment-level to Segment.",
        ],
        [
            "line_type",
            "Line Class",
            "Line Class",
            "LINE TYPE",
            "field_renamed_and_legacy_source_nonconforming",
            "The v3 schema restricts the field to Distribution/Transmission, "
            "but SDG&E used Primary Overhead Conductor. Infer Distribution "
            "for analysis, retain the raw value, and flag the inference.",
        ],
        [
            "segment_length_mi",
            "Circuit/ Segment/ Span Length (mi)",
            "Circuit/ Segment/ Span Length (mi)",
            "CIRCUIT / SEGMENT / SPAN LENGTH (MI)",
            "capitalization_standardized",
            "Require a nonnegative numeric value. R0-versus-R1 changes are "
            "reported separately.",
        ],
        [
            "hftd_tier",
            "HFTD Area",
            "HFTD Area",
            "HFTD TIER",
            "numeric_legacy_values_replaced_by_standard_labels",
            "Normalize 2/3 and uppercase v4 values to HFTD Tier 2/3.",
        ],
        [
            "wildfire_risk",
            "Ignition Risk",
            "Ignition Risk",
            "WILDFIRE RISK",
            "renamed",
            "Map legacy Ignition Risk to canonical wildfire_risk.",
        ],
        [
            "outage_program_risk",
            None,
            None,
            "OUTAGE PROGRAM RISK",
            "new_in_v4_01",
            "Leave legacy values blank. Do not substitute legacy PSPS Risk.",
        ],
        [
            "wildfire_likelihood",
            None,
            None,
            "WILDFIRE LIKELIHOOD",
            "new_in_v4_01",
            "Leave legacy values blank.",
        ],
        [
            "equipment_caused_likelihood_of_ignition",
            "Equipment Likelihood of Ignition",
            "Equipment Likelihood of Ignition",
            "EQUIPMENT CAUSED LIKELIHOOD OF IGNITION",
            "renamed",
            "Use the v4 canonical field name.",
        ],
        [
            "burn_likelihood",
            "Burn Probability",
            "Burn Probability",
            "BURN LIKELIHOOD",
            "renamed",
            "Use the v4 canonical field name.",
        ],
        [
            "wildfire_vulnerability",
            "Wildfire Vulnerability",
            "Wildfire Vulnerability",
            None,
            "not_separately_present_in_v4_01",
            "Preserve legacy values and leave v4 records blank.",
        ],
        [
            "psps_vulnerability",
            "Vulnerability of Community to PSPS",
            "Vulnerability of Community to PSPS",
            "PSPS VULNERABILITY",
            "renamed",
            "Use the v4 canonical field name.",
        ],
        [
            "peds_components",
            None,
            None,
            "PEDS RISK / LIKELIHOOD / CONSEQUENCE / EXPOSURE / VULNERABILITY",
            "new_in_v4_01",
            "Leave all legacy PEDS fields blank.",
        ],
        [
            "comments_blank_meaning_utility_year",
            None,
            None,
            "COMMENTS / BLANK MEANING / UTILITY ID / REPORTING YEAR",
            "new_metadata_in_v4_01",
            "Preserve the source metadata and keep filing vintage separate "
            "from reporting year.",
        ],
    ]


def verify_2024_local_stability(
    input_dir: Path,
) -> tuple[bool, list[str]]:
    files: list[Path] = []
    matrices: list[list[list[Any]]] = []

    for quarter in (1, 2, 3, 4):
        candidates = list(
            input_dir.glob(f"SDGE_2024_Q{quarter}_Tables1-15*.xlsx")
        )
        if not candidates:
            continue
        selected = max(candidates, key=parse_revision)
        values = read_xlsx_sheet(selected, "Table 15")
        matrix = [
            row[:22]
            for row in values[9:]
            if clean(row[0]) is not None
        ]
        files.append(selected)
        matrices.append(matrix)

    if not matrices:
        return False, []

    stable = all(matrix == matrices[0] for matrix in matrices[1:])
    return stable, [path.name for path in files]


def build_source_summary(
    grouped_records: dict[str, list[dict[str, Any]]],
    mismatch_counts: dict[str, int],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in ("2023", "2024", "2025"):
        records = grouped_records[key]
        rows.append(
            [
                records[0]["source_vintage_year"],
                records[0]["reporting_year"],
                records[0]["schema_version"],
                records[0]["source_revision"],
                records[0]["source_file"],
                len(records),
                len({record["segment_id"] for record in records}),
                ";".join(
                    sorted(
                        {
                            str(record["source_risk_granularity_raw"])
                            for record in records
                        }
                    )
                ),
                ";".join(
                    sorted(
                        {
                            str(record["source_line_type_or_class_raw"])
                            for record in records
                        }
                    )
                ),
                ";".join(
                    sorted(
                        {
                            str(record["source_hftd_raw"])
                            for record in records
                        }
                    )
                ),
                ";".join(
                    sorted(
                        {
                            record["inclusion_reason"]
                            for record in records
                        }
                    )
                ),
                min(
                    float(record["overall_utility_risk"])
                    for record in records
                ),
                max(
                    float(record["overall_utility_risk"])
                    for record in records
                ),
                mismatch_counts[key],
            ]
        )
    return rows


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


def write_rows(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    *,
    chunk_size: int = 500,
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
    sheet.get_range(f"A1:{last_column}{last_row}").format.wrap_text = True

    wide_fields = {
        "year_alignment_status",
        "metric_number_status",
        "risk_granularity_status",
        "line_type_status",
        "comments",
        "blank_meaning",
        "revision_selection_status",
        "source_file",
        "source_url",
        "guideline_url",
        "comparability_status",
        "changed_fields",
        "note",
        "converter_action",
    }
    medium_fields = {
        "source_line_type_or_class_raw",
        "source_risk_granularity_raw",
        "crosswalk_status",
        "availability_status",
        "revision_status",
        "check_name",
        "risk_granularity_values",
        "line_type_or_class_values",
        "hftd_values",
        "inclusion_reason_values",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 18
        if header in wide_fields:
            width = 38
        elif header in medium_fields:
            width = 28
        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width


def build_workbook(
    output_path: Path,
    unified_rows: list[list[Any]],
    long_rows: list[list[Any]],
    crosswalk_rows: list[list[Any]],
    revision_rows: list[list[Any]],
    revision_summary_rows: list[list[Any]],
    schema_rows: list[list[Any]],
    recon_rows: list[list[Any]],
    issue_rows: list[list[Any]],
    source_summary_rows: list[list[Any]],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()
    readme = workbook.worksheets.add("README")
    source_summary = workbook.worksheets.add("Source Summary")
    unified = workbook.worksheets.add("Unified Wide")
    long_sheet = workbook.worksheets.add("Metrics Long")
    crosswalk = workbook.worksheets.add("Segment Crosswalk")
    revision_summary = workbook.worksheets.add("R0-R1 Summary")
    revision_detail = workbook.worksheets.add("R0-R1 Detail")
    schema = workbook.worksheets.add("Schema Crosswalk")
    reconciliation = workbook.worksheets.add("Risk Reconciliation")
    issues = workbook.worksheets.add("Validation Issues")

    readme_rows = [
        ["SDG&E Table 15 Unified Top-Risk Segment Data", "", "", ""],
        [
            "Unified rows",
            validation["unified_rows"],
            "Long metric rows",
            validation["long_metric_rows"],
        ],
        [
            "Source vintages",
            "2023, 2024, 2025",
            "Explicit reporting years",
            ", ".join(str(value) for value in validation["reporting_years"]),
        ],
        [
            "Important year caveat",
            "The selected 2025 Annual-WMP filing explicitly reports risk "
            "year 2026. It is identified as source_vintage_year=2025 and "
            "reporting_year=2026.",
            "",
            "",
        ],
        [
            "Legacy reporting cadence",
            "Versions 3.1 and 3.2 require Table 15 to be updated annually "
            "with Q4 data, so Q4 is selected for 2023 and 2024.",
            "",
            "",
        ],
        [
            "2025 revision selected",
            "R1",
            "R0 retained for audit",
            True,
        ],
        [
            "R0-R1 changes",
            (
                f"{validation['r0_r1_common_segment_ids']} common IDs; "
                f"{len(validation['r0_r1_removed_segment_ids'])} removed; "
                f"{len(validation['r0_r1_added_segment_ids'])} added"
            ),
            "R1 formula mismatches",
            validation["formula_mismatches_selected_data"],
        ],
        [
            "Legacy line-class caveat",
            "SDG&E reports Primary Overhead Conductor where the v3 schema "
            "requires Distribution or Transmission. The converter infers "
            "Distribution for analysis and retains the raw value.",
            "",
            "",
        ],
        [
            "Projection handling",
            "Table 15 has no projection columns; no projection values are "
            "included.",
            "",
            "",
        ],
        [
            "2024 revision caveat",
            validation["2024_revision_note"],
            "Local 2024 copies identical",
            validation["2024_local_table15_stable"],
        ],
        ["", "", "", ""],
        ["Official source", "Purpose", "URL", "Verified finding"],
        [
            "Data Guidelines v3.1",
            "2023 Table 15 requirements",
            GUIDELINES[2023][1],
            "Annual Q4 update; report segments contributing over 1% or in "
            "the top 5% of risk.",
        ],
        [
            "Data Guidelines v3.2",
            "2024 Table 15 requirements",
            GUIDELINES[2024][1],
            "Same legacy annual-Q4 cadence and 22-column structure.",
        ],
        [
            "Data Guidelines v4.01",
            "Annual-WMP Table 15 schema",
            GUIDELINES[2025][1],
            "Table 15 moves to Annual-WMP and adds metric numbers, line type, "
            "outage/PEDS components, comments, blank meaning, utility, year.",
        ],
        [
            "v4 Template Changelog",
            "Template corrections",
            V4_CHANGELOG_URL,
            "Removed duplicate WILDFIRE CONSEQUENCE and corrected Table 15 "
            "metric-number prefix from 115... to 215....",
        ],
        [
            "SDG&E R1 cover letter",
            "R0-to-R1 revision",
            R1_COVER_URL,
            "R1 corrected circuit-segment lengths and risk scores in Annual-"
            "WMP Table 15.",
        ],
        [
            "SDG&E 2026-2028 WMP page",
            "R0 and R1 workbooks",
            SDGE_WMP_PAGE_URL,
            "R0 and R1 Annual-WMP tabular files are publicly listed; no "
            "Annual-WMP tabular R2 file is listed.",
        ],
    ]

    readme.get_range_by_indexes(0, 0, len(readme_rows), 4).values = [
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
        "row_height": 30,
    }
    for row_number, row in enumerate(readme_rows, start=1):
        if row[0] == "Official source":
            readme.get_range(f"A{row_number}:D{row_number}").format = {
                "fill": "#1D4ED8",
                "font": {"bold": True, "color": "#FFFFFF"},
                "horizontal_alignment": "center",
                "vertical_alignment": "center",
                "wrap_text": True,
            }
    readme.get_range(f"A1:D{len(readme_rows)}").format.wrap_text = True
    for column, width in zip(("A", "B", "C", "D"), (34, 70, 58, 60)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(source_summary, SOURCE_SUMMARY_HEADERS, source_summary_rows)
    format_sheet(
        source_summary,
        SOURCE_SUMMARY_HEADERS,
        len(source_summary_rows),
        freeze_columns=5,
    )

    write_rows(unified, UNIFIED_HEADERS, unified_rows)
    format_sheet(
        unified,
        UNIFIED_HEADERS,
        len(unified_rows),
        freeze_columns=8,
    )

    for metric in RISK_METRICS + ["segment_length_mi"]:
        letter = column_letter(UNIFIED_HEADERS.index(metric))
        unified.get_range(
            f"{letter}2:{letter}{len(unified_rows) + 1}"
        ).format.number_format = "0.###############"

    alignment_col = column_letter(
        UNIFIED_HEADERS.index("year_alignment_status")
    )
    unified.get_range(
        f"{alignment_col}2:{alignment_col}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${alignment_col}2="2025_submission_vintage_reports_2026_risk_year"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    line_status_col = column_letter(
        UNIFIED_HEADERS.index("line_type_status")
    )
    unified.get_range(
        f"{line_status_col}2:{line_status_col}{len(unified_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${line_status_col}2="inferred_distribution_from_nonconforming_legacy_line_class"',
        {"fill": "#DBEAFE", "font": {"color": "#1E3A8A"}},
    )

    write_rows(long_sheet, LONG_HEADERS, long_rows)
    format_sheet(
        long_sheet,
        LONG_HEADERS,
        len(long_rows),
        freeze_columns=8,
    )
    value_col = column_letter(LONG_HEADERS.index("value"))
    long_sheet.get_range(
        f"{value_col}2:{value_col}{len(long_rows) + 1}"
    ).format.number_format = "0.###############"

    write_rows(crosswalk, CROSSWALK_HEADERS, crosswalk_rows)
    format_sheet(
        crosswalk,
        CROSSWALK_HEADERS,
        len(crosswalk_rows),
        freeze_columns=1,
    )

    write_rows(
        revision_summary,
        REVISION_SUMMARY_HEADERS,
        revision_summary_rows,
    )
    format_sheet(
        revision_summary,
        REVISION_SUMMARY_HEADERS,
        len(revision_summary_rows),
        freeze_columns=1,
    )

    write_rows(revision_detail, REVISION_HEADERS, revision_rows)
    format_sheet(
        revision_detail,
        REVISION_HEADERS,
        len(revision_rows),
        freeze_columns=2,
    )
    revision_status_col = column_letter(
        REVISION_HEADERS.index("revision_status")
    )
    revision_detail.get_range(
        f"A2:{column_letter(len(REVISION_HEADERS)-1)}"
        f"{len(revision_rows)+1}"
    ).conditional_formats.add_custom(
        f'=${revision_status_col}2="added_in_r1"',
        {"fill": "#DCFCE7", "font": {"color": "#166534"}},
    )
    revision_detail.get_range(
        f"A2:{column_letter(len(REVISION_HEADERS)-1)}"
        f"{len(revision_rows)+1}"
    ).conditional_formats.add_custom(
        f'=${revision_status_col}2="removed_in_r1"',
        {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
    )

    write_rows(schema, SCHEMA_HEADERS, schema_rows)
    format_sheet(
        schema,
        SCHEMA_HEADERS,
        len(schema_rows),
        freeze_columns=1,
    )

    write_rows(reconciliation, RECON_HEADERS, recon_rows)
    format_sheet(
        reconciliation,
        RECON_HEADERS,
        len(recon_rows),
        freeze_columns=4,
    )
    recon_status_col = column_letter(RECON_HEADERS.index("status"))
    reconciliation.get_range(
        f"A2:J{len(recon_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${recon_status_col}2="mismatch"',
        {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
    )

    write_rows(issues, ISSUE_HEADERS, issue_rows)
    format_sheet(
        issues,
        ISSUE_HEADERS,
        len(issue_rows),
        freeze_columns=4,
    )
    if issue_rows:
        severity_col = column_letter(ISSUE_HEADERS.index("severity"))
        issues.get_range(
            f"A2:I{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_col}2="error"',
            {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
        )
        issues.get_range(
            f"A2:I{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_col}2="warning"',
            {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
        )

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 15 top-risk circuit/segment/span data for "
            "the 2023, 2024, and 2025 source vintages. Legacy Q4 data are "
            "crosswalked to the v4.01 Annual-WMP schema. R0 and R1 are "
            "compared, and R1 is selected for the 2025 filing vintage."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing 2023 and 2024 source workbooks.",
    )
    parser.add_argument(
        "--annual-wmp-r1",
        default=(
            "/mnt/data/SDGE_2026-WMP_R1_Tabular Wildfire Mitigation "
            "Annual-WMP Data.xlsx"
        ),
        help="Path to the selected R1 Annual-WMP workbook.",
    )
    parser.add_argument(
        "--annual-wmp-r0",
        default=(
            "/mnt/data/SDGE_2026-WMP_R0_Tabular_Wildfire_Mitigation_"
            "Annual-WMP_Data.xlsx"
        ),
        help="Path to the R0 Annual-WMP workbook used for revision audit.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table15_output",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    annual_wmp_r1 = Path(args.annual_wmp_r1)
    annual_wmp_r0 = Path(args.annual_wmp_r0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(
        input_dir,
        annual_wmp_r1,
        annual_wmp_r0,
    )
    issues: list[list[Any]] = []

    records_2023 = parse_legacy(
        read_xlsx_sheet(sources["2023"]["path"], "Table 15"),
        sources["2023"],
        issues,
    )
    records_2024 = parse_legacy(
        read_xlsx_sheet(sources["2024"]["path"], "Table 15"),
        sources["2024"],
        issues,
    )
    records_r0 = parse_v4(
        read_xlsx_sheet(sources["2025_r0"]["path"], "Table 15"),
        sources["2025_r0"],
        expected_metric_prefix=1150000000,
        metric_status="v4_r0_superseded_incorrect_115_prefix",
        issues=None,
    )
    records_2025 = parse_v4(
        read_xlsx_sheet(sources["2025"]["path"], "Table 15"),
        sources["2025"],
        expected_metric_prefix=2150000000,
        metric_status="v4_r1_corrected_215_prefix",
        issues=issues,
    )

    sources["2025"]["reporting_year"] = records_2025[0]["reporting_year"]
    sources["2025_r0"]["reporting_year"] = records_r0[0]["reporting_year"]

    selected_records = (
        records_2023 + records_2024 + records_2025
    )
    selected_records = sorted(
        selected_records,
        key=lambda record: (
            record["source_vintage_year"],
            record["overall_risk_rank"],
            record["segment_id"],
        ),
    )

    validate_records(selected_records, issues)

    local_stability, local_2024_files = verify_2024_local_stability(
        input_dir
    )
    if not local_stability:
        issues.append(
            [
                "2024_local_quarterly_table15_instability",
                "warning",
                2024,
                2024,
                sources["2024"]["path"].name,
                "Table 15",
                None,
                len(local_2024_files),
                "The available 2024 quarterly copies do not contain identical "
                "Table 15 values.",
            ]
        )

    revision_rows, revision_summary_rows, revision_stats = (
        build_revision_comparison(records_r0, records_2025)
    )

    r0_recon_rows, r0_formula_mismatches = build_reconciliation(
        records_r0,
        tolerance=1e-8,
    )
    selected_recon_rows, selected_formula_mismatches = build_reconciliation(
        selected_records,
        tolerance=1e-5,
    )

    revision_summary_rows.extend(
        [
            [
                "R0 aggregate-risk formula mismatches",
                r0_formula_mismatches,
                "corrected_in_r1",
                "R0 fails the v4 aggregate checks because Overall Utility "
                "Risk omits Outage Program Risk and most Outage Program Risk "
                "values omit PEDS Risk. R1 reconciles.",
            ],
            [
                "R1 aggregate-risk formula mismatches",
                count_formula_mismatches(records_2025, 1e-8),
                "validated",
                "All five tested v4 risk relationships reconcile in R1.",
            ],
        ]
    )

    issues.append(
        [
            "annual_wmp_r0_superseded",
            "info",
            2025,
            2026,
            sources["2025_r0"]["path"].name,
            "Table 15 revision",
            "R0",
            len(records_r0),
            "R0 is retained only for revision audit. R1 corrected lengths "
            "and risk scores and is used in the unified output.",
        ]
    )

    grouped_records = {
        "2023": records_2023,
        "2024": records_2024,
        "2025": records_2025,
    }
    mismatch_counts = {
        "2023": count_formula_mismatches(records_2023, 1e-5),
        "2024": count_formula_mismatches(records_2024, 1e-5),
        "2025": count_formula_mismatches(records_2025, 1e-8),
    }

    unified_rows = [
        record_to_unified_row(record)
        for record in selected_records
    ]
    long_rows = build_long_rows(selected_records)
    crosswalk_rows = build_segment_crosswalk(selected_records)
    schema_rows = build_schema_rows()
    source_summary_rows = build_source_summary(
        grouped_records,
        mismatch_counts,
    )

    workbook_path = (
        output_dir / "sdge_table15_2023_2025_unified.xlsx"
    )
    unified_csv = (
        output_dir / "sdge_table15_2023_2025_unified_wide.csv"
    )
    long_csv = (
        output_dir / "sdge_table15_2023_2025_metrics_long.csv"
    )
    crosswalk_csv = (
        output_dir / "sdge_table15_segment_crosswalk.csv"
    )
    revision_csv = (
        output_dir / "sdge_table15_r0_r1_revision_comparison.csv"
    )
    revision_summary_csv = (
        output_dir / "sdge_table15_r0_r1_revision_summary.csv"
    )
    schema_csv = (
        output_dir / "sdge_table15_schema_crosswalk.csv"
    )
    recon_csv = (
        output_dir / "sdge_table15_risk_reconciliation.csv"
    )
    r0_recon_csv = (
        output_dir / "sdge_table15_r0_risk_reconciliation_audit.csv"
    )
    issues_csv = (
        output_dir / "sdge_table15_validation_issues.csv"
    )
    source_summary_csv = (
        output_dir / "sdge_table15_source_summary.csv"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(unified_csv, UNIFIED_HEADERS, unified_rows)
    write_csv(long_csv, LONG_HEADERS, long_rows)
    write_csv(crosswalk_csv, CROSSWALK_HEADERS, crosswalk_rows)
    write_csv(revision_csv, REVISION_HEADERS, revision_rows)
    write_csv(
        revision_summary_csv,
        REVISION_SUMMARY_HEADERS,
        revision_summary_rows,
    )
    write_csv(schema_csv, SCHEMA_HEADERS, schema_rows)
    write_csv(recon_csv, RECON_HEADERS, selected_recon_rows)
    write_csv(r0_recon_csv, RECON_HEADERS, r0_recon_rows)
    write_csv(issues_csv, ISSUE_HEADERS, issues)
    write_csv(
        source_summary_csv,
        SOURCE_SUMMARY_HEADERS,
        source_summary_rows,
    )

    validation = {
        "unified_rows": len(unified_rows),
        "long_metric_rows": len(long_rows),
        "distinct_segment_ids": len(
            {record["segment_id"] for record in selected_records}
        ),
        "source_vintages": [2023, 2024, 2025],
        "reporting_years": sorted(
            {record["reporting_year"] for record in selected_records}
        ),
        "rows_by_source_vintage": {
            "2023": len(records_2023),
            "2024": len(records_2024),
            "2025": len(records_2025),
        },
        "2025_source_vintage_reporting_year": records_2025[0][
            "reporting_year"
        ],
        "formula_mismatches_selected_data": selected_formula_mismatches,
        "formula_mismatches_by_vintage": mismatch_counts,
        "r0_formula_mismatches": r0_formula_mismatches,
        "r0_r1_common_segment_ids": revision_stats[
            "common_segment_ids"
        ],
        "r0_r1_removed_segment_ids": revision_stats[
            "removed_segment_ids"
        ],
        "r0_r1_added_segment_ids": revision_stats[
            "added_segment_ids"
        ],
        "r0_r1_field_change_counts": revision_stats[
            "field_change_counts"
        ],
        "2024_local_table15_stable": local_stability,
        "2024_local_files_compared": local_2024_files,
        "2024_revision_note": (
            "The highest local Q4 workbook is R1. An official Q4 R2 is "
            "docketed but was not available in the input directory. All four "
            "available 2024 quarterly Table 15 copies are identical, and the "
            "documented R2 filing responded to a separate NOV."
        ),
        "validation_issue_rows": len(issues),
        "projection_columns_found": 0,
        "selected_sources": {
            key: {
                "name": source["path"].name,
                "revision": source["revision"],
                "source_url": source["source_url"],
            }
            for key, source in sources.items()
        },
    }
    validation_path.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    build_workbook(
        workbook_path,
        unified_rows,
        long_rows,
        crosswalk_rows,
        revision_rows,
        revision_summary_rows,
        schema_rows,
        selected_recon_rows,
        issues,
        source_summary_rows,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {unified_csv}")
    print(f"Created: {long_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {revision_csv}")
    print(f"Created: {revision_summary_csv}")
    print(f"Created: {schema_csv}")
    print(f"Created: {recon_csv}")
    print(f"Created: {r0_recon_csv}")
    print(f"Created: {issues_csv}")
    print(f"Created: {source_summary_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(unified_rows)}")
    print(f"Distinct segment IDs: {validation['distinct_segment_ids']}")
    print(
        "Selected-data formula mismatches:",
        selected_formula_mismatches,
    )
    print(
        "R0 formula mismatches corrected by R1:",
        r0_formula_mismatches,
    )


if __name__ == "__main__":
    main()
