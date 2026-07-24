#!/usr/bin/env python3
"""Convert SDG&E Table 1 for 2023-2025 into one verified unified dataset.

Rules
-----
1. Use the highest-revision Q4 workbook for each year because each Q4 Table 1
   contains the cumulative Q1, Q1-2, Q1-3, and Q1-4 actual-progress fields.
2. Drop only the four projected quantitative-progress columns.
3. Keep every other named Table 1 field.
4. Preserve ambiguous legacy/new WMP hierarchy fields separately.
5. Populate 2025 annual quantitative and qualitative targets from Table 12.
   The official revised 2025 target workbook can be supplied directly. If it is
   unavailable, reconstruct the approved target set from the 2024 Q4 Table 12
   baseline, the six approved 2025 amendments, and the two new 2025 activities.

Dependencies:
    pip install artifact-tool

Raw-workbook run:
    python unify_sdge_table1_with_2025_targets.py \
        --input-dir /path/to/workbooks \
        --output-dir /path/to/output \
        [--annual-wmp-2025 /path/to/SDGE_2025_Petition_to_Amend_Tables.xlsx]

Fast target-enrichment run after Table 1 was already normalized:
    python unify_sdge_table1_with_2025_targets.py \
        --input-dir /path/to/workbooks \
        --base-unified-csv /path/to/sdge_table1_2023_2025_verified.csv \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from artifact_tool import Blob, SpreadsheetFile, Workbook


YEARS = (2023, 2024, 2025)
SHEET_NAME = "Table 1"

FILE_PERIOD_RE = re.compile(r"(?P<year>20\d{2})_Q(?P<quarter>[1-4])", re.IGNORECASE)
REVISION_RE = re.compile(r"(?:^|_)(?:R|REV)(?P<revision>\d+)(?:\.|_|$)", re.IGNORECASE)


TABLE12_SHEET_NAME = "Table 12"
OFFICIAL_REVISED_TARGET_WORKBOOK_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2024-QDR&fileid=59052&shareable=true"
)
OFFICIAL_2025_AMENDMENT_DECISION_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2023-2025-WMPs&fileid=58911&shareable=true"
)
OFFICIAL_2025_WMP_UPDATE_URL = (
    "https://www.sdge.com/sites/default/files/regulatory/"
    "2024-07-05_SDGE_2025_WMP-Update_R2_redline.pdf"
)
OFFICIAL_2025_Q4_NOTIFICATION_URL = (
    "https://www.sdge.com/sites/default/files/regulatory/"
    "SDGE_2025_Q4_Quarterly%20Notification%20Letter_Attachment%20A_R1.pdf"
)
OFFICIAL_2025_AIR_URL = (
    "https://efiling.energysafety.ca.gov/EFiling/GetFile.aspx?"
    "filePath=D%3A%5CFileThat%5CFileServer%5CPublic%5CPublicDocuments%5C"
    "2025+EC+AIR%2FTN17846_20260401T153041_SDGE_2025_AIRpdf.pdf"
)

# Six 2025 target amendments approved by Energy Safety.
APPROVED_2025_TARGET_AMENDMENTS = {
    "WMP.473": 28,
    "WMP.455": 50,
    "WMP.1189": 200,
    "WMP.543": 2,
    "WMP.550": 90,
    "WMP.549": 5,
}

# WMP.970 was retired/split. These new activities do not appear in the old
# 2024 Q4 Table 12 baseline.
NEW_2025_QUANT_TARGETS = {
    "WMP.1430": (216, "Weather Stations"),
    "WMP.1431": (192, "Sensors"),
}

# The two qualitative activities in the approved 2023-2025 WMP.
NEW_2025_QUAL_TARGETS = {
    "WMP.466": (
        "Continue to provide portable backup power solutions to vulnerable, "
        "electricity-dependent customers."
    ),
    "WMP.467": (
        "Continue to provide rebates on portable backup power solutions to "
        "customers who experience PSPS."
    ),
}

# These are the only source fields intentionally excluded.
LEGACY_PROJECTION_COLUMNS = {
    "ProjectedQuantProgressQ1",
    "ProjectedQuantProgressQ1-2",
    "ProjectedQuantProgressQ1-3",
    "ProjectedQuantProgressQ1-4",
}

# Field mappings verified against the guideline definitions.
#
# Important: the v4 redesign introduced a WMP CATEGORY -> WMP INITIATIVE ->
# WMP ACTIVITY hierarchy. Some old fields do not have a clean one-to-one mapping,
# so they are retained as legacy-specific columns instead of being forced together.
LEGACY_MAP = {
    "UtilityID": "utility_id",
    "SubmissionDate": "submission_date",
    "InitiativeClassification": "activity_classification",
    "ProjectStartDate": "project_start_date",
    "ProjectEndDate": "project_end_date",

    # Official v4 template changelog identifies this as the predecessor of
    # WMP ACTIVITY NAME.
    "UtilityInitiativeName": "wmp_activity_name",
    "InitiativeDescription": "activity_description",
    "InitiativeObjective": "activity_objective",
    "WMPInitiativeCategory": "wmp_category",

    # Legacy-only category section; v4.01 has no direct equivalent field.
    "WMPInitiativeCategory#": "legacy_wmp_category_section",

    # The old Energy Safety activity classification is not forced into the new
    # WMP INITIATIVE field because the hierarchy and controlled values changed.
    "WMPInitiativeActivity": "legacy_energy_safety_activity",
    "ActivityNameifOther": "legacy_activity_name_if_other",

    # This is the section for the old WMPInitiativeActivity and corresponds to
    # the v4.01 WMP SECTION concept.
    "WMPInitiativeActivity#": "wmp_section",
    "UtilityInitiativeTrackingID": "activity_tracking_id",
    "WMPInitiativeCode": "wmp_activity_code",
    "WMPPageNumber": "wmp_page_number",
    "RiskTargetReduction": "risk_target_reduction",
    "MidYearTarget (Yes/No)": "midyear_target",
    "QuantTargetUnits": "quant_target_units",
    "AnnualQuantTarget": "annual_quant_target",

    "QuantActualProgressQ1": "quant_actual_progress_q1",
    "QuantActualProgressQ1-2": "quant_actual_progress_q1_2",
    "QuantActualProgressQ1-3": "quant_actual_progress_q1_3",
    "QuantActualProgressQ1-4": "quant_actual_progress_q1_4",

    "AnnualQualTarget": "annual_qual_target",
    "QualActualProgressQ1": "qual_actual_progress_q1",
    "QualActualProgressQ1-2": "qual_actual_progress_q1_2",
    "QualActualProgressQ1-3": "qual_actual_progress_q1_3",
    "QualActualProgressQ1-4": "qual_actual_progress_q1_4",

    "Status": "status",
    "CorrectiveActionsIfDelayed": "corrective_actions_if_delayed",

    # Legacy-only oversight/contact columns. Retained because only projections
    # may be dropped.
    "REFERENCE: Compliance Branch Requirements -->": "compliance_reference",
    "Audit": "audit",
    "Audit File Documentation Requested": "audit_file_documentation_requested",
    "FolderLink": "folder_link",
    "PersonInChargeName": "person_in_charge_name",
    "PersonInChargeEmail": "person_in_charge_email",
}

NEW_2025_MAP = {
    "METRIC NUMBER": "metric_number",
    "UTILITY ID": "utility_id",
    "SUBMISSION DATE": "submission_date",
    "ACTIVITY CLASSIFICATION": "activity_classification",
    "PROJECT START DATE": "project_start_date",
    "PROJECT END DATE": "project_end_date",
    "WMP ACTIVITY NAME": "wmp_activity_name",
    "ACTIVITY DESCRIPTION": "activity_description",
    "ACTIVITY OBJECTIVE": "activity_objective",
    "WMP CATEGORY": "wmp_category",
    "WMP INITIATIVE": "wmp_initiative",
    "WMP INITIATIVE NAME IF OTHER": "wmp_initiative_name_if_other",
    "WMP SECTION": "wmp_section",
    "UTILITY MITIGATION ACTIVITY TRACKING ID": "activity_tracking_id",
    "WMP ACTIVITY CODE": "wmp_activity_code",
    "WMP PAGE NUMBER": "wmp_page_number",
    "RISK TARGET REDUCTION": "risk_target_reduction",
    "MIDYEAR TARGET (YES / NO)": "midyear_target",
    "QUANT TARGET UNITS": "quant_target_units",
    "COMMENTS": "comments",
    "REPORTING YEAR": "reporting_year",
    "REPORTING QUARTER": "reporting_quarter",

    "QUANTITATIVE ACTUAL PROGRESS Q1": "quant_actual_progress_q1",
    "QUANTITATIVE ACTUAL PROGRESS Q1-2": "quant_actual_progress_q1_2",
    "QUANTITATIVE ACTUAL PROGRESS Q1-3": "quant_actual_progress_q1_3",
    "QUANTITATIVE ACTUAL PROGRESS Q1-4": "quant_actual_progress_q1_4",

    "QUALITATIVE ACTUAL PROGRESS Q1": "qual_actual_progress_q1",
    "QUALITATIVE ACTUAL PROGRESS Q1-2": "qual_actual_progress_q1_2",
    "QUALITATIVE ACTUAL PROGRESS Q1-3": "qual_actual_progress_q1_3",
    "QUALITATIVE ACTUAL PROGRESS Q1-4": "qual_actual_progress_q1_4",

    "STATUS": "status",
    "CORRECTIVE ACTIONS IF DELAYED": "corrective_actions_if_delayed",
}

OUTPUT_COLUMNS = [
    # Generated lineage fields.
    "record_id",
    "reporting_year",
    "reporting_quarter",
    "source_schema",
    "source_file",
    "source_sheet",
    "source_row",
    "source_revision",

    # New v4.01 identifier.
    "metric_number",

    # Safely comparable fields.
    "utility_id",
    "submission_date",
    "activity_classification",
    "project_start_date",
    "project_end_date",
    "wmp_activity_name",
    "activity_description",
    "activity_objective",
    "wmp_category",

    # Hierarchy fields kept separate where v3 -> v4 equivalence is not clean.
    "legacy_wmp_category_section",
    "legacy_energy_safety_activity",
    "legacy_activity_name_if_other",
    "wmp_initiative",
    "wmp_initiative_name_if_other",

    "wmp_section",
    "activity_tracking_id",
    "wmp_activity_code",
    "wmp_page_number",
    "risk_target_reduction",
    "midyear_target",
    "quant_target_units",

    # Targets are in the v3 quarterly workbook, but moved to the v4 Annual-WMP
    # workbook. They will therefore be blank for 2025 when only Q4 is supplied.
    "annual_quant_target",
    "annual_qual_target",

    # Target lineage. Table 1 lineage remains in source_* above.
    "annual_target_source_type",
    "annual_target_source_file",
    "annual_target_source_sheet",
    "annual_target_source_row",
    "annual_target_source_url",
    "annual_target_join_status",

    # All cumulative actual-progress snapshots retained.
    "quant_actual_progress_q1",
    "quant_actual_progress_q1_2",
    "quant_actual_progress_q1_3",
    "quant_actual_progress_q1_4",
    "qual_actual_progress_q1",
    "qual_actual_progress_q1_2",
    "qual_actual_progress_q1_3",
    "qual_actual_progress_q1_4",

    "status",
    "corrective_actions_if_delayed",
    "comments",

    # Legacy-only fields retained.
    "compliance_reference",
    "audit",
    "audit_file_documentation_requested",
    "folder_link",
    "person_in_charge_name",
    "person_in_charge_email",
]

DATE_FIELDS = {"submission_date", "project_start_date", "project_end_date"}
TEXT_ID_FIELDS = {
    "activity_tracking_id",
    "wmp_activity_code",
    "wmp_section",
    "legacy_wmp_category_section",
}


def clean_header(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in {"", "na", "n/a", "none", "null"}:
            return None
        return stripped
    return value


def excel_date_to_iso(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)) and 1 <= value <= 100000:
        try:
            return (datetime(1899, 12, 30) + timedelta(days=float(value))).date().isoformat()
        except (OverflowError, ValueError):
            return value
    if isinstance(value, str):
        # Preserve textual dates rather than guessing locale-specific values.
        return value.strip()
    return value


def parse_period(path: Path) -> tuple[int, int]:
    match = FILE_PERIOD_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse reporting period from filename: {path.name}")
    return int(match.group("year")), int(match.group("quarter"))


def parse_revision(path: Path) -> int:
    match = REVISION_RE.search(path.stem + "_")
    return int(match.group("revision")) if match else 0


def discover_q4_files(input_dir: Path) -> list[Path]:
    candidates: dict[int, list[Path]] = {year: [] for year in YEARS}
    for path in input_dir.glob("*.xlsx"):
        match = FILE_PERIOD_RE.search(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        quarter = int(match.group("quarter"))
        if year in candidates and quarter == 4:
            candidates[year].append(path)

    selected: list[Path] = []
    for year in YEARS:
        if not candidates[year]:
            raise FileNotFoundError(f"No Q4 workbook found for {year} in {input_dir}")
        selected.append(max(candidates[year], key=lambda p: (parse_revision(p), p.name)))
    return selected


def get_table_values(path: Path) -> list[list[Any]]:
    workbook = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet = workbook.worksheets.get_item(SHEET_NAME)
    region = sheet.get_range("A1").get_current_region()
    values = region.values
    if not values or len(values) < 2:
        raise ValueError(f"No Table 1 data found in {path.name}")
    return values


def detect_schema(headers: Iterable[str]) -> str:
    header_set = set(headers)
    if "METRIC NUMBER" in header_set and "REPORTING YEAR" in header_set:
        return "v4.01_2025"
    if "UtilityInitiativeTrackingID" in header_set:
        return "v3.x_2023_2024"
    raise ValueError("Unrecognized Table 1 schema")


def validate_source_mapping(headers: list[str], schema: str) -> None:
    named = {h for h in headers if h}
    if schema == "v3.x_2023_2024":
        expected_mapped = set(LEGACY_MAP)
        allowed_unmapped = LEGACY_PROJECTION_COLUMNS
        unknown = named - expected_mapped - allowed_unmapped
        missing_expected = expected_mapped - named
        if unknown:
            raise ValueError(f"Unmapped legacy columns (not projections): {sorted(unknown)}")
        if missing_expected:
            raise ValueError(f"Expected legacy columns missing from source: {sorted(missing_expected)}")
    else:
        unknown = named - set(NEW_2025_MAP)
        missing_expected = set(NEW_2025_MAP) - named
        if unknown:
            raise ValueError(f"Unmapped 2025 columns: {sorted(unknown)}")
        if missing_expected:
            raise ValueError(f"Expected 2025 columns missing from source: {sorted(missing_expected)}")


def normalize_text_id(value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def convert_file(path: Path) -> list[dict[str, Any]]:
    file_year, file_quarter = parse_period(path)
    if file_quarter != 4:
        raise ValueError(f"Expected a Q4 workbook, got {path.name}")

    values = get_table_values(path)
    headers = [clean_header(v) for v in values[0]]

    # Remove trailing unnamed columns while retaining the original indexes for named fields.
    named_indexes = [i for i, h in enumerate(headers) if h]
    if not named_indexes:
        raise ValueError(f"Table 1 has no named columns in {path.name}")
    last_named_index = max(named_indexes)
    headers = headers[: last_named_index + 1]

    schema = detect_schema(headers)
    validate_source_mapping(headers, schema)
    mapping = LEGACY_MAP if schema.startswith("v3") else NEW_2025_MAP

    records: list[dict[str, Any]] = []
    for excel_row, source_values in enumerate(values[1:], start=2):
        source_values = list(source_values[: len(headers)])
        if len(source_values) < len(headers):
            source_values.extend([None] * (len(headers) - len(source_values)))
        source = dict(zip(headers, source_values))

        # Ignore only truly empty/footer rows.
        legacy_key = source.get("UtilityInitiativeTrackingID")
        new_key = source.get("UTILITY MITIGATION ACTIVITY TRACKING ID")
        legacy_name = source.get("UtilityInitiativeName")
        new_name = source.get("WMP ACTIVITY NAME")
        if all(clean_value(v) is None for v in (legacy_key, new_key, legacy_name, new_name)):
            continue

        record = {column: None for column in OUTPUT_COLUMNS}
        for source_column, output_column in mapping.items():
            value = clean_value(source.get(source_column))
            if output_column in DATE_FIELDS:
                value = excel_date_to_iso(value)
            elif output_column in TEXT_ID_FIELDS:
                value = normalize_text_id(value)
            record[output_column] = value

        # v3.x has no explicit reporting period fields in Table 1.
        if schema.startswith("v3"):
            record["reporting_year"] = file_year
            record["reporting_quarter"] = file_quarter
        else:
            sheet_year = record["reporting_year"]
            sheet_quarter = record["reporting_quarter"]
            if sheet_year not in (None, file_year) or sheet_quarter not in (None, file_quarter):
                raise ValueError(
                    f"Filename period and in-sheet period disagree in {path.name}, row {excel_row}"
                )
            record["reporting_year"] = file_year
            record["reporting_quarter"] = file_quarter

        record["source_schema"] = schema
        record["source_file"] = path.name
        record["source_sheet"] = SHEET_NAME
        record["source_row"] = excel_row
        record["source_revision"] = parse_revision(path)

        tracking_id = record["activity_tracking_id"] or "NO_TRACKING_ID"
        record["record_id"] = (
            f"{record['utility_id'] or 'UNKNOWN'}_{file_year}_Q4_TABLE1_"
            f"{tracking_id}_ROW{excel_row}"
        )
        records.append(record)

    return records



def normalize_tracking_id(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    match = re.fullmatch(r"WMP\.(\d+)", text)
    if match:
        return f"WMP.{int(match.group(1))}"
    return text


def get_sheet_values(path: Path, sheet_name: str) -> list[list[Any]]:
    workbook = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet = workbook.worksheets.get_item(sheet_name)
    # Table 12 has title/metadata rows and internal blanks, so A1 current-region
    # detection is unreliable. This bounded range covers all target rows.
    values = sheet.get_range("A1:AZ500").values
    while values and all(clean_value(v) is None for v in values[-1]):
        values.pop()
    if values:
        last_col = max(
            (i for row in values for i, value in enumerate(row) if clean_value(value) is not None),
            default=-1,
        )
        values = [list(row[: last_col + 1]) for row in values]
    return values


def find_legacy_table12_target_column(values: list[list[Any]], year: int) -> tuple[int, int]:
    for row_index, row in enumerate(values):
        headers = [clean_header(value) for value in row]
        if "UtilityInitiativeTrackingID" not in headers or "TargetType" not in headers:
            continue
        period_row = values[row_index - 1] if row_index > 0 else []
        for column_index, value in enumerate(row):
            year_value = clean_value(value)
            period_value = clean_header(
                period_row[column_index] if column_index < len(period_row) else None
            )
            try:
                is_year = int(float(year_value)) == year
            except (TypeError, ValueError):
                is_year = False
            if is_year and period_value.casefold() in {
                "end of year", "ytd target - y1 q4", "q4"
            }:
                return row_index, column_index
        raise ValueError(f"Found Table 12 header but no {year} end-of-year target column")
    raise ValueError("Could not find legacy Table 12 header")


def parse_legacy_table12(path: Path, target_year: int = 2025) -> dict[str, dict[str, Any]]:
    values = get_sheet_values(path, TABLE12_SHEET_NAME)
    header_row, target_column = find_legacy_table12_target_column(values, target_year)
    headers = [clean_header(value) for value in values[header_row]]
    index = {header: i for i, header in enumerate(headers) if header}
    required = {"UtilityInitiativeTrackingID", "TargetType", "Units"}
    if not required.issubset(index):
        raise ValueError(f"Table 12 missing columns: {sorted(required - set(index))}")

    result = {}
    for excel_row, row in enumerate(values[header_row + 1:], start=header_row + 2):
        tracking_id = normalize_tracking_id(
            row[index["UtilityInitiativeTrackingID"]]
            if index["UtilityInitiativeTrackingID"] < len(row) else None
        )
        if not tracking_id:
            continue
        result[tracking_id] = {
            "target_type": clean_value(row[index["TargetType"]]),
            "target": clean_value(row[target_column] if target_column < len(row) else None),
            "units": clean_value(row[index["Units"]]),
            "source_file": path.name,
            "source_sheet": TABLE12_SHEET_NAME,
            "source_row": excel_row,
        }
    return result


def parse_v4_annual_wmp_table12(path: Path, target_year: int = 2025) -> dict[str, dict[str, Any]]:
    values = get_sheet_values(path, TABLE12_SHEET_NAME)
    if not values:
        raise ValueError(f"No Table 12 data in {path.name}")
    headers = [clean_header(value) for value in values[0]]
    index = {header: i for i, header in enumerate(headers) if header}
    required = {
        "UTILITY MITIGATION ACTIVITY TRACKING ID", "TARGET TYPE", "REPORTING YEAR"
    }
    if not required.issubset(index):
        raise ValueError("Not a v4 Annual-WMP Table 12")
    target_header = next(
        (name for name in ["YTD TARGET - Y1 Q4", "ANNUAL TARGET", "END OF YEAR TARGET"]
         if name in index),
        None,
    )
    if target_header is None:
        raise ValueError("Annual-WMP Table 12 has no supported annual-target column")

    result = {}
    units_index = index.get("UNIT(S)")
    for excel_row, row in enumerate(values[1:], start=2):
        try:
            report_year = int(float(clean_value(row[index["REPORTING YEAR"]])))
        except (TypeError, ValueError, IndexError):
            continue
        if report_year != target_year:
            continue
        tracking_id = normalize_tracking_id(
            row[index["UTILITY MITIGATION ACTIVITY TRACKING ID"]]
        )
        if not tracking_id:
            continue
        result[tracking_id] = {
            "target_type": clean_value(row[index["TARGET TYPE"]]),
            "target": clean_value(row[index[target_header]]),
            "units": clean_value(row[units_index]) if units_index is not None else None,
            "source_file": path.name,
            "source_sheet": TABLE12_SHEET_NAME,
            "source_row": excel_row,
        }
    if not result:
        raise ValueError(
            f"{path.name} has no Table 12 rows for 2025. A 2026-2028 Annual-WMP "
            "workbook must not be used for 2025 targets."
        )
    return result


def parse_legacy_table1_targets(path: Path) -> dict[str, dict[str, Any]]:
    """Read annual quantitative and qualitative targets from legacy Table 1.

    The uploaded 2025 Petition-to-Amend workbook contains both AnnualQuantTarget
    and AnnualQualTarget in Table 1. Table 12 contains quantitative end-of-year
    values but leaves qualitative target cells blank, so Table 1 is the complete
    authoritative source for this workbook.
    """
    values = get_table_values(path)
    headers = [clean_header(value) for value in values[0]]
    index = {header: i for i, header in enumerate(headers) if header}
    required = {
        "UtilityInitiativeTrackingID",
        "AnnualQuantTarget",
        "AnnualQualTarget",
        "QuantTargetUnits",
    }
    if not required.issubset(index):
        raise ValueError("Not a legacy Table 1 target workbook")

    result: dict[str, dict[str, Any]] = {}
    for excel_row, row in enumerate(values[1:], start=2):
        tracking_id = normalize_tracking_id(
            row[index["UtilityInitiativeTrackingID"]]
            if index["UtilityInitiativeTrackingID"] < len(row) else None
        )
        if not tracking_id:
            continue
        quant_target = clean_value(
            row[index["AnnualQuantTarget"]]
            if index["AnnualQuantTarget"] < len(row) else None
        )
        qual_target = clean_value(
            row[index["AnnualQualTarget"]]
            if index["AnnualQualTarget"] < len(row) else None
        )
        units = clean_value(
            row[index["QuantTargetUnits"]]
            if index["QuantTargetUnits"] < len(row) else None
        )
        if quant_target is None and qual_target is None:
            continue
        result[tracking_id] = {
            "target_type": "Qualitative" if qual_target is not None else "Quantitative",
            "target": qual_target if qual_target is not None else quant_target,
            "units": units,
            "source_file": path.name,
            "source_sheet": SHEET_NAME,
            "source_row": excel_row,
            "source_type": "official_2025_petition_workbook_table1",
            "source_url": OFFICIAL_REVISED_TARGET_WORKBOOK_URL,
        }
    if not result:
        raise ValueError(f"No annual targets found in {path.name} Table 1")
    return result


def add_2025_new_activity_targets(
    targets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add new 2025 activities absent from the Petition-to-Amend workbook.

    WMP.1430 and WMP.1431 replaced/continued older initiatives after the
    petition workbook's activity list. Their 2025 targets are documented in
    SDG&E's 2025 WMP Update and confirmed in the 2025 Annual Implementation
    Report.
    """
    enriched = dict(targets)
    for tracking_id, (target, units) in NEW_2025_QUANT_TARGETS.items():
        if tracking_id in enriched:
            continue
        enriched[tracking_id] = {
            "target_type": "Quantitative",
            "target": target,
            "units": units,
            "source_file": "SDGE 2025 WMP Update / 2025 Annual Implementation Report",
            "source_sheet": None,
            "source_row": None,
            "source_type": "official_2025_supplemental_source",
            "source_url": OFFICIAL_2025_WMP_UPDATE_URL + "; " + OFFICIAL_2025_AIR_URL,
            "supplemental_new_activity": True,
        }
    return enriched


def parse_2025_target_workbook(path: Path) -> dict[str, dict[str, Any]]:
    # The uploaded Petition-to-Amend workbook uses the legacy schema and stores
    # complete quantitative + qualitative annual targets in Table 1. Prefer it
    # over Table 12, whose qualitative target cells are blank.
    try:
        targets = parse_legacy_table1_targets(path)
    except (KeyError, ValueError):
        values = get_sheet_values(path, TABLE12_SHEET_NAME)
        headers = {clean_header(value) for value in values[0]} if values else set()
        if "REPORTING YEAR" in headers and "TARGET TYPE" in headers:
            targets = parse_v4_annual_wmp_table12(path, 2025)
        else:
            targets = parse_legacy_table12(path, 2025)
    return add_2025_new_activity_targets(targets)


def discover_official_2025_target_workbook(input_dir: Path) -> Path | None:
    candidates = [
        path for path in input_dir.glob("*.xlsx")
        if "2025" in path.name
        and "petition" in path.name.casefold()
        and "table" in path.name.casefold()
        and "12" in path.name
    ]
    return sorted(candidates)[-1] if candidates else None


def build_fallback_2025_targets(legacy_table12_path: Path) -> dict[str, dict[str, Any]]:
    targets = parse_legacy_table12(legacy_table12_path, 2025)
    for tracking_id, amended_value in APPROVED_2025_TARGET_AMENDMENTS.items():
        if tracking_id not in targets:
            raise ValueError(f"Amended activity absent from baseline Table 12: {tracking_id}")
        targets[tracking_id]["target"] = amended_value
        targets[tracking_id]["amended"] = True

    targets.pop("WMP.970", None)
    for tracking_id, (target, units) in NEW_2025_QUANT_TARGETS.items():
        targets[tracking_id] = {
            "target_type": "Quantitative",
            "target": target,
            "units": units,
            "source_file": "2025 WMP Update / Q4 notification",
            "source_sheet": None,
            "source_row": None,
            "new_2025_activity": True,
        }
    for tracking_id, target in NEW_2025_QUAL_TARGETS.items():
        existing = targets.get(tracking_id, {})
        targets[tracking_id] = {
            **existing,
            "target_type": "Qualitative",
            "target": target,
            "qualitative_objective": True,
        }
    return targets


def enrich_2025_targets(
    records: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    direct_target_workbook: Path | None,
) -> None:
    for record in records:
        year = int(record["reporting_year"])
        if year in (2023, 2024):
            record["annual_target_source_type"] = "quarterly_table1_native"
            record["annual_target_source_file"] = record.get("source_file")
            record["annual_target_source_sheet"] = record.get("source_sheet")
            record["annual_target_source_row"] = record.get("source_row")
            record["annual_target_join_status"] = "native_legacy_target"
            continue
        if year != 2025:
            continue

        tracking_id = normalize_tracking_id(record.get("activity_tracking_id"))
        target_record = targets.get(tracking_id or "")
        if target_record is None:
            record["annual_target_join_status"] = "no_matching_2025_target"
            continue

        target_type = str(target_record.get("target_type") or "").casefold()
        target = clean_value(target_record.get("target"))
        if target_type.startswith("qual"):
            record["annual_qual_target"] = target
            record["annual_quant_target"] = None
        else:
            record["annual_quant_target"] = target
            record["annual_qual_target"] = None
            if not record.get("quant_target_units") and target_record.get("units"):
                record["quant_target_units"] = target_record["units"]

        if direct_target_workbook is not None:
            record["annual_target_source_type"] = (
                target_record.get("source_type") or "official_2025_target_workbook"
            )
            record["annual_target_source_file"] = (
                target_record.get("source_file") or direct_target_workbook.name
            )
            record["annual_target_source_sheet"] = target_record.get("source_sheet")
            record["annual_target_source_row"] = target_record.get("source_row")
            record["annual_target_source_url"] = (
                target_record.get("source_url") or OFFICIAL_REVISED_TARGET_WORKBOOK_URL
            )
            record["annual_target_join_status"] = (
                "matched_supplemental_new_2025_target"
                if target_record.get("supplemental_new_activity")
                else "matched_direct_official_workbook"
            )
        else:
            record["annual_target_source_type"] = "reconstructed_approved_2025_target_set"
            record["annual_target_source_file"] = target_record.get("source_file")
            record["annual_target_source_sheet"] = target_record.get("source_sheet")
            record["annual_target_source_row"] = target_record.get("source_row")
            if target_record.get("new_2025_activity"):
                record["annual_target_source_url"] = (
                    OFFICIAL_2025_WMP_UPDATE_URL + "; " + OFFICIAL_2025_Q4_NOTIFICATION_URL
                )
                record["annual_target_join_status"] = "matched_new_2025_activity_target"
            elif target_record.get("qualitative_objective"):
                record["annual_target_source_url"] = OFFICIAL_2025_WMP_UPDATE_URL
                record["annual_target_join_status"] = "matched_2025_qualitative_objective"
            elif target_record.get("amended"):
                record["annual_target_source_url"] = (
                    OFFICIAL_REVISED_TARGET_WORKBOOK_URL + "; "
                    + OFFICIAL_2025_AMENDMENT_DECISION_URL
                )
                record["annual_target_join_status"] = "matched_approved_amended_target"
            else:
                record["annual_target_source_url"] = OFFICIAL_REVISED_TARGET_WORKBOOK_URL
                record["annual_target_join_status"] = "matched_unchanged_approved_target"


CSV_INTEGER_FIELDS = {"reporting_year", "reporting_quarter", "source_row", "source_revision"}
CSV_NUMERIC_FIELDS = {
    "annual_quant_target",
    "quant_actual_progress_q1", "quant_actual_progress_q1_2",
    "quant_actual_progress_q1_3", "quant_actual_progress_q1_4",
}


def parse_csv_scalar(column: str, value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return None
    if column in CSV_INTEGER_FIELDS:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if column in CSV_NUMERIC_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def load_base_unified_csv(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for source in reader:
            record = {column: None for column in OUTPUT_COLUMNS}
            for column in reader.fieldnames:
                if column in record:
                    record[column] = parse_csv_scalar(column, source.get(column))
            records.append(record)
    return records

def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def write_xlsx(records: list[dict[str, Any]], output_path: Path) -> None:
    workbook = Workbook.create()
    sheet = workbook.worksheets.add("Unified Table 1")

    rows = [OUTPUT_COLUMNS] + [[record.get(column) for column in OUTPUT_COLUMNS] for record in records]
    sheet.get_range_by_indexes(0, 0, len(rows), len(OUTPUT_COLUMNS)).values = rows

    header = sheet.get_range_by_indexes(0, 0, 1, len(OUTPUT_COLUMNS))
    header.format = {
        "fill": "#1F4E78",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
    }
    header.format.row_height = 38
    sheet.freeze_panes.freeze_rows(1)

    # Apply sensible widths without allowing very long narrative fields to dominate.
    narrative_columns = {
        "activity_description",
        "activity_objective",
        "risk_target_reduction",
        "annual_qual_target",
        "qual_actual_progress_q1",
        "qual_actual_progress_q1_2",
        "qual_actual_progress_q1_3",
        "qual_actual_progress_q1_4",
        "corrective_actions_if_delayed",
        "comments",
        "annual_target_source_url",
    }
    for index, column in enumerate(OUTPUT_COLUMNS):
        col_range = sheet.get_range_by_indexes(0, index, len(rows), 1)
        col_range.format.wrap_text = column in narrative_columns
        col_range.format.column_width = 38 if column in narrative_columns else 18

    # Preserve identifiers as text and make source rows/years integer-like.
    for column in TEXT_ID_FIELDS | {"record_id", "metric_number", "wmp_activity_code"}:
        index = OUTPUT_COLUMNS.index(column)
        sheet.get_range_by_indexes(1, index, max(len(records), 1), 1).format.number_format = "@"

    table = sheet.tables.add(
        sheet.get_range_by_indexes(0, 0, len(rows), len(OUTPUT_COLUMNS)),
        True,
        "UnifiedTable1",
    )
    _ = table

    notes = workbook.worksheets.add("Mapping Notes")
    notes_rows = [
        ["Topic", "Verified treatment"],
        ["Source selection", "Highest-revision Q4 workbook for each year; Q4 contains Q1, Q1-2, Q1-3, and Q1-4 cumulative actual progress."],
        ["Dropped fields", "Only ProjectedQuantProgressQ1, Q1-2, Q1-3, and Q1-4."],
        ["Actual progress", "All four quantitative and all four qualitative actual-progress fields are retained. Definitions are YTD/cumulative in v3.1, v3.2, and v4.01."],
        ["WMP ACTIVITY NAME", "2023-2024 UtilityInitiativeName is mapped here per the official v4 template changelog; 2025 uses WMP ACTIVITY NAME."],
        ["Legacy WMPInitiativeActivity", "Retained separately as legacy_energy_safety_activity; not forced into the redesigned 2025 WMP INITIATIVE hierarchy."],
        ["WMP CATEGORY", "Combined, but original labels are preserved because some controlled values were revised."],
        ["WMP SECTION", "2023-2024 WMPInitiativeActivity# is mapped to 2025 WMP SECTION."],
        ["2025 annual targets", "Read from Table 1 of the uploaded official Petition-to-Amend workbook because it contains both AnnualQuantTarget and AnnualQualTarget. WMP.1430 and WMP.1431 are absent from that workbook and are supplemented from SDG&E official 2025 WMP Update / Annual Implementation Report."],
        ["Official revised target workbook", OFFICIAL_REVISED_TARGET_WORKBOOK_URL],
        ["Target-amendment decision", OFFICIAL_2025_AMENDMENT_DECISION_URL],
        ["2025 WMP Update", OFFICIAL_2025_WMP_UPDATE_URL],
        ["2025 Q4 notification", OFFICIAL_2025_Q4_NOTIFICATION_URL],
        ["2025 Annual Implementation Report", OFFICIAL_2025_AIR_URL],
        ["Important cycle warning", "The Annual-WMP workbook filed for the 2026-2028 WMP cycle contains 2026-2028 targets and must not populate 2025 values."],
        ["Stable cross-year key", "Use activity_tracking_id. Do not use wmp_activity_code as a stable key because it contains the reporting year."],
    ]
    notes.get_range_by_indexes(0, 0, len(notes_rows), 2).values = notes_rows
    notes.get_range("A1:B1").format = {
        "fill": "#1F4E78",
        "font": {"bold": True, "color": "#FFFFFF"},
        "wrap_text": True,
    }
    notes.get_range("A:B").format.wrap_text = True
    notes.get_range("A:A").format.column_width = 28
    notes.get_range("B:B").format.column_width = 90
    notes.freeze_panes.freeze_rows(1)

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def validate_output(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No output records generated")
    years = {int(record["reporting_year"]) for record in records}
    if years != set(YEARS):
        raise ValueError(f"Expected output years {YEARS}, found {sorted(years)}")
    if any(record["reporting_quarter"] != 4 for record in records):
        raise ValueError("Output contains a non-Q4 source record")
    ids = [record["record_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record_id values generated")

    # Confirm all actual-progress columns exist in the output schema.
    required_progress = {
        "quant_actual_progress_q1",
        "quant_actual_progress_q1_2",
        "quant_actual_progress_q1_3",
        "quant_actual_progress_q1_4",
        "qual_actual_progress_q1",
        "qual_actual_progress_q1_2",
        "qual_actual_progress_q1_3",
        "qual_actual_progress_q1_4",
    }
    if not required_progress.issubset(OUTPUT_COLUMNS):
        raise ValueError("Output schema is missing actual-progress fields")


def validate_2025_targets(records: list[dict[str, Any]]) -> None:
    qualitative_ids = set(NEW_2025_QUAL_TARGETS)
    missing = []
    for record in records:
        if int(record["reporting_year"]) != 2025:
            continue
        tracking_id = normalize_tracking_id(record.get("activity_tracking_id"))
        if tracking_id in qualitative_ids:
            if clean_value(record.get("annual_qual_target")) is None:
                missing.append(tracking_id)
        elif clean_value(record.get("annual_quant_target")) is None:
            missing.append(tracking_id or record["record_id"])
    if missing:
        raise ValueError(f"2025 records missing annual targets: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--annual-wmp-2025",
        type=Path,
        default=None,
        help="Optional official revised 2025 workbook containing Table 12.",
    )
    parser.add_argument(
        "--base-unified-csv",
        type=Path,
        default=None,
        help="Optional previously normalized Table 1 CSV; skips re-reading Q4 Table 1 sheets.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_q4_files(args.input_dir)
    if args.base_unified_csv is not None:
        if not args.base_unified_csv.exists():
            raise FileNotFoundError(args.base_unified_csv)
        records = load_base_unified_csv(args.base_unified_csv)
    else:
        records = []
        for path in selected:
            records.extend(convert_file(path))

    direct_target_workbook = (
        args.annual_wmp_2025 or discover_official_2025_target_workbook(args.input_dir)
    )
    if direct_target_workbook is not None:
        if not direct_target_workbook.exists():
            raise FileNotFoundError(direct_target_workbook)
        targets_2025 = parse_2025_target_workbook(direct_target_workbook)
    else:
        legacy_2024_q4 = next(path for path in selected if parse_period(path)[0] == 2024)
        targets_2025 = build_fallback_2025_targets(legacy_2024_q4)

    enrich_2025_targets(records, targets_2025, direct_target_workbook)
    records.sort(
        key=lambda row: (
            int(row["reporting_year"]),
            str(row["activity_tracking_id"] or ""),
            int(row["source_row"]),
        )
    )
    validate_output(records)
    validate_2025_targets(records)

    csv_path = args.output_dir / "sdge_table1_2023_2025_with_annual_targets.csv"
    xlsx_path = args.output_dir / "sdge_table1_2023_2025_with_annual_targets.xlsx"
    write_csv(records, csv_path)
    write_xlsx(records, xlsx_path)

    print("Selected Q4 sources:")
    for path in selected:
        print(f"  {path.name}")
    print(
        f"2025 target source: {direct_target_workbook.name}"
        if direct_target_workbook is not None
        else "2025 target source: reconstructed approved target set"
    )
    print(f"Rows written: {len(records)}")
    for year in YEARS:
        print(f"  {year}: {sum(1 for row in records if int(row['reporting_year']) == year)}")
    print(f"CSV:  {csv_path}")
    print(f"XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
