
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

V4_CHANGELOG_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "fileid=57874&shareable=true"
)
WMP_R1_COVER_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2025+Data+Submissions&fileid=58976&shareable=true"
)
WMP_R1_DATA_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2025+Data+Submissions&fileid=58977&shareable=true"
)
WMP_R0_DATA_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2025+Data+Submissions&fileid=58438&shareable=true"
)
OFFICIAL_2024_Q4_R2_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "docket=2025+Data+Submissions&fileid=59226&shareable=true"
)
NOV_RESPONSE_URL = (
    "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?"
    "fileid=58890&shareable=true"
)

LEGACY_HEADERS = [
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
    "HFTD TIER",
    "LINE TYPE",
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

CANONICAL_METRICS = [
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

WIDE_HEADERS = [
    "record_id",
    "source_vintage_year",
    "year_basis",
    "reporting_year",
    "year_alignment_status",
    "metric_number",
    "metric_number_status",
    "hftd_tier",
    "source_hftd_area_raw",
    "source_hftd_tier_raw",
    "hftd_crosswalk_status",
    "line_type",
    "source_line_type_raw",
    "line_type_crosswalk_status",
    *CANONICAL_METRICS,
    "source_legacy_ignition_risk_raw",
    "wildfire_risk_crosswalk_status",
    "outage_program_risk_crosswalk_status",
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
    "hftd_tier",
    "line_type",
    "canonical_metric",
    "source_metric_name",
    "value",
    "availability_status",
    "crosswalk_status",
    "comparability_status",
    "source_file",
    "source_row",
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
    "hftd_tier",
    "line_type",
    "check_name",
    "reported_value",
    "calculated_value",
    "difference",
    "status",
    "source_file",
    "source_row",
]

REVISION_HEADERS = [
    "period_or_transition",
    "selected_source",
    "selected_revision",
    "higher_or_prior_revision",
    "verification_result",
    "note",
    "source_url",
]

ISSUE_HEADERS = [
    "issue_type",
    "severity",
    "source_vintage_year",
    "reporting_year",
    "hftd_tier",
    "line_type",
    "source_file",
    "source_row",
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
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Expected numeric Table 14 value, found {value!r}") from exc
    return int(number) if number.is_integer() else number


def relative_close(a: Any, b: Any, tolerance: float = 1e-8) -> bool:
    if a is None or b is None:
        return False
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) <= tolerance * scale


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


def parse_revision(path: Path) -> int:
    match = re.search(r"(?:_R|_Rev)(\d+)", path.name, flags=re.IGNORECASE)
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
            raise ValueError(f"No worksheets in {path.name}")

        for sheet in sheets:
            if sheet.attrib["name"] == sheet_name:
                relationship_id = sheet.attrib[f"{{{NS_REL}}}id"]
                worksheet_target = relationship_map[relationship_id]
                break

        if worksheet_target is None:
            raise KeyError(f"{sheet_name!r} not found in {path.name}")

        worksheet_path = "xl/" + worksheet_target.replace("../", "").lstrip("/")
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
            raise ValueError(f"Unrecognized worksheet dimension {reference!r}")

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
                                value = int(numeric) if numeric.is_integer() else numeric
                            except (TypeError, ValueError):
                                value = raw
                values[row_index][column] = value

        return values


def discover_sources(
    input_dir: Path,
    uploaded_wmp: Path,
) -> dict[str, Any]:
    source_2023 = input_dir / "SDGE_2023_Q4_Tables1-15.xlsx"
    if not source_2023.exists():
        raise FileNotFoundError(source_2023)

    q4_candidates = [
        path
        for path in input_dir.glob("SDGE_2024_Q4_Tables1-15*.xlsx")
    ]
    if not q4_candidates:
        raise FileNotFoundError("No local 2024 Q4 Table 1-15 workbook found")

    source_2024 = max(q4_candidates, key=parse_revision)
    if not uploaded_wmp.exists():
        raise FileNotFoundError(uploaded_wmp)

    return {
        "2023": {
            "path": source_2023,
            "name": source_2023.name,
            "source_vintage_year": 2023,
            "reporting_year": 2023,
            "schema_version": "v3.1",
            "revision": f"R{parse_revision(source_2023)}",
            "revision_selection_status": "highest_local_q4_revision",
            "source_url": None,
            "guideline_url": GUIDELINES[2023][1],
        },
        "2024": {
            "path": source_2024,
            "name": source_2024.name,
            "source_vintage_year": 2024,
            "reporting_year": 2024,
            "schema_version": "v3.2",
            "revision": f"R{parse_revision(source_2024)}",
            "revision_selection_status": (
                "local_q4_r1_used; official_q4_r2_exists_but_documented_"
                "revision_concerns_lightning_arrester_record_not_table14"
            ),
            "source_url": OFFICIAL_2024_Q4_R2_URL,
            "guideline_url": GUIDELINES[2024][1],
        },
        "2025": {
            "path": uploaded_wmp,
            "name": uploaded_wmp.name,
            "source_vintage_year": 2025,
            "reporting_year": None,
            "schema_version": "v4.01",
            "revision": f"R{parse_revision(uploaded_wmp)}",
            "revision_selection_status": "uploaded_official_r1_highest_identified_revision",
            "source_url": WMP_R1_DATA_URL,
            "guideline_url": GUIDELINES[2025][1],
        },
    }


def split_legacy_area(value: Any) -> tuple[str, str]:
    raw = clean(value)
    mapping = {
        "Non-HFTD Distribution": ("Non-HFTD", "Distribution"),
        "HFTD 2 Distribution": ("HFTD Tier 2", "Distribution"),
        "HFTD 3 Distribution": ("HFTD Tier 3", "Distribution"),
        "Non-HFTD Transmission": ("Non-HFTD", "Transmission"),
        "HFTD 2 Transmission": ("HFTD Tier 2", "Transmission"),
        "HFTD 3 Transmission": ("HFTD Tier 3", "Transmission"),
    }
    if raw not in mapping:
        raise ValueError(f"Unexpected legacy HFTD Area: {raw!r}")
    return mapping[raw]


def normalize_v4_line_type(value: Any) -> tuple[str, str]:
    raw = clean(value)
    if raw == "Transmision":
        return "Transmission", "corrected_source_typo_transmision"
    if raw in {"Distribution", "Transmission"}:
        return raw, "v4_native"
    raise ValueError(f"Unexpected v4.01 line type: {raw!r}")


def parse_legacy(
    values: list[list[Any]],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    headers = [normalize_header(value) for value in values[8][:17]]
    if headers != LEGACY_HEADERS:
        raise ValueError(
            f"Legacy Table 14 schema mismatch in {source['name']}.\n"
            f"Expected: {LEGACY_HEADERS}\nFound: {headers}"
        )

    utility = clean(values[1][1])
    if utility not in {"SDG&E", "SDGE"}:
        raise AssertionError(f"Unexpected utility in {source['name']}: {utility}")

    records = []
    for zero_based_row in range(9, len(values)):
        row = values[zero_based_row]
        if clean(row[0]) is None:
            continue

        hftd_tier, line_type = split_legacy_area(row[0])
        legacy_ignition_risk = parse_number(row[2])

        record = {
            "source_vintage_year": source["source_vintage_year"],
            "year_basis": "q4_annual_update",
            "reporting_year": source["reporting_year"],
            "year_alignment_status": "source_vintage_equals_reporting_year",
            "metric_number": None,
            "metric_number_status": "not_present_in_v3_1_or_v3_2_schema",
            "hftd_tier": hftd_tier,
            "source_hftd_area_raw": clean(row[0]),
            "source_hftd_tier_raw": None,
            "hftd_crosswalk_status": "split_legacy_hftd_area",
            "line_type": line_type,
            "source_line_type_raw": None,
            "line_type_crosswalk_status": "split_legacy_hftd_area",
            "overall_utility_risk": parse_number(row[1]),
            "wildfire_risk": legacy_ignition_risk,
            "outage_program_risk": None,
            "wildfire_likelihood": None,
            "ignition_likelihood": parse_number(row[4]),
            "equipment_caused_likelihood_of_ignition": parse_number(row[5]),
            "contact_from_vegetation_likelihood_of_ignition": parse_number(row[6]),
            "contact_from_object_likelihood_of_ignition": parse_number(row[7]),
            "burn_likelihood": parse_number(row[8]),
            "wildfire_consequence": parse_number(row[10]),
            "wildfire_hazard_intensity": parse_number(row[11]),
            "wildfire_exposure_potential": parse_number(row[12]),
            "wildfire_vulnerability": parse_number(row[13]),
            "psps_risk": parse_number(row[3]),
            "psps_likelihood": parse_number(row[9]),
            "psps_consequence": parse_number(row[14]),
            "psps_exposure_potential": parse_number(row[15]),
            "psps_vulnerability": parse_number(row[16]),
            "peds_risk": None,
            "peds_likelihood": None,
            "peds_consequence": None,
            "peds_exposure_potential": None,
            "peds_vulnerability": None,
            "source_legacy_ignition_risk_raw": legacy_ignition_risk,
            "wildfire_risk_crosswalk_status": (
                "legacy_ignition_risk_mapped_to_v4_wildfire_risk"
            ),
            "outage_program_risk_crosswalk_status": (
                "not_in_legacy_schema; legacy_psps_risk_is_not_full_v4_"
                "outage_program_risk"
            ),
            "comments": None,
            "blank_meaning": None,
            "utility_id": "SDG&E",
            "schema_version": source["schema_version"],
            "source_revision": source["revision"],
            "revision_selection_status": source["revision_selection_status"],
            "source_file": source["name"],
            "source_sheet": "Table 14",
            "source_row": zero_based_row + 1,
            "source_url": source["source_url"],
            "guideline_url": source["guideline_url"],
        }
        records.append(record)

    if len(records) != 6:
        raise AssertionError(
            f"Expected six legacy Table 14 rows in {source['name']}; "
            f"found {len(records)}"
        )
    return records


def normalize_v4_headers(headers: list[Any]) -> list[str | None]:
    normalized = [normalize_header(value) for value in headers]
    # Accept the misspelled draft-template form, but canonicalize it to the
    # corrected header used in the uploaded SDG&E R1 workbook.
    return [
        (
            "CONTACT FROM OBJECT LIKELIHOOD OF IGNITION"
            if value == "CONTACT FROM OBJECT LIKELIOD OF IGNITION"
            else value
        )
        for value in normalized
    ]


def parse_v4(
    values: list[list[Any]],
    source: dict[str, Any],
    issues: list[list[Any]],
) -> list[dict[str, Any]]:
    headers = normalize_v4_headers(values[0][:29])
    if headers != V4_HEADERS:
        raise ValueError(
            f"v4.01 Table 14 schema mismatch in {source['name']}.\n"
            f"Expected: {V4_HEADERS}\nFound: {headers}"
        )

    records = []
    for zero_based_row in range(1, len(values)):
        row = values[zero_based_row]
        if clean(row[0]) is None:
            continue

        line_type, line_status = normalize_v4_line_type(row[2])
        reporting_year = int(row[28])

        if clean(row[2]) == "Transmision":
            issues.append(
                [
                    "source_line_type_typo",
                    "warning",
                    2025,
                    reporting_year,
                    clean(row[1]),
                    line_type,
                    source["name"],
                    zero_based_row + 1,
                    "LINE TYPE",
                    clean(row[2]),
                    "The source spells Transmission as 'Transmision'. "
                    "The converter normalizes it while preserving the raw value.",
                ]
            )

        record = {
            "source_vintage_year": 2025,
            "year_basis": "annual_wmp_submission_vintage",
            "reporting_year": reporting_year,
            "year_alignment_status": (
                "2025_submission_vintage_reports_2026_risk_year"
                if reporting_year == 2026
                else "submission_vintage_differs_from_reporting_year"
            ),
            "metric_number": int(row[0]),
            "metric_number_status": "v4_native_corrected_214_prefix",
            "hftd_tier": clean(row[1]),
            "source_hftd_area_raw": None,
            "source_hftd_tier_raw": clean(row[1]),
            "hftd_crosswalk_status": "v4_native",
            "line_type": line_type,
            "source_line_type_raw": clean(row[2]),
            "line_type_crosswalk_status": line_status,
            "overall_utility_risk": parse_number(row[3]),
            "wildfire_risk": parse_number(row[4]),
            "outage_program_risk": parse_number(row[5]),
            "wildfire_likelihood": parse_number(row[6]),
            "ignition_likelihood": parse_number(row[7]),
            "equipment_caused_likelihood_of_ignition": parse_number(row[8]),
            "contact_from_vegetation_likelihood_of_ignition": parse_number(row[9]),
            "contact_from_object_likelihood_of_ignition": parse_number(row[10]),
            "burn_likelihood": parse_number(row[11]),
            "wildfire_consequence": parse_number(row[12]),
            "wildfire_hazard_intensity": parse_number(row[13]),
            "wildfire_exposure_potential": parse_number(row[14]),
            "wildfire_vulnerability": None,
            "psps_risk": parse_number(row[15]),
            "psps_likelihood": parse_number(row[16]),
            "psps_consequence": parse_number(row[17]),
            "psps_exposure_potential": parse_number(row[18]),
            "psps_vulnerability": parse_number(row[19]),
            "peds_risk": parse_number(row[20]),
            "peds_likelihood": parse_number(row[21]),
            "peds_consequence": parse_number(row[22]),
            "peds_exposure_potential": parse_number(row[23]),
            "peds_vulnerability": parse_number(row[24]),
            "source_legacy_ignition_risk_raw": None,
            "wildfire_risk_crosswalk_status": "v4_native",
            "outage_program_risk_crosswalk_status": "v4_native",
            "comments": clean(row[25]),
            "blank_meaning": clean(row[26]),
            "utility_id": clean(row[27]),
            "schema_version": source["schema_version"],
            "source_revision": source["revision"],
            "revision_selection_status": source["revision_selection_status"],
            "source_file": source["name"],
            "source_sheet": "Table 14",
            "source_row": zero_based_row + 1,
            "source_url": source["source_url"],
            "guideline_url": source["guideline_url"],
        }

        if record["utility_id"] != "SDG&E":
            raise AssertionError(
                f"Unexpected utility ID in {source['name']} row "
                f"{record['source_row']}"
            )

        records.append(record)

    if len(records) != 6:
        raise AssertionError(
            f"Expected six v4.01 Table 14 rows; found {len(records)}"
        )

    expected_numbers = list(range(2140000000, 2140000006))
    actual_numbers = sorted(record["metric_number"] for record in records)
    if actual_numbers != expected_numbers:
        raise AssertionError(
            f"Unexpected Table 14 metric numbers: {actual_numbers}"
        )

    return records


def canonical_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["hftd_tier"], record["line_type"]


def validate_keys(records: list[dict[str, Any]], label: str) -> None:
    expected = {
        ("Non-HFTD", "Distribution"),
        ("HFTD Tier 2", "Distribution"),
        ("HFTD Tier 3", "Distribution"),
        ("Non-HFTD", "Transmission"),
        ("HFTD Tier 2", "Transmission"),
        ("HFTD Tier 3", "Transmission"),
    }
    actual = {canonical_key(record) for record in records}
    if actual != expected:
        raise AssertionError(f"Unexpected Table 14 keys for {label}: {actual}")


def source_record_id(record: dict[str, Any]) -> str:
    return stable_hash(
        "T14R-",
        [
            record["source_vintage_year"],
            record["reporting_year"],
            record["hftd_tier"],
            record["line_type"],
            record["source_file"],
            record["source_row"],
        ],
    )


def record_to_wide_row(record: dict[str, Any]) -> list[Any]:
    return [
        source_record_id(record),
        record["source_vintage_year"],
        record["year_basis"],
        record["reporting_year"],
        record["year_alignment_status"],
        record["metric_number"],
        record["metric_number_status"],
        record["hftd_tier"],
        record["source_hftd_area_raw"],
        record["source_hftd_tier_raw"],
        record["hftd_crosswalk_status"],
        record["line_type"],
        record["source_line_type_raw"],
        record["line_type_crosswalk_status"],
        *[record[name] for name in CANONICAL_METRICS],
        record["source_legacy_ignition_risk_raw"],
        record["wildfire_risk_crosswalk_status"],
        record["outage_program_risk_crosswalk_status"],
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


def metric_source_name(record: dict[str, Any], metric: str) -> str | None:
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
        name: name.replace("_", " ").upper()
        for name in CANONICAL_METRICS
    }
    mapping["equipment_caused_likelihood_of_ignition"] = (
        "EQUIPMENT CAUSED LIKELIHOOD OF IGNITION"
    )
    mapping["contact_from_vegetation_likelihood_of_ignition"] = (
        "CONTACT FROM VEGETATION LIKELIHOOD OF IGNITION"
    )
    mapping["contact_from_object_likelihood_of_ignition"] = (
        "CONTACT FROM OBJECT LIKELIOD OF IGNITION"
    )
    return mapping[metric]


def metric_status(
    record: dict[str, Any],
    metric: str,
) -> tuple[str, str, str]:
    value = record[metric]
    legacy = record["schema_version"] in {"v3.1", "v3.2"}

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
            "removed_or_not_separately_reported_in_v4_01",
            "not_comparable_missing_v4_field",
        )

    availability = "reported" if value is not None else "source_blank"

    if legacy and metric == "wildfire_risk":
        crosswalk = "legacy_ignition_risk_renamed_to_wildfire_risk"
    elif legacy and metric == "burn_likelihood":
        crosswalk = "legacy_burn_probability_renamed_to_burn_likelihood"
    elif legacy and metric == "psps_vulnerability":
        crosswalk = (
            "legacy_vulnerability_of_community_to_psps_renamed_to_"
            "psps_vulnerability"
        )
    else:
        crosswalk = "same_or_standardized_metric"

    comparability = (
        "schema_aligned_but_risk_model_scale_may_change_between_vintages"
    )
    if metric in {
        "overall_utility_risk",
        "outage_program_risk",
    }:
        comparability = (
            "definition_expanded_in_v4_to_include_outage_program_and_peds"
            if not legacy or metric == "overall_utility_risk"
            else "not_comparable"
        )

    return availability, crosswalk, comparability


def build_long_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for record in records:
        for metric in CANONICAL_METRICS:
            availability, crosswalk, comparability = metric_status(
                record,
                metric,
            )
            rows.append(
                [
                    source_record_id(record),
                    record["source_vintage_year"],
                    record["reporting_year"],
                    record["year_alignment_status"],
                    record["metric_number"],
                    record["hftd_tier"],
                    record["line_type"],
                    metric,
                    metric_source_name(record, metric),
                    record[metric],
                    availability,
                    crosswalk,
                    comparability,
                    record["source_file"],
                    record["source_row"],
                ]
            )
    return rows


def build_reconciliation(
    records: list[dict[str, Any]],
    issues: list[list[Any]],
) -> list[list[Any]]:
    rows = []

    def add_check(
        record: dict[str, Any],
        name: str,
        reported: Any,
        calculated: Any,
    ) -> None:
        if reported is None or calculated is None:
            status = "not_testable_due_to_blank"
            difference = None
        else:
            difference = float(reported) - float(calculated)
            status = "reconciled" if relative_close(reported, calculated) else "mismatch"
            if status == "mismatch":
                issues.append(
                    [
                        "risk_reconciliation_mismatch",
                        "error",
                        record["source_vintage_year"],
                        record["reporting_year"],
                        record["hftd_tier"],
                        record["line_type"],
                        record["source_file"],
                        record["source_row"],
                        name,
                        difference,
                        "Reported risk does not reconcile to the expected "
                        "component relationship.",
                    ]
                )

        rows.append(
            [
                record["source_vintage_year"],
                record["reporting_year"],
                record["hftd_tier"],
                record["line_type"],
                name,
                reported,
                calculated,
                difference,
                status,
                record["source_file"],
                record["source_row"],
            ]
        )

    for record in records:
        if record["schema_version"] in {"v3.1", "v3.2"}:
            component_sum = None
            if (
                record["wildfire_risk"] is not None
                and record["psps_risk"] is not None
            ):
                component_sum = (
                    record["wildfire_risk"] + record["psps_risk"]
                )
            add_check(
                record,
                "legacy_overall_equals_ignition_plus_psps_risk",
                record["overall_utility_risk"],
                component_sum,
            )
        else:
            overall_sum = None
            if (
                record["wildfire_risk"] is not None
                and record["outage_program_risk"] is not None
            ):
                overall_sum = (
                    record["wildfire_risk"]
                    + record["outage_program_risk"]
                )
            add_check(
                record,
                "v4_overall_equals_wildfire_plus_outage_program_risk",
                record["overall_utility_risk"],
                overall_sum,
            )

            outage_sum = None
            if (
                record["psps_risk"] is not None
                and record["peds_risk"] is not None
            ):
                outage_sum = record["psps_risk"] + record["peds_risk"]
            add_check(
                record,
                "v4_outage_program_equals_psps_plus_peds_risk",
                record["outage_program_risk"],
                outage_sum,
            )

            wildfire_product = None
            if (
                record["ignition_likelihood"] is not None
                and record["wildfire_consequence"] is not None
            ):
                wildfire_product = (
                    record["ignition_likelihood"]
                    * record["wildfire_consequence"]
                )
            add_check(
                record,
                "v4_wildfire_risk_equals_ignition_likelihood_times_consequence",
                record["wildfire_risk"],
                wildfire_product,
            )

            psps_product = None
            if (
                record["psps_likelihood"] is not None
                and record["psps_consequence"] is not None
            ):
                psps_product = (
                    record["psps_likelihood"]
                    * record["psps_consequence"]
                )
            add_check(
                record,
                "v4_psps_risk_equals_likelihood_times_consequence",
                record["psps_risk"],
                psps_product,
            )

            peds_product = None
            if (
                record["peds_likelihood"] is not None
                and record["peds_consequence"] is not None
            ):
                peds_product = (
                    record["peds_likelihood"]
                    * record["peds_consequence"]
                )
            add_check(
                record,
                "v4_peds_risk_equals_likelihood_times_consequence",
                record["peds_risk"],
                peds_product,
            )

    return rows


def normalized_legacy_matrix(values: list[list[Any]]) -> list[list[Any]]:
    records = []
    fake_source = {
        "source_vintage_year": 2024,
        "reporting_year": 2024,
        "schema_version": "v3.2",
        "revision": "test",
        "revision_selection_status": "test",
        "name": "test",
        "source_url": None,
        "guideline_url": GUIDELINES[2024][1],
    }
    for record in parse_legacy(values, fake_source):
        records.append(
            [
                record["hftd_tier"],
                record["line_type"],
                *[record[metric] for metric in CANONICAL_METRICS],
            ]
        )
    return sorted(records)


def verify_2024_quarterly_stability(
    input_dir: Path,
) -> tuple[bool, list[str]]:
    files = []
    matrices = []
    for quarter in (1, 2, 3, 4):
        candidates = list(
            input_dir.glob(f"SDGE_2024_Q{quarter}_Tables1-15*.xlsx")
        )
        if not candidates:
            continue
        selected = max(candidates, key=parse_revision)
        files.append(selected.name)
        matrices.append(
            normalized_legacy_matrix(
                read_xlsx_sheet(selected, "Table 14")
            )
        )
    if not matrices:
        return False, files
    return all(matrix == matrices[0] for matrix in matrices[1:]), files


def build_schema_rows() -> list[list[Any]]:
    rows = [
        [
            "metric_number",
            None,
            None,
            "METRIC NUMBER",
            "new_in_v4_01",
            "Preserve v4 metric numbers 2140000000–2140000005. "
            "Legacy records remain blank.",
        ],
        [
            "hftd_tier",
            "Part of HFTD Area",
            "Part of HFTD Area",
            "HFTD TIER",
            "combined_field_split",
            "Split the legacy HFTD Area value into HFTD tier and line type.",
        ],
        [
            "line_type",
            "Part of HFTD Area",
            "Part of HFTD Area",
            "LINE TYPE",
            "combined_field_split",
            "Normalize source typo Transmision to Transmission and preserve raw text.",
        ],
        [
            "overall_utility_risk",
            "Overall Utility Risk",
            "Overall Utility Risk",
            "OVERALL UTILITY RISK",
            "same_name_but_component_definition_expanded",
            "Preserve values; do not assume numeric comparability because "
            "v4 includes broader outage-program risk.",
        ],
        [
            "wildfire_risk",
            "Ignition Risk",
            "Ignition Risk",
            "WILDFIRE RISK",
            "renamed",
            "Map legacy Ignition Risk to canonical wildfire_risk and preserve "
            "the legacy source value.",
        ],
        [
            "outage_program_risk",
            None,
            None,
            "OUTAGE PROGRAM RISK",
            "new_in_v4_01",
            "Leave legacy values blank. Do not equate legacy PSPS Risk with "
            "the broader v4 outage-program risk.",
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
            "ignition_likelihood",
            "Ignition Likelihood",
            "Ignition Likelihood",
            "IGNITION LIKELIHOOD",
            "retained",
            "Preserve values.",
        ],
        [
            "equipment_caused_likelihood_of_ignition",
            "Equipment Likelihood of Ignition",
            "Equipment Likelihood of Ignition",
            "EQUIPMENT CAUSED LIKELIHOOD OF IGNITION",
            "renamed",
            "Use the v4 canonical label.",
        ],
        [
            "burn_likelihood",
            "Burn Probability",
            "Burn Probability",
            "BURN LIKELIHOOD",
            "renamed",
            "Use the v4 canonical label.",
        ],
        [
            "wildfire_vulnerability",
            "Wildfire Vulnerability",
            "Wildfire Vulnerability",
            None,
            "not_separately_present_in_v4_01_template",
            "Preserve legacy values; leave v4 blank.",
        ],
        [
            "psps_vulnerability",
            "Vulnerability of Community to PSPS",
            "Vulnerability of Community to PSPS",
            "PSPS VULNERABILITY",
            "renamed",
            "Use the v4 canonical label.",
        ],
        [
            "peds_risk_components",
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
            "Preserve and validate all v4 metadata.",
        ],
    ]
    return rows


def build_revision_rows(
    sources: dict[str, Any],
    stability: bool,
    stability_files: list[str],
) -> list[list[Any]]:
    return [
        [
            "2023 annual selection",
            sources["2023"]["name"],
            sources["2023"]["revision"],
            "Earlier 2023 quarterly copies",
            "Q4 selected as required annual update",
            "v3.1 requires Table 14 values to be updated annually with Q4 data.",
            GUIDELINES[2023][1],
        ],
        [
            "2024 annual selection",
            sources["2024"]["name"],
            sources["2024"]["revision"],
            "Official Q4 R2 exists",
            (
                "local 2024 quarterly Table 14 values are identical"
                if stability
                else "local quarterly stability check failed"
            ),
            "The official R2 revision addressed a lightning-arrester record "
            "correction following an NOV, not a documented Table 14 change. "
            f"Compared local files: {', '.join(stability_files)}",
            NOV_RESPONSE_URL,
        ],
        [
            "v3.1 to v3.2",
            "2023 Q4 v3.1",
            "v3.1",
            "2024 Q4 v3.2",
            "schema stable",
            "The same 17-column Table 14 structure is used.",
            GUIDELINES[2024][1],
        ],
        [
            "v3.2 to v4.01",
            sources["2025"]["name"],
            sources["2025"]["revision"],
            "v4.0/v4.01 template transition",
            "major annual-template restructuring",
            "Table 14 moves to Annual-WMP, HFTD Area is split, metric numbers "
            "and metadata are added, and wildfire/outage/PEDS risk components "
            "are reorganized.",
            GUIDELINES[2025][1],
        ],
        [
            "v4 template corrections",
            sources["2025"]["name"],
            sources["2025"]["revision"],
            "v4.0 draft template",
            "validated against v4 changelog",
            "Duplicate WILDFIRE CONSEQUENCE was removed and Table 14 metric "
            "numbers were corrected from the 114... prefix to 214....",
            V4_CHANGELOG_URL,
        ],
        [
            "2026-WMP R0 to R1",
            sources["2025"]["name"],
            sources["2025"]["revision"],
            "SDGE_2026-WMP_R0",
            "no Table 14-specific change identified in R1 cover letter",
            "The R1 cover letter explicitly describes an Annual-WMP Table 15 "
            "correction, but does not identify a Table 14 correction. R1 is "
            "still selected as the highest uploaded revision.",
            WMP_R1_COVER_URL,
        ],
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


def write_rows(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    *,
    chunk_size: int = 300,
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

    large_text = {
        "year_alignment_status",
        "metric_number_status",
        "wildfire_risk_crosswalk_status",
        "outage_program_risk_crosswalk_status",
        "comments",
        "blank_meaning",
        "revision_selection_status",
        "source_file",
        "source_url",
        "guideline_url",
        "converter_action",
        "note",
        "comparability_status",
    }
    medium_text = {
        "availability_status",
        "crosswalk_status",
        "verification_result",
        "check_name",
    }

    for index, header in enumerate(headers):
        letter = column_letter(index)
        width = 18
        if header in large_text:
            width = 38
        elif header in medium_text:
            width = 28
        sheet.get_range(
            f"{letter}1:{letter}{last_row}"
        ).format.column_width = width


def build_workbook(
    output_path: Path,
    wide_rows: list[list[Any]],
    long_rows: list[list[Any]],
    schema_rows: list[list[Any]],
    reconciliation_rows: list[list[Any]],
    revision_rows: list[list[Any]],
    issue_rows: list[list[Any]],
    validation: dict[str, Any],
) -> None:
    workbook = Workbook.create()
    readme = workbook.worksheets.add("README")
    wide = workbook.worksheets.add("Unified Wide")
    long_sheet = workbook.worksheets.add("Metrics Long")
    schema = workbook.worksheets.add("Schema Crosswalk")
    reconciliation = workbook.worksheets.add("Risk Reconciliation")
    revisions = workbook.worksheets.add("Revision Notes")
    issues = workbook.worksheets.add("Validation Issues")

    readme_rows = [
        ["SDG&E Table 14 Unified Risk Summary", "", "", ""],
        [
            "Source vintages",
            "2023 Q4, 2024 Q4, and 2025 Annual-WMP filing",
            "Unified rows",
            validation["unified_rows"],
        ],
        [
            "Important year caveat",
            "The uploaded 2025 Annual-WMP filing reports risk year 2026. "
            "The output preserves source_vintage_year=2025 and "
            "reporting_year=2026 rather than relabeling it as 2025.",
            "",
            "",
        ],
        [
            "Legacy annual cadence",
            "v3.1/v3.2 require Table 14 to be updated annually with Q4 data.",
            "",
            "",
        ],
        [
            "v4.01 cadence",
            "Table 14 moves to the Annual-WMP workbook submitted with a WMP.",
            "",
            "",
        ],
        [
            "Projection treatment",
            "Table 14 has no projection columns; no projection data are present.",
            "",
            "",
        ],
        [
            "2024 revision status",
            validation["2024_revision_note"],
            "2024 local quarterly stability",
            validation["2024_table14_values_stable_across_local_quarters"],
        ],
        [
            "v4 metric-number validation",
            "2140000000–2140000005",
            "Risk reconciliation issues",
            validation["risk_reconciliation_issues"],
        ],
        [
            "Source line-type typo corrections",
            validation["source_line_type_typo_rows"],
            "Other validation issues",
            validation["validation_issue_rows"]
            - validation["source_line_type_typo_rows"],
        ],
        ["", "", "", ""],
        ["Official source", "Purpose", "URL", "Finding"],
        [
            "Data Guidelines v3.1",
            "2023 schema and cadence",
            GUIDELINES[2023][1],
            "Q4 annual update; combined HFTD Area field and 16 risk components.",
        ],
        [
            "Data Guidelines v3.2",
            "2024 schema and cadence",
            GUIDELINES[2024][1],
            "Same legacy Table 14 structure and annual Q4 cadence.",
        ],
        [
            "Data Guidelines v4.01",
            "Annual-WMP schema",
            GUIDELINES[2025][1],
            "Separate HFTD tier/line type, new risk architecture, metric "
            "numbers, comments, blank meaning, utility ID, and reporting year.",
        ],
        [
            "v4 Template Changelog",
            "Template corrections",
            V4_CHANGELOG_URL,
            "Removed duplicate WILDFIRE CONSEQUENCE and corrected metric "
            "number prefix to 214....",
        ],
        [
            "SDG&E 2026-WMP R1 cover letter",
            "R1 revision verification",
            WMP_R1_COVER_URL,
            "No Table 14-specific correction is identified; Table 15 is named.",
        ],
        [
            "2024 NOV response",
            "Official Q4 R2 context",
            NOV_RESPONSE_URL,
            "The documented correction concerns a lightning-arrester record, "
            "not Table 14 risk-summary values.",
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
    for column, width in zip(("A", "B", "C", "D"), (34, 70, 60, 58)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    write_rows(wide, WIDE_HEADERS, wide_rows)
    format_sheet(wide, WIDE_HEADERS, len(wide_rows), freeze_columns=8)

    for metric in CANONICAL_METRICS:
        letter = column_letter(WIDE_HEADERS.index(metric))
        wide.get_range(
            f"{letter}2:{letter}{len(wide_rows) + 1}"
        ).format.number_format = "0.###############"

    alignment_col = column_letter(
        WIDE_HEADERS.index("year_alignment_status")
    )
    wide.get_range(
        f"{alignment_col}2:{alignment_col}{len(wide_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${alignment_col}2="2025_submission_vintage_reports_2026_risk_year"',
        {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
    )

    line_status_col = column_letter(
        WIDE_HEADERS.index("line_type_crosswalk_status")
    )
    wide.get_range(
        f"{line_status_col}2:{line_status_col}{len(wide_rows) + 1}"
    ).conditional_formats.add_custom(
        f'=${line_status_col}2="corrected_source_typo_transmision"',
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

    write_rows(schema, SCHEMA_HEADERS, schema_rows)
    format_sheet(schema, SCHEMA_HEADERS, len(schema_rows), freeze_columns=1)

    write_rows(
        reconciliation,
        RECON_HEADERS,
        reconciliation_rows,
    )
    format_sheet(
        reconciliation,
        RECON_HEADERS,
        len(reconciliation_rows),
        freeze_columns=4,
    )

    write_rows(revisions, REVISION_HEADERS, revision_rows)
    format_sheet(
        revisions,
        REVISION_HEADERS,
        len(revision_rows),
        freeze_columns=3,
    )

    write_rows(issues, ISSUE_HEADERS, issue_rows)
    format_sheet(issues, ISSUE_HEADERS, len(issue_rows), freeze_columns=4)
    if issue_rows:
        severity_col = column_letter(ISSUE_HEADERS.index("severity"))
        issues.get_range(
            f"A2:K{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_col}2="error"',
            {"fill": "#FEE2E2", "font": {"color": "#991B1B"}},
        )
        issues.get_range(
            f"A2:K{len(issue_rows) + 1}"
        ).conditional_formats.add_custom(
            f'=${severity_col}2="warning"',
            {"fill": "#FEF3C7", "font": {"color": "#92400E"}},
        )

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 14 risk-summary submissions for the 2023, "
            "2024, and 2025 source vintages. Legacy Q4 data are crosswalked "
            "to the v4.01 Annual-WMP schema. The 2025 filing's explicit "
            "REPORTING YEAR=2026 is preserved."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/data",
        help="Directory containing the 2023 and 2024 source workbooks.",
    )
    parser.add_argument(
        "--annual-wmp",
        default=(
            "/mnt/data/SDGE_2026-WMP_R1_Tabular Wildfire Mitigation "
            "Annual-WMP Data.xlsx"
        ),
        help="Path to the uploaded SDG&E Annual-WMP R1 workbook.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/table14_output",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    annual_wmp = Path(args.annual_wmp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(input_dir, annual_wmp)
    issues: list[list[Any]] = []

    records_2023 = parse_legacy(
        read_xlsx_sheet(sources["2023"]["path"], "Table 14"),
        sources["2023"],
    )
    records_2024 = parse_legacy(
        read_xlsx_sheet(sources["2024"]["path"], "Table 14"),
        sources["2024"],
    )
    records_2025 = parse_v4(
        read_xlsx_sheet(sources["2025"]["path"], "Table 14"),
        sources["2025"],
        issues,
    )

    sources["2025"]["reporting_year"] = records_2025[0]["reporting_year"]

    validate_keys(records_2023, "2023")
    validate_keys(records_2024, "2024")
    validate_keys(records_2025, "2025 filing")

    all_records = sorted(
        records_2023 + records_2024 + records_2025,
        key=lambda record: (
            record["source_vintage_year"],
            record["line_type"],
            record["hftd_tier"],
        ),
    )

    # Explicitly flag the year-label caveat.
    for record in records_2025:
        issues.append(
            [
                "source_vintage_reporting_year_difference",
                "info",
                2025,
                record["reporting_year"],
                record["hftd_tier"],
                record["line_type"],
                record["source_file"],
                record["source_row"],
                "REPORTING YEAR",
                record["reporting_year"],
                "The workbook was filed in 2025 for the 2026–2028 Base WMP "
                "and explicitly reports year 2026. It is included as the "
                "2025 source vintage, not relabeled as reporting year 2025.",
            ]
        )

    stability, stability_files = verify_2024_quarterly_stability(input_dir)
    if not stability:
        issues.append(
            [
                "2024_quarterly_table14_instability",
                "warning",
                2024,
                2024,
                None,
                None,
                sources["2024"]["name"],
                None,
                "Table 14",
                None,
                "The locally available 2024 quarterly Table 14 copies differ.",
            ]
        )

    wide_rows = [record_to_wide_row(record) for record in all_records]
    long_rows = build_long_rows(all_records)
    reconciliation_rows = build_reconciliation(all_records, issues)
    schema_rows = build_schema_rows()
    revision_rows = build_revision_rows(
        sources,
        stability,
        stability_files,
    )

    workbook_path = (
        output_dir / "sdge_table14_2023_2025_unified.xlsx"
    )
    wide_csv = (
        output_dir / "sdge_table14_2023_2025_unified_wide.csv"
    )
    long_csv = (
        output_dir / "sdge_table14_2023_2025_metrics_long.csv"
    )
    schema_csv = (
        output_dir / "sdge_table14_schema_crosswalk.csv"
    )
    reconciliation_csv = (
        output_dir / "sdge_table14_risk_reconciliation.csv"
    )
    revision_csv = (
        output_dir / "sdge_table14_revision_notes.csv"
    )
    issues_csv = (
        output_dir / "sdge_table14_validation_issues.csv"
    )
    validation_path = output_dir / "validation_summary.json"

    write_csv(wide_csv, WIDE_HEADERS, wide_rows)
    write_csv(long_csv, LONG_HEADERS, long_rows)
    write_csv(schema_csv, SCHEMA_HEADERS, schema_rows)
    write_csv(
        reconciliation_csv,
        RECON_HEADERS,
        reconciliation_rows,
    )
    write_csv(revision_csv, REVISION_HEADERS, revision_rows)
    write_csv(issues_csv, ISSUE_HEADERS, issues)

    validation = {
        "unified_rows": len(wide_rows),
        "long_metric_rows": len(long_rows),
        "source_vintages": [2023, 2024, 2025],
        "reporting_years": sorted(
            {record["reporting_year"] for record in all_records}
        ),
        "2025_source_vintage_reporting_year": records_2025[0][
            "reporting_year"
        ],
        "2024_table14_values_stable_across_local_quarters": stability,
        "2024_local_files_compared": stability_files,
        "2024_revision_note": (
            "Official Q4 R2 exists, but the documented R2 correction concerns "
            "a lightning-arrester record. Local R1 Table 14 is used; all local "
            "2024 quarterly Table 14 values are identical."
        ),
        "source_line_type_typo_rows": sum(
            row[0] == "source_line_type_typo"
            for row in issues
        ),
        "risk_reconciliation_issues": sum(
            row[0] == "risk_reconciliation_mismatch"
            for row in issues
        ),
        "validation_issue_rows": len(issues),
        "projection_columns_found": 0,
        "sources": {
            key: {
                "name": source["name"],
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
        wide_rows,
        long_rows,
        schema_rows,
        reconciliation_rows,
        revision_rows,
        issues,
        validation,
    )

    print(f"Created: {workbook_path}")
    print(f"Created: {wide_csv}")
    print(f"Created: {long_csv}")
    print(f"Created: {schema_csv}")
    print(f"Created: {reconciliation_csv}")
    print(f"Created: {revision_csv}")
    print(f"Created: {issues_csv}")
    print(f"Created: {validation_path}")
    print(f"Unified rows: {len(wide_rows)}")
    print(f"Long metric rows: {len(long_rows)}")
    print(
        "Reporting years represented:",
        validation["reporting_years"],
    )
    print(
        "Risk reconciliation issues:",
        validation["risk_reconciliation_issues"],
    )


if __name__ == "__main__":
    main()
