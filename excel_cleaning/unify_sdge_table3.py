
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
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=53475&shareable=true",
    ),
    2024: (
        "v3.2",
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=56226&shareable=true",
    ),
    2025: (
        "v4.01",
        "https://efiling.energysafety.ca.gov/eFiling/Getfile.aspx?fileid=58132&shareable=true",
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

HEADERS = [
    "metric_number",
    "metric_group_crosswalk",
    "legacy_metric_number_crosswalk",
    "source_metric_group_raw",
    "source_legacy_metric_number_raw",
    "crosswalk_status",
    "metric",
    "definition",
    "purpose",
    "assumptions_made_to_connect_metric_to_purpose",
    "third_party_validation",
    "unit_raw",
    "unit_canonical",
    "actual_value_raw",
    "actual_value_canonical",
    "value_conversion",
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
    "source_number_format",
    "guideline_url",
]


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.replace("\xa0", " ").split())
        return normalized or None
    return value


def normalized_key(value: Any) -> str | None:
    value = clean(value)
    return value.casefold() if isinstance(value, str) else value


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
            raise ValueError(f"Expected a numeric Table 3 value, found {value!r}") from exc
        return int(parsed) if parsed.is_integer() else parsed
    raise TypeError(f"Unsupported numeric type: {type(value).__name__}")


def excel_column_letter(zero_based_index: int) -> str:
    number = zero_based_index + 1
    output = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        output = chr(65 + remainder) + output
    return output


def column_index(letters: str) -> int:
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


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
        if parsed:
            candidates.append(parsed)

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for year in (2023, 2024, 2025):
        quarters = (4,) if year == 2023 else (1, 2, 3, 4)
        for quarter in quarters:
            matches = [
                item
                for item in candidates
                if item["year"] == year and item["quarter"] == quarter
            ]
            if not matches:
                raise FileNotFoundError(
                    f"No SDG&E {year} Q{quarter} XLSX workbook found in {input_dir}"
                )
            selected[(year, quarter)] = max(
                matches, key=lambda item: item["revision"]
            )

    return selected


class XlsxSheet:
    def __init__(
        self,
        values: list[list[Any]],
        styles: list[list[int]],
        number_formats: dict[int, str],
    ) -> None:
        self.values = values
        self.styles = styles
        self.number_formats = number_formats

    def number_format(self, row_index: int, column_index_: int) -> str:
        style_id = self.styles[row_index][column_index_]
        return self.number_formats.get(style_id, "General")


def read_xlsx_sheet(path: Path, sheet_name: str) -> XlsxSheet:
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
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in rel_root
        }

        target = None
        sheets = workbook_root.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            raise ValueError(f"No sheets found in {path.name}")

        for sheet in sheets:
            if sheet.attrib["name"] == sheet_name:
                rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
                target = rel_map[rel_id]
                break

        if target is None:
            raise KeyError(f"{sheet_name!r} not found in {path.name}")

        worksheet_path = "xl/" + target.replace("../", "").lstrip("/")
        worksheet_root = ET.fromstring(archive.read(worksheet_path))

        dimension = worksheet_root.find(f"{{{NS_MAIN}}}dimension")
        ref = dimension.attrib.get("ref", "A1") if dimension is not None else "A1"
        last_cell = ref.split(":")[-1]
        dimension_match = re.match(r"([A-Z]+)(\d+)", last_cell)
        if not dimension_match:
            raise ValueError(f"Invalid worksheet dimension {ref!r}")

        max_columns = column_index(dimension_match.group(1)) + 1
        max_rows = int(dimension_match.group(2))
        values = [[None] * max_columns for _ in range(max_rows)]
        styles = [[0] * max_columns for _ in range(max_rows)]

        sheet_data = worksheet_root.find(f"{{{NS_MAIN}}}sheetData")
        if sheet_data is not None:
            for row in sheet_data:
                row_index = int(row.attrib["r"]) - 1
                for cell in row:
                    address_match = re.match(r"([A-Z]+)(\d+)", cell.attrib["r"])
                    if not address_match:
                        continue

                    column_index_ = column_index(address_match.group(1))
                    style_id = int(cell.attrib.get("s", "0"))
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

                    values[row_index][column_index_] = value
                    styles[row_index][column_index_] = style_id

        style_root = ET.fromstring(archive.read("xl/styles.xml"))
        custom_formats: dict[int, str] = {}
        num_formats = style_root.find(f"{{{NS_MAIN}}}numFmts")
        if num_formats is not None:
            for item in num_formats:
                custom_formats[int(item.attrib["numFmtId"])] = item.attrib[
                    "formatCode"
                ]

        built_in_formats = {
            0: "General",
            1: "0",
            2: "0.00",
            9: "0%",
            10: "0.00%",
            14: "mm-dd-yy",
        }
        cell_xfs = style_root.find(f"{{{NS_MAIN}}}cellXfs")
        style_number_formats: dict[int, str] = {}
        if cell_xfs is not None:
            for style_id, item in enumerate(cell_xfs):
                number_format_id = int(item.attrib.get("numFmtId", "0"))
                style_number_formats[style_id] = custom_formats.get(
                    number_format_id,
                    built_in_formats.get(number_format_id, str(number_format_id)),
                )

        return XlsxSheet(values, styles, style_number_formats)


def is_percent_number_format(number_format: str) -> bool:
    return "%" in (number_format or "")


def canonicalize_value(
    raw_value: Any,
    unit_raw: Any,
    number_format: str,
) -> tuple[int | float | None, str | None]:
    numeric = parse_number(raw_value)
    if numeric is None:
        return None, None

    if (
        normalized_key(unit_raw) == "percent"
        and is_percent_number_format(number_format)
    ):
        if numeric > 1:
            raise ValueError(
                "A percent-formatted cell contained a value greater than one; "
                "automatic fraction-to-percent conversion would be unsafe."
            )
        converted = numeric * 100
        return converted, "excel_percent_fraction_to_percent_points"

    return numeric, None


def parse_v3_1_template(sheet: XlsxSheet) -> list[dict[str, Any]]:
    values = sheet.values
    rows = []
    group = None

    for row_index in range(9, len(values)):
        row = values[row_index]
        if clean(row[3]):
            group = clean(row[3])

        legacy_metric_number = clean(row[4])
        metric_name = clean(row[5])
        if metric_name is None:
            continue

        rows.append(
            {
                "template_index": len(rows),
                "source_row": row_index + 1,
                "source_metric_group_raw": group,
                "source_legacy_metric_number_raw": legacy_metric_number,
                "metric": metric_name,
                "definition": metric_name,
                "purpose": clean(row[6]),
                "assumptions": clean(row[7]),
                "third_party_validation": clean(row[8]),
                "unit_raw": clean(row[28]),
                "comments": clean(row[29]),
                "blank_meaning": clean(row[30]),
            }
        )

    return rows


def locate_v3_1_actual_columns(sheet: XlsxSheet) -> dict[int, int]:
    values = sheet.values
    result: dict[int, int] = {}

    for column in range(len(values[8])):
        year = clean(values[8][column])
        quarter = clean(values[7][column])
        if year == 2023 and quarter in {1, 2, 3, 4}:
            result[int(quarter)] = column

    if sorted(result) != [1, 2, 3, 4]:
        raise AssertionError(
            f"Expected 2023 Q1-Q4 actual columns, found {result}"
        )
    return result


def parse_v3_2_template(sheet: XlsxSheet) -> tuple[list[dict[str, Any]], int, int, int]:
    values = sheet.values

    actual_columns = [
        column
        for column, value in enumerate(values[6])
        if clean(value) == "Actual"
    ]
    if len(actual_columns) != 1:
        raise AssertionError(
            f"Expected one v3.2 Actual column, found {actual_columns}"
        )

    actual_column = actual_columns[0]
    quarter_value = clean(values[7][actual_column])
    year_value = clean(values[8][actual_column])

    if isinstance(quarter_value, str):
        quarter_match = re.fullmatch(r"Q([1-4])", quarter_value)
        if not quarter_match:
            raise AssertionError(f"Invalid v3.2 quarter header {quarter_value!r}")
        reporting_quarter = int(quarter_match.group(1))
    else:
        reporting_quarter = int(quarter_value)

    reporting_year = int(year_value)

    rows = []
    group = None
    for row_index in range(9, len(values)):
        row = values[row_index]
        if clean(row[3]):
            group = clean(row[3])

        metric_name = clean(row[4])
        if metric_name is None:
            continue

        rows.append(
            {
                "template_index": len(rows),
                "source_row": row_index + 1,
                "source_metric_group_raw": group,
                "source_legacy_metric_number_raw": None,
                "metric": metric_name,
                "definition": metric_name,
                "purpose": clean(row[5]),
                "assumptions": clean(row[6]),
                "third_party_validation": clean(row[7]),
                "unit_raw": clean(row[12]),
                "comments": clean(row[13]),
                "blank_meaning": clean(row[14]),
            }
        )

    return rows, actual_column, reporting_year, reporting_quarter


def parse_v4_template(sheet: XlsxSheet) -> list[dict[str, Any]]:
    expected_headers = [
        "METRIC NUMBER",
        "METRIC",
        "DEFINITION",
        "PURPOSE",
        "ASSUMPTIONS MADE TO CONNECT METRIC TO PURPOSE",
        "THIRD-PARTY VALIDATION (IF ANY)",
        "UNIT(S)",
        "COMMENTS",
        "BLANK MEANING",
        "UTILITY ID",
        "REPORTING YEAR",
        "REPORTING QUARTER",
        "ACTUAL VALUE",
    ]
    actual_headers = [clean(value) for value in sheet.values[0][:13]]
    if actual_headers != expected_headers:
        raise ValueError(
            "The 2025 Table 3 header does not match Data Guidelines v4.01.\n"
            f"Expected: {expected_headers}\nFound: {actual_headers}"
        )

    rows = []
    for row_index in range(1, len(sheet.values)):
        row = sheet.values[row_index]
        if clean(row[1]) is None:
            continue

        rows.append(
            {
                "template_index": len(rows),
                "source_row": row_index + 1,
                "metric_number": int(row[0]),
                "metric": clean(row[1]),
                "definition": clean(row[2]),
                "purpose": clean(row[3]),
                "assumptions": clean(row[4]),
                "third_party_validation": clean(row[5]),
                "unit_raw": clean(row[6]),
                "comments": clean(row[7]),
                "blank_meaning": clean(row[8]),
                "utility_id": clean(row[9]),
                "reporting_year": int(row[10]),
                "reporting_quarter": int(row[11]),
                "actual_value_raw": row[12],
                "actual_value_column": 12,
            }
        )

    return rows


def build_crosswalk(
    v3_1_rows: list[dict[str, Any]],
    v3_2_rows: list[dict[str, Any]],
    v4_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(v3_1_rows) == len(v3_2_rows) == len(v4_rows) == 37):
        raise AssertionError(
            "Expected 37 Table 3 metrics in each schema generation; got "
            f"{len(v3_1_rows)}, {len(v3_2_rows)}, and {len(v4_rows)}."
        )

    v4_by_metric = {}
    for row in v4_rows:
        key = normalized_key(row["metric"])
        if key in v4_by_metric:
            raise AssertionError(f"Duplicate v4 metric name {row['metric']!r}")
        v4_by_metric[key] = row

    crosswalk = []
    for old_2023, old_2024 in zip(v3_1_rows, v3_2_rows):
        if normalized_key(old_2023["metric"]) != normalized_key(old_2024["metric"]):
            raise AssertionError(
                "The v3.1 and v3.2 Table 3 metric order/name differs at "
                f"template row {old_2023['template_index']}."
            )

        if normalized_key(old_2023["unit_raw"]) != normalized_key(old_2024["unit_raw"]):
            raise AssertionError(
                "The v3.1 and v3.2 Table 3 unit differs for "
                f"{old_2023['metric']!r}."
            )

        key = normalized_key(old_2023["metric"])
        if key not in v4_by_metric:
            raise AssertionError(
                f"Legacy metric {old_2023['metric']!r} has no v4.01 match."
            )

        v4_row = v4_by_metric[key]
        if normalized_key(old_2023["unit_raw"]) != normalized_key(v4_row["unit_raw"]):
            raise AssertionError(
                f"Unit changed unexpectedly for {old_2023['metric']!r}: "
                f"{old_2023['unit_raw']!r} -> {v4_row['unit_raw']!r}"
            )

        crosswalk.append(
            {
                "template_index": old_2023["template_index"],
                "metric_number": v4_row["metric_number"],
                "metric_group_crosswalk": old_2023["source_metric_group_raw"],
                "legacy_metric_number_crosswalk": old_2023[
                    "source_legacy_metric_number_raw"
                ],
                "metric": v4_row["metric"],
                "unit_canonical": v4_row["unit_raw"],
                "crosswalk_status": "exact_metric_name_and_unit",
            }
        )

    if len({item["metric_number"] for item in crosswalk}) != 37:
        raise AssertionError("v4.01 metric numbers are not unique.")

    return crosswalk


def crosswalk_by_metric(
    crosswalk: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        normalized_key(item["metric"]): item
        for item in crosswalk
    }


def create_output_row(
    *,
    source_record: dict[str, Any],
    mapping: dict[str, Any],
    raw_value: Any,
    number_format: str,
    reporting_year: int,
    reporting_quarter: int,
    schema_version: str,
    guideline_url: str,
    source: dict[str, Any],
    source_report_quarter: int,
    source_value_column: int,
    utility_id: str,
) -> list[Any]:
    canonical_value, value_conversion = canonicalize_value(
        raw_value,
        source_record["unit_raw"],
        number_format,
    )

    return [
        mapping["metric_number"],
        mapping["metric_group_crosswalk"],
        mapping["legacy_metric_number_crosswalk"],
        source_record.get("source_metric_group_raw"),
        source_record.get("source_legacy_metric_number_raw"),
        mapping["crosswalk_status"],
        clean(source_record["metric"]),
        clean(source_record["definition"]),
        clean(source_record["purpose"]),
        clean(source_record["assumptions"]),
        clean(source_record["third_party_validation"]),
        clean(source_record["unit_raw"]),
        clean(mapping["unit_canonical"]),
        parse_number(raw_value),
        canonical_value,
        value_conversion,
        clean(source_record["comments"]),
        clean(source_record["blank_meaning"]),
        utility_id,
        reporting_year,
        reporting_quarter,
        schema_version,
        source["revision"],
        source_report_quarter,
        source["name"],
        "Table 3",
        source_record["source_row"],
        (
            f"{excel_column_letter(source_value_column)}"
            f"{source_record['source_row']}"
        ),
        number_format,
        guideline_url,
    ]


def build_actual_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
    loaded: dict[tuple[int, int], XlsxSheet],
    crosswalk: list[dict[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    mapping_by_name = crosswalk_by_metric(crosswalk)

    # 2023: the Q4 v3.1 workbook contains all 2023 Q1-Q4 actual columns.
    source = selected[(2023, 4)]
    sheet = loaded[(2023, 4)]
    template = parse_v3_1_template(sheet)
    actual_columns = locate_v3_1_actual_columns(sheet)

    for record in template:
        mapping = mapping_by_name[normalized_key(record["metric"])]
        for quarter, column in sorted(actual_columns.items()):
            raw_value = sheet.values[record["source_row"] - 1][column]
            number_format = sheet.number_format(
                record["source_row"] - 1, column
            )
            rows.append(
                create_output_row(
                    source_record=record,
                    mapping=mapping,
                    raw_value=raw_value,
                    number_format=number_format,
                    reporting_year=2023,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2023][0],
                    guideline_url=GUIDELINES[2023][1],
                    source=source,
                    source_report_quarter=4,
                    source_value_column=column,
                    utility_id="SDG&E",
                )
            )

    # 2024: each v3.2 workbook contains only its subject-quarter actual.
    for quarter in (1, 2, 3, 4):
        source = selected[(2024, quarter)]
        sheet = loaded[(2024, quarter)]
        template, actual_column, year_header, quarter_header = parse_v3_2_template(
            sheet
        )

        if year_header != 2024 or quarter_header != quarter:
            raise AssertionError(
                f"{source['name']} reports {year_header} Q{quarter_header}, "
                f"not 2024 Q{quarter}."
            )

        for record in template:
            mapping = mapping_by_name[normalized_key(record["metric"])]
            raw_value = sheet.values[record["source_row"] - 1][actual_column]
            number_format = sheet.number_format(
                record["source_row"] - 1, actual_column
            )
            rows.append(
                create_output_row(
                    source_record=record,
                    mapping=mapping,
                    raw_value=raw_value,
                    number_format=number_format,
                    reporting_year=2024,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2024][0],
                    guideline_url=GUIDELINES[2024][1],
                    source=source,
                    source_report_quarter=quarter,
                    source_value_column=actual_column,
                    utility_id="SDG&E",
                )
            )

    # 2025: v4.01 workbooks are already long-form.
    for quarter in (1, 2, 3, 4):
        source = selected[(2025, quarter)]
        sheet = loaded[(2025, quarter)]
        records = parse_v4_template(sheet)

        for record in records:
            if (
                record["reporting_year"] != 2025
                or record["reporting_quarter"] != quarter
            ):
                raise AssertionError(
                    f"Incorrect reporting period in {source['name']} "
                    f"row {record['source_row']}."
                )

            mapping = mapping_by_name[normalized_key(record["metric"])]
            if record["metric_number"] != mapping["metric_number"]:
                raise AssertionError(
                    f"Metric-number mismatch for {record['metric']!r}."
                )

            number_format = sheet.number_format(
                record["source_row"] - 1,
                record["actual_value_column"],
            )
            rows.append(
                create_output_row(
                    source_record=record,
                    mapping=mapping,
                    raw_value=record["actual_value_raw"],
                    number_format=number_format,
                    reporting_year=2025,
                    reporting_quarter=quarter,
                    schema_version=GUIDELINES[2025][0],
                    guideline_url=GUIDELINES[2025][1],
                    source=source,
                    source_report_quarter=quarter,
                    source_value_column=record["actual_value_column"],
                    utility_id=record["utility_id"],
                )
            )

    if len(rows) != 37 * 4 * 3:
        raise AssertionError(
            f"Expected 444 unified actual records; got {len(rows)}."
        )

    return rows


def validate_quarterly_schemas(
    loaded: dict[tuple[int, int], XlsxSheet],
) -> None:
    # v3.2: all four quarter files must retain the same 37 metric names and units.
    reference_2024, _, _, _ = parse_v3_2_template(loaded[(2024, 4)])
    reference_2024_key = [
        (normalized_key(row["metric"]), normalized_key(row["unit_raw"]))
        for row in reference_2024
    ]

    for quarter in (1, 2, 3, 4):
        current, _, _, _ = parse_v3_2_template(loaded[(2024, quarter)])
        current_key = [
            (normalized_key(row["metric"]), normalized_key(row["unit_raw"]))
            for row in current
        ]
        if current_key != reference_2024_key:
            raise AssertionError(
                f"2024 Q{quarter} Table 3 metric/unit structure differs from Q4."
            )

    # v4.01: metric number, descriptive fields, and unit must be stable across revisions.
    reference_2025 = parse_v4_template(loaded[(2025, 4)])
    stable_fields = (
        "metric_number",
        "metric",
        "definition",
        "purpose",
        "assumptions",
        "third_party_validation",
        "unit_raw",
        "utility_id",
    )
    reference_key = [
        tuple(clean(row[field]) for field in stable_fields)
        for row in reference_2025
    ]

    for quarter in (1, 2, 3, 4):
        current = parse_v4_template(loaded[(2025, quarter)])
        current_key = [
            tuple(clean(row[field]) for field in stable_fields)
            for row in current
        ]
        if current_key != reference_key:
            raise AssertionError(
                f"2025 Q{quarter} Table 3 descriptive schema differs from Q4."
            )


def write_csv(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def crosswalk_rows(
    crosswalk: list[dict[str, Any]]
) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "metric_number",
        "metric_group_crosswalk",
        "legacy_metric_number_crosswalk",
        "metric",
        "unit_canonical",
        "crosswalk_status",
    ]
    rows = [
        [
            item["metric_number"],
            item["metric_group_crosswalk"],
            item["legacy_metric_number_crosswalk"],
            item["metric"],
            item["unit_canonical"],
            item["crosswalk_status"],
        ]
        for item in crosswalk
    ]
    return headers, rows


def build_workbook(
    output_path: Path,
    actual_rows: list[list[Any]],
    crosswalk: list[dict[str, Any]],
    selected: dict[tuple[int, int], dict[str, Any]],
) -> None:
    workbook = Workbook.create()
    readme = workbook.worksheets.add("README")
    actuals = workbook.worksheets.add("Unified Actuals")
    crosswalk_sheet = workbook.worksheets.add("Metric Crosswalk")
    changes = workbook.worksheets.add("Schema Changes")

    conversion_count = sum(
        1
        for row in actual_rows
        if row[HEADERS.index("value_conversion")] is not None
    )

    readme_rows = [
        ["SDG&E Table 3 Unified Dataset, 2023–2025", "", "", ""],
        ["Actual rows", len(actual_rows), "Metrics per quarter", 37],
        ["Reporting periods", "2023 Q1–Q4, 2024 Q1–Q4, 2025 Q1–Q4", "", ""],
        [
            "Projection treatment",
            "Projection columns are intentionally excluded.",
            "",
            "",
        ],
        [
            "2023 extraction",
            "All four 2023 quarters are read from the selected Q4 v3.1 workbook.",
            "",
            "",
        ],
        [
            "2024 extraction",
            "Each selected v3.2 quarterly workbook contributes its subject quarter.",
            "",
            "",
        ],
        [
            "2025 extraction",
            "Each selected v4.01 workbook is already one record per metric and quarter.",
            "",
            "",
        ],
        [
            "Percent normalization",
            f"{conversion_count} percent-formatted Excel cells were converted "
            "from stored fractions to percentage points; raw values are retained.",
            "",
            "",
        ],
        [
            "Crosswalk validation",
            "All 37 legacy metric names and units match the 2025 v4.01 metrics.",
            "Unexpected changes",
            0,
        ],
        [
            "Important legacy mapping",
            "The SDG&E v3.1/v3.2 'Definition' cell matches the v4.01 METRIC "
            "and DEFINITION text. The legacy group heading and number are "
            "preserved separately.",
            "",
            "",
        ],
        ["", "", "", ""],
        ["Official source", "Applicable period", "URL", "Verified change"],
        [
            "Data Guidelines v3.1",
            "2023",
            GUIDELINES[2023][1],
            "Table 3 fields include metric, definition, purpose, assumptions, "
            "validation, quarter values, units, comments, and blank meaning.",
        ],
        [
            "Data Guidelines v3.2",
            "2024",
            GUIDELINES[2024][1],
            "Same Table 3 field definitions as v3.1.",
        ],
        [
            "v3.2 Template Change Log",
            "2024 transition",
            V3_2_CHANGELOG_URL,
            "Historical actual columns removed; only the current reporting "
            "quarter remains in each workbook.",
        ],
        [
            "Data Guidelines v4.01",
            "2025",
            GUIDELINES[2025][1],
            "Long-form quarterly fields add metric number, utility ID, "
            "reporting year, reporting quarter, and actual value.",
        ],
        [
            "v4.0 Template Changelog",
            "2025 transition",
            V4_CHANGELOG_URL,
            "Table 3 projection worksheets moved to Annual-WMP workbooks.",
        ],
        ["", "", "", ""],
        ["Selected source file", "Year", "Quarter", "Revision"],
    ]
    for key in sorted(selected):
        source = selected[key]
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
        "font": {"bold": True, "color": "#FFFFFF", "font_size": 15},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "row_height": 30,
    }
    for row_index, row in enumerate(readme_rows, start=1):
        if row[0] in {"Official source", "Selected source file"}:
            readme.get_range(f"A{row_index}:D{row_index}").format = {
                "fill": "#1D4ED8",
                "font": {"bold": True, "color": "#FFFFFF"},
                "horizontal_alignment": "center",
                "vertical_alignment": "center",
                "wrap_text": True,
            }
    readme.get_range(f"A1:D{len(readme_rows)}").format.wrap_text = True
    for column, width in zip(("A", "B", "C", "D"), (34, 56, 68, 50)):
        readme.get_range(
            f"{column}1:{column}{len(readme_rows)}"
        ).format.column_width = width
    readme.freeze_panes.freeze_rows(1)

    actuals.get_range_by_indexes(
        0, 0, 1, len(HEADERS)
    ).values = [HEADERS]
    chunk_size = 250
    for start in range(0, len(actual_rows), chunk_size):
        chunk = actual_rows[start : start + chunk_size]
        actuals.get_range_by_indexes(
            start + 1, 0, len(chunk), len(HEADERS)
        ).values = chunk

    last_column = excel_column_letter(len(HEADERS) - 1)
    last_row = len(actual_rows) + 1
    actuals.get_range(f"A1:{last_column}1").format = {
        "fill": "#0F766E",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 34,
    }
    actuals.freeze_panes.freeze_rows(1)
    actuals.freeze_panes.freeze_columns(7)
    actuals.get_range(f"G1:G{last_row}").format.column_width = 48
    for column in ("H", "I", "J", "K", "Q", "R"):
        actuals.get_range(
            f"{column}1:{column}{last_row}"
        ).format.column_width = 42
        actuals.get_range(
            f"{column}2:{column}{last_row}"
        ).format.wrap_text = True
    for column in ("A", "B", "C", "D", "E", "F", "L", "M", "P", "S", "V", "W"):
        actuals.get_range(
            f"{column}1:{column}{last_row}"
        ).format.column_width = 20
    actuals.get_range(f"N2:O{last_row}").format.number_format = "0.########"
    actuals.get_range(f"Y1:Y{last_row}").format.column_width = 42
    actuals.get_range(f"AD1:AD{last_row}").format.column_width = 64

    conversion_column = excel_column_letter(HEADERS.index("value_conversion"))
    actuals.get_range(
        f"{conversion_column}2:{conversion_column}{last_row}"
    ).conditional_formats.add_custom(
        f'=${conversion_column}2<>""',
        {"fill": "#DBEAFE", "font": {"color": "#1E3A8A"}},
    )

    cross_headers, cross_rows = crosswalk_rows(crosswalk)
    crosswalk_sheet.get_range_by_indexes(
        0, 0, 1, len(cross_headers)
    ).values = [cross_headers]
    crosswalk_sheet.get_range_by_indexes(
        1, 0, len(cross_rows), len(cross_headers)
    ).values = cross_rows
    crosswalk_sheet.get_range("A1:F1").format = {
        "fill": "#1D4ED8",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
    }
    crosswalk_sheet.freeze_panes.freeze_rows(1)
    for column, width in zip(("A", "B", "C", "D", "E", "F"), (20, 38, 24, 58, 24, 28)):
        crosswalk_sheet.get_range(
            f"{column}1:{column}{len(cross_rows) + 1}"
        ).format.column_width = width
    crosswalk_sheet.get_range(
        f"A1:F{len(cross_rows) + 1}"
    ).format.wrap_text = True

    change_rows = [
        ["Change", "2023 v3.1", "2024 v3.2", "2025 v4.01", "Converter action"],
        [
            "Actual-value layout",
            "Historical quarter columns, including 2023 Q1–Q4",
            "Only the subject quarter remains",
            "One long-form record with REPORTING YEAR, REPORTING QUARTER, ACTUAL VALUE",
            "Unpivot 2023; append each 2024/2025 quarter.",
        ],
        [
            "Metric identifier",
            "SDG&E legacy number such as 1.a.",
            "Legacy number column removed from SDG&E workbook",
            "Standard METRIC NUMBER such as 1030000000",
            "Use v4.01 metric number and preserve the 2023 legacy number as crosswalk metadata.",
        ],
        [
            "Metric/group placement",
            "Group heading in Metric column; unique metric text in Definition",
            "Same SDG&E arrangement",
            "Unique metric text in both METRIC and DEFINITION",
            "Map legacy Definition to unified metric/definition; preserve the group separately.",
        ],
        [
            "Projection location",
            "Projection columns in quarterly workbook",
            "Projection columns in quarterly workbook",
            "Projections belong in Annual-WMP workbook",
            "Drop all projections as requested.",
        ],
        [
            "Percent storage",
            "Some percentage-point values and some Excel-formatted fractions",
            "Same mixed Excel storage",
            "Same mixed Excel storage across revisions",
            "Preserve raw values; use cell number format to normalize fractions to percentage points.",
        ],
        [
            "Descriptive fields",
            "Purpose/assumption fields are frequently blank in SDG&E data",
            "Frequently blank",
            "Populated and stable across Q1–Q4 revisions",
            "Preserve the value from each source record; do not backfill historical text.",
        ],
    ]
    changes.get_range_by_indexes(
        0, 0, len(change_rows), 5
    ).values = change_rows
    changes.get_range("A1:E1").format = {
        "fill": "#0F766E",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 32,
    }
    changes.get_range(f"A1:E{len(change_rows)}").format.wrap_text = True
    for column, width in zip(("A", "B", "C", "D", "E"), (28, 45, 45, 52, 58)):
        changes.get_range(
            f"{column}1:{column}{len(change_rows)}"
        ).format.column_width = width
    changes.freeze_panes.freeze_rows(1)

    SpreadsheetFile.export_xlsx(workbook).save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine SDG&E Table 3 actual values for 2023-2025 into a "
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
        default="/mnt/data/table3_output",
        help="Directory for CSV, XLSX, and validation outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover_sources(input_dir)
    loaded = {
        key: read_xlsx_sheet(source["path"], "Table 3")
        for key, source in selected.items()
    }

    validate_quarterly_schemas(loaded)

    v3_1_rows = parse_v3_1_template(loaded[(2023, 4)])
    v3_2_rows, _, _, _ = parse_v3_2_template(loaded[(2024, 4)])
    v4_rows = parse_v4_template(loaded[(2025, 4)])
    crosswalk = build_crosswalk(v3_1_rows, v3_2_rows, v4_rows)

    actual_rows = build_actual_rows(selected, loaded, crosswalk)

    actual_csv = output_dir / "sdge_table3_2023_2025_unified_actuals.csv"
    crosswalk_csv = output_dir / "sdge_table3_metric_crosswalk.csv"
    workbook_path = output_dir / "sdge_table3_2023_2025_unified.xlsx"
    validation_path = output_dir / "validation_summary.json"

    write_csv(actual_csv, HEADERS, actual_rows)
    cross_headers, cross_rows = crosswalk_rows(crosswalk)
    write_csv(crosswalk_csv, cross_headers, cross_rows)

    conversion_count = sum(
        1
        for row in actual_rows
        if row[HEADERS.index("value_conversion")] is not None
    )
    summary = {
        "actual_rows": len(actual_rows),
        "metrics_per_quarter": 37,
        "reporting_periods": 12,
        "crosswalk_rows": len(crosswalk),
        "crosswalk_status_counts": dict(
            Counter(item["crosswalk_status"] for item in crosswalk)
        ),
        "percent_fraction_conversions": conversion_count,
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

    build_workbook(workbook_path, actual_rows, crosswalk, selected)

    print(f"Created: {actual_csv}")
    print(f"Created: {crosswalk_csv}")
    print(f"Created: {workbook_path}")
    print(f"Created: {validation_path}")
    print(f"Actual rows: {len(actual_rows)}")
    print(f"Percent-formatted fraction conversions: {conversion_count}")


if __name__ == "__main__":
    main()
