
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TABLE_PATTERN = re.compile(r"^sdge_table0*(\d+)", re.IGNORECASE)


def output_filename(source_name: str) -> str:
    """
    Convert a source filename such as:
        sdge_table2_2023_2025_unified.csv
        sdge_table02.csv
    into:
        sdge_table02_rag_ready.csv
    """
    match = TABLE_PATTERN.match(source_name)
    if not match:
        raise ValueError(
            f"Filename does not begin with 'sdge_table<number>': {source_name}"
        )

    table_number = int(match.group(1))
    return f"sdge_table{table_number:02d}_rag_ready.csv"


def remove_raw_columns(
    source_path: Path,
    destination_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Copy one CSV while removing every column whose header ends with '_raw'
    (case-insensitive).

    Returns:
        (columns_kept, columns_removed, removed_column_names)
    """
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination_path}. "
            "Use --overwrite to replace it."
        )

    # utf-8-sig reads both ordinary UTF-8 and UTF-8 files with a BOM.
    with source_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source_file:
        reader = csv.reader(source_file)

        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {source_path}") from exc

        keep_indexes = [
            index
            for index, header in enumerate(headers)
            if not header.strip().lower().endswith("_raw")
        ]
        removed_columns = [
            header
            for header in headers
            if header.strip().lower().endswith("_raw")
        ]

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Write UTF-8 with BOM for good Excel compatibility.
        with destination_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as destination_file:
            writer = csv.writer(destination_file)
            writer.writerow([headers[index] for index in keep_indexes])

            for row_number, row in enumerate(reader, start=2):
                # Pad short rows so indexed access remains safe. Extra cells are
                # rejected because they do not correspond to a named column.
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    raise ValueError(
                        f"{source_path.name}, row {row_number}: "
                        f"{len(row)} cells but {len(headers)} headers."
                    )

                writer.writerow([row[index] for index in keep_indexes])

    return len(keep_indexes), len(removed_columns), removed_columns


def process_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_dir}")

    source_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and TABLE_PATTERN.match(path.name)
        and not path.name.lower().endswith("_rag_ready.csv")
    )

    if not source_files:
        raise FileNotFoundError(
            f"No CSV files beginning with 'sdge_table' found in {input_dir}"
        )

    # The requested output convention creates one filename per table number.
    # Stop rather than silently overwrite when multiple input files belong to
    # the same table.
    output_names: dict[str, Path] = {}
    for source_path in source_files:
        name = output_filename(source_path.name)
        if name in output_names:
            raise ValueError(
                "Multiple input files would produce the same output name:\n"
                f"  {output_names[name].name}\n"
                f"  {source_path.name}\n"
                f"Both map to {name}. Keep one primary CSV per table or "
                "rename the inputs before running the program."
            )
        output_names[name] = source_path

    output_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    print(f"Input folder:  {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print()

    for output_name, source_path in sorted(output_names.items()):
        destination_path = output_dir / output_name
        kept, removed, removed_names = remove_raw_columns(
            source_path,
            destination_path,
            overwrite=overwrite,
        )
        total_removed += removed

        print(f"{source_path.name}")
        print(f"  -> {destination_path.name}")
        print(f"  kept {kept} columns; removed {removed}")
        if removed_names:
            print(f"  removed: {', '.join(removed_names)}")
        else:
            print("  removed: none")
        print()

    print(
        f"Finished: {len(source_files)} files processed; "
        f"{total_removed} raw columns removed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create RAG-ready SDG&E CSVs by removing every column whose "
            "header ends with '_raw'."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("cleaned_csv"),
        help="Folder containing source CSVs. Default: cleaned_csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rag_ready_csv"),
        help=(
            "Folder for generated files. Default: rag_ready_csv. "
            "A separate default is used because the input folder is already "
            "named cleaned_csv."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing RAG-ready files.",
    )
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "Input and output folders must be different to avoid mixing "
            "source and generated files."
        )

    process_folder(
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
