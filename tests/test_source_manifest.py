import tempfile
import unittest
from pathlib import Path

from retrieval.source_manifest import (
    SourceManifest,
    SourceRecord,
    audit_local_files,
    filenames_for_document_scope,
    validate_intake_filename,
)


class SourceManifestTests(unittest.TestCase):
    def test_human_wmp_scope_resolves_both_filename_aliases(self):
        filenames = filenames_for_document_scope(
            "SDG&E 2026-2028 Wildfire Mitigation Plan filing",
            "2026-2028",
        )

        self.assertIn("SDG&E_2026-2028_Base-WMP_R2.pdf", filenames)
        self.assertIn(
            "sdge__wmp__2026-2028__r2__2025-05-23.pdf", filenames
        )
        self.assertNotIn(
            "SDG&E_2023-2023_Base-WMP_R5-redacted.pdf", filenames
        )

    def test_manifest_resolves_stable_role_to_current_filename(self):
        manifest = SourceManifest((
            SourceRecord(
                source_id="sdge-wmp-2023-2025",
                utility="sdge",
                document_role="wmp_2023_2025",
                document_type="pdf",
                original_filename="old human filename.pdf",
                canonical_filename="sdge__wmp__2023-2025__r5__2023-03-27.pdf",
                cycle_start=2023,
                cycle_end=2025,
                version="r5",
                effective_date="2023-03-27",
            ),
        ))
        self.assertEqual(
            manifest.filenames_for_role("wmp_2023_2025"),
            ("old human filename.pdf",),
        )

    def test_manifest_rejects_duplicate_source_ids(self):
        record = SourceRecord(
            source_id="same",
            utility="sdge",
            document_role="wmp",
            document_type="pdf",
            original_filename="a.pdf",
            canonical_filename="sdge__wmp__2023-2025__v1__2023-01-01.pdf",
        )
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            SourceManifest((record, record))

    def test_new_intake_requires_canonical_names(self):
        validate_intake_filename(
            "sdge__wmp__2026-2028__v1__2025-05-23.pdf", "pdf"
        )
        validate_intake_filename("sdge__qdr_metrics__2025__v3.xlsx", "excel")
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_intake_filename("FINAL revised plan.pdf", "pdf")

    def test_file_audit_reports_missing_and_untracked_without_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "known.pdf").write_text("pdf")
            (root / "untracked.pdf").write_text("pdf")
            manifest = SourceManifest((
                SourceRecord(
                    source_id="known",
                    utility="sdge",
                    document_role="wmp",
                    document_type="pdf",
                    original_filename="known.pdf",
                    canonical_filename="sdge__wmp__2023-2025__v1__2023-01-01.pdf",
                ),
                SourceRecord(
                    source_id="missing",
                    utility="sdge",
                    document_role="qdr_metrics",
                    document_type="csv",
                    original_filename="missing.csv",
                    canonical_filename="sdge__qdr_table_01__cleaned__v1.csv",
                ),
            ))
            report = audit_local_files(manifest, pdf_dir=root, csv_dir=root)
        self.assertEqual(report["missing"], ["missing.csv"])
        self.assertEqual(report["untracked"], ["untracked.pdf"])


if __name__ == "__main__":
    unittest.main()
