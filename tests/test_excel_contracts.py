from __future__ import annotations

from pathlib import Path

from retrieval.ingest.excel.contracts import load_contracts


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_contract_path_is_independent_of_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("SDGE_EXCEL_CONTRACTS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    load_contracts.cache_clear()

    try:
        contracts = load_contracts()
    finally:
        load_contracts.cache_clear()

    assert contracts.tables


def test_unix_launcher_exports_absolute_excel_contract_path():
    launcher = (REPO_ROOT / "backend" / "run_dev.sh").read_text()

    assert "export SDGE_EXCEL_CONTRACTS_PATH=" in launcher
    assert "/config/excel_contracts.yaml" in launcher
