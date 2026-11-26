"""Unit tests for Step H: independent post-export leakage scanner."""
import json
from pathlib import Path

from raksha.qa.leak_scanner import scan_and_remediate


def _setup_mock_output(tmp_path: Path, rel: str, content: str) -> Path:
    san = tmp_path / "sanitized" / rel
    san.parent.mkdir(parents=True, exist_ok=True)
    san.write_text(content, encoding="utf-8")
    (tmp_path / "records.jsonl").write_text(
        json.dumps({"file": f"sanitized/{rel}", "loc": "doc", "patient": None}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "schema": "raksha-report/1",
                "files": [{"input_sha256": "abc", "output": f"sanitized/{rel}", "status": "sanitized"}],
                "counts": {"files_total": 1, "files_sanitized": 1, "files_quarantined": 0, "records_processed": 1, "records_linked": 0},
            }
        ),
        encoding="utf-8",
    )
    return san


def test_step_h_rescrubs_residual_leak_and_preserves_doctor_clinical(tmp_path: Path):
    san = _setup_mock_output(
        tmp_path,
        "notes/doc1.txt",
        "Patient Mahesh Chauhan seen by Dr. Ramesh Srivastava. Platelet Count 240124 /uL. Mob: 9876543210.",
    )
    summary = scan_and_remediate(
        tmp_path,
        known_surfaces={"mahesh chauhan": "PATIENT_NAME", "ramesh srivastava": "PATIENT_NAME"},
        doctor_surfaces={"ramesh srivastava"},
    )
    cleaned = san.read_text(encoding="utf-8")
    assert "Mahesh Chauhan" not in cleaned
    assert "9876543210" not in cleaned
    assert "Dr. Ramesh Srivastava" in cleaned
    assert "240124 /uL" in cleaned
    assert summary.files_rescrubbed == 1
    assert summary.files_quarantined_qa == 0


def test_step_h_quarantines_unscrubbable_filename_leak(tmp_path: Path):
    san = _setup_mock_output(
        tmp_path,
        "notes/Mahesh_Chauhan_note.txt",
        "Clean text inside.",
    )
    summary = scan_and_remediate(
        tmp_path,
        known_surfaces={"mahesh chauhan": "PATIENT_NAME"},
        doctor_surfaces=set(),
    )
    assert not san.exists()
    assert summary.files_quarantined_qa == 1
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["files"][0]["status"] == "quarantined"
    assert rep["counts"]["files_sanitized"] == 0
    assert rep["counts"]["files_quarantined"] == 1
    assert (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip() == ""


def test_step_h_missing_secret_refused_and_deterministic_run(tmp_path: Path, monkeypatch):
    from raksha.pipeline import run_pipeline

    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "note.txt").write_text("Name: Zubin Khosravi\nC/O fever for 3 days.\nParacetamol 650 mg.", encoding="utf-8")

    monkeypatch.delenv("RAKSHA_SECRET", raising=False)
    out_bad = tmp_path / "out_bad"
    assert run_pipeline(inp, out_bad) == 1

    monkeypatch.setenv("RAKSHA_SECRET", "test-secret-key")
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    assert run_pipeline(inp, out1) == 0
    assert run_pipeline(inp, out2) == 0
    assert (out1 / "sanitized" / "note.txt").read_bytes() == (out2 / "sanitized" / "note.txt").read_bytes()
    assert (out1 / "records.jsonl").read_bytes() == (out2 / "records.jsonl").read_bytes()
    txt = (out1 / "sanitized" / "note.txt").read_text(encoding="utf-8")
    assert "Zubin Khosravi" not in txt
    assert "Paracetamol 650 mg" in txt


def test_step_h_txt_starting_with_bracket_timestamp_not_quarantined_as_json(tmp_path: Path):
    san = _setup_mock_output(
        tmp_path,
        "lab/whatsapp_forward_0003.txt",
        "[30/05/26, 9:32 pm] [RELATIVE_NAME]: Report of [PATIENT_NAME]\nKoi problem ho to call karo [PHONE]",
    )
    summary = scan_and_remediate(
        tmp_path,
        known_surfaces={"asha rajput": "PATIENT_NAME"},
        doctor_surfaces=set(),
    )
    assert san.is_file()
    assert summary.files_quarantined_qa == 0


