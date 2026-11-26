"""Unit tests for Step G: trained CRF model catches unseen canary names via context while preserving clinical traps."""
from raksha.detect.crf_model import CRFPredictor
from raksha.detect.identifiers import redact_spans
from raksha.detect.protected import ClinicalLexicon, resolve_pii_vs_protected


def test_step_g_crf_catches_unseen_canary_via_context_and_preserves_clinical():
    crf = CRFPredictor()
    assert crf.available

    # Completely unseen names and unseen trigger labels ("Beneficiary:", "Caretaker:")
    txt = (
        "Beneficiary: Qwertyz Xylarov\n"
        "Caretaker: Zorawar Maken\n"
        "Dr. Ramesh Srivastava reviewed. C/O fever for 3 days. "
        "K/C/O Parkinson's disease. Tab Paracetamol 650 mg 1-0-1 x 5 days."
    )
    lex = ClinicalLexicon()
    prot = lex.find_spans(txt)
    crf_spans = crf.predict_spans(txt)
    out = redact_spans(txt, resolve_pii_vs_protected(crf_spans, prot))

    assert "Qwertyz" not in out
    assert "Xylarov" not in out
    assert "Zorawar" not in out
    assert "Dr. Ramesh Srivastava" in out
    assert "Parkinson's disease" in out
    assert "Paracetamol 650 mg" in out
