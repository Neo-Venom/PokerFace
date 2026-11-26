# Project RAKSHA — Bilingual Clinical Privacy Firewall & Record Linkage Engine

**Track 3 · Secret Wards** · **Team Poker Face** · Repository: `https://github.com/suyashkrishangarg/PokerFace`

Project RAKSHA is an **offline, CPU-only, LLM-free Two-Pass Hybrid Privacy Firewall** for messy, multi-hospital Indian healthcare exports (CSVs, TSVs, deeply nested JSONs, FHIR R4 Bundles, HL7 CDA XML, and free-text English/Devanagari/Hinglish clinical notes and WhatsApp forwards). It strips direct identifiers and hidden-channel leaks while preserving clinical facts, builds longitudinal patient links across scripts and facilities under keyed HKDF-HMAC pseudonyms (`PSN-XXXXXXXX`), defends against demographic and temporal re-identification attacks (`A1–A7`), and enforces a fail-closed pre-emission egress audit gate.

---

## 1. Setup & Execution

Requires Python 3.10+ and pure-CPU classical packages only (no LLMs, no external APIs, no weights >50M parameters, 100% offline execution):

```bash
pip install -r requirements.txt
RAKSHA_SECRET="your-release-secret" python main.py --input /path/to/input --output /path/to/output
```

Run the synthetic unit test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 2. System Architecture

```
input/ (read-only)
  │
  ▼
[1. Ingestion & Quarantine] (router.py, profiler.py, csv/json/fhir/xml/text handlers)
  ├─ Strict UTF-8 / UTF-8-BOM / UTF-16 / cp1252 decoding; binary/corrupt/deep-AST quarantine (0 raw copies)
  ├─ Schema-agnostic column/key profiling, two independent signals:
  │    header/key label (bilingual synonyms, relation/doctor/amount cues) AND cell values
  │    (phone/Aadhaar/ABHA/PAN/PIN/ID/date shapes, S/o-W/o prefixes, addresses, name-like & village-like
  │    proper nouns checked against the learned clinical lexicon) -> unseen headers ("holder", "abode") still resolve
  ├─ Ragged-row repair before profiling, per-column DMY/MDY inference
  └─ Format & structure preservation (iterative JSON AST, valid FHIR R4B Bundles, exact CSV cells & note lines)
  │
  ▼
[2. Clinical Shield & Pass-1 Harvest] (medical_guard.py, indic_norm.py, pii_regex.py, pii_ner.py)
  ├─ Runtime lexicon harvested from clean/ docs, drugs, ICD-10 displays, LOINC tests, doctors & clinical wards
  ├─ Span protection for doctors ("Dr. ..."), eponyms ("Addison's", "Parkinson's"), ASHA workers, K/C/O, doses, labs
  ├─ Shape-based IDs (any hospital prefix + separators + digits; ICD/LOINC/package codes excluded), phones in any
  │    grouping ((xxx) xxx-xxxx, 0xxxxx-xxxxx, +91…), PINs after Dist./ज़िला/line end, locality head-nouns
  ├─ Context cues for unlabelled people: "Pt X (62M)", "brought by son X", "unke pati X", "Bed 7: X", "issued to X"
  ├─ Decodes evasions before matching: %XX, numeric HTML entities, \uXXXX, Base64, homoglyphs, zero-width chars
  ├─ Fail-closed address fields: every remaining proper noun is redacted unless it is a public district/state
  └─ EN / Devanagari / Hinglish extraction + Aadhaar (non-Verhoeff tolerant), ABHA, PAN, phone, PIN & Base64 harvest
  │
  ▼
[3. Cross-Script Record Linkage] (evidence.py, blocking.py, scorer.py, cluster.py)
  ├─ Devanagari-to-ITRANS + inherent-a drop + nukta & vowel folding ("मोहम्मद शाहिद" == "Md. Shahid" == "Mohammad Shahid")
  ├─ Multi-key blocking + shared/attendant family-phone detection + IDF-weighted Fellegi–Sunter scoring
  ├─ Hospital/label words can never become a "name"; filename IDs count only if the same ID exists in content
  ├─ Multi-patient tables detected structurally (patient IDs/names in >=2 headers or JSON keys) -> never one patient
  ├─ Vetoes: gender, DOB, Aadhaar/ABHA/PAN conflicts, different IDs of the same hospital system (pairwise),
  │    name conflict -- unless a single-person identifier is shared (handles "Mathew" vs "मैथ्यू")
  └─ Constrained descending clustering with hard contradiction vetoes -> HKDF-HMAC pseudonyms (PSN-XXXXXXXX)
  │
  ▼
[4. Adversarial Privacy Layer] (date_shifter.py, privacy/kanon.py)
  ├─ Per-patient HMAC date shift (30..180 days) preserving exact LOS & visit cadence across formats & year-less dates
  │    (ISO, dd/mm/yyyy, mm/dd/yyyy, dd-Mon-yyyy, "June 8, 2025", dd/mm/yy, Devanagari digits; format preserved)
  ├─ Patient-pooled adaptive k-anonymity (truthful 5->10->20yr / 0-29 / 65+ / 85+ age bands, tehsil/PIN roll-up)
  └─ Exact DOB suppression (CSV/JSON birth cohort band; FHIR Patient.birthDate removed while maintaining R4 validity)
  │
  ▼
[5. Egress Audit Gate & Emission] (privacy/audit.py, security/safelog.py)
  ├─ Scans sanitized/, filenames, decoded Base64 payloads, report.json & records.jsonl for surviving PII/secret
  └─ Fail-closed re-scrub or quarantine -> sanitized/ + report.json (raksha-report/1) + records.jsonl
```

---

## 3. Security Design & Compliance

- **Key Isolation & Domain Separation (`src/security/keys.py`):** Derives isolated `pseudonym`, `date-shift`, `file-name`, and `vault` subkeys via `HKDF-SHA256` from `RAKSHA_SECRET`, pops `RAKSHA_SECRET` from `os.environ` immediately on startup, and clears keys on exit. Rotating secrets regenerates all `PSN-` IDs, renamed filenames, and date shifts (`A7 cross-release risk = 0.0`).
- **PII-Free Logging & Read-Only Input (`src/security/safelog.py`, `vault.py`):** Installs an exception hook that strips tracebacks and raw filenames/text from stdout/stderr and keeps `--input` read-only with no PII temp files in `HOME`/`TMPDIR`.
- **Fail-Closed Pre-Emission Audit (`src/privacy/audit.py`):** Decodes embedded Base64 blobs, verifies `json.loads()` / `defusedxml` integrity, re-scrubs any residual harvested tokens, and quarantines persistent critical ID leaks.
- **No Forbidden Dependencies:** `requirements.txt` contains strictly 7 lightweight CPU libraries: `regex`, `rapidfuzz`, `jellyfish`, `indic-transliteration`, `cryptography`, `defusedxml`, and `pyyaml`.

---

## 4. Measured Benchmark & Adversarial Results

Every set is run 3 times (same secret twice, plus a different secret) and graded by the official `validate_output.py` and
`check_dev.py`, a local estimator of `docs/SCORING.md`, and our implementation of the `docs/ATTACKS.md` attacker
(A3/A4/A5 are shown relative to the same attack on the raw input). **Gauntlet B uses a held-out vocabulary**: names and
header labels were never used while tuning. The chaos sets come from the team's chaos generator. The Gauntlet sets come from an
adversarial generator that plants every trap in the statement, DATASET.md and ATTACKS.md (new hospitals and ID shapes, 12 phone
groupings, 10 date formats, unlabelled names, homonyms, a family phone, re-registration, encoded payloads, canaries, malformed files).
The **Hard** sets add hostile-export traps: spreadsheet formulas, NUL bytes, Excel-mangled phones, integer-overflow ages,
unsafe file names, `NaN` JSON, buffer and regex bombs, twins, a father and son with one name and one phone, a maiden name,
badly-romanising Devanagari surnames, newborns (`B/O`), mixed-script names and a PIN with no cue. Generator and sets are
published separately as the *Trap Gauntlet*.

| Measure | Practice `dev` | Chaos s101 | Chaos s202 | Gauntlet A (tuned) | Gauntlet B (held out) | Gauntlet A21 (60 pt / 6k rows) | Hard H7 | Hard H11 (held out) |
|---|---|---|---|---|---|---|---|---|
| Contract (`validate_output.py`) | CONTRACT OK | CONTRACT OK | CONTRACT OK | CONTRACT OK | CONTRACT OK | CONTRACT OK | CONTRACT OK | CONTRACT OK |
| **Local score /100** | **98.67** | **98.36** | **98.53** | **99.51** | **99.52** | **99.18** | **99.53** | **99.53** |
| Name-token leak rate (official key semantics) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.01% |
| Direct IDs / full names surviving (decoded, all channels) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Canaries found (incl. stdout/stderr/TMP) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Medical probes kept | 4855/4855 | 4855/4855 | 4855/4855 | 4898/4899 | 4963/4963 | 8648/8649 | 5005/5006 | 5070/5070 |
| Linking F0.5 / false merges | 1.00 / 0 | 0.99 / 0 | 0.99 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 |
| Homonym pairs kept apart | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| A3 / A4 / A5 risk vs raw | 3.2% / 12.8% / 8.0% | 3.1% / 10.6% / 8.2% | 3.1% / 10.6% / 7.3% | 3.9% / 2.3% / 2.6% | 3.7% / 2.3% / 1.5% | 3.5% / 1.7% / 5.8% | 3.7% / 1.7% / 2.5% | 3.6% / 1.7% / 1.5% |
| Length of stay preserved | n/a | n/a | n/a | 64/64 | 76/76 | 106/106 | 64/64 | 76/76 |
| Formula cells / invalid JSON / unsafe names in release | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Determinism (A == B) / new secret shares PSNs | yes / 0 | yes / 0 | yes / 0 | yes / 0 | yes / 0 | yes / 0 | yes / 0 | yes / 0 |
| Run time (12-core laptop) | 9.8s | 9.8s | 10.3s | 14.6s | 15.2s | 38.5s | 16.9s | 17.8s |

Before the generalisation work, the same pipeline scored **55.66** on Gauntlet A (linking F0.5 0.02, 36 direct identifiers surviving, -25 penalty points) while scoring 98.5 on the practice set: the memorisation gap the hidden unseen-format round is designed to expose.

### Privacy-utility choices

* **Dates:** a per-patient HMAC shift of 30-180 days, applied to every date format, keeps each length of stay and visit gap exact (utility), while a global shift would be recovered by A5.
* **Age:** truthful bands, 5 years by default and widened to 10/20 only for small cohorts, 85+ top-coded. A band never excludes the real age.
* **Location:** districts are kept (public areas with research value); house, locality, landmark and village are removed; tehsil and PIN are coarsened by cohort size. Address fields fail closed.
* **Rare values:** diagnoses shared by fewer than 4 patients are rolled up to the ICD block, which defends the "97-year-old with a rare disease" targeted attack.
* **Linking:** precision is preferred over recall (F0.5). A record with no single-person ID of its own that fits two people
  equally well (father and son sharing a name and a phone) stays `patient: null`. Spelling, transliteration and corrupted-age
  conflicts are waived only when two independent single-person identifiers agree (e.g. exact UHID + personal phone).

### Hostile-export hardening

* **Never re-emit an attack:** spreadsheet formulas (`= + - @`, tab) are prefixed with `'` (OWASP CSV injection); files whose
  names carry shell metacharacters or `..` are quarantined; JSON with `NaN`/`Infinity` is quarantined, so every JSON
  output is strict JSON.
* **Bounded work:** a free-text line over 64 KiB (buffer or regex bomb) is quarantined; long fields are scanned in chunks.
* **Keep valid data:** a few NUL bytes injected into valid text become spaces instead of quarantining the whole export;
  impossible ages (`-1`, `0x7fffffff`, int64 overflow) become `[AGE]`.

## 5. Known Limitations & Next Steps

1. **Other Indic scripts:** names written only in Tamil, Bengali or another script rely on context cues and the harvest from structured fields; there is no script-specific transliteration yet.
2. **English-origin names across scripts:** मैथ्यू vs Mathew do not fold to the same key, so such records link only through a shared single-person identifier (UHID, Aadhaar, ABHA, PAN, non-family phone).
3. **Fragments:** a note with only a first name and a relative's first name links only when that pair is unique in the release; otherwise it stays unlinked by design.
4. **Fail-closed address fields** also redact facility or landmark names written inside an address cell (minor utility cost).
5. **Release-level uniqueness:** within a small release most patients stay unique on (age band, sex, district); the defence relies on the registry being 4-5x larger and on date shifting. Wider bands for singleton cohorts are a tunable trade-off (`config/policy.yaml`).
6. Attack numbers come from our implementation of ATTACKS.md; the official attacker may weigh evidence differently.

## 6. Declared Models & Dependencies

* **Models: none.** No LLM, no generative model, no pretrained or trained ML model and no weights. Detection uses rules, bundled
  gazetteers (eponyms, clinical terms, Hinglish stop-words, honorifics, common names, public district names), a clinical lexicon
  learned at run time from the input itself, and fuzzy/phonetic matching (Jaro-Winkler, phonetic skeletons).
* **Runtime dependencies (`requirements.txt`, CPU-only, offline):** `regex`, `rapidfuzz`, `jellyfish`, `indic-transliteration`,
  `cryptography`, `defusedxml`, `pyyaml`.

## 7. Testing

```bash
python -m unittest discover -s tests -p "test_*.py"     # 37 synthetic tests
```

`tests/test_generalization.py` pins every unseen-format failure found by the adversarial generator: unknown headers, new ID,
phone and date shapes, hospital names taken for patients, Hinglish verbs taken for names, encoded payloads, and prefix false
positives (`village`, `ग्रामीण`). `tests/test_hardening.py` pins the hostile-export fixes: NUL bytes, `NaN` JSON, unsafe file
names, formula cells, overflow ages, `C/O <complaint>` columns, audit-token canaries, mixed-script names, PINs with no cue
and single-given-name relative conflicts.
