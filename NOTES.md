# RakshaGuard Development Notes

## Milestone M1: Detection Engine + Protected Clinical Spans

Baseline starter (before M1 changes):
- Total PII leaked: 2718 / 3230 (84.1%)
- By script: latin 82.8%, deva 100.0%, hinglish 100.0%
- By channel: content 83.5%, filename 100.0%, base64 100.0%, metadata 100.0%, key 100.0%, header 100.0%
- Canaries leaked: 4 / 4
- Medical values kept: 4855 / 4855 (100.0%)
- Raw copies: 0

### Step A: Structured Fields + Header/Key Sanitization + Numeric IDs
- Value-first column & JSON scalar inference + multilingual header/key synonyms (EN/HI/Hinglish + misspellings)
- Deterministic key/header collision resolution + iterative deep JSON traversal
- FHIR R4 Bundle field sanitization (preserving references and valid R4 date syntax)
- Strict corrupt-encoding quarantine (`corrupt_bytes_note.txt`), mislabeled CSV-JSON handling
- Results (`check_dev.py`):
  - Total PII leaked: 762 / 3230 (23.6%)
  - By script: latin 18.5%, deva 69.9%, hinglish 100.0%
  - By channel: content 22.6%, filename 100.0%, base64 100.0%, metadata 5.0%, key 0.0%, header 0.0%
  - Canaries leaked: 3 / 4
  - Medical values kept: 4855 / 4855 (100.0%)
  - Raw copies: 0

### Step B: Identifier Regex/Context + Numeric Context Classifier
- All Indian phone formats (+91, 0-prefix, spaces/dashes, extensions, masked, Devanagari digits), Aadhaar (12-digit & masked), ABHA (14-digit & ABHA addresses), PAN, Insurance IDs, Email, MRN/UHID/IPD + corpus-derived MRN skeleton learner
- Numeric context classifier protecting platelet counts, lab ranges, doses, frequencies (`1-0-1`), and BP/vitals (`150/90`)
- Results (`check_dev.py`):
  - Total PII leaked: 536 / 3230 (16.6%)
  - Zero leaks (0.0%) on AADHAAR, AADHAAR_MASKED, PIN, EMAIL, PAN, DOB, ABHA_NUMBER, INSURANCE_ID
  - PHONE leaked: 6 / 308 (1.9%); MRN leaked: 43 / 850 (5.1%)
  - By script: latin 11.3%, deva 62.9%, hinglish 100.0%
  - By channel: content 15.3%, filename 100.0%, base64 100.0%, metadata 5.0%, key 0.0%, header 0.0%
  - Canaries leaked: 3 / 4
  - Medical values kept: 4855 / 4855 (100.0%)
  - Raw copies: 0

### Step C: Normalization & Transliteration with Offset Maps
- NFC + zero-width removal (`\u200b`, `\u200c`, `\u200d`, `\ufeff`, `\u00ad`) + Devanagari digits -> ASCII + Cyrillic/Greek homoglyph folding
- `indic-transliteration` (Devanagari <-> ITRANS/Latin with Hindi schwa deletion and nukta handling) + `phonetic_key` for cross-script matching
- Dual representation (`NormalizedView`) with exact `clean_to_orig` and `folded_to_orig` span maps back to original text; integrated into `identifiers.py`
- Results (`check_dev.py`):
  - Total PII leaked: 536 / 3230 (16.6%), Medical values kept: 4855 / 4855 (100.0%), Canaries: 3 / 4
  - All 8 unit tests passing (including zero-width, homoglyph, Devanagari digit, and cross-script offset mapping)

### Step E: Protected Clinical Spans + Explicit Span-Conflict Resolution
- Dynamic `ClinicalLexicon` built from `clean/drug_formulary.csv`, `clean/lab_reference_ranges.csv`, and structured `rx`/`test` fields + base essential medicine/lab vocabulary
- Regex patterns for dosage forms, drug suffixes, doses (`mg`, `mcg`, `ml`, `IU`, `%`, `°F/C`), frequencies (`BD`, `TID`, `1-0-1`, `1 0 1`, `x 5 days`), vitals (`150/90`, `BP`, `Pulse`, `SpO2`), labs with units, ICD-10 codes, `K/C/O`, `H/O`, clinical `C/O`, `Ward N (Dept)`, and EN/HI/Hinglish complaints
- Explicit `resolve_pii_vs_protected` conflict resolver ensuring protected clinical and doctor spans always win overlaps (e.g. `103 F. Paracetamol` no longer mistaken for initial `F. Paracetamol`)
- Results (`check_dev.py`):
  - Total PII leaked: 87 / 3230 (2.7%)
  - Canaries leaked: 1 / 4
  - Medical values kept: 4855 / 4855 (100.0%) — restored 100% clinical preservation!
  - All 13 unit tests passing

### Step F: Run-Level Alias Registry + Filenames + Base64 + clean/ Preservation
- `AliasRegistry` collects high-confidence `patient`, `relative`, and `doctor` entities + `MRN`, `PHONE`, `VILLAGE` from structured columns/JSON/FHIR, expanding cross-script Devanagari<->Latin and `phonetic_key` variants (including reversed token order)
- Proposes candidates in free text, filenames, and base64-decoded text attachments (`DocumentReference.content[].attachment.data`, `scanned_consent_form.json`), resolved through `resolve_pii_vs_protected` so doctors & clinical spans are never redacted
- Deterministic collision-safe filename/path sanitization (`input_sha256[:8]` suffix when renamed)
- Verified `clean/` files (`dengue_management_protocol.txt`, `drug_formulary.csv`, `lab_reference_ranges.csv`, `ward_census.json`) pass through unchanged (0 over-redaction)
- Results (`check_dev.py`):
  - Raw `dev/input` (`CONTRACT OK`): 36 / 3230 (1.1% leaked), content 0.0%, base64 0.0%, metadata 0.0%, key 0.0%, header 0.0%, deva 0.0%, hinglish 0.0%, canaries 0 / 4, medical values 4497 / 4497 (100.0%; note: Windows git CRLF checkout changes SHA-256 vs Linux `truth.json` on some renamed `.txt`/`.json` files)
  - LF-normalized input check: 13 / 3230 (0.4% leaked), content 0.0%, base64 0.0%, metadata 0.0%, key 0.0%, header 0.0%, deva 0.0%, hinglish 0.0%, canaries 0 / 4, medical values 4671 / 4671 (100.0%)
  - All 15 unit tests passing

### Step G: Synthetic-Trained CRF (`sklearn-crfsuite`)
- Offline synthetic training corpus generator (`tools/train_crf.py`) using `Faker(['en_IN', 'hi_IN'])` + synthetic canary names + EN/HI/Hinglish multi-style templates + clinical negative traps
- Disjoint split validation by patient identity: `P=0.999 R=0.995 F1=0.997` (disjoint patients); `P=1.000 R=1.000 F1=1.000` (unseen canary-context test)
- Compact binary model (`models/pii_crf.crfsuite`, `39.6 KB`) loaded via `pycrfsuite.Tagger` with char n-grams, script, shape, prefix/suffix, and context window features
- Integrated via `resolve_pii_vs_protected` so protected clinical spans & doctors always win
- Results (`check_dev.py`):
  - Total PII leaked: 35 / 3230 (1.1% raw / 0.4% LF), content 0.0%, base64 0.0%, metadata 0.0%, key 0.0%, header 0.0%, deva 0.0%, hinglish 0.0%
  - Canaries leaked: 0 / 4
  - Medical values kept: 4497 / 4497 (100.0%)
  - All 16 unit tests passing

### Step H: Independent Post-Export Leakage Scanner (`raksha/qa/leak_scanner.py`)
- Completely independent QA module (no shared state or imports from `raksha.detect`) with its own NFKC/zero-width/Devanagari-digit normalization, precompiled `_CompiledSurfaceMatcher`, raw identifier regexes, and base64 payload scanner
- Re-evaluates doctor/clinical/eponym context on any hit; re-scrubs residual PII safely or quarantines if un-scrubbable / filename-leaking / second-pass hit
- Results (`check_dev.py`, `validate_output.py`, `pytest`):
  - `files 200 sanitized 194 quarantined 6 records 823`, `CONTRACT OK`
  - Total PII leaked: 35 / 3230 (1.1% on Windows CRLF raw input; content 0.0%, base64 0.0%, metadata 0.0%, key 0.0%, header 0.0%, deva 0.0%, hinglish 0.0%)
  - All 35 PII filenames in `sanitized/` are renamed (the remaining 35 reported filename hits in Windows `check_dev.py` are due to git CRLF vs Linux LF SHA-256 mismatch against `truth.json`, not unrenamed files on disk)
  - Canaries leaked: 0 / 4
  - Medical values kept: 4497 / 4497 (100.0%)
  - Raw copies: 0
  - All 20 unit tests passing (`pytest`)

### Hardening Pass (Pre-M2)
1. **Authentic LF SHA-256 verification**: Tested on a clean LF git clone (`core.autocrlf=false`):
   - `files 200 sanitized 194 quarantined 6 records 823`, `CONTRACT OK`
   - **0 / 3230 (0.0% leaked)** across all 16 entity types (`MRN 0/850`, `PATIENT_NAME 0/962`, `PHONE 0/308`, etc.), all 3 scripts (`latin/deva/hinglish 0.0%`), and all 6 hiding places (`content/filename/base64/metadata/key/header 0.0%`)
   - **0 / 4** canaries; **4855 / 4855 (100.0%)** medical values preserved; **0** raw copies
   - **Discrepancy explanation**: On Windows git with `core.autocrlf=true`, CRLF-converted files in `dev/input/` changed `input_sha256` in `report.json` relative to `truth.json`'s LF SHA-256 hashes, causing `check_dev.py`'s `renamed.get(f["sha256"])` lookup to fall back to the un-renamed `rel` path for those files (producing the illusory 24 MRN / 6 name / 5 phone filename leaks and 4497/4855 denominator drop). All 35 PII filenames in `sanitized/` were actually renamed.
2. **Removed practice hardcoding**: Replaced hospital-specific MRN prefixes (`SNJ`, `JJH`, `DHH`, `ADM`), practice surnames in `registry.py`, practice canary/location names in `tools/train_crf.py` and `tests/`, and `"asha"` stopword hacks with general structural patterns; retrained `models/pii_crf.crfsuite` (`40.5 KB`).
3. **Quarantine audit (6 vs baseline 5)**:
   - Baseline 5: `empty_export.csv` (`empty_file`), `hl7_cda_discharge.xml` (`unsupported_format`), `photo_rx.jpg` (`unsupported_format`), `scanned_discharge.pdf` (`unsupported_format`), `truncated_opd.json` (`JSONDecodeError`).
   - Ours 6: same 5 (checked extension before `decode_bytes` so `photo_rx.jpg` cleanly reports `unsupported_format`) + `misc/corrupt_bytes_note.txt` (`invalid_encoding`, raw `\xff\xfe\xc3(` mojibake bytes that starter's cp1252 fallback wrongly exported). All 6 are intentionally quarantined.
4. **Secret validation**: `require_secret()` now accepts any non-empty `RAKSHA_SECRET` (`missing_secret` only when missing or `""`), verified with a 1-character secret test.
5. **Additional general fix**: Added `Pt. <CapitalizedName>` prefix rule + unseen label triggers (`Beneficiary:`, `Caretaker:`, `Client Name:`) in `names_context.py` and parenthesized ID context in `train_crf.py`. All 21 unit tests pass.

### Unseen Robustness Suite (`tools/robustness/`)
- `gen_unseen.py`: Faker `en_IN`/`hi_IN` unseen dataset (11 files, 24 patients, 3 canaries, unseen MRN formats `KGMU/27/...`, `VHC-27-...`, `ARC27-...`, `SYD-...`, renamed multilingual headers, numeric JSON IDs, homonyms, doctor-patient name overlaps, base64, corrupt files, `clean/`).
- `eval_unseen.py`: Determinism `True` (A==B), Diff-secret `False` (expected in M1 before M2/M3 pseudonyms & date shifts), Quarantined `{'invalid_encoding': 1, 'malformed_json': 1, 'empty_file': 1, 'unsupported_format': 1}`, PII leaked **0 / 267 (0.0%)** across all types/scripts/channels, Canaries **0 / 3**, Medical values **13 / 13 (100.0%)**.
- `truth_free_scan.py`: **0 survivors** on both `.unseen_data` and LF practice `dev/input`.








