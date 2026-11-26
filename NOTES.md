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

---

## Milestone M2: Patient Identity Resolution + Pseudonyms ("Link first, redact second")

1. **Multi-script evidence extraction (`raksha/identity/evidence.py`)**:
   - Extracts normalized + Latin-folded + phonetic name tokens, relatives (`S/o`, `D/o`, `W/o`, `C/O`, Hindi/English caretaker fields), DOB (per-column DD/MM vs MM/DD inference), age + record year, sex, villages/PINs, phones, Aadhaar (full & last4, guarding against matching trailing 12 digits of 14-digit ABHA numbers), ABHA numbers & addresses, PAN, email, insurance IDs, and hospital-scoped/prefixed MRNs (`(hospital, id)`).
   - Uses iterative traversal (`get_json_record_units` + `_extract_dict_evidence`) so deeply nested exports never hit recursion limits.
2. **Multi-channel name comparison (`raksha/identity/name_compare.py`)**:
   - Evaluates exact normalized, cross-script transliteration/phonetic matching (with diphthong/romanization vowel folding), corroborated `Mohammad/Mohd/Md/Muhammad/मोहम्मद` alias matching (`Md.` rejected unless corroborated), initials, token-order permutations, `rapidfuzz` similarity, and strict single-token fragmentary name rules.
3. **Blocking, Fellegi–Sunter scoring, and constrained clustering (`raksha/identity/linker.py`)**:
   - 5-pass blocking union (strong IDs, transliterated/phonetic full names + initials, name+relative, name+overlapping age bucket, phone+name/phone candidates).
   - Hand-calibrated Fellegi–Sunter weights with exclusive ID conflict vetoes, weak phone weight, DOB-to-age interval consistency, same-hospital vs cross-hospital generic UHID handling, and all-pairs cluster validation preventing transitive over-merging (`A~B, B~C, A!=C`).
   - Stage A deterministic clustering (sorted by content hash and descending weight) + Stage B fragmentary note attachment requiring threshold and margin (otherwise `patient: null`).
4. **Deterministic >=80-bit HMAC pseudonyms (`raksha/identity/pseudonymizer.py` & `raksha/pipeline.py`)**:
   - Derives `PSN-<16 Crockford base32 chars>` (`80 bits`) from `HMAC-SHA256(key_pid, deterministic_cluster_fingerprint)` with explicit collision-detection nonce loop.
   - Injects resolved `PSN-...` directly inside sanitized CSV/TSV cells, JSON records (iterative, skipping `_export_meta`), FHIR R4 `Patient` elements, and free-text documents without adding columns or breaking schemas; emits `records.jsonl` with exact `loc` (`null` when unlinked/ambiguous) and updates `counts.records_linked`.
5. **Evaluation & Verification**:
   - `pytest -v`: **27 / 27 passed** (includes cross-script `Md.`, homonyms, anti-transitive merge, fragmentary uniqueness/ambiguity, >=80-bit PSN entropy & forced collision avoidance, and end-to-end shuffle invariance).
   - LF `validate_output.py`: **`CONTRACT OK`** (`files 200 sanitized 194 quarantined 6 records 823`).
   - LF `check_dev.py`: **0 / 3230 (0.0%)** PII leaked across all scripts/channels, **0 / 4** canaries, **4855 / 4855 (100.0%)** medical values kept, **Patient Linking precision 1.00, recall 1.00, F0.5 1.00, 0 wrong merges**.
   - `tools/eval_linkage.py` against `dev/truth/truth.json`: **771 / 771** linkable units across **48 / 48** true clusters, **Pairwise P/R/F0.5 = 1.0000 / 1.0000 / 1.0000**, **B-cubed P/R/F1 = 1.0000 / 1.0000 / 1.0000**, **Wrong merges = 0**, **PSN in-record consistency = 776 / 776 (100.0%)**.
   - `tools/robustness/`: **0 / 267 (0.0%)** unseen PII leaked, **0 / 3** canaries, **13 / 13 (100.0%)** medical values kept, **Determinism = True**, **Different-secret diff = True**, **0** truth-free survivors on both practice LF and unseen inputs.

---

## Milestone M3: Adaptive Privacy Engine + Local Attacker Simulator (`tools/attacker/`)

1. **Offline Attacker Simulator (`tools/attacker/simulator.py`, dev-only, A1–A7 per `docs/ATTACKS.md`)**:
   - Approximates A1 (direct exposure), A2/A3 (Sweeney record linkage with filter-skip cascade and transliteration-aware weighting against `registry.csv`), A4 (profile aggregation across exported `PSN-...` and reference tokens), A5 (targeted known-facts attack with global date-shift recovery, condition rarity weighting, stay-length intervals, and membership-inference decoys), A6 (dictionary hash/base64/reverse pseudonym inversion), and A7 (cross-release linkability across two `RAKSHA_SECRET` values).
2. **Keyed Per-Patient Date Shifting (`raksha/privacy/dates.py`)**:
   - Deterministic signed shift derived via HMAC-SHA256 from `derive_subkey(secret, b"date_shift")` with magnitude `15..180` days (never `0` or near-zero), keyed by `PSN` for linked patients across all CSV/TSV/JSON/FHIR/TXT files and by `content_hash` for unlinked records.
   - Preserves all intra-patient date intervals and admission-to-discharge stay lengths (`100.0%`), preserves input formats (ISO8601 + timezone, `DD/MM/YYYY`, `MM/DD/YYYY`, `DD.MM.YYYY`, `DD-MM-YYYY`, `[DD/MM/YY,`), shifts `_export_meta`/`exported_at` timestamps, and recomputes adjacent English/Hindi weekday names to match shifted dates.
3. **Truthful Age Bands, Location Broadening & Runtime Risk Engine (`raksha/privacy/risk.py`, `raksha/privacy/transform.py`)**:
   - Runtime risk estimator computes catchment candidate-set size `k_est` and rare-condition exposure purely from in-batch demographic distributions and empirical diagnosis/procedure counts (never reading external truth or attacker files).
   - Escalates QI generalization per profile in order: remove `house/landmark/locality/village/PIN` & `tehsil` (`[TEHSIL]`) -> widen truthful age band (`5 -> 10 -> 20`, with `80+`/`85+` cap, skipping clinical durations/lab ranges) -> broaden/suppress `district` (`[DISTRICT]` & `Organization.address[].city`) -> coarsen `gender` -> suppress age (`[AGE]`); FHIR `Patient.birthDate` omitted when generalized/risky while keeping R4 bundles valid; `clean/` and non-patient static docs untouched.
   - Populates exact `status: "generalized"` in `records.jsonl` and true `counts.records_generalized` / `counts.records_suppressed` in `report.json`.
4. **Before / After Attacker & Utility Results (Practice LF Dataset)**:
   - `pytest -v`: **31 / 31 passed** (`0.79s`).
   - `validate_output.py`: **`CONTRACT OK`** (`files 200 sanitized 194 quarantined 6 records 823`).
   - `check_dev.py`: **0 / 3230 (0.0%)** PII leaked, **0 / 4** canaries, Linking **P/R/F0.5 = 1.00 / 1.00 / 1.00 (0 wrong merges)**, **4763 / 4763 (100.0%)** clinical probes preserved (`4763 / 4855 = 98.1%` including `92` districts intentionally broadened for k-anonymity), **100.0%** LoS/interval preservation.
   - **A1–A7 Comparison (`tools/attacker/simulator.py`, `m3-final-secret-alpha` vs `beta`)**:
     - `A1_direct_exposure`: `1.000` (raw) -> **`0.000`**
     - `A3_record_linkage`: `0.684` (raw) -> `0.108` (uniform 5yr+dist) -> **`0.025`** (adaptive `k=5`, `-96.3%` vs raw)
     - `A4_profile_reid`: `0.979` (raw) -> `0.833` (M2 ungeneralized) -> `0.542` (uniform 5yr+dist) -> **`0.062`** (adaptive `k=5`, `-93.6%` vs raw)
     - `A5_targeted_known_facts`: `1.000` (raw) -> **`0.486`** (`0.446` sweep; `-51%`–`-55%` vs raw; `0/4` decoy FPs)
     - `A6_pseudonym_inversion`: **`0.000`**
     - `A7_cross_release_link`: `1.000` (raw / M2 unshifted) -> **`0.000`**
     - Overall weighted attack resistance (`k=5`): **`0.8827`** (`Overall risk: 0.1173`).
   - Utility-vs-Privacy sweep (`tools/eval_privacy_utility.py`): documented across Raw, Uniform (`5yr+dist`, `10yr+nodist`), and Adaptive `k=3 / 5 / 8` policies.

---

## Milestone M4: Generalization Pass (No Holdout / Truth Usage)

1. **Contextual Canaries & Digit-Guard FP Measurement (`raksha/detect/names_context.py`, `raksha/detect/crf_model.py`, `raksha/detect/structured.py`, `raksha/qa/leak_scanner.py`)**:
   - Diagnosed all 30/30 unseen canary leaks: `gen_unseen.py` injected digits inside synthetic names (`Rapihivx0`, deviating from `DATASET.md` person+number descriptions), plus 4 genuine detector gaps (`रेफरल नोट:` Hindi referral label, unpunctuated `Attendant <Name>`, `Signed consent of <Name>`, `_meta.source_file` path tokens + `"by_beneficiary"` keys).
   - Fixed the general detectors + QA dict-key scrubbing and allowed synthetic-canary alphanumeric tokens strictly inside explicit person-slot triggers (`_SYNTHETIC_CANARY_TOK_RE` + stopword/clinical exclusion).
   - Measured FPs on clinical alphanumeric sentences & 6 adversarial clinical slots (`Dolo 650`, `Vitamin B12`, `Covid 19`, `Ward 5`, `Stage 3 CKD`, `Phase 2`, `HbA1c`, `SpO2`, `HER2`, `CA125`, `BRCA1`, `A15.0`): **0/7** negative sentence FPs, **0/6** clinical-slot FPs (vs **6/6** when unconditionally removing the digit guard), **3/3** canary TPs, and **0** digit hits across all `dev/input` files.
2. **Compositional Header/Key Schema & Identity Inference (`raksha/detect/structured.py`, `raksha/detect/identifiers.py`, `raksha/identity/evidence.py`, `raksha/detect/registry.py`, `raksha/privacy/*`)**:
   - Replaced literal header lists with morphological part decomposition (`split_key_parts` + `classify_header_parts`), unified table (`TableSchema` / `infer_table_schema`) and nested JSON (`DictSchema` / `infer_dict_schema`) neighbor-context inference (preserving `panels.name`, `diagnosis.display`, clinicians, and clinical enums while masking uncertain high-cardinality non-clinical columns to `[REDACTED_PII]`).
   - Added 2-to-4-digit year prefix support (`[A-Z]{2,6}\d{2,4}-\d{3,}`) and `.txt`-derived multi-word names in `AliasRegistry` without loosening `linker.py` thresholds or conflict vetoes (**0** wrong merges).
3. **Composed/Decomposed Accented Unicode & `cp1252` Support (`raksha/ingest/loader.py`, `normalize.py`, `names_context.py`, `crf_model.py`, `registry.py`, `evidence.py`, `qa/leak_scanner.py`)**:
   - Added NFC/NFD-safe accented Latin character classes, `NFKD` accent folding (`fold_latin_accents`), and valid accented `cp1252` letter decoding while continuing to quarantine corrupted/mojibake bytes (`\xff\xfe`, `\xfe\xff`, `\xc3(`).
4. **Content-Based Non-Patient Classification (`raksha/detect/structured.py`, `raksha/identity/evidence.py`, `raksha/privacy/risk.py`, `raksha/pipeline.py`)**:
   - Removed `"clean/"` path matching (`_NON_PATIENT_PATH_TOKS`, `_NON_PATIENT_FILES`) and replaced with content-based `is_non_patient_item` across CSV/JSON/TXT in linking, risk planning, and privacy transforms.
5. **Before / After Verification (`tests/test_generalization.py`: 36/36 pytest passing)**:
   - **Practice LF (`dev/input`, `validate_output.py`, `check_dev.py`, `eval_linkage.py`)**: kept `CONTRACT OK`, `files 200 sanitized 194 quarantined 6 records 823`, **0 / 3230 (0.0%)** PII leaks, **0 / 4** canaries, **4763 / 4855 (98.1%)** medical preservation, Linking **1.00 / 1.00 / 1.00 (0 wrong merges, 48/48 clusters)**, **0** truth-free survivors, deterministic output (`alpha1 == alpha2`, `alpha != beta`).
   - **Saved Old Unseen (`seed=2028, n=320`)**: PII leaks **153 / 3115 (4.91%) -> 2 / 3115 (0.06%)** (both remaining strings are `Dr. Roland Bernard` / `Dr. Adèle Gosselin` preserved under `doc_consultant`); Canaries **30 / 30 -> 0 / 30**; Medical preservation **794/794 -> 930/930 (100.00%)**; Quarantines **9 -> 6** (decoded `coastal_opd_cp1252.csv` + both accented filenames); Linking Precision **1.0000 -> 1.0000**, Recall **0.5622 -> 0.9204**, F0.5 **0.8652 -> 0.9830**, Wrong merges **0** (remaining 16 splits are synthetic mismatched-name cross-script pairs).




---

## Generalization & Synthetic Debugging Pass (No Runtime Truth Usage / No M3 Policy Changes)

1. **Multi-Patient Keyed Dict & Wrong-Merge Prevention (`raksha/identity/evidence.py`, `raksha/identity/linker.py`, `raksha/detect/structured.py`)**:
   - Detected multi-patient keyed-dict collections (`has_multi_patient_keyed_dict`, e.g. `results_by_patient`) so distinct patients keep `patient: null` rather than collapsing into `"doc"`; protected dict keys/values (`"operator_account"`, `"gram_sabha"` word-boundary check so it never falsely maps to `"ABHA"`).
   - Guarded `_extract_free_text_clues` against relative-modifying Hindi/English `"नाम:"`/`"Name:"` prefixes (e.g. `"पिता का नाम:"`), marked `NAME_LOCAL` / parenthesized secondary aliases as `is_local_alias=True`, pruned phones/MRNs shared across >=3 mutually disagreeing patient names (`_prune_repeated_ambiguous_evidence`), and avoided penalizing short single-token first-name variants sharing a hospital MRN.
   - Added signature compression and star-hub blocking in `_build_candidate_pairs` so high-visit-count (>50 records) patients still link across hospitals without quadratic blowup or wrong merges.
2. **General Field Inference, MRNs, Bilingual Aliases, URLs & Ragged CSVs (`raksha/detect/identifiers.py`, `raksha/detect/names_context.py`, `raksha/detect/registry.py`, `raksha/detect/structured.py`, `raksha/export/filenames.py`, `raksha/qa/leak_scanner.py`)**:
   - Added structural MRN/UHID regex shapes (`KLS:UHID:..`, `PGC_REG_..`, `RGH-B25-..` style branch formats without hardcoding prefixes) and labeled `Reg:`/`ID:`/`पंजीयन क्रमांक:` cues while excluding `"LAB-"`/`"ORD-"` order/lab IDs.
   - Fixed `classify_header_parts` so `visit_uid`/`Encounter_Code` stay `MRN` (not `CLINICAL_DATE`), `opd_remark` is `FREE_TEXT`, and `_DATE_PROP_PARTS` includes `mdy`/`dmy`/`ymd` (`Visit_MDY`, `Birth_MDY`), while guarding `sanitize_header_or_key` against corrupting valid multi-word schema headers (`Domicile_Addr`, `Clinical_Remarks`, `ABHA_Health_Acc`).
   - Added bilingual parenthesized alias detection, URL query-parameter scrubbing (`Bundle.meta.source`), space-delimited FHIR `Patient.text.div` scrubbing, expanded English/Hindi `LOCALITY`/`LANDMARK` ordering, adjacent-to-ID filename name scrubbing + base64 JSON name propagation, and ragged CSV alignment/cell scrubbing (`First | Last` split cells, `LANDMARK`, standalone adjacent-to-sex age banding, and `District` preservation).
   - Protected Hindi clinical symptom/history/dosing prose in `protected.py`, prevented `1 ग्राम` (`gram` unit) from matching `ग्राम` (`Village`) in `names_context.py`/`evidence.py`, and fixed `_AGE_COMPACT_RE` negative lookahead `(?!ों)` so `"6 वर्षों से"` durations are never age-banded.
3. **Before / After Verification (`tests/test_synthetic_regressions.py`: 45/45 pytest passing)**:
   - **Practice LF (`dev/input`, `validate_output.py`, `check_dev.py`, `eval_linkage.py`)**: kept `CONTRACT OK` (`files 200 sanitized 194 quarantined 6 records 823`), **0 / 3230 (0.0%)** PII leaks across all types/scripts/channels, **0 / 4** canaries, **4763 / 4855 (98.1%)** medical preservation, Linking **1.0000 / 1.0000 / 1.0000 (48 / 48 clusters, 6892 / 6892 pairs, 0 wrong merges, PSN in-record 776 / 776)**, deterministic (`DET_SAME = True`, `DIFF_SECRET = True`), wall clock **11.216s**.
   - **Generated Unseen (`tools/robustness/gen_unseen.py --seed 2028 --patients 320`, `eval_unseen.py`, `eval_linkage.py`)**: PII leaks **2 / 3115 (0.06%)** (`Roland`, `Adèle` in doctor slot `doc_consultant`), Canaries **0 / 30**, Medical values **930 / 930 (100.00%)**, Linking **precision 1.0000, recall 0.9005, F0.5 0.9784, B-cubed F1 0.9800 (0 wrong merges, 681 / 681 PSN in-record)** (vs `fe6065e`: `2 / 3115`, `0 / 30`, `930 / 930`, `P=1.0000, R=0.9204, F0.5=0.9830, wrong_merges=0`), deterministic (`DET_SAME = True`, `DIFF_SECRET = True`), wall clock **5.273s**.
   - **`synthetic_test_set` (`fe6065e` -> current, `validate_output.py`, `evaluate.py`)**:
     - Leaks: **3949 / 42866 (9.21%) -> 0 / 42866 (0.00%)** (MRN `3681 -> 0`, LOCALITY `181 -> 0`, PATIENT_NAME `68 -> 0`, VILLAGE `14 -> 0`, PIN `4 -> 0`, RELATIVE_NAME `1 -> 0`; all scripts & channels `0.0%`).
     - Canaries: **5 / 6 -> 0 / 6**.
     - Medical preservation: **45776 / 45883 (99.8%) -> 45883 / 45883 (100.0%)** (`83 / 83` exact stay lengths, `8672 / 8672` truthful ages/bands).
     - Linking: **precision `0.9876 -> 1.0000`**, **recall `0.6265 -> 0.9887`**, **F0.5 `0.8855 -> 0.9977`**, **B3-F1 `0.8494 -> 0.9927`**, **wrong merges `1 cluster (1638 FP pairs) -> 0 (0 FP pairs)`**, homonym merges **`0 / 12`**, PSN in-record **`9858 / 9858`**.
     - FHIR R4 valid **`30 / 30`**, Quarantines **`6` (`3` safely sanitized, `0` raw copies)**, **`CONTRACT OK`**, deterministic (`DET_SAME = True`), wall clock **`108.484s -> 85.854s`**.
