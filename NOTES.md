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



