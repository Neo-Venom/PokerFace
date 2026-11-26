# RakshaGuard

**RakshaGuard** is an offline, CPU-only, deterministic healthcare privacy firewall for messy multi-hospital Indian clinical exports (CSV, TSV, nested JSON, FHIR R4 Bundles, and English/Hindi/Hinglish free text). Configured via `RAKSHA_SECRET`, it resolves patient identities across files and scripts first, applies keyed date shifts and adaptive quasi-identifier generalization, scrubs all personal identifiers while preserving clinical meaning, sanitizes hidden channels (filenames, headers, JSON keys, metadata, base64 payloads), and audits every output with an independent zero-shared-state leakage scanner.

---

## Setup & Usage

RakshaGuard is pure Python (`>= 3.11`) with pinned offline-capable runtime dependencies and zero network calls at execution time (`HF_HUB_OFFLINE=1`, zero LLM/hosted-AI packages).

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Run the privacy firewall (RAKSHA_SECRET must be non-empty)
RAKSHA_SECRET="your-secret-key" python main.py --input /path/to/input --output /path/to/output
```

Configuration defaults live in `config/policy.yaml` (`age_band_years: 5`, `date_shift_max_days: 180`, `min_group_size: 5`), and submission entrypoints are declared in `raksha.toml`.

Run the unit and regression test suite:

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## Architecture Overview

RakshaGuard follows a strict **"link first, redact second"** two-pass pipeline orchestrated by [`raksha/pipeline.py`](raksha/pipeline.py) (detailed diagram and module provenance live in [`ARCHITECTURE.md`](ARCHITECTURE.md)):

```text
input/ ─▶ Pass 1 (ingest/loader.py: UTF-8/BOM/UTF-16/clean cp1252, delimiter & JSON sniff, quarantine broken files)
       ─▶ Corpus Learning & Detection Setup (protected clinical lexicon, MRN skeleton learner, cross-script AliasRegistry)
       ─▶ Stage M2 Linking (identity/evidence.py ─▶ name_compare.py ─▶ linker.py ─▶ pseudonymizer.py)
       ─▶ Stage M3 Privacy Planning (privacy/risk.py ─▶ privacy/dates.py)
       ─▶ Pass 2 Redaction & Export (detect/structured.py, identifiers.py, names_context.py, crf_model.py, privacy/transform.py, export/filenames.py)
       ─▶ Independent QA Audit (qa/leak_scanner.py) ─▶ sanitized/, records.jsonl, report.json
```

### How Each Component Works

1. **Safe Ingestion (`raksha/ingest/loader.py`)**: Sniffs formats (`csv`, `tsv`, `json`, `txt`) and CSV delimiters (`;`, `\t`, `|`, `,`), normalizes `NaN`/`Infinity` JSON constants safely, recovers isolated sparse NUL padding when valid printable UTF-8, enforces deep-nesting recursion safety, and quarantines empty, corrupt-encoding, binary, XML, PDF, image, or malformed files under fixed reason codes (`empty_file`, `invalid_encoding`, `unsupported_format`, `malformed_json`). Never copies broken inputs raw.
2. **Multi-Script Normalization & Context Detection (`raksha/detect/`)**:
   - `normalize.py`: Strips zero-width characters (`\u200b`–`\u200d`, `\ufeff`), folds Cyrillic/Greek homoglyphs, converts Devanagari numerals to ASCII, and transliterates Devanagari↔Latin (`indic-transliteration`) while tracking exact span offset maps back to unmodified source text.
   - `structured.py` & `identifiers.py`: Uses value-first column and nested-key schema inference, scrubs PII from headers, keys, metadata paths, URL query strings, and FHIR R4 resources (preserving R4 syntax, references, codes, and units), and synthesizes per-corpus MRN patterns at run time.
   - `protected.py`, `names_context.py` & `registry.py`: Builds a runtime clinical lexicon from formularies and structured medical columns; resolves overlap conflicts so medicines, doses (`mg`, `1-0-1`), vitals, labs, clinical `C/O`/`K/C/O`/`H/O`, hospitals, eponyms, and clinicians (`Dr.`, `डॉ.`, `ASHA`) always win over PII spans.
   - `export/filenames.py`: Redacts PII across directory paths and filenames with deterministic `input_sha256[:8]` collision suffixes, decodes printable base64 attachments, re-scrubs them, and neutralizes CSV formula prefixes (`=`, `+`, `-`, `@`).
3. **Cross-Document Identity Resolution (`raksha/identity/`)**:
   - Extracts normalized/phonetic names, relatives (`S/o`/`D/o`/`W/o`/Hindi relations), age/DOB, gender, locality, phones, Aadhaar/ABHA/PAN, and hospital-scoped `(hospital, MRN)` keys without hardcoding dataset literals.
   - Evaluates name pairs over independent channels (`name_compare.py`: exact norm, cross-script transliteration, corroborated `Mohammad/Mohd/Md` aliases, initials, token order, `rapidfuzz` ratio) inside a 5-pass blocker and Fellegi–Sunter scorer (`linker.py`) with cluster-level cannot-link checks preventing homonym, twin, father–son, and shared-family-phone false merges. Ambiguous records become `"patient": null`.
4. **Adaptive Privacy Layer (`raksha/privacy/`)**:
   - Derives per-patient 15–180 day calendar shifts keyed by `PSN` (or `content_hash` when unlinked), preserving intra-patient visit intervals, lengths of stay, and synchronized EN/HI weekdays (`dates.py`).
   - Computes empirical batch candidate-set sizes (`risk.py`) and applies minimal truthful generalization (`5/10/20`-year bands, `80+`/`85+` caps, district/tehsil/gender coarsening in `transform.py`) while leaving clinical facts and static reference files intact.
5. **Independent QA Scanner (`raksha/qa/leak_scanner.py`)**: Scans `sanitized/`, decoded base64 strings, filenames, `records.jsonl`, and `report.json` with zero shared imports from `raksha/detect/`, re-scrubbing residual hits or quarantining if un-scrubbable.

---

## Model Declaration (No LLMs / <= 50M Params)

- **Bundled Model**: [`models/pii_crf.crfsuite`](models/pii_crf.crfsuite) (`41,456 bytes` / `40.5 KB`).
- **Type & Training**: Compact linear-chain Conditional Random Field (`python-crfsuite` / `sklearn-crfsuite`) trained offline via [`tools/train_crf.py`](tools/train_crf.py) exclusively on synthetic `Faker(['en_IN', 'hi_IN'])` clinical notes and synthetic canary contexts (never `dev/input` or `dev/truth`).
- **No LLMs / No Generative Models**: Zero pretrained transformers, zero neural weights, zero hosted APIs.

---

## Security & Cryptographic Design

- **Key Separation**: Validates non-empty `RAKSHA_SECRET` in memory (`raksha/config.py`) and derives independent 256-bit subkeys via **HKDF-SHA256** (`salt=b"rakshaguard-v1"`) for pseudonyms (`b"pseudonym"`) and date shifts (`b"date_shift"`).
- **Unlinkable Pseudonyms**: `PSN-<16 Crockford base32 chars>` (`80` bits) computed via HMAC-SHA256 over canonical cluster fingerprints with collision checking. Output under a second secret produces distinct pseudonyms and date shifts.
- **Strict Output Hygiene**: `RAKSHA_SECRET`, raw PII, and stack traces with data values are never printed to stdout/stderr/temp files or `report.json` (which records fixed reason codes and SHA-256 checksums only). Same secret + same inputs produce byte-identical `sanitized/` trees and `records.jsonl` sorted deterministically by content hash.

---

## Verified Evaluation Results

| Dataset / Benchmark | Leaks | Canaries | Medical Kept | Linking (`P / R / F0.5`, Wrong Merges) | Contract / Score |
|---|---|---|---:|---|---|
| Unit & Regression Suite (`pytest`) | `49 / 49` passed (`2.15s`) | — | — | — | Exit `0` |
| Practice LF (`Secret-Wards-ToolKit`, `7.94s` / `8.63s`) | **`0 / 3230` (`0.0%`)** | **`0 / 4`** | `4763 / 4855` (`98.1%`) | **`1.0000 / 1.0000 / 1.0000`**, `0` wrong (`771` units, `48/48` clusters) | **`CONTRACT OK`** (`195` files deterministic) |
| Independent `synthetic_test_set` (`69.60s`) | **`0 / 42866` (`0.00%`)** | **`0 / 6`** | `45883 / 45883` (`100.0%`) | `1.0000 / 0.9814 / 0.9962`, `0` wrong (`0/12` homonyms) | **`CONTRACT OK`**, `30/30` FHIR R4 |
| Trap Gauntlet `A7` (`15.76s`) | `0` direct IDs | **`0 / 5`** | `4814 / 4899` | `F0.5=0.9966, B3=0.9917`, `0` wrong | **`99.34 / 100`** (`0` penalties) |
| Trap Gauntlet `H7` (`30.72s`) | `0` direct IDs | **`0 / 8`** | `4922 / 5006` | `F0.5=0.9966, B3=0.9918`, `0` wrong (`twins 1/1`, `father_son 1/1`) | **`98.68 / 100`** (`0` penalties) |
| Trap Gauntlet `B11` *(held-out)* (`15.72s`) | `0` direct IDs (`0.0008` token) | **`0 / 5`** | `4792 / 4963` | `F0.5=0.9965, B3=0.9914`, `0` wrong (`PT00014` `2 psn`) | **`98.16 / 100`** (`0` penalties) |
| Trap Gauntlet `H11` *(held-out)* (`31.34s`) | `0` direct IDs (`0.0008` token) | **`0 / 8`** | `4892 / 5070` | `F0.5=0.9965, B3=0.9915`, `0` wrong (`twins 1/1`, `father_son 1/1`) | **`97.58 / 100`** (`0` penalties) |

---

## Known Limitations

- **Rare-Condition Exposure (`A5`) & Zero Record Suppression**: Because diagnoses, procedures, and exact admission-to-discharge lengths of stay are intentionally preserved (`records_suppressed` stays `0`), rare-condition facts paired with broad demographics can still identify outlier patients in targeted known-facts attacks.
- **Small-Batch District Coarsening**: Under `min_group_size: 5`, small ~48-patient batches frequently broaden `district` across profiles (`-1.9%` probe count on practice).
- **Script Scope**: Transliteration and normalization cover Latin (with Cyrillic/Greek homoglyph folding) and Devanagari Hindi; other regional Indic scripts rely primarily on structured inference and the compact synthetic CRF.
- **Unseen Edge Cases**: On held-out gauntlet sets (`B11`/`H11`), minor residual edge cases remain (`4` landmark tokens, `18` PINs in `H11`, `6` standalone Latin first names in discharge text, one cross-script split on `PT00014`, and some package/procedure over-redaction under unfamiliar labels).
