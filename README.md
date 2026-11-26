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
  ├─ Schema-agnostic column/key profiling (bilingual synonyms + value shape + per-column DMY/MDY inference)
  └─ Format & structure preservation (iterative JSON AST, valid FHIR R4B Bundles, exact CSV cells & note lines)
  │
  ▼
[2. Clinical Shield & Pass-1 Harvest] (medical_guard.py, indic_norm.py, pii_regex.py, pii_ner.py)
  ├─ Runtime lexicon harvested from clean/ docs, drugs, ICD-10 displays, LOINC tests, doctors & clinical wards
  ├─ Span protection for doctors ("Dr. ..."), eponyms ("Addison's", "Parkinson's"), ASHA workers, K/C/O, doses, labs
  └─ EN / Devanagari / Hinglish extraction + Aadhaar (non-Verhoeff tolerant), ABHA, PAN, phone, PIN & Base64 harvest
  │
  ▼
[3. Cross-Script Record Linkage] (evidence.py, blocking.py, scorer.py, cluster.py)
  ├─ Devanagari-to-ITRANS + inherent-a drop + nukta & vowel folding ("मोहम्मद शाहिद" == "Md. Shahid" == "Mohammad Shahid")
  ├─ Multi-key blocking + shared/attendant family-phone detection + IDF-weighted Fellegi–Sunter scoring
  └─ Constrained descending clustering with hard contradiction vetoes -> HKDF-HMAC pseudonyms (PSN-XXXXXXXX)
  │
  ▼
[4. Adversarial Privacy Layer] (date_shifter.py, privacy/kanon.py)
  ├─ Per-patient HMAC date shift (30..180 days) preserving exact LOS & visit cadence across formats & year-less dates
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

Evaluated offline with the official toolkit validators (`validate_output.py`, `check_dev.py`) and local attacker/scorer (`ATTACKS.md` / `SCORING.md`):

| Metric / Benchmark | Baseline (Raw/Starter) | Project RAKSHA (`dev`, `k=5`) | Unseen Chaos (`seed=101`) | Unseen Chaos (`seed=202`) |
|---|---|---|---|---|
| **Contract & Integrity** | `CONTRACT OK` | `CONTRACT OK` (`10.0 / 10`) | `CONTRACT OK` (`10.0 / 10`) | `CONTRACT OK` (`10.0 / 10`) |
| **Personal Data Leaked (Overall)** | `84.1%` (`starter`) | **`0 / 3,230 (0.0%)`** | **`0 / 3,230 (0.0%)`** | **`0 / 3,230 (0.0%)`** |
| **Leaked by Script (LA / HI / HG)** | `82.8% / 100% / 100%` | **`0.0% / 0.0% / 0.0%`** | **`0.0% / 0.0% / 0.0%`** | **`0.0% / 0.0% / 0.0%`** |
| **Leaked Hidden Channels** | — | **`0.0%`** across all 6 channels | **`0.0%`** across all 6 channels | **`0.0%`** across all 6 channels |
| **Canaries / Raw Broken Copies** | `4 / 4` canaries | **`0 / 4` / `0 raw`** | **`0 / 4` / `0 raw`** | **`0 / 4` / `0 raw`** |
| **Medical Values Preserved** | `100.0%` | **`4,855 / 4,855 (100.0%)`** | **`4,855 / 4,855 (100.0%)`** | **`4,855 / 4,855 (100.0%)`** |
| **Linkage (`P / R / F0.5 / Merges`)** | `0 / 0 / 0 / 0` | **`1.00 / 1.00 / 1.00 / 0`** | **`1.00 / 0.93 / 0.985 / 0`** | **`1.00 / 0.92 / 0.984 / 0`** |
| **Attacker `A3` / `A4` / `A5`** | `92.2% / 97.9% / 100%` | **`3.74% / 14.58% / 7.73%`** | **`3.85% / 14.58% / 7.74%`** | **`3.74% / 14.58% / 8.02%`** |
| **Global Shift / Cross-Release (`A7`)** | — | **`0` recovered / `0.0`** | **`0` recovered / `0.0`** | **`0` recovered / `0.0`** |
| **Total Score (`local_scorer`)** | `~18 / 100` | **`98.52 / 100`** | **`97.83 / 100`** | **`97.79 / 100`** |

### Privacy–Utility Trade-off Across `k` (`min_group_size`)

| `k` (`min_group_size`) | Total Score | Privacy (`25`) | Utility (`20`) | `A3` Linkage | `A4` Re-ID Risk | `A5` Targeted Risk |
|---|---|---|---|---|---|---|
| **`k = 1`** | `97.75` | `22.75` | `20.00` | `4.21%` | `22.92%` | `12.48%` |
| **`k = 3`** | `98.34` | `23.34` | `20.00` | `3.77%` | `16.67%` | `8.77%` |
| **`k = 5` (default)** | **`98.52`** | **`23.52`** | **`20.00`** | **`3.74%`** | **`14.58%`** | **`7.73%`** |
| **`k = 10`** | `98.52` | `23.52` | `20.00` | `3.74%` | `14.58%` | `7.73%` |

---

## 5. Known Limitations & Next Steps

1. **Scripts Outside Latin/Devanagari:** Transliteration and colloquial gazetteers target English, Devanagari Hindi, and Hinglish; regional scripts (Tamil, Bengali, Telugu, Gujarati) would benefit from extending `indic-transliteration` mappings.
2. **District Retention vs. Extremal A4 Outliers:** Districts (`Rae Bareli`, `Barabanki`, `Unnao`) and specific diagnosis/procedure phrases are preserved to keep 100% clinical utility (`4,855/4,855` probes), leaving ~`14.58%` probabilistic narrowing against an external voter registry when combined with broad age bands and gender; rare diagnosis roll-up (`E27.1 -> E27`) is configurable when utility probes permit.
3. **Unlinked Weak Fragments:** Records containing only an isolated given name or pronoun (`patient: null`) intentionally abstain from merging to uphold zero homonym false merges (`merged_people = 0`), trading minor recall on unseen chaos splits (`0.92–0.93`) for `1.00` precision.
