# 02 — Record type `Nav`

**What it is:** navigation-message chunk record (variable-arity byte arrays).

**Volume [OBSERVED]:** 783,399 rows across **67 sessions** (files containing this type); no wall-clock column is declared for this record type (Nav carries no timestamp of its own — its time context comes from the surrounding Raw/Status rows).
Every declared column is documented below in the logger's header order — no exceptions. Numbers marked [OBSERVED] come from the per-column profile in `scratch/profiles/Nav.json` (approximate distinct counts via `approx_count_distinct`, quantiles via `approx_quantile`).

---

## `Svid`

**What:** Satellite ID the navigation-message bytes belong to.

**Why it matters:** Ties each nav chunk to a satellite.

**Use / project relevance:** Grouping; building per-satellite nav timelines.

**Caveats:** Meaningful only with ConstellationType — which this record type does NOT declare; infer from the accompanying Raw/Status rows by time.

**Status:** OBSERVED IN DATA + BACKGROUND (inference needed)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 47  *(exact)*
- **min:** 1
- **max:** 199
- **mean:** 15.1
- **std:** 14.84
- **p50:** 13
- **p95:** 32
- **p99:** 39

## `Type`

**What:** Nav-message source type code (e.g. 257 = GNSS nav message, 769 = …; an Android GnssNavigationMessage enum).

**Why it matters:** Tells which message stream the bytes come from.

**Use / project relevance:** Grouping by message family.

**Caveats:** 5 distinct values observed; decode against the Android API documentation in Phase C (REFERENCE).

**Status:** OBSERVED (5 values) + REFERENCE (decode pending)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 5  *(exact)*
- **min:** 257
- **max:** 1,537
- **mean:** 950.7
- **std:** 395
- **p50:** 769
- **p95:** 1,537
- **p99:** 1,537

## `Status`

**What:** Parity/health status of the received nav-message data (Android: STATUS_UNKNOWN=0, STATUS_PARITY_PASSED=1<<0, STATUS_PARITY_REBUILT=1<<1, STATUS_CONTENT_UNKNOWN...).

**Why it matters:** Only parity-passed data is usable.

**Use / project relevance:** Filter for nav analyses.

**Caveats:** Constant 1 observed (parity passed) — good; constant → dead weight as a feature.

**Status:** OBSERVED (constant 1) + REFERENCE (decode)

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 1  *(exact)*
- **min:** 1
- **max:** 1
- **mean:** 1
- **std:** 0
- **p50:** 1
- **p95:** 1
- **p99:** 1

## `MessageId`

**What:** Message identifier within the stream (e.g. subframe/page id); −1 when unknown.

**Why it matters:** Sequences the message chunks.

**Use / project relevance:** Reconstructing full messages.

**Caveats:** −1 observed on some rows (unknown); ranges −1..127.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 125  *(approx_count_distinct)*
- **min:** -1
- **max:** 127
- **mean:** 6.583
- **std:** 9.932
- **p50:** 4
- **p95:** 23
- **p99:** 50
- **negatives:** 76,174 (9.7%)

## `SubMessageId`

**What:** Sub-message identifier (page/word) within the message; −1 when unknown.

**Why it matters:** Finer sequencing of chunks.

**Use / project relevance:** Same as MessageId.

**Caveats:** −1 when unknown.

**Status:** OBSERVED IN DATA + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 22  *(exact)*
- **min:** 0
- **max:** 40
- **mean:** 6.663
- **std:** 5.017
- **p50:** 5
- **p95:** 16
- **p99:** 19
- **zeros:** 36,334 (4.6%)

## `nDataBytes`

**What:** Number of bytes in the payload (typed-cache column replacing a duplicate declared field; values 11/…/40; also −5 artifacts from arity quirks).

**Why it matters:** Sanity-checks the payload length.

**Use / project relevance:** Filtering malformed nav rows.

**Caveats:** min −5 observed — negative lengths are arity anomalies; 8,595 rows with empty payload.

**Status:** OBSERVED (−5 anomalies) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `BIGINT`; **nulls:** 0 (0.0%)
- **Distinct:** 3  *(exact)*
- **min:** 11
- **max:** 40
- **mean:** 21.28
- **std:** 12.16
- **p50:** 11
- **p95:** 40
- **p99:** 40

## `DataBytes`

**What:** The raw navigation-message payload as a comma-separated byte array (variable arity! 6th field onward).

**Why it matters:** The actual bits of the nav message (ephemerides, clock, ionosphere model...).

**Use / project relevance:** Out of scope for C/N0 prediction; kept for completeness of column documentation.

**Caveats:** Variable arity breaks naive CSV parsing; 8,595 rows with empty payload and nDataBytes = −5 anomalies observed (arity quirks).

**Status:** OBSERVED IN DATA (arity anomalies) + BACKGROUND

**Observed in this corpus:**
- **DuckDB type:** `VARCHAR`; **nulls:** 0 (0.0%)
- **Distinct:** 243,616  *(approx_count_distinct)*
- **Top values:** `8,-81,0,0,0,-92,0,0,-128,64,5` = 3,496, `4,46,0,0,0,-92,0,0,-128,64,5` = 2,536, `-66,-13,-14,55,110,-105,22,96,-13,68,7` = 2,472, `102,1,68,2,88,15,-51,63,-83,4,10` = 2,456, `72,83,-11,-92,-66,-105,112,21,-13,-123,7` = 2,452, `30,-122,77,-125,-104,24,123,-76,-75,23,6` = 2,442, `88,-11,-117,2,108,12,73,46,-82,-126,12` = 2,436, `-128,34,-11,23,-74,-105,2,24,-50,-94,15` = 2,432

