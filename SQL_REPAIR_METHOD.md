# SQL verification and repair (`sql_repair/`)

> This stage is **not part of the paper submission repository**, but the reported BIRD-dev 73.7 was
> obtained on the repaired candidate pool.
> Reference implementation: the `sql_revision` module of DeepEye-SQL (SIGMOD 2026). This directory is
> an independent reimplementation of it, keeping only the reward-blind parts and adapting it to our
> candidate-pool format.

## 1. Where it sits and what it does

```
32 candidate SQLs --> (1) execution cache --> (2) [THIS MODULE] serial checker chain
                  --> (3) rebuilt execution cache --> (4) clustering + tournament
```

- **Repair raises the ceiling**: it rescues candidates that error out, return an empty table, or hit a
  known anti-pattern, so pass@32 goes up.
- **The tournament does the choosing**: it picks between clusters. The two are complementary, not
  competing.
- Repair changes SQL => the execution signature changes => you **must** rebuild the execution cache,
  re-cluster, and re-run the tournament under a new verdict-cache name.

## 2. The checker chain

One candidate passes **serially** through every checker in the chain, the output of one feeding the
next (`revise_chain`). Production uses `DEFAULT_CHAIN`:

| Order | checker | Fires when | How it repairs | Calls an LLM |
|---|---|---|---|---|
| 1 | `syntax` | result type not in {success, empty_result, all_null_result} (i.e. an error or a timeout) | feed question + schema + SQL + **the error message** to the LLM, sampling `budget` repaired versions | yes (temp 0.8) |
| 2 | `time` | a bare year number such as `strftime(...) >= 2020` | pure regex: quote the year | no |
| 3 | `select` | (a) `SELECT a \|\| ' ' \|\| b` (string concatenation) (b) `SELECT T.*` (ambiguous star) | (a) replace with comma-separated columns (deterministic) (b) a targeted suggestion asking the LLM to use the corresponding id column | (a) no (b) yes (temp 0) |
| 4 | `result` | result type != success (**empty tables and all-NULL results are repaired too**) | same as `syntax`, but only repaired versions that return a non-empty result are accepted | yes (temp 0.8) |

`FULL_CHAIN` contains four further rule-based checkers (`join` / `order_by_limit` / `max_min` /
`order_by_null`). The code is kept complete, but **the reported numbers do not use them**.

### Key design of the execution-driven checkers: self-consistency over repairs

`_exec_repair` has the LLM sample `budget` repaired versions -> **executes all of them** -> keeps only
those that run (result type in the accepted set) -> picks one by **majority vote over execution-result
signatures** (`_self_consistency_pick`). This is reward-blind self-consistency, the same idea as
"cluster by execution signature and take the majority" in stage 4.

### Key design of the rule-based checkers: a tightly constrained prompt

`COMMON_CHECKER_PROMPT` states explicitly `You are NOT ALLOWED to do any other modifications which are
not listed in the given suggestions`, with `n=1` and `temperature=0`. The point is to stop the model
from improvising and making things worse.

## 3. Parameters

```bash
BON_DATASET=bird BON_DB_DIR=<local db dir> BIRD_FILE=<candidate pool> EXEC_CACHE=<original exec cache> \
python repair_candidates_generic.py \
    --out-file <repaired_pool.json> --out-cache <repaired_exec_cache.json> \
    --budget 6 --chain default --workers 32 [--no-gold]
```

| Parameter | Value | Meaning / rationale |
|---|---|---|
| `--chain` | `default` | the production chain (table above); `full` is the complete 8-checker chain and was not used for the reported numbers |
| `--budget` | `6` | how many repaired versions the LLM samples when an execution-driven checker fires; production value |
| `--workers` | `32` | how many unique phrasings are repaired concurrently |
| `--no-gold` | add it for the test submission | do not recompute reward (no gold available). The repair logic itself is independent of gold |
| `BON_DATASET` | `bird` / `spider` | only affects the reward-relabelling function for **changed** candidates: bird = set comparison (column order strict, row order ignored); spider = eval_exec_match (enumerating column permutations) |
| `REPAIR_MODEL` | default `glm-5.2` | the LLM used for repair, with thinking off. Credentials come from `JUDGE_API_KEY` / `JUDGE_API_BASE` |
| execution timeout | 15s (`EXEC_TIMEOUT`) | a thread Timer plus `conn.interrupt()`, so it can never hang |

### Scope: only the multi-result questions

Questions are grouped by the signature in the original execution cache:
- **single-result** (all 32 share one signature) -> skipped. There is no disagreement on this question,
  so repairing it cannot change the final answer.
- **multi-result** (>=2 signatures) -> repaired.

Verified: repairing only the multi-result questions gives the **same final metric** as repairing
everything, while saving about half of the LLM calls.
(Main pool: 913 single-result questions skipped / 621 multi-result questions repaired; 13.6k unique
phrasings, of which 7.9k needed repair.)

## 4. Gain (BIRD old-dev, 14B step192, 1534 questions)

| Metric | Before repair | After repair | Delta |
|---|---|---|---|
| majority-vote EX | 71.58 | 72.36 | **+0.78** |
| pass@32 ceiling | 80.77 | 81.36 | **+0.59** |
| contested (questions needing adjudication) | 435 | 423 | -12 |
| final best-of-32 EX | 73.40 | **73.66** | **+0.26** |

Amount changed: 648 of 13,626 unique phrasings (824 candidates); only a minority of erroring or
empty-result candidates actually trigger an LLM call.

> **Warning: on near-saturated datasets repair is a liability.** Spider-dev majority voting is already
> around 89%, and repair breaks a few already-correct SQLs and lowers pass@32. The test: run
> `report_baseline.py` after repair, and if majority voting or pass@n **drops**, skip repair and use the
> original pool. On BIRD it is a net gain.

## 5. Hard rules

1. **Reward-blind**: only "does it execute / result shape / regex anti-patterns / self-consistency" may
   be used. Never gold, never reward.
2. **Never overwrite the original candidate pool**: the output goes to a new `_repaired_v1` file
   (`--out-file`), together with a new execution cache (`--out-cache`).
3. **The number of candidates never changes**: `trials[].pred_sql` is replaced in place, keeping all 32
   — the `n_i/N` prior of the downstream clustering depends on it.
4. **Reward is recomputed only at final evaluation**, and only for the candidates that were **changed**;
   unchanged ones keep their original label.
5. **The databases must be on a local disk**, otherwise random sqlite I/O errors make the very signal
   "can this execute" unreliable.
