# sqlegit

Judge/evaluate NL2SQL predictions (Spider/BIRD/NL2SQL-Bugs style datasets) with either:

- an LLM judge (single model decision), or
- a "Guardian" test suite (semantic checks + metamorphic/oracle/self-consistency style tests).

This repo does **not** ship benchmark data/DBs; you point it at your local dataset JSON + SQLite DB folder.

## Quickstart (Docker)

Build:

```bash
docker build -t sqlegit .
```

Run (mount your data; pass env via `--env-file`):

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/work" \
  -w /work \
  sqlegit run_judgment.py --help
```

## Local Setup (Python)

Python 3.10 is recommended (the Docker image uses Python 3.10).

Install dependencies (see `Dockerfile` for the pinned list) and export env vars (or use a local `.env`).

At minimum you typically need:

- `TEST_INSTANCE_ROOT_PATH`: where testers write artifacts/caches (e.g. `test_cases/`)
- one LLM provider credential set:
  - Azure OpenAI via `langchain-openai` (e.g. `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`), or
  - DeepSeek via `langchain-deepseek` (`DEEPSEEK_API_KEY`)

Notes:
- `.env` is loaded by the tester stack (via `python-dotenv`) and should **not** be committed.
- You can control Azure call timeout with `CHECKLIST_LLM_TIMEOUT_S` / `LLM_TIMEOUT_S` (seconds).

## Run Judgments (`run_judgment.py`)

This is the main entrypoint. It reads a dataset JSON and writes a JSONL file under a sibling `results/` directory.

Show CLI:

```bash
python run_judgment.py --help
```

### Spider/BIRD-style run (with a prediction file)

Requirements:
- dataset JSON contains at least `db_id`, `question`, and `query` or `SQL`
- for BIRD, `evidence` is used as a hint
- `--predicted_sql_path` points at a text file with one SQL per line (BIRD/Spider); the code reads the first tab-separated field if present
- `--schema_file_path` is a JSON list with `db_id` entries (Spider `tables.json`-style)
- `--db_root_path` contains per-db folders: `<db_id>/<db_id>.sqlite`

Example (LLM judge):

```bash
python run_judgment.py \
  --judge_name gpt-4o-mini-0708 \
  --benchmark_name spider \
  --db_root_path /path/to/spider/database \
  --data_file_path /path/to/spider/dev.json \
  --schema_file_path /path/to/spider/tables.json \
  --predicted_sql_path /path/to/preds.sql
```

### NL2SQL-Bugs-style run (dataset provides SQL + label)

Requirements:
- dataset JSON contains at least `db_id`, `question`, `evidence`, `sql`, `label`
- DB layout is still `<db_root_path>/<db_id>/<db_id>.sqlite`

Example (Guardian suite, multiple checks):

```bash
python run_judgment.py \
  --judge_name "guardian+sem+nos+crs+orc+slf+nlr(gpt-4o-mini-0708)" \
  --benchmark_name nl2sql-bugs \
  --db_root_path /path/to/databases \
  --data_file_path /path/to/NL2SQL-Bugs.json \
  --schema_file_path /path/to/tables.json
```

How `--judge_name` is interpreted:
- if it contains `guardian` (case-insensitive), the runner builds a `GuardianJudge`
  - backbone model is parsed from the name if it contains `gpt-...` (defaults to `gpt-4o-mini-0708`)
  - any of these substrings enable the matching check: `sem`, `nos`, `crs`, `orc`, `nlr`, `slf`
- otherwise it runs `LLMJudge` with `model_name = --judge_name`

### Resume an interrupted run

`--append_mode` continues writing into the same output JSONL (it counts existing lines and resumes from there):

```bash
python run_judgment.py ... --append_mode
```

### Evaluate an existing judgment file

`--eval_mode` reads the produced JSONL and prints accuracy/confusion-matrix style metrics:

```bash
python run_judgment.py \
  --eval_mode \
  --judge_name guardian+sem+nos+crs+orc+slf+nlr \
  --benchmark_name spider \
  --db_root_path /path/to/dbs \
  --data_file_path /path/to/dev.json \
  --schema_file_path /path/to/tables.json \
  --predicted_sql_path /path/to/preds.sql
```

## Combine Per-Check Outputs (`run_consensus.py`)

If you run Guardian checks **separately** (one JSONL per check), you can combine them with a short-circuit + weighted vote policy:

```bash
python run_consensus.py --help
```

To generate the per-check JSONLs, run Guardian with only one check enabled each time, e.g.:

```bash
python run_judgment.py \
  --judge_name "guardian+sem(gpt-4o-mini-0708)" \
  --benchmark_name nl2sql-bugs \
  --db_root_path /path/to/databases \
  --data_file_path /path/to/NL2SQL-Bugs.json \
  --schema_file_path /path/to/tables.json
```

Example:

```bash
python run_consensus.py \
  --benchmark-name nl2sql-bugs \
  --db-root-path /path/to/databases \
  --data-file-path /path/to/NL2SQL-Bugs.json \
  --sem-jsonl /path/to/judgments_sem.jsonl \
  --nos-jsonl /path/to/judgments_nos.jsonl \
  --orc-jsonl /path/to/judgments_orc.jsonl \
  --crs-jsonl /path/to/judgments_crs.jsonl \
  --slf-jsonl /path/to/judgments_slf.jsonl \
  --nlr-jsonl /path/to/judgments_nlr.jsonl \
  --out-jsonl /path/to/final_consensus.jsonl
```

Safety note: `run_consensus.py` may execute the predicted SQL against SQLite to check if the result set is empty; it opens DBs in **read-only** mode and only executes queries that look like `SELECT`/`WITH`.

## Output Files

`run_judgment.py` writes under:

- a sibling `results/` directory next to your `--data_file_path` (i.e. `<data_dir>/results/`)
- file name pattern:
  - `judgments,dataset=<benchmark>[+<nl2sql_model>],judge=<judge_name>[+3-shots][+cot].jsonl`

Each line is a JSON dict containing:
- `final_judgment`: `true` / `false` / `"UNDETERMINED"`
- one key per enabled Guardian check (e.g. `semantic_check`, `oracle_result`, ...), each with details like:
  - `judgment`, `confidence`, `tokens_used`, `elapsed`, per-test-case results, traces, etc.

## Repo Layout

- `src/`: core library (judges, testers, DB utils, RED parser)
- `templates/`: prompt templates used by the LLM-based components
- `scripts/`: helper scripts for analysis/plotting/one-off utilities
- `run_judgment.py`: run a judge and write JSONL
- `evalution.py`: evaluate judgments against gold correctness/labels (name kept for compatibility)
- `run_consensus.py`: combine per-check Guardian outputs into a single final decision
