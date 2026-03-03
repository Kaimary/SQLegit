# SQLegit: A Testing Framework for Trustworthy NL2SQL
> Determine the correctness of NL2SQL translations using testing techniques.

<p align="center">
   <a href="https://github.com/Kaimary/SQLegit/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/Kaimary/SQLegit" />
   </a>
   <a href="https://github.com/Kaimary/SQLegit/network/members">
       <img alt="forks" src="https://img.shields.io/github/forks/Kaimary/SQLegit?color=FF8000" />
   </a>
   <a href="https://github.com/Kaimary/SQLegit/issues">
      <img alt="issues" src="https://img.shields.io/github/issues/Kaimary/SQLegit?color=0088ff" />
   </a>
   <br />
</p>

## Overview

## About SQLegit

**TL;DR:** SQLegit is an evaluation harness for NL2SQL systems. Given an NL2SQL translation, it runs a test suite with fast-running test cases to provide a correctness judgment.

## Quick Start

### Prerequisites

- Python 3.10 is recommended (the Docker image uses Python 3.10).
- A dataset JSON (see the run modes below) and SQLite DB folder.
- One LLM provider credential set:
  - Azure OpenAI via `langchain-openai` (e.g. `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`), or
  - DeepSeek via `langchain-deepseek` (`DEEPSEEK_API_KEY`)

### Docker

Build:

```bash
docker build -t sqlegit .
```

Run (mount your data; pass env via `--env-file`):

```bash
docker run --rm -it \
  --env-file .env \
  -v "/your-data-path:/SQLegit/data" \
  sqlegit /bin/bash
```

## Try It

`run_judgment.py` is the main entrypoint. It reads a dataset JSON and writes a JSONL file under a sibling `results/` directory.

```bash
python run_judgment.py --help
```


Requirements:
- dataset JSON contains at least `db_id`, `question`, and `query` or `SQL`
- for BIRD, `evidence` is used as a hint
- `--predicted_sql_path` points at a text file with one SQL per line (BIRD/Spider); the code reads the first tab-separated field if present
- `--schema_file_path` is a JSON list with `db_id` entries (Spider `tables.json`-style)
- `--db_root_path` contains per-db folders: `<db_id>/<db_id>.sqlite`

Example:

```bash
python run_judgment.py \
  --judge_name "sqlegit+sem+nos+crs+orc+slf+nlr(gpt-4o-mini-0708)" \
  --benchmark_name nl2sql-bugs \
  --db_root_path /path/to/databases \
  --data_file_path /path/to/NL2SQL-Bugs.json \
  --schema_file_path /path/to/tables.json
```

How `--judge_name` is interpreted:
- if it contains `sqlegit` (case-insensitive), the runner builds a `SQLegitJudge`
  - backbone model is parsed from the name if it contains `gpt-...` (defaults to `gpt-4o-mini-0708`)
  - any of these substrings enable the matching check: `sem`, `nos`, `crs`, `orc`, `nlr`, `slf`

## Consensus (`run_consensus.py`)

If you run SQLegit checks **separately** (one JSONL per check), you can combine them with a short-circuit + weighted vote policy:

```bash
python run_consensus.py --help
```

To generate the per-check JSONLs, run SQLegit with only one check enabled each time, e.g.:

```bash
python run_judgment.py \
  --judge_name "sqlegit+sem(gpt-4o-mini-0708)" \
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

## Output Files

`run_judgment.py` writes under:

- a sibling `results/` directory next to your `--data_file_path` (i.e. `<data_dir>/results/`)
- file name pattern:
  - `judgments,dataset=<benchmark>[+<nl2sql_model>],judge=<judge_name>[+3-shots][+cot].jsonl`

Each line is a JSON dict containing:
- `final_judgment`: `true` / `false` / `"UNDETERMINED"`
- one key per enabled SQLegit check (e.g. `semantic_check`, `oracle_result`, ...), each with details like:
  - `judgment`, `confidence`, `tokens_used`, `elapsed`, per-test-case results, traces, etc.

## Code Structure

```shell
|-- src/
|   |-- judges/         # LLMJudge + SQLegitJudge implementations
|   |-- testers/        # semantic/noise-row/cross-model/oracle/self-consistency/... checks
|   |-- db_utils/       # schema parsing + execution helpers
|   |-- eval/           # evaluation scripts (Spider/BIRD-style)
|   |-- evalution.py    # evaluation entrypoints (name kept for compatibility)
|-- templates/          # prompt templates for LLM-based components
|-- assets/             # design diagrams (PDF)
|-- run_judgment.py     # run a judge and write JSONL
|-- run_consensus.py    # combine per-check SQLegit outputs into a final decision
```

## Contributing

Contributions and suggestions are welcome.

If you find bugs, encounter problems when running the code, or have suggestions for SQLegit, please submit an issue or reach out to me (kaimary1221@163.com).
