import os
import re
import math
import json
import sqlite3
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

from checklist.eval.bird.evaluation import execute_model

_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    sql = _COMMENT_BLOCK_RE.sub(" ", sql)
    sql = _COMMENT_LINE_RE.sub(" ", sql)
    return sql


def _strip_sql_quoted_literals(sql: str) -> str:
    """
    Best-effort removal of quoted regions so keyword checks (e.g. WHERE) don't
    accidentally match inside strings/identifiers.
    """
    out = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"', "`"):
            q = ch
            out.append(" ")
            i += 1
            while i < n:
                if sql[i] == q:
                    # SQL escapes quotes by doubling them: '' or "".
                    if i + 1 < n and sql[i + 1] == q:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _sql_has_where(sql: str) -> bool:
    if not sql:
        return False
    s = _strip_sql_quoted_literals(_strip_sql_comments(sql)).lower()
    return re.search(r"\bwhere\b", s) is not None


def _sql_looks_readonly(sql: str) -> bool:
    """
    Guardrail: only execute queries that look like SELECT/WITH statements.
    NL2SQL benchmarks should be SELECT-only; avoid any mutation.
    """
    if not sql:
        return False
    s = _strip_sql_comments(sql).lstrip().lower()
    return s.startswith("select") or s.startswith("with")


@dataclass(frozen=True)
class _Decision:
    verdict: bool | None
    confidence: float | None


def _extract_decision(j: dict, abbrev_key: str) -> _Decision:
    verdict = j.get("final_judgment", None)
    if verdict == "UNDETERMINED":
        verdict = None
    if not isinstance(verdict, bool):
        verdict = None

    conf = None
    payload = j.get(abbrev_key)
    if isinstance(payload, dict):
        c = payload.get("confidence", None)
        if isinstance(c, (int, float)):
            conf = float(c)

        # Some logs only store logprobs; convert them to a [0, 1] confidence.
        # Convention in our judgment JSONLs: confidence ~= mean(exp(avg_logprob_i)).
        def _conf_from_logprobs(v) -> float | None:
            if isinstance(v, (int, float)):
                try:
                    return float(math.exp(float(v)))
                except OverflowError:
                    return 0.0
            if isinstance(v, list):
                nums = []
                for x in v:
                    if isinstance(x, (int, float)):
                        nums.append(float(x))
                if not nums:
                    return None
                probs = []
                for lp in nums:
                    try:
                        probs.append(float(math.exp(lp)))
                    except OverflowError:
                        probs.append(0.0)
                return sum(probs) / len(probs)
            return None

        # If explicit confidence is missing or looks like a placeholder (0.0),
        # try deriving it from available logprobs.
        if conf is None or conf == 0.0:
            derived = None
            for k in ("avg_logprobs", "logprobs", "avg_logprob", "logprob"):
                if k in payload:
                    derived = _conf_from_logprobs(payload.get(k))
                    if derived is not None:
                        break
            if derived is not None:
                conf = derived
    return _Decision(verdict=verdict, confidence=conf)


def _pick_by_confidence(named: dict[str, _Decision]) -> bool | None:
    """
    Weighted confidence vote:
      - For each tester: add +confidence if verdict is True, else -confidence.
      - Sum all (available) confidences.
      - Return True if sum > 0, False if sum < 0, else None.
    """
    total = 0.0
    used = 0
    for _, d in named.items():
        if not isinstance(d.verdict, bool):
            continue
        if not isinstance(d.confidence, (int, float)):
            continue
        used += 1
        w = float(d.confidence)
        total += w if d.verdict else -w

    if used == 0:
        return None
    if total > 0:
        return True
    if total < 0:
        return False
    return None


def _open_sqlite_ro(db_path: str) -> sqlite3.Connection:
    # Open read-only to prevent accidental writes.
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda x: x.decode("utf-8", errors="replace")
    return conn


def _sql_result_is_empty(conn: sqlite3.Connection, sql: str) -> bool | None:
    """
    Returns:
      - True/False for SELECT/WITH queries that execute successfully
      - None for non-readonly-looking SQL or execution errors
    """
    if not _sql_looks_readonly(sql):
        return None
    try:
        cur = conn.execute(sql.replace("\\", ""))
        row = cur.fetchone()
        return row is None
    except sqlite3.Error:
        return None


_DEFAULT_TESTER_MEAN_SECONDS: dict[str, float] = {
    # Mean runtime per tester (seconds/example). Used to estimate end-to-end
    # pipeline time and summarize time distribution (min/q1/median/q3/max).
    "sem": 0.7044,
    "crs": 8.3,
    "slf": 5.4422,
    "orc": 8.8256,
    "nos": 9.1549,
    "nlr": 8.1283,
}
# _DEFAULT_TESTER_MEAN_SECONDS: dict[str, float] = {
#     # Mean runtime per tester (seconds/example). Used to estimate end-to-end
#     # pipeline time and summarize time distribution (min/q1/median/q3/max).
#     "sem": 0.1132,
#     "crs": 2.6271,
#     "slf": 4.2964,
#     "orc": 5.4558,
#     "nos": 4.5815,
#     "nlr": 5.3699,
# }
# Spider use



def _percentile_from_sorted(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 1], like numpy's default."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _add_tester_time(
    *,
    tester: str,
    time_box: list[float] | None,
    counted: set[str] | None,
    mean_seconds: dict[str, float] | None,
) -> None:
    if time_box is None:
        return
    if counted is not None and tester in counted:
        return
    if mean_seconds is None:
        mean_seconds = _DEFAULT_TESTER_MEAN_SECONDS
    t = mean_seconds.get(tester, None)
    if not isinstance(t, (int, float)):
        return
    time_box[0] += float(t)
    if counted is not None:
        counted.add(tester)


def one_vote_veto(
    *,
    idx: int,
    judgments: dict[str, list[dict]],
    keys: dict[str, str],
    check_used: defaultdict,
    db_id: str | None,
    db_path: str | None,
    conns: dict[str, sqlite3.Connection],
    sql: str,
    time_box: list[float] | None = None,
    counted: set[str] | None = None,
    mean_seconds: dict[str, float] | None = None,
) -> bool:
    """
    One-vote vetoing:
      1) SEM: if False -> short-circuit to final False
      2) ORC: only gate when SQL execution result is empty
    Returns True when veto triggers, else False.
    """
    sem_d = _extract_decision(judgments["sem"][idx], keys["sem"])
    check_used["sem"] += 1
    _add_tester_time(tester="sem", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if sem_d.verdict is False:
        return True

    empty_res = None
    if db_path and os.path.exists(db_path):
        # Only open DB when it exists; keep a per-db_id cache.
        conn_key = db_id if isinstance(db_id, str) else db_path
        if conn_key not in conns:
            conns[conn_key] = _open_sqlite_ro(db_path)
        empty_res = _sql_result_is_empty(conns[conn_key], sql)

    if empty_res is True:
        orc_d = _extract_decision(judgments["orc"][idx], keys["orc"])
        # orc_d = _Decision(verdict=None, confidence=orc_d.confidence)
        check_used["orc"] += 1
        _add_tester_time(tester="orc", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
        if orc_d.verdict is False:
            return True

    return False


def global_judgment(
    *,
    idx: int,
    judgments: dict[str, list[dict]],
    keys: dict[str, str],
    check_used: defaultdict,
    db_id: str | None,
    db_path: str | None,
    conns: dict[str, sqlite3.Connection],
    sql: str,
    majority: int = 3,
    time_box: list[float] | None = None,
    counted: set[str] | None = None,
    mean_seconds: dict[str, float] | None = None,
) -> bool | None:
    """
    Global decision logic; returns:
      - True/False when a final verdict can be produced
      - None when the example should be considered invalid/undetermined
    """
    # 1) One-vote veto.
    if one_vote_veto(
        idx=idx,
        judgments=judgments,
        keys=keys,
        check_used=check_used,
        db_id=db_id,
        db_path=db_path,
        conns=conns,
        sql=sql,
        time_box=time_box,
        counted=counted,
        mean_seconds=mean_seconds,
    ):
        return False

    # 2) majority voting
    vote = 0.0
    voted = {}
    crs_d = _extract_decision(judgments["crs"][idx], keys["crs"])
    # crs_d = _Decision(verdict=None, confidence=crs_d.confidence)
    check_used["crs"] += 1.0
    _add_tester_time(tester="crs", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if isinstance(crs_d.verdict, bool):
        vote = 1.5 if crs_d.verdict else -1.5
        voted["crs"] = crs_d
    
    orc_d = _extract_decision(judgments["orc"][idx], keys["orc"])
    # orc_d = _Decision(verdict=None, confidence=orc_d.confidence)
    check_used["orc"] += 1.0
    _add_tester_time(tester="orc", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if isinstance(orc_d.verdict, bool):
        voted["orc"] = orc_d
        if orc_d.verdict:
            vote += 1.0 
        else:
            vote += -1.5

    if abs(vote) >= majority: 
        return vote > 0

    slf_d = _extract_decision(judgments["slf"][idx], keys["slf"])
    # slf_d = _Decision(verdict=None, confidence=slf_d.confidence)
    check_used["slf"] += 1
    _add_tester_time(tester="slf", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if isinstance(slf_d.verdict, bool):
        voted["slf"] = slf_d
        if slf_d.verdict:
            vote += 1.0
        else:
            vote += -1.0
    
    if abs(vote) >= majority: 
        return vote > 0
    
    nos_d = _extract_decision(judgments["nos"][idx], keys["nos"])
    # nos_d = _Decision(verdict=None, confidence=nos_d.confidence)
    check_used["nos"] += 1.0
    _add_tester_time(tester="nos", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if isinstance(nos_d.verdict, bool):
        voted["nos"] = nos_d
        if _sql_has_where(sql) and not nos_d.verdict:
            vote += -1.5
        else:
            vote = vote + 1.0 if nos_d.verdict else vote - 1.0

    if abs(vote) >= majority:
        return vote > 0
    
    nlr_d = _extract_decision(judgments["nlr"][idx], keys["nlr"])
    nlr_d = _Decision(verdict=None, confidence=nlr_d.confidence)
    check_used["nlr"] += 1
    _add_tester_time(tester="nlr", time_box=time_box, counted=counted, mean_seconds=mean_seconds)
    if isinstance(nlr_d.verdict, bool):
        voted["nlr"] = nlr_d
        if nlr_d.verdict:
            vote += 1.0
        else:
            vote += -1.0

    if abs(vote) >= majority: 
        return vote > 0
    
    return _pick_by_confidence(voted)


def run_guardian_pipeline_evalution(
    *,
    data_file_path: str,
    db_root_path: str,
    sem_jsonl: str,
    nos_jsonl: str,
    orc_jsonl: str,
    crs_jsonl: str,
    slf_jsonl: str,
    nlr_jsonl: str,
    judge_name: str = "guardian-pipeline",
    benchmark_name: str = "nl2sql-bugs",
    out_jsonl: str | None = None,
):
    """
    Combine 6 Guardian check results with short-circuit logic to produce a final
    True/False per example, then report accuracy / confusion-matrix metrics and
    per-check usage counts.
    """
    data = json.load(open(data_file_path))
    # Abbrev key mapping (file name) -> top-level key in each JSONL line.
    keys = {
        "sem": "semantic_check",
        "orc": "oracle_result",
        "nos": "metamorphic_noise",
        "crs": "cross_model",
        "slf": "query_consistency",
        "nlr": "nl_review",
    }

    fps = {
        "sem": open(sem_jsonl),
        "orc": open(orc_jsonl),
        "nos": open(nos_jsonl),
        "crs": open(crs_jsonl),
        "slf": open(slf_jsonl),
        "nlr": open(nlr_jsonl),
    }

    # Read all lines once to ensure consistent lengths and allow random access.
    judgments = {k: [json.loads(line) for line in fp] for k, fp in fps.items()}
    for fp in fps.values():
        fp.close()

    lens = {"data": len(data), **{k: len(v) for k, v in judgments.items()}}
    n = min(lens.values())
    if n == 0:
        raise ValueError("No overlapping examples between dataset and judgment files.")
    if len(set(lens.values())) != 1:
        print(f"[warn] length mismatch; using first n={n} examples. lengths={lens}")

    if "spider" in benchmark_name:
        predicted_sql_path = "data/spider/results/resdsql-3b.sql"
        preds = [line.strip() for line in open(predicted_sql_path).readlines()]
        assert len(preds) == n

    check_used = defaultdict(int)
    invalids = 0
    acc = 0
    TP = FN = FP = TN = 0
    # Estimated pipeline runtime distribution (seconds/example). We use mean
    # per-tester times and the actual short-circuiting behavior of the pipeline.
    time_elapse = 0.0
    per_example_times: list[float] = []

    conns: dict[str, sqlite3.Connection] = {}
    out_fp = None
    if out_jsonl:
        out_dir = os.path.dirname(out_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(out_jsonl, "w", encoding="utf-8")
    try:
        for idx in tqdm(range(n), total=n):
            ex = data[idx]
            db_id = ex.get("db_id", None)
            db_path = None
            if isinstance(db_id, str):
                db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

            if "spider" in benchmark_name:
                gold_sql = ex['query']
                ret = execute_model(preds[idx], gold_sql, db_path, idx=-1, meta_time_out=30.0)
                gold = True if ret['res'] == 1 else False
            else:
                gold = ex.get("label", None)
            if not isinstance(gold, bool):
                if out_fp is not None:
                    json.dump({"final_judgment": "UNDETERMINED"}, out_fp, ensure_ascii=False)
                    out_fp.write("\n")
                invalids += 1
                continue
            
            if "spider" in benchmark_name:
                sql = preds[idx]
            else:
                sql = ex.get("sql", "")

            time_box = [0.0]
            counted: set[str] = set()
            pred = global_judgment(
                idx=idx,
                judgments=judgments,
                keys=keys,
                check_used=check_used,
                db_id=db_id,
                db_path=db_path,
                conns=conns,
                sql=sql,
                time_box=time_box,
                counted=counted,
                mean_seconds=_DEFAULT_TESTER_MEAN_SECONDS,
            )
            if out_fp is not None:
                json.dump(
                    {"final_judgment": pred if isinstance(pred, bool) else "UNDETERMINED"},
                    out_fp,
                    ensure_ascii=False,
                )
                out_fp.write("\n")
            per_example_times.append(time_box[0])
            time_elapse += time_box[0]
            if pred is None:
                invalids += 1
                continue
                

            if pred == gold:
                acc += 1

            if pred and gold:
                TP += 1
            elif (not pred) and gold:
                FN += 1
            elif pred and (not gold):
                FP += 1
            else:
                TN += 1
    finally:
        if out_fp is not None:
            try:
                out_fp.close()
            except Exception:
                pass
        for c in conns.values():
            try:
                c.close()
            except Exception:
                pass

    denom = n - invalids
    print(f"Evaluation Results of `{judge_name}` on `{benchmark_name}`:")
    if denom <= 0:
        print(f"No valid examples. Total={n}, Invalid={invalids}")
        return

    print(f"Total Accuracy: {acc/denom} ({acc}/{denom}), Invalid/Skipped: {invalids}")
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    positive_precision = TP / (TP + FP) if TP + FP > 0 else 0
    positive_recall = TP / (TP + FN) if TP + FN > 0 else 0
    negative_precision = TN / (TN + FN) if TN + FN > 0 else 0
    negative_recall = TN / (TN + FP) if TN + FP > 0 else 0
    f1 = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall) if positive_precision + positive_recall > 0 else 0
    print(f"PP: {positive_precision}, PR: {positive_recall}, NP: {negative_precision}, NR: {negative_recall}, F1: {f1}")
    print(
        "Checks used: "
        + ", ".join(f"{k}={check_used.get(k, 0)}" for k in ["sem", "orc", "nos", "crs", "slf", "nlr"])
    )
    if out_jsonl:
        print(f"Wrote final judgments to: {out_jsonl}")

    # Match run_judgment-style timing summary (but estimated).
    if per_example_times:
        times_sorted = sorted(per_example_times)
        n_time = len(times_sorted)
        avg_time = time_elapse / n_time
        t_min = times_sorted[0]
        q1 = _percentile_from_sorted(times_sorted, 0.25)
        median = _percentile_from_sorted(times_sorted, 0.50)
        q3 = _percentile_from_sorted(times_sorted, 0.75)
        t_max = times_sorted[-1]
        print(
            f"Test avg execution time <{judge_name}({benchmark_name})>: "
            f"{avg_time:.4f} seconds ({time_elapse:.4f}/{n_time})"
        )
        print(
            f"Execution time stats <{judge_name}({benchmark_name})>: "
            f"avg={avg_time:.4f}s, min={t_min:.4f}s, q1={q1:.4f}s, median={median:.4f}s, "
            f"q3={q3:.4f}s, max={t_max:.4f}s (total={time_elapse:.4f}s, n={n_time})"
        )
    else:
        print(f"Execution time stats <{judge_name}({benchmark_name})>: no samples (n=0)")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-name", default="guardian-pipeline")
    parser.add_argument("--benchmark-name", default="nl2sql-bugs")
    parser.add_argument("--db-root-path", default="data/bird/databases")
    parser.add_argument(
        "--data-file-path",
        default="data/nl2sql-bugs/NL2SQL-Bugs-with-evidence-and-difficulty-with-gold-sql.json",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="If set, file paths default to `{prefix}(sem|nos|orc|crs|slf|nlr).jsonl` unless overridden.",
    )
    parser.add_argument("--sem-jsonl", default=None)
    parser.add_argument("--nos-jsonl", default=None)
    parser.add_argument("--orc-jsonl", default=None)
    parser.add_argument("--crs-jsonl", default=None)
    parser.add_argument("--slf-jsonl", default=None)
    parser.add_argument("--nlr-jsonl", default=None)
    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="If set, write one JSON dict per example: {\"final_judgment\": <True|False|\"UNDETERMINED\">}.",
    )
    args = parser.parse_args()

    def _path_or_from_prefix(abbrev: str, explicit: str | None) -> str:
        if explicit:
            return explicit
        if args.prefix:
            return f"{args.prefix}({abbrev}).jsonl"
        raise SystemExit(f"Missing --{abbrev}-jsonl (or pass --prefix).")

    run_guardian_pipeline_evalution(
        data_file_path=args.data_file_path,
        db_root_path=args.db_root_path,
        sem_jsonl=_path_or_from_prefix("sem", args.sem_jsonl),
        nos_jsonl=_path_or_from_prefix("nos", args.nos_jsonl),
        orc_jsonl=_path_or_from_prefix("orc", args.orc_jsonl),
        crs_jsonl=_path_or_from_prefix("crs", args.crs_jsonl),
        slf_jsonl=_path_or_from_prefix("slf", args.slf_jsonl),
        nlr_jsonl=_path_or_from_prefix("nlr", args.nlr_jsonl),
        judge_name=args.judge_name,
        benchmark_name=args.benchmark_name,
        out_jsonl=args.out_jsonl,
    )
