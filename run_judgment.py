import os
import sys
import time
import json
import math
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from evalution import run_evalution, run_nl2sql_bugs_evalution
from checklist.judges.llm_judge import LLMJudge
from checklist.utils import createJudge, get_data_from_bench, get_red_schemas, print_summary

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--judge_name', type=str, required=True, default='')
    args_parser.add_argument('--enable_few_shots', action='store_true')
    args_parser.add_argument('--enable_cot', action='store_true')
    args_parser.add_argument('--benchmark_name', type=str, required=True, default='')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--data_file_path', type=str, required=True, default='')
    args_parser.add_argument('--schema_file_path', type=str, required=True, default='')
    args_parser.add_argument('--nl2sql_model_name', type=str)
    args_parser.add_argument('--predicted_sql_path', type=str)
    args_parser.add_argument('--eval_mode', action='store_true')
    args_parser.add_argument('--append_mode', action='store_true')
    args_parser.add_argument('--debug', action='store_true')
    args = args_parser.parse_args()
    # prepare output file
    output_file_dir = os.path.join(os.path.dirname(args.data_file_path), "results")
    nl2sql = getattr(args, 'nl2sql_model_name', None)
    output_file_name = (
        f"judgments,dataset={args.benchmark_name}"
        f"{f'+{nl2sql}' if nl2sql else ''},"
        f"judge={args.judge_name}"
        f"{'+3-shots' if args.enable_few_shots else ''}"
        f"{'+cot' if args.enable_cot else ''}.jsonl"
    )
    output_file_path = os.path.join(output_file_dir, output_file_name)
    logging.basicConfig(
        level=logging.INFO,
        filename=(
            f'output/logs/{datetime.now():%Y%m%d_%H%M%S}_{Path(output_file_name).stem}.log'
            if not args.debug else None
        ),
        filemode='a',
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        force=True,
    )
    # load to-test data
    data = json.load(open(args.data_file_path))
    # initialize the judge
    judge = createJudge(args.judge_name, enable_few_shots=args.enable_few_shots, enable_cot =args.enable_cot)
    start_idx = 0
    if args.append_mode and os.path.exists(output_file_path):
        start_idx = sum(1 for _ in open(output_file_path))
        data = data[start_idx:]
    if args.eval_mode:
        run_evalution(judge_name=args.judge_name, benchmark_name=args.benchmark_name,
                      judgment_file_path=output_file_path, data_file_path=args.data_file_path, 
                      db_root_path=args.db_root_path, predicted_sql_path=args.predicted_sql_path)
        sys.exit(0)

    time_elapse = 0
    per_example_times = []
    token_elapse = 0
    per_example_tokens = []
    missing_token_used = 0
    # Progress logging (helps with large datasets / mid-run interruptions).
    progress_every = 50
    run_wall_start = time.time()
    window = {
        "time_elapse": 0.0,
        "token_elapse": 0.0,
        "token_samples": 0,
        "missing_token_used": 0,
        "n": 0,
    }

    def _fmt_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:05.2f}"
        return f"{m:d}:{s:05.2f}"

    def _maybe_print_progress(done: int, total: int) -> None:
        if done <= 0:
            return
        if done % progress_every != 0 and done != total:
            return
        wall_elapsed = time.time() - run_wall_start
        avg_time = (time_elapse / done) if done else float("nan")
        eta = avg_time * (total - done) if math.isfinite(avg_time) else float("nan")
        avg_tok = (token_elapse / len(per_example_tokens)) if per_example_tokens else float("nan")
        window_avg_time = (window["time_elapse"] / window["n"]) if window["n"] else float("nan")
        window_avg_tok = (
            (window["token_elapse"] / window["token_samples"]) if window["token_samples"] else float("nan")
        )

        msg = (
            f"[progress {done}/{total}] "
            f"time wall={_fmt_seconds(wall_elapsed)}, judge_sum={_fmt_seconds(time_elapse)}, "
            f"avg={avg_time:.4f}s/ex, last{window['n']}={window_avg_time:.4f}s/ex, "
            f"ETA={_fmt_seconds(eta)}; "
            f"tokens total={token_elapse:.0f}, avg={avg_tok:.2f}/ex "
            f"(avail={len(per_example_tokens)}/{done}, missing={missing_token_used}), "
            f"last{window['token_samples']} avg={window_avg_tok:.2f}/ex "
            f"(missing={window['missing_token_used']})"
        )
        print(msg, flush=True)
        logging.info(msg)
        # Reset window counters for the next chunk.
        window["time_elapse"] = 0.0
        window["token_elapse"] = 0.0
        window["token_samples"] = 0
        window["missing_token_used"] = 0
        window["n"] = 0

    def _percentile_from_sorted(sorted_vals, p: float) -> float:
        """Linear-interpolated percentile, p in [0, 1], like numpy's default."""
        if not sorted_vals:
            return float("nan")
        if p <= 0:
            return sorted_vals[0]
        if p >= 1:
            return sorted_vals[-1]
        k = (len(sorted_vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    judgments = open(output_file_path, 'a+')
    red_schemas = get_red_schemas(data, args.db_root_path, args.schema_file_path)
    for idx, ex in tqdm(enumerate(data), total=len(data), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        db_id, db_path, nl, hint, pred, gold, judgment_gold_label = get_data_from_bench(
            ex, idx+start_idx, args.benchmark_name, args.predicted_sql_path, args.db_root_path)
        
        # debugging info
        logging.debug(f"NL query: {nl}\nPredicted SQL: {pred}\nGold: {gold}")
        print(f"\033[94m\nNatural Language: \033[92m{nl}\033[0m")
        print(f"\033[94mPredicted: \033[92m{pred}\033[0m")
        if gold: print(f"\033[94mGold: \033[91m{gold}\033[0m")

        judge.set(nl, hint, pred, gold, db_id, db_root_path=args.db_root_path, red_schema=red_schemas[db_id])
        # run the judge
        start=time.time()
        ret, munch = judge.run()
        elapsed = time.time() - start

        token_used = None
        if isinstance(judge, LLMJudge):
            token_used = ret.get("token_used", None)
        else:
            for key, value in ret.items():
                if isinstance(value, dict) and "tokens_used" in value:
                    token_used = value["tokens_used"]
                    break
        if token_used is None:
            missing_token_used += 1
            window["missing_token_used"] += 1
        else:
            # token_used is expected to be numeric (usually int), but coerce defensively.
            try:
                token_used_num = float(token_used)
            except (TypeError, ValueError):
                missing_token_used += 1
                window["missing_token_used"] += 1
            else:
                per_example_tokens.append(token_used_num)
                token_elapse += token_used_num
                window["token_elapse"] += token_used_num
                window["token_samples"] += 1

        # print out judgment summary if not beat baseline/ground-truth
        print_summary(judge, ret, munch, idx+start_idx, judgment_gold_label, output_file_name, output_file_dir)
        # update output
        json.dump(ret, judgments)
        judgments.write('\n')
        judgments.flush()
        per_example_times.append(elapsed)
        time_elapse += elapsed
        window["time_elapse"] += elapsed
        window["n"] += 1

        _maybe_print_progress(done=idx + 1, total=len(data))

    if per_example_times:
        times_sorted = sorted(per_example_times)
        n = len(times_sorted)
        avg = time_elapse / n
        t_min = times_sorted[0]
        q1 = _percentile_from_sorted(times_sorted, 0.25)
        median = _percentile_from_sorted(times_sorted, 0.50)
        q3 = _percentile_from_sorted(times_sorted, 0.75)
        t_max = times_sorted[-1]
        # Backward-compatible line (some users/scripts may grep this).
        print(
            f"Test avg execution time <{args.judge_name}({args.benchmark_name})>: "
            f"{avg:.4f} seconds ({time_elapse:.4f}/{n})"
        )
        print(
            f"Execution time stats <{args.judge_name}({args.benchmark_name})>: "
            f"avg={avg:.4f}s, min={t_min:.4f}s, q1={q1:.4f}s, median={median:.4f}s, "
            f"q3={q3:.4f}s, max={t_max:.4f}s (total={time_elapse:.4f}s, n={n})"
        )
    else:
        print(f"Execution time stats <{args.judge_name}({args.benchmark_name})>: no samples (n=0)")

    if per_example_tokens:
        tokens_sorted = sorted(per_example_tokens)
        n_tok = len(tokens_sorted)
        avg_tok = token_elapse / n_tok
        tok_min = tokens_sorted[0]
        tok_q1 = _percentile_from_sorted(tokens_sorted, 0.25)
        tok_median = _percentile_from_sorted(tokens_sorted, 0.50)
        tok_q3 = _percentile_from_sorted(tokens_sorted, 0.75)
        tok_max = tokens_sorted[-1]
        # Use integer-ish formatting if the values are whole numbers.
        fmt = (lambda x: f"{int(x)}" if float(x).is_integer() else f"{x:.4f}")
        print(
            f"Token usage stats <{args.judge_name}({args.benchmark_name})>: "
            f"avg={fmt(avg_tok)}, min={fmt(tok_min)}, q1={fmt(tok_q1)}, median={fmt(tok_median)}, "
            f"q3={fmt(tok_q3)}, max={fmt(tok_max)} "
            f"(total={fmt(token_elapse)}, n={n_tok}, missing={missing_token_used})"
        )
    else:
        print(
            f"Token usage stats <{args.judge_name}({args.benchmark_name})>: "
            f"no samples (n=0, missing={missing_token_used})"
        )

    run_evalution(judge_name=args.judge_name, benchmark_name=args.benchmark_name,
                  judgment_file_path=output_file_path, data_file_path=args.data_file_path, 
                  db_root_path=args.db_root_path, predicted_sql_path=args.predicted_sql_path)
    # if not args.benchmark_name in ["spider", "bird"]:
    #     run_nl2sql_bugs_evalution(
    #         judge_name=args.judge_name, judgment_file_path=output_file_path, data_file_path=args.data_file_path)
    # print(f"Log is saved to {log_file_path}\nJudgment results saved to {output_file_path}")