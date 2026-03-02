import json
import os
import re
import pickle
from pathlib import Path
from tqdm import tqdm

from src.judges.llm_judge import LLMJudge
from src.judges.guardian_judge import GuardianJudge
from src.eval.bird.evaluation import execute_model
from src.red.parser.schema import Schema
from src.db_utils.db_info import get_db_schema_from_json
from src.testers.cross_model_tester import CrossModelTester
from src.testers.nl_review_tester import NLReviewTester
from src.testers.noise_row_tester import NoiseRowTester
from src.testers.oracle_result_tester import OracleResultTester
from src.testers.self_consistency_tester import SelfConsistencyTester
from src.testers.semantic_check_tester import SemanticCheckTester

DEFAULT_BACKBONE_MODEL_NAME = "gpt-4o-mini-0708"
TEST_CLASS_MAP = {
    "sem": SemanticCheckTester,
    "nos": NoiseRowTester,
    "crs": CrossModelTester,
    "orc": OracleResultTester,
    "nlr": NLReviewTester,
    "slf": SelfConsistencyTester,
}
    # "test": TestingTester,
    # "qry": QueryReviewTester,    
    # "syn": MinimumSyntaxTester,
    # "lax": NLRelaxTester,
    # "stn": NLStrengthenTester,
_PRED_CACHE = {}

def get_data_from_bench(ex, idx, bench_name, predicted_sql_path, db_root_path):
    db_id = ex['db_id']
    db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
    nl = ex['question']
    hint = ""
    pred, gold, judgment_gold_label=None, None, None
    if  "spider" in bench_name or "bird" in bench_name:
        if "bird" in bench_name: hint = ex['evidence']
        assert predicted_sql_path is not None
        if predicted_sql_path not in _PRED_CACHE:
            with open(predicted_sql_path) as f:
                _PRED_CACHE[predicted_sql_path] = [
                    line.strip().split("\t")[0] for line in f
                ]
        pred = _PRED_CACHE[predicted_sql_path][idx]
        gold = ex['SQL'] if 'SQL' in ex else ex['query']
        ret = execute_model(pred, gold, db_path, idx=-1, meta_time_out=30.0)
        judgment_gold_label = ret['res'] == 1
    elif "nl2sql-bugs" in bench_name:
        hint = ex['evidence']
        pred = ex['sql']
        gold = ex.get('gold_sql', None)
        judgment_gold_label = ex['label']
    
    return db_id, db_path, nl, hint, pred, gold, judgment_gold_label
    
def createJudge(judge_name, enable_few_shots=False, enable_cot=False):
    judge = None
    if "guardian" in judge_name.lower():
        match = re.search(r"gpt[-\w.]+", judge_name.lower())
        backbone = match.group() if match else DEFAULT_BACKBONE_MODEL_NAME
        tests = [cls for key, cls in TEST_CLASS_MAP.items() if key in judge_name.lower()]
        if not tests:
            raise ValueError(f"No matching test class found for '{judge_name}'")
        judge = GuardianJudge(judge_name, backbone, *tests)
    else:
        judge = LLMJudge(judge_name, model_name=judge_name, 
                         enable_few_shot=enable_few_shots,
                         enable_cot=enable_cot)
    
    return judge

def print_summary(judge, ret, munch, idx, judgment_gold_label, output_file_name, output_file_dir):
    if not isinstance(judge, GuardianJudge): return
    if ret['final_judgment'] == "UNDETERMINED": return

    judgment_baseline_label = None
    baseline_output_file_name = re.sub(r"guardian\+|\([^)]*\)", "", output_file_name)
    baseline_output_file_path = os.path.join(output_file_dir, baseline_output_file_name)
    if not os.path.exists(baseline_output_file_path): 
        print(f"\033[31m\nWarning! Baseline output file does not exist ...\033[0m")
    else:
        lines = open(baseline_output_file_path).readlines()
        judgment_baseline_label = json.loads(lines[idx])["final_judgment"]
    
    judge.summary(ret, munch, judgment_baseline_label, judgment_gold_label)

def build_red_schemas(
    data,
    db_root_path,
    schema_file_path,
    output_path="data/spider/spider_red_schemas.pkl"
):
    red_schemas = {}
    db_ids = sorted({ex["db_id"] for ex in data})
    for db_id in tqdm(db_ids, desc=f"Building RED schemas"):
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        schema_json = get_db_schema_from_json(db_id, schema_file_path)
        red_schemas[db_id] = Schema(schema_json, db_path)
    with open(output_path, "wb") as f:
        pickle.dump(red_schemas, f)

    print(f"Saved {len(red_schemas)} schemas to {output_path}")

def get_red_schemas(data, db_root_path, schema_file_path):
    # Read the offline RED schema file to avoid on-the-fly parsing overhead
    out_path = os.path.join(Path(db_root_path).parent, "red_schemas.pkl")
    if not os.path.exists(out_path):
        build_red_schemas(data, db_root_path, schema_file_path, output_path=out_path)

    return pickle.load(open(out_path, "rb"))