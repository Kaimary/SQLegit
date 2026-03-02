import os
import random
import logging
import threading
from munch import Munch
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# from checklist.spinners import Spinner
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.base_tester import SchemaPruningMixin, BaseTester, ValidationError
from checklist.models import MODEL_CLASS_MAP, GenericLLM
from checklist.db_utils.execution import validate_sql_query


class CrossModelTester(SchemaPruningMixin, BaseTester):
    def __init__(self):
        super().__init__("Cross Models Tester", "cross_model", "differential")
        
    def set(self, pruning_threshold=20, **kwargs):
        super().set(**kwargs)
        self.num=3
        self.max_retry = self.num * 2
        self.active_model_num = 3
        model_list=(["resdsql", "codes15b", "dailsql", "llm:gpt-5.1"] \
                    if "spider" in self.db_root_path else ["chess", "cscsql32b", "omnisql32b", "llm:gpt-5.1"])
        self.parser = get_parser(parser_name="nl2sql_translation")
        self.model_pool = self._create_nl2sql_model_pool(model_list)
        self.parallel_workers = min(self.num, len(self.model_pool)) if self.model_pool else self.num
        self.schema, self.schema_pruned = self._get_db_schema(pruning_threshold)

    def _create_nl2sql_model_pool(self, model_list):
        models = []
        for name in model_list:
            if name in MODEL_CLASS_MAP:
                models.append(MODEL_CLASS_MAP[name]())
            elif name.startswith("llm:"):
                # e.g. "llm:gpt-4o-mini-0708"
                _, model_name = name.split(":", 1)
                models.append(GenericLLM(model_name=model_name))
        return models
    
    def _compare_query_results(self, pred_list, origin):
        if origin is None: return False
        vote = 0
        for pred in pred_list:
            if set(pred) == set(origin): vote+=1
        return vote >= len(pred_list) / 2
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        res = [validate_sql_query(self.db_path, sql, max_returned_rows="all") for sql in ret.test_fixtures.candidates]
        ret.results.pred = [r['RESULT'] if r['STATUS'] == 'OK' else None for r in res]
        res1 = validate_sql_query(self.db_path, self.sql, max_returned_rows="all")
        ret.results.target = res1['RESULT'] if res1['STATUS'] == 'OK' else None
        ret.results.standard = "vote(pred) == target"
        passed = self._compare_query_results(ret.results.pred, ret.results.target)
        return passed, ret.test_fixtures, ret.results, ret.avg_logprob, ret.trace
    
    def _validate_test_fixture(self, candidates):
        def __sql_executable_check(candidate, db_path):
            res = validate_sql_query(db_path, candidate)
            if res["STATUS"] != "OK":
                raise ValidationError(
                        f"SQL executable check failed. "
                        f"Fail log from DBMS: {res['RESULT']}"
                    )
            return True
        def __candidate_number_check(candidates, active_model_num):
            if len(candidates) < active_model_num:
                raise ValidationError(f"Candidate number check failed. Expected {active_model_num} candidates, but got {len(candidates)}.")
            return True

        if isinstance(candidates, str): __sql_executable_check(candidates, self.db_path)
        else: __candidate_number_check(candidates, self.active_model_num)
        
    def _form_instance(self, idx, ret):
        """
        Form each single test case, and save related test fixture for serialization. 
        Format as: <`db-file`, `candidate-sqls`, `original-sql`>
        
        Parameters
        ----------
        ret: Dict with `candidates` key
        No return value
        """
        TEST_INSTANCE_ROOT_PATH = os.path.join(self.instance_saved_path, f"{idx}")
        os.makedirs(TEST_INSTANCE_ROOT_PATH, exist_ok=True)
        
        # test case serialization
        self.write_test_fixture_file(output_dir=TEST_INSTANCE_ROOT_PATH, 
            candidates=ret.test_fixtures.candidates,
            sql=self.sql)
        
        return ret
    
    def _generator(self, verbose=True):
        def __error_to_string(invalids):
            return "\n".join(
                f"invalid sql {idx+1}:\n{invalid[0]}\nerror:{invalid[1]}"
                for idx, invalid in enumerate(invalids)
            )

        invalids = set()
        retry = 0
        state_lock = threading.Lock()

        def _current_invalid_string(snapshot_string, local_invalids):
            local_string = __error_to_string(local_invalids) if local_invalids else None
            if snapshot_string and local_string:
                return "\n".join([snapshot_string, local_string])
            return snapshot_string or local_string

        def _generate_candidate(snapshot_invalids):
            local_invalids = []
            snapshot_string = __error_to_string(snapshot_invalids) if snapshot_invalids else None
            candidates = []
            ret = Munch()
            ret.test_fixtures = Munch()
            trace_lines = ["->>Parallel Test Case Tracelog<<-"]
            total_tokens = 0
            logprob_values = []

            for model in random.sample(self.model_pool, len(self.model_pool)):
                one_retry = 0
                candidate_sql = None
                metadata = {}
                model_name = getattr(model, "model_name", model.__class__.__name__)
                trace_lines.append(f"[{model_name}] start generation")
                while one_retry < self.max_retry:
                    try:
                        if isinstance(model, GenericLLM):
                            prompt = get_prompt(
                                template_name="nl2sql_translation",
                                schema_string=self.schema_string,
                                invalid_queries_string=_current_invalid_string(snapshot_string, local_invalids)
                            )
                            candidate_sql, metadata = model(
                                prompt=prompt,
                                parser=self.parser,
                                request_kwargs={"HINT": self.hint, "QUESTION": self.nl}
                            )

                            self.calls += 1
                            self.token_used += metadata.get("token_used", 0)
                            
                            logprob = metadata.get("avg_logprob")
                            logprob_values.append(logprob)
                        else:
                            candidate_sql = model(nl=self.nl)
                            logprob_values.append(-0.23)
                        self._validate_test_fixture(candidate_sql)
                        trace_lines.append(f"[{model_name}] candidate accepted: {candidate_sql}")
                        break
                    except ValidationError as e:
                        logging.warning(f"Candidate SQL validation failed: {e}")
                        trace_lines.append(f"[{model_name}] validation failed: {e}")
                        if isinstance(model, GenericLLM) and candidate_sql:
                            local_invalids.append((candidate_sql, str(e)))
                        if isinstance(model, GenericLLM):
                            one_retry += 1
                        else:
                            one_retry = self.max_retry
                            candidate_sql = None
                            break
                    except Exception as exc:
                        logging.exception("Cross model candidate generation error", exc_info=exc)
                        trace_lines.append(f"[{model_name}] generation error: {exc}")
                        one_retry = self.max_retry
                        candidate_sql = None
                        break
                if candidate_sql is not None and one_retry < self.max_retry:
                    candidates.append(candidate_sql)
                    if len(candidates) == self.active_model_num:
                        break
                else:
                    trace_lines.append(f"[{model_name}] generation aborted after retries")

            try:
                self._validate_test_fixture(candidates)
            except ValidationError as e:
                return None, local_invalids, e

            ret.test_fixtures.candidates = candidates
            ret.token_used = total_tokens
            ret.avg_logprob = (sum(logprob_values) / len(logprob_values)) if logprob_values else None
            ret.trace = "\n".join(trace_lines)
            return ret, local_invalids, None

        def submit_task(executor, futures):
            with state_lock:
                outstanding = len(self.test_cases) + len(futures)
                if outstanding >= self.num or retry >= self.max_retry:
                    return False
                snapshot_invalids = list(invalids) if invalids else None
            future = executor.submit(_generate_candidate, snapshot_invalids)
            futures.add(future)
            return True

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = set()
            for _ in range(self.parallel_workers):
                if not submit_task(executor, futures):
                    break

            stop_generation = False
            while futures and not stop_generation:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.remove(fut)
                    try:
                        ret, new_invalids, error = fut.result()
                    except Exception as exc:
                        logging.exception("Cross model generation worker failed", exc_info=exc)
                        with state_lock:
                            retry += 1
                            stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
                        continue

                    if new_invalids:
                        with state_lock:
                            invalids.update(new_invalids)

                    if error:
                        logging.warning(f"Test fixture validation failed (attempt {retry + 1}/{self.max_retry}): {error}")
                        with state_lock:
                            retry += 1
                    elif ret:
                        with state_lock:
                            self.test_cases.append(self._form_instance(len(self.test_cases), ret))

                    with state_lock:
                        stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry

                    if stop_generation or not submit_task(executor, futures):
                        break

            for fut in futures:
                fut.cancel()

        return
