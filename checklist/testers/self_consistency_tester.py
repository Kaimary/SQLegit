import os
import logging
import threading
from munch import Munch
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# from checklist.spinners import Spinner
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.base_tester import SchemaPruningMixin, BaseTester, ValidationError
from checklist.db_utils.execution import validate_sql_query


class SelfConsistencyTester(SchemaPruningMixin, BaseTester):
    def __init__(self):
        super().__init__("Query Consistency Tester", "query_consistency", "differential")

    def set(self, pruning_threshold=20, **kwargs):
        super().set(**kwargs)
        self.num = 3
        self.criteria = 0.6
        self.max_retry = self.num * 2
        self.parallel_workers = self.num
        self.schema, self.schema_pruned = self._get_db_schema(pruning_threshold)
        self.prompt = get_prompt(template_name="nl_paraphrase_generation", schema_string=self.schema_string)
        self.prompt2 = get_prompt(template_name="nl2sql_translation_with_example", schema_string=self.schema_string)
        self.parser = get_parser(parser_name="nl_paraphrase_generation")
        self.parser2 = get_parser(parser_name="nl2sql_translation_with_example")
    
    def _compare_query_results(self, pred, target):
        if pred is None or target is None: return False
        return set(target) == set(pred)
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        res = validate_sql_query(self.db_path, ret.test_fixtures.sql, max_returned_rows="all")
        ret.results.pred = res['RESULT'] if res['STATUS'] == 'OK' else None
        res = validate_sql_query(self.db_path, self.sql, max_returned_rows="all")
        ret.results.target = res['RESULT'] if res['STATUS'] == 'OK' else None
        ret.results.standard = "pred == target"
        passed = self._compare_query_results(ret.results.pred, ret.results.target)
        return passed, ret.test_fixtures, ret.results, ret.avg_logprob, ret.trace
        
    def _form_instance(self, idx, ret):
        """
        Form each single test case, and save related test fixture for serialization. 
        
        Parameters
        ----------
        ret: Dict with `turns` key
        No return value
        """
        TEST_INSTANCE_ROOT_PATH = os.path.join(self.instance_saved_path, f"{idx}")
        os.makedirs(TEST_INSTANCE_ROOT_PATH, exist_ok=True)
        
        # test case serialization
        self.write_test_fixture_file(output_dir=TEST_INSTANCE_ROOT_PATH, 
            nl=self.nl,
            sql=self.sql, 
            paraphrase=ret.test_fixtures.paraphrase,
            paraphrase_sql=ret.test_fixtures.sql)
        
        return ret
    
    def _validate_test_fixture(self, response, key="paraphrases"):
        def __output_format_check(response, key):
            if not isinstance(response, dict):
                raise ValidationError(
                    f"Output format(type) check failed. "
                    f"response type: {type(response)}, "
                    f"Expected type: dict"
                )
            if key not in response.keys():
                raise ValidationError(
                    f"Output format(key) check failed. "
                    f"Keys found in response: {','.join(response.keys())}, "
                    f"Expected keys: `{key}`"
                )
            return True
        def __sql_executable_check(candidate):
            res = validate_sql_query(self.db_path, candidate)
            if res["STATUS"] != "OK":
                raise ValidationError(
                        f"SQL executable check failed. "
                        f"Fail log from DBMS: {res['RESULT']}"
                    )
            return True
        
        __output_format_check(response, key)
        if key == "SQL": __sql_executable_check(response["SQL"])
       
    def _generator(self, verbose=True):
        def _prepare_paraphrases():
            retry = 0
            paraphrases = []
            while len(paraphrases) < self.num and retry < self.max_retry:
                try:
                    response, metadata = self.backbone(
                        self.prompt,
                        self.parser,
                        request_kwargs={
                            "HINT": self.hint,
                            "QUESTION": self.nl,
                            "SQL": self.sql,
                            "NUM": self.num,
                        },
                    )
                except Exception as exc:
                    retry += 1
                    logging.warning(
                        f"Paraphrase generation failed (attempt {retry}/{self.max_retry}): {exc}"
                    )
                    continue
                
                self.calls += 1
                self.token_used += metadata.get("token_used", 0)
                
                try:
                    self._validate_test_fixture(response)
                except ValidationError as e:
                    retry += 1
                    logging.warning(
                        f"Paraphrase validation failed (attempt {retry}/{self.max_retry}): {e}"
                    )
                    continue
                if not isinstance(response.get("paraphrases", None), list) or not response["paraphrases"]:
                    retry += 1
                    logging.warning(
                        f"Paraphrase output missing/empty (attempt {retry}/{self.max_retry})."
                    )
                    continue
                paraphrases.extend(response["paraphrases"])
            # Keep downstream logic predictable even if model over-produces.
            return paraphrases[: self.num]
        def _generate_candidate(paraphrase):
            ret = Munch()
            ret.test_fixtures = Munch()
            trace = f"->>Parallel Test Case Tracelog<<-\n"
            response, metadata = self.backbone(
                self.prompt2,
                self.parser2,
                request_kwargs={
                    "HINT": self.hint,
                    "QUESTION": paraphrase,
                    "SQL": self.sql,
                },
            )

            self.calls += 1
            self.token_used += metadata.get("token_used", 0)

            self._validate_test_fixture(response, key="SQL")
            ret.avg_logprob = metadata.get("avg_logprob", None)
            ret.test_fixtures.paraphrase = paraphrase
            ret.test_fixtures.sql = response["SQL"].strip()
            trace += f"predicted sql: {response['SQL']}\n"
            ret.trace = trace
            return ret

        paraphrases = _prepare_paraphrases()
        if not paraphrases:
            logging.warning("No paraphrases generated; skipping self-consistency test case generation.")
            return

        retry = 0
        state_lock = threading.Lock()
        paraphrase_queue = deque(paraphrases)
        paraphrase_attempts = {}  # paraphrase -> num submissions
        futures = {}

        def submit_task(executor):
            nonlocal retry
            with state_lock:
                outstanding = len(self.test_cases) + len(futures)
                if outstanding >= self.num or retry >= self.max_retry:
                    return False
                paraphrase = None
                while paraphrase_queue:
                    cand = paraphrase_queue.popleft()
                    if paraphrase_attempts.get(cand, 0) < self.max_retry:
                        paraphrase = cand
                        break
                if paraphrase is None:
                    return False
                paraphrase_attempts[paraphrase] = paraphrase_attempts.get(paraphrase, 0) + 1
            future = executor.submit(_generate_candidate, paraphrase)
            futures[future] = paraphrase
            return True

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            for _ in range(min(self.parallel_workers, len(paraphrases))):
                if not submit_task(executor):
                    break

            stop_generation = False
            while futures and not stop_generation:
                done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    paraphrase = futures.pop(fut)
                    try:
                        case_ret = fut.result()
                        with state_lock:
                            self.test_cases.append(self._form_instance(len(self.test_cases), case_ret))
                    except ValidationError as e:
                        with state_lock:
                            retry += 1
                            if retry < self.max_retry and paraphrase_attempts.get(paraphrase, 0) < self.max_retry:
                                paraphrase_queue.append(paraphrase)
                        logging.warning(f"Self-consistency test fixture validation failed (attempt {retry}/{self.max_retry}): {e}")
                    except Exception as exc:
                        with state_lock:
                            retry += 1
                            if retry < self.max_retry and paraphrase_attempts.get(paraphrase, 0) < self.max_retry:
                                paraphrase_queue.append(paraphrase)
                        logging.warning(f"Self-consistency test case generation failed (attempt {retry}/{self.max_retry}): {exc}")

                    with state_lock:
                        stop_generation = (
                            len(self.test_cases) >= self.num or retry >= self.max_retry
                        )

                    if stop_generation or not submit_task(executor):
                        break

            for fut in futures:
                fut.cancel()

        return