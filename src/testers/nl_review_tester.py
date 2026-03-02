import os
import logging
import threading
from munch import Munch
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# from src.spinners import Spinner
from src.parsers import get_parser
from src.prompts import get_prompt
from src.base_tester import SchemaPruningMixin, BaseTester, ValidationError
from src.db_utils.execution import validate_sql_query


class NLReviewTester(SchemaPruningMixin, BaseTester):
    def __init__(self):
        super().__init__("Step-through Natural Language Review Tester", "nl_review", "explore")

    def set(self, pruning_threshold=20, **kwargs):
        super().set(**kwargs)
        self.num = 3
        self.criteria = 0.6
        self.max_retry = self.num * 2
        self.parallel_workers = self.num
        self.schema, self.schema_pruned = self._get_db_schema(pruning_threshold)
        self.prompt = get_prompt(template_name="nl_paraphrase_generation", schema_string=self.schema_string)
        self.prompt2 = get_prompt(template_name="nl_rubber_duck_debugging", schema_string=self.schema_string)
        self.parser = get_parser(parser_name="nl_paraphrase_generation")
        self.parser2 = get_parser(parser_name="nl_rubber_duck_debugging")
        # self.test_cases = self._generator()
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        ret.results.pred = ret.test_fixtures.turns['judgment']
        ret.results.target = True
        ret.results.standard = "pred == target"
        passed = ret.results.pred
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
        self.write_test_fixture_file(output_dir=TEST_INSTANCE_ROOT_PATH, turns=ret.test_fixtures.turns)
        
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

        __output_format_check(response, key)
        if not isinstance(response[key], list) or len(response[key]) < 2:
            raise ValidationError(
                f"Paraphrase count check failed. "
                f"Expected at least 2 paraphrases, got {len(response[key]) if isinstance(response[key], list) else 'invalid'}."
            )
       
    def _generator(self, verbose=True):
        def _prepare_paraphrases():
            retry = 0
            paraphrases = [self.nl]
            while len(paraphrases) < self.num and retry < self.max_retry:
                try:
                    response, metadata = self.backbone(
                        self.prompt,
                        self.parser,
                        request_kwargs={
                            "HINT": self.hint,
                            "QUESTION": self.nl,
                            "SQL": self.sql,
                            "NUM": self.num - 1,
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
                    "RESULT": "\n".join(f"{tup}" for tup in preview)
                    if isinstance(preview, list)
                    else preview,
                },
            )

            self.calls += 1
            self.token_used += metadata.get("token_used", 0)

            ret.avg_logprob = metadata.get("avg_logprob", None)
            ret.test_fixtures.turns = response
            trace += f"{response['chain_of_thought_reasoning']} -> {response['judgment']}\n"
            ret.trace = trace
            return ret

        paraphrases = _prepare_paraphrases()
        exec = validate_sql_query(self.db_path, self.sql, max_returned_rows=5)
        preview = exec.get("RESULT")
        # spinner = Spinner(f"Generating test cases of `{self.name}` ...")
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
                        logging.warning(
                            f"NL review test fixture validation failed (attempt {retry}/{self.max_retry}): {e}"
                        )
                    # except Exception as exc:
                    #     with state_lock:
                    #         retry += 1
                    #         if retry < self.max_retry and paraphrase_attempts.get(paraphrase, 0) < self.max_retry:
                    #             paraphrase_queue.append(paraphrase)
                    #     logging.warning(
                    #         f"NL review test case generation failed (attempt {retry}/{self.max_retry}): {exc}"
                    #     )

                    with state_lock:
                        stop_generation = (
                            len(self.test_cases) >= self.num or retry >= self.max_retry
                        )
                        
                    if stop_generation or not submit_task(executor):
                        break

            for fut in futures:
                fut.cancel()

        return
