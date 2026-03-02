import os
import re
import json
import shutil
import logging
import threading
from collections import Counter
from munch import Munch
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# from checklist.spinners import Spinner
from checklist.db_utils.sql_parser import is_sql_do_math, is_sql_select_distinct, is_sql_to_cast, is_sql_use_join, is_sql_use_limit
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.db_manager import DatabaseManager
from checklist.db_utils.schema_generator import DatabaseSchemaGenerator
from checklist.base_tester import SchemaPruningMixin, BaseTester, ValidationError
from checklist.db_utils.execution import validate_sql_query
from checklist.db_utils.db_opt import create_sqlite_database, duplicate_sqlite_database, insert_rows_into_table, sqlite_type_map


class NoiseRowTester(SchemaPruningMixin, BaseTester):
    def __init__(self):
        super().__init__("Noise Row Injection Tester", "metamorphic_noise", "metamorphic")

    @staticmethod
    def _results_multiset_equal(a, b, *, unordered_row: bool = False) -> bool:
        """
        Order-insensitive equality for SQL results, preserving duplicates.
        Results are typically a list of tuples from sqlite3 cursor.fetchall().

        Args:
            a, b: SQL execution results (iterables of rows).
            unordered_row: If True, treats each row as an unordered multiset of its elements
                (so (1, "a") equals ("a", 1)). Use this only if you want to ignore
                column/element order differences in upstream outputs.
        """
        def _make_hashable(x):
            if isinstance(x, list):
                return tuple(_make_hashable(i) for i in x)
            if isinstance(x, tuple):
                return tuple(_make_hashable(i) for i in x)
            if isinstance(x, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
            if isinstance(x, set):
                return tuple(sorted(_make_hashable(i) for i in x))
            if isinstance(x, memoryview):
                return x.tobytes()
            try:
                hash(x)
                return x
            except TypeError:
                return repr(x)

        try:
            def _sort_key(v):
                # Avoid direct comparisons across mixed types during sorting.
                return (type(v).__name__, repr(v))

            def _norm_row(r):
                hr = _make_hashable(r)
                if unordered_row and isinstance(hr, tuple):
                    return tuple(sorted(hr, key=_sort_key))
                return hr

            return Counter([_norm_row(r) for r in a]) == Counter([_norm_row(r) for r in b])
        except Exception:
            return False

    def set(self, pruning_threshold=20, **kwargs):
        super().set(**kwargs)
        self.criteria=0.6
        self.num=3
        self.max_retry = self.num * 2
        self.parallel_workers = self.num
        self.err = ""
        self.parser = get_parser(parser_name="noise_data_injection")
        self.schema, self.schema_pruned = self._get_db_schema(pruning_threshold)
        self._schedule_pruned_db_materialization(self.schema_string, copy_existing_rows=True)
        # self.test_cases = self._generator()

    def _compare_query_results(self, preds, oracles, type="relevant", 
                               do_cast=False, do_math=False, do_distinct=False, do_limit=False, do_join_but_add_one=False):
        if preds is None or oracles is None: return False
        # if difference larger than one or error observed during db creation, most probably the injection made something wrong...
        if abs(len(preds)-len(oracles)) > 1 or getattr(self, "err", ""): return True

        if type == "relevant":
            # handling corner cases to ensure high precision:
            # case#1: distinct/cast semantics
            # case#2: limit semantics
            # case#3: math semantics may not change len, so require at least one value change.
            # case#4: doing join but only add one-table data (most probably made something wrong...)
            if len(preds) == len(oracles):
                if do_cast or do_math or do_distinct or do_limit or do_join_but_add_one: return True
                return not self._results_multiset_equal(preds, oracles, unordered_row=True)
            
            return len(preds) - 1 == len(oracles)
        elif type == "irrelevant":
            return len(preds) == len(oracles)
        
        return True
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        # Test the original SQL over the noise-injected database with expected execution results
        res = validate_sql_query(ret.test_fixtures.db, self.sql, max_returned_rows="all")
        logging.info(f"Validating SQL: {self.sql}")
        ret.results.pred = res['RESULT'] if res['STATUS'] == 'OK' else None
        res = validate_sql_query(self.db_path, self.sql, max_returned_rows="all")
        ret.results.target = res['RESULT'] if res['STATUS'] == 'OK' else None
        logging.info(f"Predicted Result: {ret.results.pred}, Target Result: {ret.results.target}")
        ret.results.standard = "abs(len(pred) - len(target)) == 1"
        passed = self._compare_query_results(
            ret.results.pred, 
            ret.results.target, 
            ret.test_fixtures.type,
            do_cast=is_sql_to_cast(self.sql),
            do_math=is_sql_do_math(self.sql),
            do_distinct=is_sql_select_distinct(self.sql),
            do_limit=is_sql_use_limit(self.sql),
            do_join_but_add_one = (
                is_sql_use_join(self.sql) 
                and len(ret.test_fixtures.data) == 1
            )
        )
        # clean up
        os.remove(ret.test_fixtures.db)
        return passed, ret.test_fixtures, ret.results, ret.avg_logprob, ret.trace
    
    def _validate_test_fixture(self, response, history):
        def __attempt_row_alignment_fix(table_name, row, column_names, column_types):
            """
            Attempt to fix minor row-schema misalignment when column count differs by <=1.

            Strategy:
            - If missing 1 column: insert None at reasonable position.
            - If extra 1 column: drop likely redundant column (prefer first if ID-like).
            """

            expected_len = len(column_types)
            actual_len = len(row)

            if abs(expected_len - actual_len) != 1:
                return None

            # -----------------------------
            # Case 1: Missing one column
            # -----------------------------
            if actual_len == expected_len - 1:
                # Heuristic: if first column looks like primary key (id-like), insert None at front
                if column_names and column_names[0].lower() in ("id", f"{table_name.lower()}_id"):
                    return [None] + row
                # otherwise append None at end
                return row + [None]

            # -----------------------------
            # Case 2: One extra column
            # -----------------------------
            if actual_len == expected_len + 1:
                # If first column is integer-like and expected first type is INTEGER
                first_expected = column_types[0].upper()
                if (
                    first_expected == "INTEGER"
                    and isinstance(row[0], int)
                ):
                    # likely redundant ID
                    return row[1:]

                # otherwise drop last column
                return row[:-1]

            return None
        
        def __output_format_check(response):
            if not isinstance(response, dict):
                raise ValidationError(
                    f"Output format(type) check failed. "
                    f"response type: {type(response)}, "
                    f"Expected type: dict"
                )
            if "injected_rows" not in response.keys() or "injection_type" not in response.keys():
                raise ValidationError(
                    f"Output format(key) check failed. "
                    f"Keys found in response: {','.join(response.keys())}, "
                    f"Expected keys: `injected_rows` | `injection_type`"
                )
            return True
        def __schema_data_alignment_check(response, tables, column_types):
            def __normalize_sqlite_type(tp: str) -> str:
                """Normalize SQLite type (case-insensitive, strip length, etc.)."""
                tp = tp.upper().strip()
                # Remove size qualifiers, e.g., VARCHAR(20) -> VARCHAR
                tp = re.sub(r'\s*\(.*\)', '', tp)
                return tp
            # table name validity check
            tables_in_data = response["injected_rows"].keys()
            for td in tables_in_data:
                if td not in tables:
                    raise ValidationError(
                        f"Table name checking failed. "
                        f"Non-existed table name found in generated data: {td} "
                        f"Existing table names: {','.join(tables)}"
                    )
            # column count and data types consistent check
            data = response["injected_rows"]
            for t, row in data.items():
                if not row: continue
                # hard-code to convert ``incorrect'' nested list into list for parsing
                if isinstance(row[0], list): 
                    row = row[0]
                    data[t] = row
                expected_len = len(column_types[t])
                if expected_len != len(row):
                    fixed_row = None
                    len_delta = abs(expected_len - len(row))
                    if len_delta == 1:
                        fixed_row = __attempt_row_alignment_fix(
                            table_name=t,
                            row=row,
                            column_names=self.schema.get(t, []),
                            column_types=column_types[t]
                        )
                    if fixed_row is None:
                        raise ValidationError(
                            f"Schema-data column count mismatch. "
                            f"Column count of table `{t}` in data row: {len(row)} (e.g., {row}), "
                            f"Expected column count: {expected_len}({','.join(self.schema[t])})"
                        )
                    data[t] = fixed_row
                    row = fixed_row
                
                for v, tp in zip(row, column_types[t]):
                    # print(tp)
                    normalized = __normalize_sqlite_type(tp)
                    expected_type = sqlite_type_map.get(normalized, str)
                    try:
                        if v is not None:
                            expected_type(v)
                    except (ValueError, TypeError):
                        raise ValidationError(
                            f"Schema-data column type mismatch. "
                            f"Column type Data: {v} "
                            f"Expected column type: {expected_type}"
                        )
            return True
        def __extract_column_types_from_schema_string(schema_string):
            constraints = ('primary key', 'foreign key', 'unique', 'check', 'constraint')

            res = {}
            ddl_regex = re.compile(r"CREATE TABLE.*?\);", re.DOTALL | re.IGNORECASE)
            ddl_commands = ddl_regex.findall(schema_string)
            for ddl_command in ddl_commands:
                create_table_match = re.match(r'CREATE TABLE "?`?([\w -]+)`?"?\s*\((.*)\)', ddl_command, re.DOTALL)
                table_name = create_table_match.group(1).strip()
                column_definitions = create_table_match.group(2).strip()
                definitions = DatabaseSchemaGenerator._separate_column_definitions(column_definitions)
                type_regex = re.compile(r'.*\b(TEXT|FLOAT|INT|INTEGER|REAL|NUMERIC|VARCHAR|BLOB|bool|BOOLEAN|DATE|DATETIME)\b', re.IGNORECASE)
                types = []
                for column_def in definitions:
                    column_def = column_def.strip()
                    # if 'foreign key' in column_def.lower(): continue
                    # 跳过表级约束
                    if column_def.lower().startswith(constraints): continue
                    match = type_regex.search(column_def)
                    if match:
                        types.append(match.group(1).upper())
                res[table_name] = types
            return res
        def __response_history_compatible_check(response, history):
            def __dicts_equal___(d1, d2):
                if d1.keys() != d2.keys():
                    return False
                
                for key in d1:
                    v1, v2 = d1[key], d2[key]
                    # If both are lists, check order-insensitive equality
                    if isinstance(v1, list) and isinstance(v2, list) and set(v1) != set(v2): return False
                return True
            
            for h in history:
                if __dicts_equal___(response["injected_rows"], h["data"]):
                    raise ValidationError("Duplicate(`injected_rows`) test case.")
            return True
        
        # output format check
        __output_format_check(response)
        # schema-data alignment check
        table_names = DatabaseManager().get_db_all_tables() if not self.schema_pruned else [k for k in self.schema.keys()]
        column_types= DatabaseManager().get_all_column_types() \
            if not self.schema_pruned else __extract_column_types_from_schema_string(self.schema_string)
        __schema_data_alignment_check(response, table_names, column_types)
        # response duplication check
        __response_history_compatible_check(response, history)

    def _form_instance(self, idx, ret):
        """
        Form each single test case, and save related test fixture for serialization. 
        Format as: <`db-file-with-generated-data`, `to-executed-sql`, `expected-executed-result`>
        
        Parameters
        ----------
        ret: Dict with `data` and `result` keys
        No return value
        """
        TEST_INSTANCE_ROOT_PATH = os.path.join(self.instance_saved_path, f"{idx}")
        os.makedirs(TEST_INSTANCE_ROOT_PATH, exist_ok=True)
        
        # Create test data test_cases
        ret.test_fixtures.db = os.path.join(TEST_INSTANCE_ROOT_PATH, f"{self.db_id}.sqlite")
        logging.info(f"Creating test database at \"{ret.test_fixtures.db}\" ...")
        if not self.schema_pruned:
            duplicate_sqlite_database(src_db_path=self.db_path, dest_db_path=ret.test_fixtures.db, reset=False)
        else:
            snapshot_path = self._ensure_pruned_db_snapshot_ready()
            if snapshot_path and os.path.exists(snapshot_path):
                shutil.copy2(snapshot_path, ret.test_fixtures.db)
            else:
                logging.warning("Pruned schema snapshot unavailable; rebuilding synchronously for NoiseRowTester.")
                create_sqlite_database(ret.test_fixtures.db, self.schema_string)
                self._copy_rows_into_pruned_db(ret.test_fixtures.db, self.schema)
        for t, row in ret.test_fixtures.data.items(): 
            res = insert_rows_into_table(ret.test_fixtures.db, table_name=t, rows=[row])
            if res: self.err = res
        return ret
    
    def _generator(self, verbose=True):
        def __history_to_string(history):
            return "\n".join(
                f"--- Example {i+1} ---\n"
                f"injected rows: {json.dumps(h['data'], indent=4)}\n\n"
                for i, h in enumerate(history)
            )

        retry = 0
        history = []
        # spinner = Spinner(f"Generating test cases of `{self.name}` ...")
        state_lock = threading.Lock()

        def _generate_candidate(history_string):
            ret = Munch()
            ret.test_fixtures = Munch()
            trace = f"->>Parallel Test Case Tracelog<<-\n"

            prompt = get_prompt(
                template_name="noise_data_injection",
                schema_string=self.schema_string,
                history_string=history_string
            )
            response, metadata = self.backbone(prompt, self.parser, request_kwargs={"QUESTION": self.nl, "HINT": self.hint})
            
            self.calls += 1
            self.token_used += metadata.get("token_used", 0)

            ret.avg_logprob = metadata.get("avg_logprob", None)
            ret.test_fixtures.data = response.get("injected_rows", {}) or None
            if ret.test_fixtures.data:
                trace += f"[injected rows]: {response.get('injected_rows', '')}\n"
            ret.test_fixtures.type = response.get("injection_type", {}) or None
            if ret.test_fixtures.type:
                trace += f"[injected type]: {response.get('injection_type', '')}\n"
                trace += f"[reasoning]: {response.get('chain_of_thought_reasoning', '')}\n"
            ret.trace = trace 
            return response, ret

        def submit_task(executor, futures):
            with state_lock:
                outstanding = len(self.test_cases) + len(futures)
                if outstanding >= self.num or retry >= self.max_retry:
                    return False
                history_string = __history_to_string(history) if history else None
            future = executor.submit(_generate_candidate, history_string)
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
                        response, ret = fut.result()
                    except Exception as exc:
                        logging.exception("Noise data generation worker failed", exc_info=exc)
                        with state_lock:
                            retry += 1
                            # if verbose:
                            #     spinner.set_message(f"Test fixture generation failed (attempt {retry}/{self.max_retry})...")
                        continue

                    appended_to_history = False
                    try:
                        with state_lock:
                            self._validate_test_fixture(response, history)
                            history.append(ret.test_fixtures)
                            appended_to_history = True
                            self.test_cases.append(self._form_instance(len(self.test_cases), ret))
                            # spinner.set_message(f"Generated {len(outputs)} test cases ...")
                            stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
                            if stop_generation: break
                    except ValidationError as e:
                        print(e)
                        with state_lock:
                            if appended_to_history and history:
                                history.pop()
                            retry += 1
                            # if verbose:
                            #     spinner.set_message(f"Test fixture validation failed (attempt {retry}/{self.max_retry})...")
                        logging.warning(f"Test fixture validation failed: {e}")
                    # except Exception as err:
                    #     with state_lock:
                    #         if appended_to_history and history:
                    #             history.pop()
                    #         retry += 1
                    #         # if verbose:
                    #         #     spinner.set_message(f"Test fixture materialization failed (attempt {retry}/{self.max_retry})...")
                    #     logging.exception("Failed to materialize test instance", exc_info=err)

                    with state_lock:
                        stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry

                    if stop_generation or not submit_task(executor, futures):
                        break

            for fut in futures:
                fut.cancel()

        return
