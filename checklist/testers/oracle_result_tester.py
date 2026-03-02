import os
import re
import json
import shutil
import logging
import threading
import numpy as np
from munch import Munch
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# from checklist.spinners import Spinner
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.db_manager import DatabaseManager
from checklist.db_utils.schema_generator import DatabaseSchemaGenerator
from checklist.base_tester import SchemaPruningMixin, BaseTester, ValidationError
from checklist.db_utils.sql_parser import is_sql_do_math
from checklist.db_utils.execution import validate_sql_query
from checklist.db_utils.db_opt import create_sqlite_database, duplicate_sqlite_database, insert_rows_into_table, sqlite_type_map


class OracleResultTester(SchemaPruningMixin, BaseTester):
    def __init__(self):
        super().__init__("Oracle Result Tester", "oracle_result", "oracle", key="nl")

    def set(self, pruning_threshold=20, **kwargs):
        super().set(**kwargs)
        self.num=3
        self.criteria=0.6
        self.max_retry = self.num * 2
        self.parallel_workers = self.num
        self.parser = get_parser(parser_name="oracle_data_generation")
        self.schema, self.schema_pruned = self._get_db_schema(pruning_threshold)
        self._schedule_pruned_db_materialization(self.schema_string)

    def _compare_query_results(self, preds, oracles, both_empty=False, do_math=False):
        def __normalize_scalar(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return round(float(value), 2)
            if isinstance(value, str):
                stripped = value.strip()
                try:
                    return round(float(stripped), 2)
                except ValueError:
                    return value
            return value
        def __freeze(obj):
            """Recursively convert unhashable objects into hashable equivalents."""
            if isinstance(obj, dict):
                return tuple(sorted((k, __freeze(v)) for k, v in obj.items()))
            elif isinstance(obj, tuple) or isinstance(obj, list):
                items = [__freeze(x) for x in obj]
                try:
                    return tuple(sorted(items))
                except TypeError:
                    return tuple(sorted(items, key=lambda v: repr(v)))
            elif isinstance(obj, set):
                items = [__freeze(x) for x in obj]
                try:
                    return tuple(sorted(items))
                except TypeError:
                    return tuple(sorted(items, key=lambda v: repr(v)))
            elif isinstance(obj, np.ndarray):
                return __freeze(tuple(obj.tolist()))
            else:
                return __normalize_scalar(obj)
        def __is_subset(pred, oracle_row):
            """Check if pred tuple is a subsequence of oracle_row tuple."""
            n, m = len(pred), len(oracle_row)
            if n > m:
                return False
            # 尝试匹配 pred 在 oracle_row 中的某个连续子序列
            for i in range(m - n + 1):
                if oracle_row[i:i+n] == pred:
                    return True
            return False
        # Case# 1: if the simulated database can't execute pred sql (but original database can), most probably the simulated database made something wrong...
        # Case# 2: if the SQL is doing math calculation, most probably the oracle result is incorrect, so we forgive it...
        if not oracles or (not preds and not both_empty) or do_math: return True
        if len(set(preds)) != len([tuple(x) for x in oracles]): return False

        preds_frozen = [__freeze(p) for p in preds]
        oracle_frozen = [__freeze(o) for o in oracles]

        for p in preds_frozen:
            if not any(__is_subset(p, o) or __is_subset(o, p) for o in oracle_frozen):
                return False
        return True
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        # Test the original SQL over a faked database with expected execution results
        res = validate_sql_query(ret.test_fixtures.db, self.sql, max_returned_rows="all")
        logging.info(f"Validating SQL: {self.sql}")
        ret.results.pred = res['RESULT'] if res['STATUS'] == 'OK' else None
        # if empty result both observed in original db and simulated db, mostly making some mistake
        both_empty = False
        if ret.results.pred == []:
            res1 = validate_sql_query(self.db_path, self.sql, max_returned_rows="all")
            if res1['STATUS'] == 'OK' and res1['RESULT'] == []:
                both_empty = True
        ret.results.target = ret.test_fixtures.oracle["rows"] if "rows" in ret.test_fixtures.oracle.keys() else []
        logging.info(f"Predicted Result: {ret.results.pred}, Target Result: {ret.results.target}")
        ret.results.standard = "pred == target"
        passed = self._compare_query_results(
            ret.results.pred, 
            ret.results.target, 
            both_empty=both_empty,
            do_math=is_sql_do_math(self.sql)
        )
        # clean up
        os.remove(ret.test_fixtures.db)
        return passed, ret.test_fixtures, ret.results, ret.avg_logprob, ret.trace

    def _validate_test_fixture(self, response, history):
        def __output_format_check(response):
            if not isinstance(response, dict):
                raise ValidationError(
                    f"Output format(type) check failed. "
                    f"response type: {type(response)}, "
                    f"Expected type: dict"
                )
            if 'data' not in response.keys() or 'result' not in response.keys():
                raise ValidationError(
                    f"Output format(keys) check failed. "
                    f"Keys found in response: {','.join(response.keys())}, "
                    f"Expected keys: `data` and `result`"
                )
            if any(k not in response["result"].keys() for k in ["columns", "rows"]): 
                raise ValidationError(
                    f"Output format(key in key) check failed. "
                    f"Keys found in `result`: {','.join(response['result'].keys())}, "
                    f"Expected keys: `columns` and `rows`"
                )
            # quick fix format issues frequently observed
            # (1) the data in ``rows`` key is expected to be list of list
            # (2) the first row in each key (table) of ``data`` key is column names
            if isinstance(response["result"]["rows"], list) and \
                response["result"]["rows"] and not isinstance(response["result"]["rows"][0], list):
                response["result"]["rows"] = [response["result"]["rows"]]
            for table, rows in response["data"].items():
                if len(rows) <= 1: continue
                if all(isinstance(c,str) for c in rows[0]) and any(not isinstance(c,str) for c in rows[1]):
                    response["data"][table] = rows[1:]
            return True
        # def __attempt_row_alignment_fix(table_name, row, column_names, column_types):
        #     """
        #     Best-effort fix for common LLM mistakes where a generated row has exactly
        #     one extra/missing value compared to the table schema.

        #     Returns:
        #         list: A fixed row with length == len(column_types)
        #         None: If no safe fix is found
        #     """
        #     expected_len = len(column_types)

        #     def __normalize_sqlite_type_(tp: str) -> str:
        #         tp = str(tp).upper().strip()
        #         tp = re.sub(r"\s*\(.*\)", "", tp)  # VARCHAR(20) -> VARCHAR
        #         return tp

        #     def __nullish(v) -> bool:
        #         if v is None:
        #             return True
        #         if isinstance(v, str) and v.strip().lower() in {"null", "none", "nil", "n/a", "na", ""}:
        #             return True
        #         return False

        #     def __expected_py(tp: str):
        #         return sqlite_type_map.get(__normalize_sqlite_type_(tp), str)

        #     expected_types = [__expected_py(tp) for tp in column_types]

        #     def __compatible(v, py_tp) -> bool:
        #         if __nullish(v):
        #             return True
        #         if py_tp is bool:
        #             if isinstance(v, bool):
        #                 return True
        #             if isinstance(v, (int, float)) and v in (0, 1):
        #                 return True
        #             if isinstance(v, str) and v.strip().lower() in {
        #                 "0", "1", "true", "false", "t", "f", "yes", "no", "y", "n"
        #             }:
        #                 return True
        #             return False
        #         try:
        #             py_tp(v)
        #             return True
        #         except Exception:
        #             return False

        #     # If the model returned a dict, align by column name (pad with None for missing).
        #     if isinstance(row, dict):
        #         if not column_names:
        #             return None
        #         fixed = [row.get(c, None) for c in column_names]
        #         if len(fixed) != expected_len:
        #             if len(fixed) > expected_len:
        #                 fixed = fixed[:expected_len]
        #             else:
        #                 fixed = fixed + [None] * (expected_len - len(fixed))
        #         return [None if __nullish(v) else v for v in fixed]

        #     if not isinstance(row, list):
        #         return None
        #     if len(row) == expected_len:
        #         return [None if __nullish(v) else v for v in row]
        #     if abs(len(row) - expected_len) != 1:
        #         return None

        #     candidates = []
        #     col_name_lc = [str(c).strip().lower() for c in (column_names or [])]
        #     tbl_lc = str(table_name).strip().lower()

        #     if len(row) == expected_len - 1:
        #         for i in range(expected_len):
        #             cand = row[:i] + [None] + row[i:]
        #             bonus = 0
        #             if i < len(col_name_lc):
        #                 cn = col_name_lc[i]
        #                 if expected_types[i] is int:
        #                     # SQLite will auto-generate INTEGER PRIMARY KEY when inserting NULL.
        #                     if cn == "id":
        #                         bonus += 4
        #                     elif cn.endswith("_id"):
        #                         bonus += 2
        #                     if i == 0 and (cn == "id" or cn.endswith("_id")):
        #                         bonus += 1
        #             candidates.append((cand, bonus))
        #     else:  # len(row) == expected_len + 1
        #         for i in range(len(row)):
        #             cand = row[:i] + row[i + 1:]
        #             bonus = 0
        #             v = row[i]
        #             if isinstance(v, str):
        #                 vv = v.strip().strip('"`').lower()
        #                 if vv == tbl_lc:
        #                     bonus += 2
        #                 if vv in col_name_lc:
        #                     bonus += 2
        #                 if vv in {"row", "values", "value"}:
        #                     bonus += 1
        #             candidates.append((cand, bonus))

        #     best = None
        #     best_key = None
        #     best_count = 0
        #     for cand, bonus in candidates:
        #         cand = [None if __nullish(v) else v for v in cand]
        #         mismatches = sum(
        #             1 for v, py_tp in zip(cand, expected_types) if not __compatible(v, py_tp)
        #         )
        #         key = (mismatches, -bonus)
        #         if best_key is None or key < best_key:
        #             best_key = key
        #             best = cand
        #             best_count = 1
        #         elif key == best_key:
        #             best_count += 1

        #     if best is None or best_key[0] > 1:
        #         return None
        #     if best_count > 1 and best_key[0] == 0 and best_key[1] == 0:
        #         return None
        #     return best
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
        def __normalize_sqlite_type(tp: str) -> str:
            """Normalize SQLite type (case-insensitive, strip length, etc.)."""
            tp = tp.upper().strip()
            # Remove size qualifiers, e.g., VARCHAR(20) -> VARCHAR
            tp = re.sub(r'\s*\(.*\)', '', tp)
            return tp          
        def __schema_data_alignment_check(response, tables, column_types):
            # table name validity check
            tables_in_data = response["data"].keys()
            for td in tables_in_data:
                if td not in tables:
                    raise ValidationError(
                        f"Table name checking failed. "
                        f"Non-existed table name found in generated data: {td} "
                        f"Existing table names: {','.join(tables)}"
                    )
            # column count and data types consistent check
            for t, rows in response["data"].items():
                if not rows: continue

                expected_len = len(column_types[t])
                for i, row in enumerate(rows):
                    if row is None:
                        continue

                    if expected_len != len(row):
                        fixed_row = None
                        if abs(expected_len - len(row)) == 1:
                            fixed_row = __attempt_row_alignment_fix(
                                table_name=t,
                                row=row,
                                column_names=self.schema.get(t, []),
                                column_types=column_types[t]
                            )
                        if fixed_row is None:
                            raise ValidationError(
                                f"Schema-data column count mismatch. "
                                f"Column count in data row: {len(row)}(e.g., {row}), "
                                f"Expected column count of table {t}: {expected_len}({','.join(self.schema[t])})"
                            )
                        rows[i] = fixed_row
                        row = fixed_row

                    for v, tp in zip(row, column_types[t]):
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
        def __response_history_compatible_check(response, history):
            def __dicts_equal___(d1, d2):
                if d1.keys() != d2.keys():
                    return False
                
                for key in d1:
                    v1, v2 = d1[key], d2[key]
                    # If both are lists, check order-insensitive equality
                    if isinstance(v1, list) and isinstance(v2, list):
                        # Convert inner lists to tuples (hashable) for set comparison
                        set1 = set(tuple(item) for item in v1)
                        set2 = set(tuple(item) for item in v2)
                        if set1 != set2:
                            return False
                    elif v1 != v2: return False
                return True
            
            for h in history:
                if not __dicts_equal___(response["data"], h["data"]): continue
                raise ValidationError("Duplicate(`data`+`result`) test case.")
            return True
        
        # output format check
        __output_format_check(response)
        # schema-data alignment check
        table_names = DatabaseManager().get_db_all_tables() if not self.schema_pruned else [k for k in self.schema.keys()]
        column_types= DatabaseManager().get_all_column_types() if not self.schema_pruned \
            else __extract_column_types_from_schema_string(self.schema_string)
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
            duplicate_sqlite_database(src_db_path=self.db_path, dest_db_path=ret.test_fixtures.db)
        else:
            snapshot_path = self._ensure_pruned_db_snapshot_ready()
            if snapshot_path and os.path.exists(snapshot_path):
                shutil.copy2(snapshot_path, ret.test_fixtures.db)
            else:
                create_sqlite_database(ret.test_fixtures.db, self.schema_string)
        for t, rows in ret.test_fixtures.data.items(): 
            insert_rows_into_table(ret.test_fixtures.db, table_name=t, rows=rows)
        
        return ret

    def _generator(self, verbose=True):
        def __history_to_string(history):
            return "\n".join(
                f"--- Example {i+1} ---\n"
                f"data: {json.dumps(h['data'], indent=4)}\n\n"
                for i, h in enumerate(history)
            )
        def __values_to_string(vals):
            return "\n".join(
                f"Column `{t}.{c}`: {', '.join(str(v))};"
                for t, c2vals in vals.items()
                for c, v in c2vals.items()
            )

        history = []
        retry = 0
        # spinner = Spinner(f"Generating test cases of `{self.name}` ...")
        cond_literals = DatabaseManager(db_id=self.db_id, db_root_path=self.db_root_path).get_sql_condition_literals(self.sql)
        state_lock = threading.Lock()

        def _generate_candidate(history_string):
            ret = Munch()
            ret.test_fixtures = Munch()
            trace = "->>Parallel Test Case Tracelog<<-\n"
            prompt = get_prompt(
                template_name="oracle_data_generation",
                schema_string=self.schema_string,
                columns_values_string=__values_to_string(cond_literals) if cond_literals else None,
                history_string=history_string
            )
            response, metadata = self.backbone(
                prompt,
                self.parser,
                request_kwargs={"QUESTION": self.nl, "HINT": self.hint}
            )
            self.calls += 1
            self.token_used += metadata.get("token_used", 0)
            
            ret.avg_logprob = metadata.get("avg_logprob", 0.0) or None
            ret.test_fixtures.data = response.get("data", {}) or None
            ret.test_fixtures.oracle = response.get("result", {}) or None
            if ret.test_fixtures.data and ret.test_fixtures.oracle:
                trace += f"[simulated DB]: {ret.test_fixtures.data}"
                trace += f"[oracle data]: {ret.test_fixtures.oracle}"
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
                        logging.exception("Oracle result generation worker failed", exc_info=exc)
                        with state_lock:
                            retry += 1
                            # if verbose:
                            #     spinner.set_message(f"Test fixture generation failed (attempt {retry}/{self.max_retry})...")
                            stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
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
                            if stop_generation:
                                break
                    except ValidationError as e:
                        with state_lock:
                            if appended_to_history and history:
                                history.pop()
                            retry += 1
                            # if verbose:
                            #     spinner.set_message(f"Test fixture validation failed (attempt {retry}/{self.max_retry})...")
                            stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
                        logging.warning(f"Test fixture validation failed: {e}")
                    # except Exception as err:
                    #     with state_lock:
                    #         if appended_to_history and history:
                    #             history.pop()
                    #         retry += 1
                    #         # if verbose:
                    #         #     spinner.set_message(f"Test fixture materialization failed (attempt {retry}/{self.max_retry})...")
                    #         stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
                    #     logging.exception("Failed to materialize oracle test instance", exc_info=err)

                    with state_lock:
                        stop_generation = len(self.test_cases) >= self.num or retry >= self.max_retry
                        
                    if stop_generation or not submit_task(executor, futures):
                        break

            for fut in futures:
                fut.cancel()

        return
