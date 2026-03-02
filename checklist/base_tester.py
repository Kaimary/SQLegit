import copy
import os
import re
import json
import hashlib
import logging
import sqlite3
import threading
import numpy as np
from pathlib import Path
from munch import Munch
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from checklist.llm import LLM
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.db_utils.db_opt import create_sqlite_database
from checklist.db_utils.schema import DatabaseSchema
from checklist.db_utils.schema_generator import DatabaseSchemaGenerator
from checklist.db_manager import DatabaseManager
from checklist.db_utils.db_catalog.csv_utils import load_tables_description

load_dotenv(override=True)
TEST_INSTANCE_ROOT_PATH = Path(os.getenv("TEST_INSTANCE_ROOT_PATH"))

def hashing(**kwargs):
    combined = ""
    nl = kwargs.get("nl", None)
    sql = kwargs.get("sql", None)
    if nl is not None: combined += f"{nl}"
    if sql is not None:
        normalized_sql = re.sub(r"\s+", " ", sql.strip().lower())
        combined += f";{normalized_sql}" if combined else f"{normalized_sql}"
    hashing_str = hashlib.md5(combined.encode()).hexdigest()[:8]
    
    return hashing_str

class ValidationError(Exception):
    pass

class SchemaCache:
    _cache = {}

    @classmethod
    def get_schema(cls, db_id, db_path, db_root_path):
        if db_id not in cls._cache:
            schema = DatabaseManager(db_id=db_id, db_root_path=db_root_path).get_db_schema()
            # schema_with_examples = load_schema_with_examples(_get_unique_values(db_path))
            schema_with_descriptions = load_tables_description(db_path, use_value_description=True)
            cls._cache[db_id] = DatabaseManager().get_database_schema_string(
                tentative_schema=schema,
                schema_with_examples=None, #schema_with_examples,
                schema_with_descriptions=schema_with_descriptions,
                include_value_description=True
            )
        return cls._cache[db_id]


_DB_SCHEMA_CACHE = {}

class SchemaPruningMixin:
    """Shared helpers for classes that optionally prune large schemas via LLM."""
    def _quote_identifier(self, identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _validate_pruned_schema(self, response):
        if not response or not isinstance(response, dict):
            raise ValidationError("Pruned schema type checking failed.")

        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(response),
            db_id=self.db_id,
            db_path=self.db_path
        )
        ddl_commands = schema_generator._extract_create_ddl_commands()
        for table_name, ddl_command in ddl_commands.items():
            ddl_command = re.sub(r'\s+', ' ', ddl_command.strip())
            create_table_match = re.match(r'CREATE TABLE "?`?([\w -]+)`?"?\s*\((.*)\)', ddl_command, re.DOTALL)
            table = create_table_match.group(1).strip()
            if table != table_name:
                logging.warning(f"Table name mismatch: {table} != {table_name}")
            column_definitions = create_table_match.group(2).strip()
            definitions = DatabaseSchemaGenerator._separate_column_definitions(column_definitions)
            for col in response[table_name]:
                if all(col not in d for d in definitions):
                    raise ValidationError(
                        f"Pruned schema column name checking failed. "
                        f"Column `{col}` should not in table `{table_name}`❌"
                    )

    def _copy_rows_into_pruned_db(self, target_db_path, schema_subset):
        dest_conn = sqlite3.connect(target_db_path)
        src_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Some benchmark DBs contain invalid UTF-8 sequences in TEXT fields. The default
        # sqlite3 text decoding is strict UTF-8 and will raise on fetch. Use a tolerant
        # decoder so pruned-DB materialization doesn't crash.
        def _safe_text_factory(raw: bytes) -> str:
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                # cp1252 is a common "accidental" encoding for Western names; fall back
                # to a replacement strategy so we always return a valid Python str.
                return raw.decode("cp1252", errors="replace")
        src_conn.text_factory = _safe_text_factory
        try:
            dest_cur = dest_conn.cursor()
            src_cur = src_conn.cursor()
            for table, columns in schema_subset.items():
                if not columns:
                    continue
                quoted_table = self._quote_identifier(table)
                quoted_columns = ', '.join(self._quote_identifier(col) for col in columns)
                select_sql = f"SELECT {quoted_columns} FROM {quoted_table}"
                try:
                    src_cur.execute(select_sql)
                except sqlite3.Error as exc:
                    logging.warning(
                        f"Skipping data backfill for table `{table}` during pruned DB build: {exc}"
                    )
                    continue
                placeholders = ', '.join('?' for _ in columns)
                insert_sql = f"INSERT INTO {quoted_table} ({quoted_columns}) VALUES ({placeholders})"
                while True:
                    rows = src_cur.fetchmany(512)
                    if not rows:
                        break
                    try:
                        dest_cur.executemany(insert_sql, rows)
                    except sqlite3.Error as exc:
                        logging.warning(
                            f"Failed inserting rows into table `{table}` for pruned DB build: {exc}"
                        )
                        break
            dest_conn.commit()
        finally:
            src_conn.close()
            dest_conn.close()

    def _prune_schema_if_needed(
        self,
        schema,
        threshold,
        kept
    ):
        self.schema_pruned = False
        self._pruned_db_build_event = None
        self._pruned_db_build_error = None
        self._pruned_db_snapshot_path = None
        if threshold is None:
            return schema, False
        # exclude table pruning as existing benchmarks do not have many tables
        tables_larger_than_threshold_cols = [tab for tab, cols in schema.items() if len(cols) > threshold]
        if not tables_larger_than_threshold_cols:
            return schema, False

        logging.warning(
            f"Database {self.db_id} has tables with more than {threshold} columns. "
            "Truncating the schema before generation ..."
        )
        large_tables = {}
        for t in tables_larger_than_threshold_cols:
            cols = [c for c in schema[t] if t.lower() not in kept.keys() or (t.lower() in kept.keys() and c not in kept[t.lower()])]
            large_tables[t] = cols

        retry = 0
        error = set()
        parser = get_parser(parser_name="schema_pruning_by_selection")

        while retry < self.max_retry:
            prompt = get_prompt(
                template_name="schema_pruning_by_selection",
                error_string='\n'.join(error) if error else None
            )
            response, metadata = self.backbone(
                prompt,
                parser,
                request_kwargs={
                    "HINT": self.hint,
                    "QUESTION": self.nl,
                    "LARGE_TABLES": json.dumps(large_tables, indent=4)
                }
            )
            # self.calls += 1
            # self.token_used += metadata.get("token_used", 0) 
            try:
                self._validate_pruned_schema(response)
                pruned_schema = {}
                for t, cols in schema.items():
                    if t not in response.keys():
                        pruned_schema[t] = schema[t]
                    else:
                        pruned_schema[t] = response[t]
                        if t.lower() in kept.keys(): pruned_schema[t].extend(kept[t.lower()])
                logging.info(f"Pruned schema: {json.dumps(response, indent=4)}")
                self.schema_pruned = True
                return pruned_schema, True
            except ValidationError as e:
                error.add(str(e).split('.')[-1])
                retry += 1
                logging.warning(f"Pruned schema validation failed: {e}. Retrying...")

        return schema, False

    def _schedule_pruned_db_materialization(self, schema_string, copy_existing_rows=False):
        if not getattr(self, "schema_pruned", False):
            return
        if not schema_string:
            return
        if getattr(self, "_pruned_db_build_event", None):
            return
        self._pruned_db_snapshot_path = os.path.join(
            self.instance_saved_path,
            f"{self.db_id}_pruned_base.sqlite"
        )
        schema_subset = copy.deepcopy(self.schema)
        event = threading.Event()
        self._pruned_db_build_event = event
        self._pruned_db_build_error = None

        def _worker():
            try:
                os.makedirs(os.path.dirname(self._pruned_db_snapshot_path), exist_ok=True)
                create_sqlite_database(self._pruned_db_snapshot_path, schema_string)
                if copy_existing_rows:
                    self._copy_rows_into_pruned_db(self._pruned_db_snapshot_path, schema_subset)
            except Exception as exc:
                self._pruned_db_build_error = exc
                logging.exception("Failed to build pruned schema database snapshot", exc_info=exc)
            finally:
                event.set()

        threading.Thread(target=_worker, daemon=True).start()

    def _ensure_pruned_db_snapshot_ready(self):
        event = getattr(self, "_pruned_db_build_event", None)
        if not event:
            return None
        event.wait()
        if getattr(self, "_pruned_db_build_error", None):
            logging.warning(
                f"Pruned schema snapshot creation failed: {self._pruned_db_build_error}"
            )
            return None
        return getattr(self, "_pruned_db_snapshot_path", None)

    def _get_db_schema(self, threshold):
        cache_key = (
            self.db_id,
            self.nl,
            self.sql
        )
        cached = _DB_SCHEMA_CACHE.get(cache_key)
        if cached:
            schema_copy = copy.deepcopy(cached["schema"])
            cached_schema_string = cached.get("schema_string")
            if cached_schema_string is not None:
                self.schema_string = cached_schema_string
            return schema_copy, cached["schema_pruned"]

        schema = DatabaseManager(db_id=self.db_id, db_root_path=self.db_root_path).get_db_schema() # type: ignore
        kept = {}
        try:
            kept = DatabaseManager(db_id=self.db_id, db_root_path=self.db_root_path).get_sql_columns_dict(self.sql)
        except Exception as exc:
            logging.warning(
                f"parse sql failed: {exc}"
            )
    
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(schema),
            db_id=self.db_id,
            db_path=self.db_path
        )
        for k, cols in schema_generator.get_all_primary_foreign_keys().items():
            key = k.lower()
            if key not in kept.keys(): 
                kept[key] = cols
            else: 
                for col in cols:
                    if col.lower() not in kept[key] and col not in kept[key] and len(kept[key]) <= 10: # avoid too many fk found in the table
                        kept[key].append(col)
        schema, schema_pruned = self._prune_schema_if_needed(
            schema=schema,
            threshold=threshold,
            kept=kept
        )
        if schema_pruned:
            # schema_with_examples = load_schema_with_examples(_get_unique_values(self.db_path))
            schema_with_descriptions = load_tables_description(self.db_path, use_value_description = True)
            self.schema_string = DatabaseManager().get_database_schema_string(
                tentative_schema=schema,
                schema_with_examples=None, # type: ignore
                schema_with_descriptions=schema_with_descriptions,
                include_value_description=True
            )
        _DB_SCHEMA_CACHE[cache_key] = {
            "schema": copy.deepcopy(schema),
            "schema_pruned": schema_pruned,
            "schema_string": getattr(self, "schema_string", None)
        }
        return schema, schema_pruned

class BaseTester(ABC):
    def __init__(self, name, abbrev_name, abbrev_type, key="nl+sql", use_cache=False):
        self.name = name
        self.abbrev_name = abbrev_name
        self.abbrev_type = abbrev_type
        self.key = key

        self.token_used = 0
        self.calls = 0

        self.use_cache=use_cache
        self.test_fn = self._test_fn

    def set(self, nl, hint, sql, gold, db_id, db_root_path, backbone_llm_model_name="gpt-4o-mini-0708", num=1, criteria=1.0):
        self.nl=nl
        self.hint=hint
        self.sql=sql
        self.gold=gold

        self.db_id=db_id
        self.db_root_path=db_root_path
        self.db_path = os.path.join(self.db_root_path, self.db_id, f"{self.db_id}.sqlite")
        self.schema_string = SchemaCache.get_schema(db_id, self.db_path, db_root_path)

        kwargs = {"nl": self.nl if "nl" in self.key else None, "sql": self.sql if "sql" in self.key else None}
        self.instance_saved_path = os.path.join(TEST_INSTANCE_ROOT_PATH, self.abbrev_type, self.abbrev_name, self.db_id, hashing(**kwargs))
        os.makedirs(self.instance_saved_path, exist_ok=True)

        self.backbone = LLM(model_name=backbone_llm_model_name)
        self.num = num
        
        self.criteria = criteria

        self.test_cases = []
        self.max_retry = self.num

    def reset(self):
        self.token_used = 0
        self.calls = 0
        if getattr(self, "_pruned_db_snapshot_path", None) and os.path.exists(self._pruned_db_snapshot_path):
            os.remove(self._pruned_db_snapshot_path)

    @abstractmethod
    def _test_fn(self, ret):
        pass
    
    @abstractmethod
    def _generator(self):
        pass
    
    def set_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def write_test_fixture_file(self, output_dir, **kwargs):
        output_path = os.path.join(output_dir, 'meta.json')
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(kwargs, f, indent=2, ensure_ascii=False)

    def _load_cached_test_cases(self):
        return None
    
    def run(self):
        """Run all generated test cases in this test class
        """
        passes, avg_logprobs, traces = [], [], []
        fixtures, results = Munch(), Munch()
        self._generator()
        for tc in self.test_cases:
            passed, fixture, result, avg_logprob, trace = self.test_fn(tc)
            avg_logprobs.append(avg_logprob)
            traces.append(trace)
            passes.append(passed)
            for k, v in fixture.items():
                if k not in fixtures: fixtures[k] = []
                fixtures[k].append(v)
            for k, v in result.items():
                if k not in results: results[k] = []
                results[k].append(v)
        if not passes: detection_result = "UNDETERMINED"
        # Verify whether the number of passed test cases meets the criteria
        else: detection_result = True if np.sum(passes)/len(passes) >= self.criteria else False

        return np.array(passes), detection_result, results, self.criteria, avg_logprobs, self.token_used, self.calls, traces
