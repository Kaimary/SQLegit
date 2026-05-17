import os
import copy
import json
import logging

from src.llm import LLM
from src.parsers import get_parser
from src.prompts import get_prompt
from src.base_judge import BaseJudge
from src.db_manager import DatabaseManager
from src.db_utils.execution import validate_sql_query
from src.red.parser.red_parser import Query
from src.db_utils.db_catalog.csv_utils import load_tables_description


class LLMJudge(BaseJudge):
    def __init__(self, name, model_name: str, enable_few_shot, enable_cot):
        super().__init__(name)
        self.model_name = model_name
        self.model = LLM(model_name=model_name)
        self.enable_few_shot = enable_few_shot
        self.enable_cot = enable_cot

    def set(self, nl, hint, pred, gold, db_id, db_root_path, red_schema =None):
        self.nl = nl
        self.hint = hint
        self.pred = pred
        self.gold = gold
        self.db_id = db_id
        self.db_root_path = db_root_path
        self.db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        self.red_schema = red_schema

        self.schema_string = self._build_schema_string(db_root_path)

    def _build_schema_string(self, db_root_path):
        schema = DatabaseManager(db_id=self.db_id, db_root_path=db_root_path).get_db_schema() # type: ignore
        # schema_with_examples = load_schema_with_examples(_get_unique_values(self.db_path))
        schema_with_descriptions = load_tables_description(self.db_path, use_value_description = True)
        return DatabaseManager().get_database_schema_string(
            tentative_schema=schema,
            schema_with_examples=None, # schema_with_examples,
            schema_with_descriptions=schema_with_descriptions,
            include_value_description=True
        )

    def run(self):
        if self.enable_cot:
            parser = get_parser(parser_name="llm_cot_nl2sql_judgment")
            prompt = get_prompt(
                template_name="llm_cot_nl2sql_judgment", 
                schema_string=self.schema_string,
                examples_string="placeholder" if self.enable_few_shot else None
            )
        else:
            parser = get_parser(parser_name="llm_nl2sql_judgment")
            prompt = get_prompt(
                template_name="llm_nl2sql_judgment", 
                schema_string=self.schema_string,
                examples_string="placeholder" if self.enable_few_shot else None
            )
        response, metadata = self.model(prompt, parser, request_kwargs={
            "HINT": self.hint, 
            "QUESTION": self.nl,
            "SQL": self.pred
            }
        )
        return response | metadata, None


class LLMExecutionReflectionJudge(LLMJudge):
    """LLM self-reflection baseline grounded with SQL execution feedback."""

    def __init__(self, name, model_name: str, enable_few_shot=False, enable_cot=False):
        super().__init__(name, model_name, enable_few_shot, enable_cot)

    def _execution_feedback(self, max_returned_rows=5):
        feedback = validate_sql_query(self.db_path, self.pred, max_returned_rows=max_returned_rows)
        return json.dumps(feedback, ensure_ascii=False, default=str)

    def run(self):
        parser = get_parser(parser_name="llm_self_reflection_nl2sql_judgment")
        prompt = get_prompt(
            template_name="llm_self_reflection_nl2sql_judgment",
            schema_string=self.schema_string
        )
        response, metadata = self.model(
            prompt,
            parser,
            request_kwargs={
                "HINT": self.hint,
                "QUESTION": self.nl,
                "SQL": self.pred,
                "EXECUTION_FEEDBACK": self._execution_feedback()
            }
        )
        return response | metadata, None


class SQLensFixAllJudge(LLMExecutionReflectionJudge):
    """SQLENS Fix-All style baseline using all available DB and LLM signals in one call."""

    LLM_SIGNAL_PROMPTS = (
        "sqlens_evidence_violation",
        "sqlens_insufficient_evidence",
        "sqlens_question_clause_linking",
        "sqlens_column_ambiguity",
        "sqlens_self_check_bool",
    )

    def _database_signals(self):
        signals = []
        try:
            parsed_query = Query(self.pred, copy.deepcopy(self.red_schema))
            for bug in parsed_query.validate():
                signals.append(self._format_red_bug(bug))
        except Exception as exc:
            logging.warning("RED signal extraction failed: %s", exc)
            signals.append({
                "signal": "red_parse_or_validation_error",
                "level": "ERROR",
                "description": f"{exc} SQL parse or validation failed."
            })

        if not signals:
            signals.append({
                "signal": "no_database_signal",
                "level": "INFO",
                "description": "No RED parser/runtime database signal was triggered."
            })
        return json.dumps(signals, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _format_red_bug(bug):
        if isinstance(bug, str):
            return {
                "signal": "red_parser_message",
                "level": "ERROR",
                "description": bug
            }
        return {
            "signal": getattr(bug, "location", "red_signal"),
            "level": str(getattr(bug, "level", "ERROR")),
            "description": getattr(bug, "description", str(bug))
        }

    def _llm_signals(self):
        signals = {}
        token_used = 0
        for signal_name in self.LLM_SIGNAL_PROMPTS:
            parser = get_parser(parser_name=signal_name)
            prompt = get_prompt(template_name=signal_name, schema_string=self.schema_string)
            response, metadata = self.model(
                prompt,
                parser,
                request_kwargs={
                    "HINT": self.hint,
                    "QUESTION": self.nl,
                    "SQL": self.pred
                }
            )
            signals[signal_name] = response
            if metadata and "token_used" in metadata:
                token_used += metadata["token_used"]
                if isinstance(signals[signal_name], dict):
                    signals[signal_name]["token_used"] = metadata["token_used"]
        return signals, token_used

    def run(self):
        parser = get_parser(parser_name="sqlens_fix_all_nl2sql_judgment")
        prompt = get_prompt(
            template_name="sqlens_fix_all_nl2sql_judgment",
            schema_string=self.schema_string
        )
        llm_signals, signal_token_used = self._llm_signals()
        response, metadata = self.model(
            prompt,
            parser,
            request_kwargs={
                "HINT": self.hint,
                "QUESTION": self.nl,
                "SQL": self.pred,
                "EXECUTION_FEEDBACK": self._execution_feedback(),
                "DATABASE_SIGNALS": json.dumps(
                    {
                        "database_signals": json.loads(self._database_signals()),
                        "llm_signals": llm_signals
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )
            }
        )
        response["llm_signals"] = llm_signals
        response["raw_final_judgment"] = response.get("final_judgment")
        if "needs_fix" in response:
            response["final_judgment"] = not response["needs_fix"]
        if metadata and "token_used" in metadata:
            metadata["token_used"] += signal_token_used
        return response | metadata, None
