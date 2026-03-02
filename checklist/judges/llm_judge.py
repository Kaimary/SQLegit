import os

from checklist.llm import LLM
from checklist.parsers import get_parser
from checklist.prompts import get_prompt
from checklist.base_judge import BaseJudge
from checklist.db_manager import DatabaseManager
from checklist.db_utils.db_catalog.csv_utils import load_tables_description


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

        schema = DatabaseManager(db_id=self.db_id, db_root_path=db_root_path).get_db_schema() # type: ignore
        # schema_with_examples = load_schema_with_examples(_get_unique_values(self.db_path))
        schema_with_descriptions = load_tables_description(self.db_path, use_value_description = True)
        self.schema_string = DatabaseManager().get_database_schema_string(
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