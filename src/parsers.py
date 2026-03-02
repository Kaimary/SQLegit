import json
import re
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class PythonListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing Python lists."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract Python list content from markdown.

        Args:
            output (str): The output string containing Python list.

        Returns:
            Any: The parsed Python list.
        """
        logging.debug(f"Parsing output with PythonListOutputParser: {output}")
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return eval(output)  # Note: Using eval is potentially unsafe, consider using ast.literal_eval if possible.

class SQLGenerationOutput(BaseModel):
    """Model for SQL generation output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    SQL: str = Field(description="The generated SQL query in a single string.")

class MarkDownOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "```sql" in output:
            output = output.split("```sql")[1].split("```")[0]
            output = re.sub(r"^\s+", "", output)
            return {"SQL": output}
        elif "```nl" in output:
            output = output.split("```nl")[1].split("```")[0]
            output = re.sub(r"^\s+", "", output)
            return {"NL": output}
        # elif "```tbl" in output:
        #     output = output.split("```tbl")[1].split("```")[0]
        #     output = re.sub(r"^\s+", "", output)
        #     return {"table": output}
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
            output = re.sub(r"^\s+", "", output)
            return {"table": output}

class OracleDataGenerationOutput(BaseModel):
    """Model for oracle data generation output."""
    data: Dict[str, Any] = Field(description="The generated data instances based on the database schema")
    result: Dict[str, Any] = Field(description="The expected resulting data queried by the given natural language over the given databse schema")

class NoiseDataInjectionOutput(BaseModel):
    """Model for oracle data generation output."""
    injected_rows: Dict[str, Any] = Field(description="The injected data rows based on the database schema")
    injection_type: str = Field(description="The injected data rows are (ir)relevant to the query")
    chain_of_thought_reasoning: str = Field(description="Your thought process.")

# class NoiseDataAlignmentFixOutput(BaseModel):
#     """Model for repairing injected noise rows."""
#     fixed_rows: Dict[str, Any] = Field(description="Corrected noise data rows keyed by table name")

# class OracleDataVerificationOutput(BaseModel):
#     """Model for oracle data verification output."""
#     chain_of_thought_reasoning: str = Field(description="Your thought process on how you think.")
#     resulting_data: Dict[str, Any] = Field(description="The expected resulting data queried by the given natural language over the given databse schema")

# class OracleResultCheckingOutput(BaseModel):
#     """Model for oracle result checking output."""
#     result: Dict[str, Any] = Field(description="The expected resulting data queried by the given natural language over the given databse schema")

class QueryRelaxingOutput(BaseModel):
    """Model for query relaxing constraint generation output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you think.")
    type: str = Field(description="Type for the relaxing (remove/relax)")
    description: str = Field(description="Brief description of the relaxing operation")
    nl_mutant: str = Field(description="the natural language mutant transformed from the orignal natural language after applying the relaxing")
    sql_mutant: str = Field(description="the query mutant transformed from the orignal SQL after applying the relaxing")

class LLMCoTJudgmentOutput(BaseModel):
    """Model for LLM judgment output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you think.")
    judgment: str = Field(description="Yes or No")

class LLMJudgmentOutput(BaseModel):
    """Model for LLM judgment output."""
    judgment: str = Field(description="Yes or No")

class RubberDuckDebuggingOutput(BaseModel):
    """Model for LLM judgment output."""
    phrase_alignment: List[str] = Field(description="phrase alignment")
    reasoning_summary: str = Field(description="reasoning summary")
    judgment: str = Field(description="Yes or No")

class NLParaphraseOutput(BaseModel):
    """Model for paraphrase generation output."""
    paraphrases: List[str] = Field(description="Two semantically equivalent paraphrases of the input question.")

class SchemaPruningParser(BaseModel):
    """Model for schema pruning output."""
    pruned_schema: Dict[str, Any] = Field(description="The pruned database schema with only the necessary tables and columns.")

def get_parser(parser_name: str) -> BaseOutputParser:
    """
    Returns the appropriate parser based on the provided parser name.

    Args:
        parser_name (str): The name of the parser to retrieve.

    Returns:
        BaseOutputParser: The appropriate parser instance.

    Raises:
        ValueError: If the parser name is invalid.
    """
    parser_configs = {
        "nl2sql_translation": MarkDownOutputParser,
        "nl2sql_translation_with_example": MarkDownOutputParser,
        # "simulate_db_generation": lambda: JsonOutputParser(pydantic_object=OracleDataGenerationOutput),
        "oracle_data_generation": lambda: JsonOutputParser(pydantic_object=OracleDataGenerationOutput),
        # "oracle_result_checking": lambda: JsonOutputParser(pydantic_object=OracleResultCheckingOutput),
        # "nl_relaxing_generation": lambda: JsonOutputParser(pydantic_object=QueryRelaxingOutput),
        # "nl_strengthening_generation": lambda: JsonOutputParser(pydantic_object=QueryRelaxingOutput),
        # "nl_mutation_generation": MarkDownOutputParser,
        # "noise_data_table_determination": MarkDownOutputParser,
        "noise_data_injection": lambda: JsonOutputParser(pydantic_object=NoiseDataInjectionOutput),
        # "noise_data_alignment_fix": lambda: JsonOutputParser(pydantic_object=NoiseDataAlignmentFixOutput),
        "llm_nl2sql_judgment": lambda: JsonOutputParser(pydantic_object=LLMJudgmentOutput),
        "nl_rubber_duck_debugging": lambda: JsonOutputParser(pydantic_object=LLMCoTJudgmentOutput),
        "nl_paraphrase_generation": lambda: JsonOutputParser(pydantic_object=NLParaphraseOutput),
        "query_rubber_duck_debugging": lambda: JsonOutputParser(pydantic_object=LLMCoTJudgmentOutput),
        "llm_cot_nl2sql_judgment": lambda: JsonOutputParser(pydantic_object=LLMCoTJudgmentOutput),
        # "schema_pruning": lambda: JsonOutputParser(pydantic_object=SchemaPruningParser),
        "schema_pruning_by_selection": lambda: JsonOutputParser(pydantic_object=SchemaPruningParser),
    }

    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")

    logging.debug(f"Retrieving parser for: {parser_name}")
    parser = parser_configs[parser_name]() if callable(parser_configs[parser_name]) else parser_configs[parser_name]
    return parser
