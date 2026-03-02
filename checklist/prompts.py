import os
import logging
from typing import Any

from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

TEMPLATES_ROOT_PATH = "templates"

def load_template(template_name: str) -> str:
    """
    Loads a template from a file.

    Args:
        template_name (str): The name of the template to load.

    Returns:
        str: The content of the template.
    """
    file_name = f"template_{template_name}.j2"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    
    try:
        with open(template_path, "r") as file:
            template = file.read()
        logging.debug(f"Template {template_name} loaded successfully.")
        return template
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading template {template_name}: {e}")
        raise

def _get_prompt_template(template_name: str, **kwargs: Any) -> HumanMessagePromptTemplate:
    """
    Creates a HumanMessagePromptTemplate based on the provided template name and parameters.

    Args:
        template_name (str): The name of the template.
        **kwargs: Additional parameters for the template.

    Returns:
        HumanMessagePromptTemplate: The configured prompt template.

    Raises:
        ValueError: If the template name is invalid.
    """
    template_configs = {
        "nl2sql_translation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "INVALIDS": kwargs.get("invalid_queries_string", "")}},
        "nl2sql_translation_with_example": {"input_variables": ["HINT", "QUESTION", "SQL"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "INVALIDS": kwargs.get("invalid_queries_string", "")}},
        # "simulate_db_generation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "VALUES": kwargs.get("columns_values_string", ""), "HISTORY": kwargs.get("history_string", "")}},
        "oracle_data_generation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "VALUES": kwargs.get("columns_values_string", ""), "HISTORY": kwargs.get("history_string", "")}},
        # "oracle_result_checking": {"input_variables": ["HINT", "QUESTION", "INSTANCES", "RESULT1", "RESULT2"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        # "nl_relaxing_generation": {"input_variables": ["HINT", "QUESTION", "SQL"], "partial_variables": {"HISTORY": kwargs.get("history_string", ""), "INVALIDS": kwargs.get("invalid_queries_string", "")}},
        # "nl_strengthening_generation": {"input_variables": ["HINT", "QUESTION", "SQL"], "partial_variables": {"HISTORY": kwargs.get("history_string", ""), "INVALIDS": kwargs.get("invalid_queries_string", "")}},
        # "nl_mutation_generation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"HISTORY": kwargs.get("history_string", "")}},
        # "noise_data_table_determination": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "noise_data_injection": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "HISTORY": kwargs.get("history_string", "")}},
        # "noise_data_alignment_fix": {"input_variables": ["HINT", "QUESTION", "TABLE_NAME", "COLUMN_SPEC", "ROW_VALUES", "ISSUE_DESCRIPTION"]},
        "nl_paraphrase_generation": {"input_variables": ["HINT", "QUESTION", "SQL", "NUM"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "query_rubber_duck_debugging": {"input_variables": ["HINT", "QUESTION", "SQL", "RESULT", "SUBSQLS"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "nl_rubber_duck_debugging": {"input_variables": ["QUESTION", "SQL", "HINT", "RESULT"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "llm_nl2sql_judgment": {"input_variables": ["HINT", "QUESTION", "SQL"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "EXAMPLES": kwargs.get("examples_string", "")}},
        "llm_cot_nl2sql_judgment": {"input_variables": ["HINT", "QUESTION", "SQL"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", ""), "EXAMPLES": kwargs.get("examples_string", "")}},
        # "schema_pruning": {"input_variables": ["HINT", "QUESTION", "DATABASE_SCHEMA"], "partial_variables": {"KEYS": kwargs.get("keys_string", ""), "COLUMNS": kwargs.get("columns_string", ""), "ERROR": kwargs.get("error_string", "")}},
        "schema_pruning_by_selection": {"input_variables": ["HINT", "QUESTION", "LARGE_TABLES"], "partial_variables": {"ERROR": kwargs.get("error_string", "")}},
    }

    if template_name not in template_configs:
        raise ValueError(f"Invalid template name: {template_name}")

    config = template_configs[template_name]
    input_variables = config["input_variables"]
    partial_variables = config.get("partial_variables", {})

    template_content = load_template(template_name)
    
    human_message_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template_content,
            template_format="jinja2",
            input_variables=input_variables,
            partial_variables=partial_variables
        )
    )

    return human_message_prompt_template

def get_prompt(template_name: str, **kwargs) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate based on the provided template name and schema string.

    Args:
        template_name (str): The name of the template.
        schema_string (str, optional): The schema string for the template. Defaults to None.

    Returns:
        ChatPromptTemplate: The combined prompt template.
    """
    human_message_prompt_template = _get_prompt_template(template_name=template_name, **kwargs)
    
    combined_prompt_template = ChatPromptTemplate.from_messages(
        [human_message_prompt_template]
    )
    
    return combined_prompt_template
