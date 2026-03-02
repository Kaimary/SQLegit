import logging
import os
import random
import re
import time

from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser


def _env_float(*names: str, default: float) -> float:
    for name in names:
        val = os.getenv(name)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except ValueError:
            logging.warning("Invalid float in env %s=%r; using default=%s", name, val, default)
            break
    return default

CONFIGS: Dict[str, Dict[str, Any]] = {
    # "gemini-pro": {
    #     "constructor": ChatGoogleGenerativeAI,
    #     "params": {"model": "gemini-pro", "temperature": 0, "convert_system_message_to_human": True},
    #     "preprocess": lambda x: x.to_messages()
    # },
    # "gemini-1.5-pro-latest": {
    #     "constructor": ChatGoogleGenerativeAI,
    #     "params": {"model": "gemini-1.5-pro-latest", "temperature": 0, "convert_system_message_to_human": True},
    #     "preprocess": lambda x: x.to_messages()
    # },
    # "gpt-3.5-turbo-0125": {
    #     "constructor": ChatOpenAI,
    #     "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    # },
    # "gpt-3.5-turbo-instruct": {
    #     "constructor": ChatOpenAI,
    #     "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0}
    # },
    # "gpt-4-1106-preview": {
    #     "constructor": ChatOpenAI,
    #     "params": {"model": "gpt-4-1106-preview", "temperature": 0}
    # },
    # "gpt-4-0125-preview": {
    #     "constructor": ChatOpenAI,
    #     "params": {"model": "gpt-4-0125-preview", "temperature": 0}
    # },
    "gpt-4-turbo": {
        "constructor": AzureChatOpenAI,
        "params": {"model": "gpt-4.1", "temperature": 0, "logprobs": True}
    },
    "gpt-5.1": {
        "constructor": AzureChatOpenAI,
        "params": {"model": "gpt-5.1", "temperature": 0, "logprobs": True}
    },
    "gpt-4o-1120": {
        "constructor": AzureChatOpenAI,
        "params": {"model": "gpt-4o-1120", "temperature": 0, "logprobs": True}
    },
    "gpt-4o-mini-0708": {
        "constructor": AzureChatOpenAI,
        "params": {"model": "gpt-4o-mini-0708", "temperature": 0, "logprobs": True}
    },
    "deepseek-chat": {
        "constructor": ChatDeepSeek,
        "params": {"model": "deepseek-chat", "temperature": 0, "logprobs": True}
    },
    "deepseek-v3.2": {
        "constructor": AzureChatOpenAI,
        "params": {"model": "DeepSeek-V3.2", "temperature": 0, "logprobs": True}
    }
    # "claude-3-opus-20240229": {
    #     "constructor": ChatAnthropic,
    #     "params": {"model": "claude-3-opus-20240229", "temperature": 0}
    # },
    # "finetuned_nl2sql": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/NL2SQL_DeepSeek_33B",
    #         "api_key": "EMPTY",
    #         "max_tokens": 400,
    #         "temperature": 0,
    #         "model_kwargs": {
    #             "stop": ["```\n", ";"]
    #         }
    #     }
    # },
    # "meta-llama/Meta-Llama-3-70B-Instruct": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 600,
    #         "temperature": 0,
    #         "model_kwargs": {
    #             "stop": [""]
    #         }
    #     }
    # }
}


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.temperature = 0.0
        self.base_uri = None
        if model_name not in CONFIGS:
            raise ValueError(f"Model {model_name} not supported yet")
        
        config = CONFIGS[model_name]
        constructor = config["constructor"]
        params = config["params"]
        params["temperature"] = self.temperature

        # Keep single-call tail latency bounded; override via env if needed.
        # NOTE: read env here (not at import time) so `.env` loaded elsewhere can take effect.
        if constructor is AzureChatOpenAI:
            params.setdefault(
                "timeout",
                _env_float("CHECKLIST_LLM_TIMEOUT_S", "LLM_TIMEOUT_S", default=30.0),
            )
        if self.base_uri and "openai_api_base" in params:
            params["openai_api_base"] = f"{self.base_uri}/v1"
        model = constructor(**params)
        if "preprocess" in config:
            self.llm_chain = config["preprocess"] | model
        else:
            self.llm_chain = model

    def __call__(self, prompt, parser, request_kwargs, max_attempts: int = 1, backoff_base: int = 2, jitter_max: int = 60) -> Any:
        output, metadata = None, None
        for attempt in range(max_attempts):
            try:
                prompt_text = prompt.invoke(request_kwargs).messages[0].content
                logging.debug(f"prompt: \n\n{prompt_text}\n\n")
                raw_output = self.llm_chain.invoke(prompt_text)
                if isinstance(parser, JsonOutputParser):
                    # 去掉 // 注释 
                    raw_output.content = re.sub(r'(?<!:)//.*', '', raw_output.content)
                    # 把 NULL 替换成字符串 "NULL"
                    raw_output.content = re.sub(r'(?<=,\s)NULL\b', '"NULL"', raw_output.content)
                    # raw_output.content = re.sub(r"\\'", "'", raw_output.content)
                    # 转义反斜杠 \
                    # raw_output.content = re.sub(r'(".*?")', lambda m: m.group(0).replace('\\', '\\\\'), raw_output.content)
                    # # 转义双引号 "
                    # raw_output.content = re.sub(r'(".*?")', lambda m: m.group(0).replace('"', '\\"'), raw_output.content)
                output = parser.invoke(raw_output)
                # logging.debug(f"`{self.llm_chain.model_name}` model response: \"{raw_output.content}\"\n"
                #         f"\t- out tokens: {raw_output.response_metadata['token_usage']['completion_tokens']}\n"
                #         f"\t- prompt tokens: {raw_output.response_metadata['token_usage']['prompt_tokens']}\n"
                #         f"\t- total tokens: {raw_output.response_metadata['token_usage']['total_tokens']}")
                total_logprob = sum(token['logprob'] for token in raw_output.response_metadata['logprobs']['content'])
                metadata={
                    "token_used": raw_output.response_metadata['token_usage']['total_tokens'],
                    "avg_logprob": total_logprob / (len(raw_output.response_metadata['logprobs']['content']))#  ** 0.8)
                }
                break
            except OutputParserException as e:
                print(e)
                new_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_chain)
                chain = prompt | self.llm_chain | new_parser
                if attempt == max_attempts - 1:
                    raise e
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_attempts - 1:
                    sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
                    time.sleep(sleep_time)
                else:
                    raise e  
        return output, metadata
