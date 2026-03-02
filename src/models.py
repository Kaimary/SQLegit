import json
from abc import ABC
from src.llm import LLM

class BaseNL2SQLModel(ABC):
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        raise NotImplementedError
    
class CSCSQL7b(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "cscsql-7b"

        dev_json = "data/bird/dev.json"
        output_file = "data/bird/results/csc-sql-7b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip() for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")

class CSCSQL32b(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "cscsql-32b"

        dev_json = "data/bird/dev.json"
        output_file = "data/bird/results/csc-sql-32b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip() for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")
    
class CHESS(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "chess"

        dev_json = "data/bird/dev.json"
        output_file = "data/bird/results/chess.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip() for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")
    
class OMNISQL32b(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "omnisql"

        dev_json = "data/bird/dev.json"
        output_file = "data/bird/results/omnisql-32b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip().encode('utf-8').decode('unicode_escape')
                        for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")

class RESDSQL(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "resdsql"

        dev_json = "data/spider/dev.json"
        output_file = "data/spider/results/resdsql-3b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip() for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")

class DAILSQL(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "dailsql"

        dev_json = "data/spider/dev.json"
        output_file = "data/spider/results/dailsql.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip().encode('utf-8').decode('unicode_escape')
                        for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")

class CODES7b(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "codes-7b"

        dev_json = "data/spider/dev.json"
        output_file = "data/spider/results/codes-7b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip().encode('utf-8').decode('unicode_escape')
                        for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")
    
class CODES15b(BaseNL2SQLModel):
    def __init__(self):
        super().__init__()
        self.model_name = "codes-15b"

        dev_json = "data/spider/dev.json"
        output_file = "data/spider/results/codes-15b.sql"
        self.dev = json.load(open(dev_json))
        self.outputs = [line.strip().encode('utf-8').decode('unicode_escape') 
                        for line in open(output_file)]

    def __call__(self, **kwargs):
        nl = kwargs.get("nl", None)
        for ex in self.dev:
            if ex["question"] == nl:
                idx = self.dev.index(ex)
                return self.outputs[idx]
        raise ValueError("No matching NL found in the dev set")

MODEL_CLASS_MAP = {
    "cscsql7b": CSCSQL7b,
    "cscsql32b": CSCSQL32b,
    "chess": CHESS,
    "omnisql32b": OMNISQL32b,
    "resdsql": RESDSQL,
    "dailsql": DAILSQL,
    "codes15b": CODES15b,
    "codes7b": CODES7b
}

class GenericLLM(BaseNL2SQLModel):
    def __init__(self, model_name: str = "gpt-4o-mini-0708"):
        super().__init__()
        self.model_name = model_name

    def __call__(self, **kwargs):
        model = LLM(model_name=self.model_name)
        response, metadata = model(
            prompt=kwargs.get("prompt"),
            parser=kwargs.get("parser"),
            request_kwargs=kwargs.get("request_kwargs", {})
        )
        sql = response["SQL"].strip()
        return sql, metadata
