import os
import json
import copy
import logging
from munch import Munch

from src.red.parser.report import BugLevel
from src.red.parser.red_parser import Query
from src.base_tester import BaseTester

class SemanticCheckTester(BaseTester):
    def __init__(self):
        super().__init__("Semantic Check Tester", "semantic_check", "semantic", key="sql")

    def set(self, red_schema, **kwargs):
        super().set(**kwargs)
        self.schema = red_schema

    def _compare_query_results(self, pred):
        if pred: return False
        return True
    
    def _test_fn(self, ret: Munch):
        ret.results = Munch()
        ret.results.warnings = [bug for bug in ret.test_fixtures.bugs if type(bug) != str and bug.level == BugLevel.WARNING]
        ret.results.pred = [bug for bug in ret.test_fixtures.bugs if type(bug) == str or bug.level == BugLevel.ERROR]
        ret.results.standard = "pred is empty"
        ret.results.target = None
        passed = self._compare_query_results(ret.results.pred)
        return passed, ret.test_fixtures, ret.results, None, ret.trace
    
    def write_test_fixture_file(self, output_dir, **kwargs):
        data = {
            "database": kwargs.get("database"),
            "sql": kwargs.get("sql"),
            "bugs": '\n'.join(str(bug) for bug in kwargs.get("bugs"))
        }
        output_path = os.path.join(output_dir, 'meta.json')
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
    def _form_instance(self, idx, ret):
        """
        Form each single test case, and save related test fixture for serialization. 
        Format as: a list of `bugs`
        
        Parameters
        ----------
        ret: Dict with `data` and `result` keys
        No return value
        """
        TEST_INSTANCE_ROOT_PATH = os.path.join(self.instance_saved_path, f"{idx}")
        os.makedirs(TEST_INSTANCE_ROOT_PATH, exist_ok=True)
        
        return ret
    
    def _generator(self):
        if self.use_cache: return self._load_cached_test_cases()
        
        bugs = []
        # spinner = Spinner(f"Generating test cases of `{self.name}` ...")
        # with spinner:
        ret = Munch()
        ret.test_fixtures = Munch()
        trace = f"->>Single Test Case Tracelog<<-\n"
        parsed_query = None
        try:
            parsed_query = Query(self.sql, copy.deepcopy(self.schema))
        except Exception as e:
            print(e)
            bugs.append(f"{e} SQL parse failed! \nSQL: {self.sql}")
        if parsed_query:
            try:
                bugs.extend(parsed_query.validate())
            except Exception as e:
                bugs.append(f"{e} Query validation process failed. \nSQL: {self.sql}")
        # for b in bugs: print(f"level: {b.level}, desc: {b.description}")
        # Hard-code for spider to ignore `column type mismathes aggregation` bugs
        if "spider" in self.db_path: bugs = [bug for bug in bugs if not isinstance(bug, str) and "but function" not in bug.description]
        for bug in bugs:
            trace += f"\nBugs found:\n{bug.description if not isinstance(bug, str) else bug}"
            logging.info(f"\nBugs found:\n{bug.description if not isinstance(bug, str) else bug}")
        ret.trace = trace
        ret.test_fixtures.bugs = bugs
        self.test_cases.append(self._form_instance(len(self.test_cases), ret))
        del parsed_query

        return
