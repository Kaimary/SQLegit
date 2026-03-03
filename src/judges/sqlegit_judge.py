from src.base_judge import BaseJudge
from src.testsuite import TestSuite


class SQLegitJudge(BaseJudge):
    def __init__(self, name, backbone_llm_model_name, *tests):
        super().__init__(name)
        self.backbone = backbone_llm_model_name
        self.suite = TestSuite()
        # Iterate over the provided test classes and add them to the suite
        for test in tests:
            test_instance = test()  # Create an instance of the test class
            self.suite.add(test_instance, name=test_instance.abbrev_name)

    def set(self, nl, hint, sql, gold, db_id, db_root_path, red_schema):
        self.suite.set(
            backbone=self.backbone,
            nl=nl,
            hint=hint,
            sql=sql,
            gold=gold,
            db_id=db_id,
            db_root_path=db_root_path,
            red_schema=red_schema
        )

    def run(self):
        return self.suite.run1()
    
    def summary(self, ret, munch, baseline_judgment, gold):
        return self.suite.summary1(ret, munch, baseline_judgment, gold)