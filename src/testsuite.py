import sys
import dill
import json
import inspect
import collections
import threading
import time
import numpy as np
from contextlib import nullcontext
from collections import defaultdict, OrderedDict

from .spinners import _Spinner


HIGH_PRECISION_TESTERS = {"SemanticCheckTester", "NoiseRowTester"}

class TestSuite:
    def __init__(self, format_example_fn=None, print_fn=None):
        self.tests = OrderedDict()
        self.info = defaultdict(lambda: defaultdict(lambda: ''))
        self.format_example_fn = format_example_fn
        self.print_fn = print_fn
        self.test_ranges = {}

    def set(self, backbone, nl, hint, sql, gold, db_id, db_root_path, red_schema):
        for t in self.tests.values():
            sig = inspect.signature(t.set)
            kwargs = dict(
                backbone_llm_model_name=backbone,
                nl=nl,
                hint=hint,
                sql=sql,
                gold=gold,
                db_id=db_id,
                db_root_path=db_root_path,
            )

            if "red_schema" in sig.parameters:
                kwargs["red_schema"] = red_schema
            
            t.set(**kwargs)

    def add(self, test, name=None):
        if name is None and test.name is None:
            raise(Exception('If test does not have test.name, you must specify a name'))
        if name is None:
            name = test.name
        self.tests[name] = test

    def remove(self, name):
        if name not in self.tests:
            raise(Exception('%s not in suite.' % name))
        del self.tests[name]
        del self.info[name]

    def to_dict(self, example_to_dict_fn=None, n=None, seed=None, new_sample=False):
        if example_to_dict_fn is None:
            try:
                example_to_dict_fn = self.example_to_dict_fn
            except AttributeError:
                raise(Exception('suite does not have example_to_dict_fn, must pass function as argument.'))
        examples = self.get_raw_examples(format_fn=lambda x:x, n=n, seed=seed, new_sample=new_sample)
        data_keys = list(example_to_dict_fn(examples[0]).keys())
        keys = data_keys + ['test_name', 'test_case', 'example_idx']
        hf_dict = { k:[] for k in keys }
        for e in examples:
            m = example_to_dict_fn(e)
            for k,v  in m.items():
                hf_dict[k].append(v)
        for test_name, r in sorted(self.test_ranges.items(), key=lambda x:x[1][0]):
            test = self.tests[test_name]
            size = r[1] - r[0]
            hf_dict['test_name'].extend([test_name for _ in range(size)])
            hf_dict['test_case'].extend(test.result_indexes)
            cnt = collections.defaultdict(lambda: 0)
            example_idx = []
            for i in test.result_indexes:
                example_idx.append(cnt[i])
                cnt[i] += 1
            hf_dict['example_idx'].extend(example_idx)
        return hf_dict

    def get_raw_examples(self, file_format=None, format_fn=None, n=None, seed=None, new_sample=True):
        if new_sample or len(self.test_ranges) == 0:
            self.test_ranges = {}
            all_examples = self.create_raw_example_list(file_format=file_format, format_fn=format_fn, n=n, seed=seed)
        else:
            all_examples = self.get_raw_example_list(file_format=file_format, format_fn=format_fn)
        return all_examples

    def get_raw_example_list(self, file_format=None, format_fn=None):
        if not self.test_ranges:
            raise(Exception('example list not created. please call create_raw_example_list, or to_raw_file first'))
        examples = []
        for test_name, r in sorted(self.test_ranges.items(), key=lambda x:x[1][0]):
            test = self.tests[test_name]
            test_examples = test.to_raw_examples(file_format=file_format, format_fn=format_fn,
                                         n=None, seed=None, new_sample=False)
            assert len(test_examples) == r[1] - r[0]
            examples.extend(test_examples)
        return examples

    def create_raw_example_list(self, file_format, format_fn, n, seed):
        self.test_ranges = {}
        current_idx = 0
        all_examples = []
        for name, t in self.tests.items():
            examples = t.to_raw_examples(file_format=file_format, format_fn=format_fn, n=n, seed=seed, new_sample=True)
            self.test_ranges[name] = (current_idx, current_idx + len(examples))
            current_idx += len(examples)
            all_examples.extend(examples)
        return all_examples

    def to_raw_file(self, path, file_format=None, format_fn=None, header=None, n=None, seed=None, new_sample=True):
        """Flatten all tests into individual examples and print them to file.
        Indices of example to test case will be stored in each test.
        If n is not None, test.run_idxs will store the test case indexes.
        The line ranges for each test will be saved in self.test_ranges.

        Parameters
        ----------
        path : string
            File path
        file_format : string, must be one of 'jsonl', 'squad', 'qqp_test', or None
            None just calls str(x) for each example in self.data
            squad assumes x has x['question'] and x['passage'], or that format_fn does this
        format_fn : function or None
            If not None, call this function to format each example in self.data
        header : string
            If not None, first line of file
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None
        new_sample: bool
            If False, will rely on a previous sample and ignore the 'n' and 'seed' parameters

        """
        ret = ''
        all_examples = []
        add_id = False
        if file_format == 'qqp_test':
            add_id = True
            file_format = 'tsv'
            header = 'id\tquestion1\tquestion2'
        if header is not None:
            ret += header.strip('\n') + '\n'
        all_examples = self.get_raw_examples(file_format=file_format, format_fn=format_fn, n=n, seed=seed, new_sample=new_sample)

        if add_id and file_format == 'tsv':
            all_examples = ['%d\t%s' % (i, x) for i, x in enumerate(all_examples)]
        if file_format == 'squad':
            ret_map = {'version': 'fake',
                       'data': []}
            for i, x in enumerate(all_examples):
                r = {'title': '',
                     'paragraphs': [{
                        'context': x['passage'],
                        'qas': [{'question' : x['question'],
                                 'id': str(i)
                                 }]
                      }]
                    }
                ret_map['data'].append(r)
            ret = json.dumps(ret_map)
        else:
            ret += '\n'.join(all_examples)
        f = open(path, 'w')
        f.write(ret)
        f.close()

    def run_from_preds_confs(self, preds, confs, overwrite):
        for n, t in self.tests.items():
            p = preds[slice(*self.test_ranges[n])]
            c = confs[slice(*self.test_ranges[n])]
            t.run_from_preds_confs(p, c, overwrite=overwrite)

    def run(self, predict_and_confidence_fn, verbose=True, **kwargs):
        """Runs all tests in the suite
        See run in abstract_test.py .

        Parameters
        ----------
        predict_and_confidence_fn : function
            Takes as input a list of examples
            Outputs a tuple (predictions, confidences)
        overwrite : bool
            If False, raise exception if results already exist
        verbose : bool
            If True, print extra information
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None

        """
        for n, t in self.tests.items():
            if verbose:
                print('Running', n)
            t.run(predict_and_confidence_fn, verbose=verbose, **kwargs)

    def run1(self, verbose=True):
        """Runs all tests in the suite
        See run in abstract_test.py .

        Parameters
        ----------
        verbose : bool
            If True, print extra information
        """
        ret = {}
        judgments = []
        munch = None
        vote_true = 0
        vote_false = 0
        score = 0
        veto = False
        cond_arr = []
        majority = len([t for t in self.tests.values() if t.__class__.__name__ not in HIGH_PRECISION_TESTERS]) / 2 + 1
        correct, incorrect = False, False
        break_triggered = False

        def _result_symbol(val):
            if val is True:
                return "✅"
            if val is False:
                return "❌"
            return "🤔"

        status_state = OrderedDict((name, ("pending", None, None, None, None)) for name in self.tests.keys())
        status_rendered = False
        render_lock = threading.Lock()

        def _render_status_line(name):
            def __strike(text):
                return ''.join(c + '\u0336' for c in text)
            state_info = status_state[name]
            state = state_info[0]
            symbol = state_info[1] if len(state_info) > 1 else None
            calls = state_info[2] if len(state_info) > 2 else None
            tokens_used = state_info[3] if len(state_info) > 3 else None
            elapsed = state_info[4] if len(state_info) > 4 else None
            if state == "pending":
                return f"[  ] {name}"
            if state == "aborted":
                return __strike(f"[  ] {name}")
            if state == "running":
                frame = symbol if symbol else "|"
                return f"[ {frame} ] {name}"
            suffix = ""
            if symbol:
                suffix_parts = [symbol]
                metrics = []
                if calls is not None:
                    metrics.append(f"(calls: {calls},")
                if tokens_used is not None:
                    metrics.append(f"tokens: {tokens_used},")
                if elapsed is not None:
                    metrics.append(f"time: {elapsed:.2f}s)")
                if metrics:
                    suffix_parts.append(" ".join(metrics))
                suffix = " " + " ".join(suffix_parts)
            return f"[ ✔️ ] {name}{suffix}"

        def _print_status_block():
            nonlocal status_rendered
            if not verbose:
                return
            with render_lock:
                lines = len(status_state)
                if status_rendered and lines:
                    sys.stdout.write(f"\033[{lines}F")
                for tn in status_state:
                    sys.stdout.write(_render_status_line(tn) + "\n")
                sys.stdout.flush()
                status_rendered = True

        def _update_spinner_line(test_name, frame):
            status_state[test_name] = ("running", frame)
            _print_status_block()

        _print_status_block()
        last_tester = None

        for name, t in self.tests.items():
            if verbose:
                status_state[name] = ("running", "|")
                _print_status_block()
                spinner_ctx = _Spinner(lambda frame, tn=name: _update_spinner_line(tn, frame))
            else:
                spinner_ctx = nullcontext()
            with spinner_ctx:
                start_time = time.time()
                passed, judgment, munch, criteria, avg_logprobs, tokens_used, calls, traces = t.run()
                t.reset()
            elapsed = time.time() - start_time
            status_state[name] = ("completed", _result_symbol(judgment), calls, tokens_used, elapsed)
            _print_status_block()
            last_tester = t.__class__.__name__
            confidence = 0
            if isinstance(judgment, bool):
                judgments.append(judgment)
                if t.__class__.__name__ not in HIGH_PRECISION_TESTERS:
                    if judgment:
                        vote_true += 1
                    else:
                        vote_false += 1
                    probs = np.exp(avg_logprobs)
                    signs = np.where(passed, 1, -1)
                    confidence = abs(np.mean(probs * signs)) # confidence magnitude
                    score += confidence if judgment else -1 * confidence
                else:
                    veto = judgment

            cond_arr.append(confidence)

            ret[name] = {
                "judgment": judgment,
                "total": len(passed),
                "passed": int(np.sum(passed)),
                "results": passed.tolist(),
                "avg_logprobs": avg_logprobs,
                "confidence": confidence,
                "tokens_used": tokens_used,
                "calls": calls,
                "criteria": criteria,
                "traces": traces,
                "elapsed": elapsed
            }
            stop_due_to = None
            if t.__class__.__name__ in HIGH_PRECISION_TESTERS and judgment is False:
                incorrect = True
                stop_due_to = "critical_failure"
            elif vote_false >= majority:
                incorrect = True
                stop_due_to = "majority_false"
            elif vote_true >= majority:
                correct = True
                stop_due_to = "majority_true"

            if stop_due_to:
                ret["last_tester"] = last_tester
                break_triggered = True
                break

        if not break_triggered and last_tester is not None:
            ret["last_tester"] = last_tester

        if break_triggered:
            pending_exists = False
            for tn, (state, _, _, _, _) in status_state.items():
                if state == "pending":
                    status_state[tn] = ("aborted", None)
                pending_exists = True
            if pending_exists:
                _print_status_block()

        if incorrect:
            ret["final_judgment"] = False
        elif correct:
            ret["final_judgment"] = True
        elif not judgments:
            ret["final_judgment"] = "UNDETERMINED"
        else:
            if verbose and len(self.tests.keys()) > 1: print(f"TIE (no clear correct or incorrect decision), use confidences ({cond_arr})")
            if score == 0: 
                ret["final_judgment"] = veto
            else: 
                ret["final_judgment"] = True if score > 0 else False
        return ret, munch

    def summary(self, types=None, capabilities=None, **kwargs):
        """Print stats and example failures for each test.
        See summary in abstract_test.py

        Parameters
        ----------
        types : list(string)
            If not None, will only show tests of these test types.
            Options are MFT, INV, and DIR
        capabilities : list(string)
            If not None, will only show tests with these capabilities.
        **kwargs : type
            Will be passed as arguments to each test.summary()

        """
        vals = collections.defaultdict(lambda: 100, {'MFT': 0, 'INV': 1, 'DIR': 2})
        tests = self.tests.keys()
        capability_order = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER',  'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']
        cap_order = lambda x:capability_order.index(x) if x in capability_order else 100
        caps = sorted(set([x['capability'] for x in self.info.values()]), key=cap_order)
        for capability in caps:
            if capabilities is not None and capability not in capabilities:
                continue
            print(capability)
            print()
            tests = [x for x in self.tests if self.info[x]['capability'] == capability]
            for n in tests:
                if types is not None and self.info[n]['type'] not in types:
                    continue
                print(n)
                if 'format_example_fn' not in kwargs:
                    kwargs['format_example_fn'] = self.info[n].get('format_example_fn', self.format_example_fn)
                if 'print_fn' not in kwargs:
                    kwargs['print_fn'] = self.info[n].get('print_fn', self.print_fn)
                self.tests[n].summary(**kwargs)
                print()
                print()
            print()
            print()

    def summary1(self, ret, munch, baseline_judgment=None, gold=None):
        """Print stats for each test comparing with baseline/gold judgments.

        Parameters
        ----------
        **kwargs : type
            Will be passed as arguments to each test.summary()

        """
        for k, v in ret.items():
            if k in ["final_judgment", "last_tester"]: continue
            test_judgment = v['judgment']
            results = v['results']
            if test_judgment is None: 
                print(f"\033[94m\n{k}: [Skip]\033[0m")
                continue

            test_evaluation = test_judgment == gold
            baseline_evaluation = baseline_judgment == gold
            if not test_evaluation:
                print(f"\033[94m\n{k}:\033[0m Judgment ({test_judgment})\n\033[92mCorrectness? ❌\033[0m")
                if gold and not test_judgment: #FN
                    print(f"\033[94m\n{k}:\033[0m Results: {results}\n\033[92mPreds: {munch.pred}\n\033[92mTarget: {munch.target}\033[0m")
            if baseline_evaluation and not test_evaluation:
                print(f"\033[92mBeat Baseline? ❌\033[0m")
                print(f"\033[92m[info] \033[0mTotal Test Cases: {v['total']}, Passed: {v['passed']}, Criteria: {v['criteria']}")
                if v['traces']:
                    print(f"\033[92m[trace]\033[0m")
                    print(v['traces'][0])
            print()

    def visual_summary_by_test(self, testname):
        """Displays visual summary for a single test.

        Parameters
        ----------
        testname : string
            name of the test

        Returns
        -------
        test.visual_summary
            summary

        """
        if not testname in self.tests:
            raise(Exception(f"There's no test named {testname} in the suite!"))
        test, info = self.tests[testname], self.info[testname]
        return test.visual_summary(
            name=testname,
            capability=info["capability"] if "capability" in info else None,
            description=info["description"] if "description" in info else None
        )

    def _on_select_test(self, testname: str):
        if not testname:
            test_info, testcases = {}, []
        else:
            if not testname in self.tests:
                raise(Exception(f"There's no test named {testname} in the suite!"))
            test, info = self.tests[testname], self.info[testname]
            test_info = test.form_test_info(
                name=testname,
                capability=info["capability"] if "capability" in info else None,
                description=info["description"] if "description" in info else None
            )
            n = 1 if self.info[testname]['type'] == 'MFT' else 2
            testcases = test.form_testcases(n_per_testcase=n)
        return test_info, testcases

    def save(self, path):
        """Serializes the suite and saves it to a file

        Parameters
        ----------
        path : string
            output file path

        """
        dill.dump(self, open(path, 'wb'), recurse=True)
