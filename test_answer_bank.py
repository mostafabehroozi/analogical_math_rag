"""Answer bank: extraction, rules, reuse, packing, lazy Pass@K and log replay."""

import json
import os
import re
import tempfile
import threading
import time
import unittest

from src.answer_bank import (
    PACKED_JUDGE, AnswerBank, extract_final_answer, forget_answer_bank, get_answer_bank,
    ground_truth_answer, normalize_answer, numeric_verdict, parse_packed_reply,
)
from src.answer_bank_replay import replay_answer_judging
from src.distributed_execution import _is_runtime_or_secret_key
from src.evaluation import LAZY_SKIP_STATUS, analyze_experiment_logs, evaluate_single_answer_with_llm


class FakeJudge:
    """An evaluator manager whose reply is computed from the prompt."""

    provider_name = "openrouter"

    def __init__(self, reply=None, gate=None):
        self.prompts = []
        self._lock = threading.Lock()
        self._reply = reply or (lambda prompt: "Extracted Model Final Answer: x\nEvaluation: true")
        self._gate = gate

    def generate_content(self, prompt, model, temperature, avalai_role=None):
        with self._lock:
            self.prompts.append(prompt)
        if self._gate is not None:
            self._gate.wait(5)
        return {"status": "SUCCESS", "text": self._reply(prompt)}


def single_reply(answer, verdict):
    return lambda prompt: f"Extracted Model Final Answer: {answer}\nEvaluation: {'true' if verdict else 'false'}"


class BankTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = {
            "ANSWER_BANK_ENABLED": True, "RESULTS_DIR": self.tmp.name,
            "OPENROUTER_MODEL_NAME_EVALUATOR": "judge", "DEFAULT_EVALUATOR_TEMPERATURE": 0.0,
            "ANSWER_BANK_AUDIT_RATE": 0.0, "ANSWER_BANK_RECHECK_HITS": 0, "ANSWER_JUDGE_PACK_MAX_ITEMS": 1,
        }
        self.addCleanup(forget_answer_bank, self.config)

    def judge(self, output, ground_truth, judge):
        return evaluate_single_answer_with_llm(output, ground_truth, judge, self.config)

    def bank_rows(self):
        with open(os.path.join(self.tmp.name, "answer_bank.jsonl"), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]


class ExtractionTests(unittest.TestCase):
    def test_final_answer_comes_from_the_later_of_boxed_and_header(self):
        self.assertEqual(extract_final_answer(r"so $\boxed{\frac{\sqrt{3}}{2}}$ holds"), r"\frac{\sqrt{3}}{2}")
        self.assertEqual(extract_final_answer("We get \\boxed{5}.\nFinal Answer:\n\n12"), "12")
        self.assertEqual(extract_final_answer("Rationale...\nFinal Answer: $\\boxed{7}$"), "7")
        self.assertEqual(extract_final_answer("**Final Answer:** The answer is 12."), "12.")
        self.assertIsNone(extract_final_answer("no answer here"))
        self.assertIsNone(extract_final_answer("Final Answer: \\boxed{12"))  # cut off
        self.assertIsNone(extract_final_answer("Final Answer: " + "word " * 40))

    def test_ground_truth_answer_rejects_multi_part_solutions(self):
        self.assertEqual(ground_truth_answer("long solution\nso the answer is $\\boxed{4}$."), "4")
        self.assertEqual(ground_truth_answer("\\boxed{4} and again \\boxed{ 4 }"), "4")
        self.assertIsNone(ground_truth_answer("(1) $\\boxed{3}$\n(2) $\\boxed{5}$"))
        self.assertEqual(ground_truth_answer("\\frac{14}{3}"), "\\frac{14}{3}")
        self.assertIsNone(ground_truth_answer("a solution\nwithout any boxed answer"))

    def test_normalization_and_numbers(self):
        self.assertEqual(normalize_answer("$x = 3$."), "3")
        self.assertEqual(normalize_answer("1,000"), "1000")
        self.assertEqual(normalize_answer("\\dfrac{1}{2}"), normalize_answer("\\frac{1}{2}"))
        self.assertEqual(normalize_answer("(B)"), "B")
        self.assertEqual(normalize_answer("\\left( 3, \\frac{\\pi}{2} \\right)"), "(3,\\frac{\\pi}{2})")
        self.assertEqual(normalize_answer("90^\\circ"), "90")
        self.assertTrue(numeric_verdict("0.5", "\\frac{1}{2}"))
        self.assertTrue(numeric_verdict("12.0", "12"))
        self.assertFalse(numeric_verdict("12", "13"))
        self.assertIsNone(numeric_verdict("0.33", "\\frac{1}{3}"))  # rounding: the judge decides
        self.assertIsNone(numeric_verdict("15%", "0.15"))

    def test_packed_reply_parsing_distrusts_unclear_items(self):
        reply = ("Item 1\nExtracted Model Final Answer: 4\nEvaluation: true\n\n"
                 "**Item 2**\nExtracted Model Final Answer: 5\nEvaluation: **false**\n\n"
                 "Item 3\nExtracted Model Final Answer: 6\nEvaluation: true\n"
                 "Item 3\nExtracted Model Final Answer: 7\nEvaluation: false\n")
        self.assertEqual(parse_packed_reply(reply, 4), [("4", True), ("5", False), None, None])


class BankJudgeTests(BankTestCase):
    def test_rules_decide_without_a_call(self):
        judge = FakeJudge()
        result = self.judge("Final Answer: 12", "12", judge)
        self.assertEqual((result["is_correct"], result["judge_method"]), (True, "exact"))
        result = self.judge("Final Answer: 13", "12", judge)
        self.assertEqual((result["is_correct"], result["judge_method"]), (False, "numeric"))
        self.assertEqual(judge.prompts, [])

    def test_a_judged_answer_is_reused_from_memory_and_from_the_file(self):
        solution = "a long solution\nso $\\boxed{\\frac{\\sqrt{3}}{2}}$"
        judge = FakeJudge(single_reply("\\frac{1}{2}\\sqrt{3}", True))
        first = self.judge("work\nFinal Answer: \\frac{1}{2}\\sqrt{3}", solution, judge)
        self.assertEqual((first["is_correct"], first["judge_method"]), (True, "llm"))
        # The judge compared with the solution's own final answer.
        self.assertIn("Ground Truth:\n\\frac{\\sqrt{3}}{2}", judge.prompts[0])
        second = self.judge("other work\nFinal Answer: $\\frac{1}{2} \\sqrt{3}$", solution, judge)
        self.assertEqual((second["is_correct"], second["judge_method"]), (True, "bank"))
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(len(self.bank_rows()), 1)
        self.assertEqual(len(self.bank_rows()[0]["keys"]), 2)

        forget_answer_bank(self.config)
        third = self.judge("third\n\\boxed{\\frac{1}{2}\\sqrt{3}}", solution, judge)
        self.assertEqual(third["judge_method"], "bank")
        self.assertEqual(len(judge.prompts), 1)

    def test_answer_is_not_reused_when_the_judge_read_another_answer(self):
        judge = FakeJudge(single_reply("7", False))
        self.judge("a\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)
        self.judge("b\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)
        self.assertEqual(len(judge.prompts), 2)
        # The identical output is still reused.
        self.assertEqual(self.judge("a\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)["judge_method"], "bank_text")
        self.assertEqual(len(judge.prompts), 2)

    def test_conflicting_verdicts_go_back_to_the_judge(self):
        judge = FakeJudge(single_reply("2\\sqrt{3}", True))
        case_output = "Final Answer: 2\\sqrt{3}"
        self.judge(case_output, "\\sqrt{12}", judge)
        bank = get_answer_bank(self.config)
        scope = next(iter(bank._verdicts))
        bank.record(scope, ["a:2\\sqrt{3}"], False)
        self.judge("another\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)
        self.assertEqual(len(judge.prompts), 2)

    def test_multi_part_ground_truth_reuses_only_identical_outputs(self):
        solution = "(1) $\\boxed{3}$\n(2) $\\boxed{5}$"
        judge = FakeJudge(single_reply("5", True))
        self.judge("x\nFinal Answer: 5", solution, judge)
        self.judge("y\nFinal Answer: 5", solution, judge)
        self.assertEqual(len(judge.prompts), 2)
        self.assertIn("Extracted Ground Truth Answer", judge.prompts[0])  # evaluator_v1, whole solution
        self.assertEqual(self.judge("x\nFinal Answer: 5", solution, judge)["judge_method"], "bank_text")

    def test_bank_off_calls_the_judge_as_before(self):
        self.config["ANSWER_BANK_ENABLED"] = False
        self.config["TARGET_BENCHMARK"] = "math500"
        judge = FakeJudge(single_reply("12", True))
        result = self.judge("Final Answer: 12", "12", judge)
        self.assertEqual(result, {"is_correct": True, "status": "SUCCESS", "error_details": None})
        self.assertEqual(len(judge.prompts), 1)

    def test_audit_uses_the_judge_verdict_when_it_disagrees(self):
        self.config["ANSWER_BANK_AUDIT_RATE"] = 1.0
        judge = FakeJudge(single_reply("12", False))
        with self.assertLogs("src.evaluation", level="WARNING"):
            result = self.judge("Final Answer: 12", "12", judge)
        self.assertEqual((result["is_correct"], result["judge_method"]), (False, "audit"))
        # The judge's verdict is banked and wins over the rule from now on.
        result = evaluate_single_answer_with_llm("again\nFinal Answer: 12", "12", judge,
                                                 {**self.config, "ANSWER_BANK_AUDIT_RATE": 0.0})
        self.assertEqual((result["is_correct"], result["judge_method"]), (False, "bank"))

    def test_heavily_reused_answer_is_rechecked_once(self):
        self.config["ANSWER_BANK_RECHECK_HITS"] = 2
        judge = FakeJudge(single_reply("2\\sqrt{3}", True))
        for number in range(5):
            self.judge(f"run {number}\nFinal Answer: 2\\sqrt{{3}}", "\\sqrt{12}", judge)
        self.assertEqual(len(judge.prompts), 2)  # the first judgment and one recheck

    def test_value_of_a_banked_answer_decides_equal_numbers(self):
        judge = FakeJudge(single_reply("0.15", True))
        self.judge("Final Answer: 0.15", "15\\%", judge)
        result = self.judge("Final Answer: \\frac{3}{20}", "15\\%", judge)
        self.assertEqual((result["is_correct"], result["judge_method"]), (True, "bank_value"))
        self.assertEqual(len(judge.prompts), 1)


class PackingTests(BankTestCase):
    VERDICTS = {"\\sqrt{12}": True, "3\\sqrt{2}": False, "\\sqrt{3}\\cdot 2": True}

    def packed_reply(self, garble=None):
        def reply(prompt):
            if "<Item 1>" not in prompt:
                answer = extract_final_answer(prompt.split("Model Output:\n", 1)[1].split("\n\nGround Truth", 1)[0])
                return f"Extracted Model Final Answer: {answer}\nEvaluation: {str(self.VERDICTS[answer]).lower()}"
            blocks = []
            for number, tail in re.findall(r"<Item (\d+)>\nModel Output:\n(.*?)\n\nGround Truth", prompt, re.S):
                if number == garble:
                    continue
                answer = extract_final_answer(tail)
                blocks.append(f"Item {number}\nExtracted Model Final Answer: {answer}\n"
                              f"Evaluation: {str(self.VERDICTS[answer]).lower()}")
            return "\n\n".join(blocks)
        return reply

    def judge_together(self, judge):
        outputs = [f"work {n}\nFinal Answer: {answer}" for n, answer in enumerate(self.VERDICTS)]
        results = [None] * len(outputs)

        def run(index):
            results[index] = self.judge(outputs[index], "2\\sqrt{3}", judge)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(len(outputs))]
        with PACKED_JUDGE.active():  # another judging thread, so the batch waits to fill
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
        return results

    def test_answers_arriving_together_share_one_call(self):
        self.config.update({"ANSWER_JUDGE_PACK_MAX_ITEMS": 3, "ANSWER_JUDGE_PACK_COLLECT_SECONDS": 5.0})
        judge = FakeJudge(self.packed_reply())
        results = self.judge_together(judge)
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual([r["is_correct"] for r in results], list(self.VERDICTS.values()))
        self.assertEqual({r["judge_method"] for r in results}, {"llm_packed"})

    def test_an_unclear_packed_item_gets_its_own_call(self):
        self.config.update({"ANSWER_JUDGE_PACK_MAX_ITEMS": 3, "ANSWER_JUDGE_PACK_COLLECT_SECONDS": 5.0})
        judge = FakeJudge(self.packed_reply(garble="2"))
        results = self.judge_together(judge)
        self.assertEqual(len(judge.prompts), 2)
        self.assertEqual([r["is_correct"] for r in results], list(self.VERDICTS.values()))
        self.assertEqual(sorted(r["judge_method"] for r in results), ["llm", "llm_packed", "llm_packed"])

    def test_identical_answer_in_flight_waits_instead_of_calling(self):
        gate = threading.Event()
        judge = FakeJudge(single_reply("2\\sqrt{3}", True), gate=gate)
        results = {}
        first = threading.Thread(target=lambda: results.update(a=self.judge("a\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)))
        second = threading.Thread(target=lambda: results.update(b=self.judge("b\nFinal Answer: 2\\sqrt{3}", "\\sqrt{12}", judge)))
        first.start()
        while not judge.prompts:
            time.sleep(0.01)
        second.start()
        time.sleep(0.2)
        gate.set()
        first.join(5)
        second.join(5)
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual((results["a"]["judge_method"], results["b"]["judge_method"]), ("llm", "llm_joined"))
        self.assertTrue(results["b"]["is_correct"])


class LazyPassAtKTests(unittest.TestCase):
    def run_analysis(self, lazy):
        with tempfile.TemporaryDirectory() as results_dir:
            judge = FakeJudge(lambda prompt: "Evaluation: " + ("true" if "RIGHT" in prompt else "false"))
            config = {
                "RESULTS_DIR": results_dir, "PASS_K_VALUES_TO_REPORT": [1, 2, 3], "BATCH_MAX_WORKERS": 1,
                "API_PROVIDER_EVALUATOR": "openrouter", "OPENROUTER_MODEL_NAME_EVALUATOR": "judge",
                "DEFAULT_EVALUATOR_TEMPERATURE": 0.0, "TARGET_BENCHMARK": "math500", "EVAL_LAZY_PASS_AT_K": lazy,
            }
            logs = {"exp": [{"target_query_original_hard_list_idx": 0, "steps": {"solving": {
                "solution_attempts": ["Final Answer: WRONG", "Final Answer: RIGHT", "Final Answer: WRONG"]}}}]}
            summary = analyze_experiment_logs(logs, ["12"], {"openrouter": judge}, config)
            with open(os.path.join(results_dir, "exp_detailed_eval.json"), encoding="utf-8") as handle:
                entry = json.load(handle)[0]
        return summary.iloc[0], entry, len(judge.prompts)

    def test_lazy_pass_at_k_skips_attempts_after_the_first_correct_one(self):
        lazy_row, entry, calls = self.run_analysis(lazy=True)
        full_row, _, full_calls = self.run_analysis(lazy=False)
        self.assertEqual((calls, full_calls), (2, 3))
        self.assertEqual(entry["evaluation_status_list"], ["SUCCESS", "SUCCESS", LAZY_SKIP_STATUS])
        for k in (1, 2, 3):
            self.assertEqual(lazy_row[f"pass@{k}"], full_row[f"pass@{k}"])
        self.assertEqual(list(lazy_row[["pass@1", "pass@2", "pass@3"]]), [0, 1, 1])


class ReplayTests(BankTestCase):
    def test_replay_measures_rules_and_seeds_the_bank(self):
        solution = "a long example solution\nso $\\boxed{\\sqrt{2}}$"

        def run(step, sub_step, text, context=None):
            return {"step_name": step, "sub_step": sub_step, "input_context": context or {},
                    "output_result": {"status": "SUCCESS", "text": text}}

        trace = [run("mirror_phase_0", f"baseline_idx_7_attempt_{i}", text, {"ground_truth": solution})
                 for i, text in enumerate(["Final Answer: \\sqrt{2}", "So Final Answer: \\sqrt{2}", "Final Answer: 2"])]
        # One correct run out of three different answers: ambiguous, so skipped.
        trace += [run("mirror_phase_2", f"cand_5_vs_val_7_run_{i}", f"Final Answer: {answer}")
                  for i, answer in enumerate(["\\sqrt{2}", "1", "2"])]
        state = {
            "candidate_set": {"5": {"candidate_text": "Final Answer: 12"},
                              "zs_0": {"candidate_text": "Final Answer: 13"},
                              "zs_1": {"candidate_text": "So Final Answer: 12"}},
            "ground_truth_labels": {"5": {"is_correct": True, "evaluation_status": "SUCCESS"},
                                    "zs_0": {"is_correct": False, "evaluation_status": "SUCCESS"},
                                    "zs_1": {"is_correct": False, "evaluation_status": "SUCCESS"}},
            "intrinsic_baselines": {"7": 2 / 3},
            "cross_evaluation_matrix": {"5": {"7": 1 / 3}},
            "execution_trace": trace,
        }
        logs = {"exp": [{"target_query_original_hard_list_idx": 0, "layer1_base_execution_state": state}]}
        with open(os.path.join(self.tmp.name, "exp_detailed_eval.json"), "w", encoding="utf-8") as handle:
            json.dump([{"hard_list_idx": 0, "strategy": "primary",
                        "attempts": ["Final Answer: 12", {"status": "FAILURE"}],
                        "is_correct_list": [True, None],
                        "evaluation_status_list": ["SUCCESS", "GENERATION_FAILED"]}], handle)

        summary, disagreements = replay_answer_judging(logs, ["12"], self.config)
        totals = summary.set_index("source").loc["all"]
        self.assertEqual(totals["items"], 7)  # 3 labels + 3 baseline runs + 1 Pass@K attempt
        self.assertEqual(totals["llm_calls"], 1)  # only "2" against \sqrt{2} needs the judge
        self.assertEqual(len(disagreements), 1)  # zs_1: "12" is exact, the LLM said false
        self.assertEqual(disagreements.iloc[0]["llm_verdict"], False)

        # The next run reuses the baseline verdict the replay recovered.
        refuse = FakeJudge(lambda prompt: self.fail("the bank should have answered"))
        result = self.judge("new run\nFinal Answer: 2", solution, refuse)
        self.assertEqual((result["is_correct"], result["judge_method"]), (False, "bank"))


class DistributedConfigTests(unittest.TestCase):
    def test_bank_switches_do_not_change_a_run_manifest(self):
        for key in ("ANSWER_BANK_ENABLED", "ANSWER_BANK_AUDIT_RATE", "ANSWER_JUDGE_PACK_MAX_ITEMS", "EVAL_LAZY_PASS_AT_K"):
            self.assertTrue(_is_runtime_or_secret_key(key), key)


if __name__ == "__main__":
    unittest.main()
