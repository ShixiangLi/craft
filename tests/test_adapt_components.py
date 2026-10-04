"""ADaPT 任务组合的有效性与配置边界。"""
import json
import unittest

from modules.adapt.components import parse_plan, planner_schema, validate_params


def plan(logic, subtasks=None, **extra):
    return json.dumps({"thought": "Decompose only after execution failed.",
                       "subtasks": subtasks or [" A ", "B", "C"],
                       "logic": logic, **extra})


class AdaptComponentsTest(unittest.TestCase):
    def test_nested_plan_and_boolean_precedence(self):
        tasks, expression = parse_plan(plan("1 AND (2 OR 3)"), 5)
        self.assertEqual(tasks, ["A", "B", "C"])
        self.assertEqual(expression, ("AND", [1, ("OR", [2, 3])]))
        self.assertEqual(parse_plan(plan("1 or 2 AnD 3"), 5)[1],
                         ("OR", [1, ("AND", [2, 3])]))
        self.assertEqual(parse_plan(plan("1", ["A"]), 1), (["A"], 1))

    def test_rejects_missing_duplicate_and_out_of_range_references(self):
        for logic in ("1 AND 2", "1 AND 1 AND 3", "0 AND 2 AND 3", "1 OR 2 OR 4"):
            with self.subTest(logic=logic), self.assertRaisesRegex(ValueError, "引用"):
                parse_plan(plan(logic), 5)

    def test_rejects_executable_or_non_boolean_expressions(self):
        for logic in ("__import__('os').system('echo bad')", "1 + 2 + 3", "not 1",
                      "True", "[1, 2, 3]", "1 < 2 < 3", "-1", "1 & 2 & 3", "("):
            with self.subTest(logic=logic), self.assertRaises(ValueError):
                parse_plan(plan(logic), 5)

    def test_rejects_malformed_planner_payloads(self):
        valid = json.loads(plan("1 AND 2 AND 3"))
        invalid = [[], {}, {**valid, "thought": []}, {**valid, "observation": "invented"},
                   {**valid, "subtasks": []}, {**valid, "subtasks": [" "]},
                   {**valid, "subtasks": [True]}, {**valid, "logic": 1},
                   {**valid, "logic": ""}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_plan(json.dumps(payload), 5)
        with self.assertRaises(ValueError):
            parse_plan(json.dumps(valid), 2)

    def test_params_defaults_overrides_and_no_mutation(self):
        params = {"max_depth": 2, "max_history_steps": None}
        self.assertEqual(validate_params(params), {"max_depth": 2, "max_executor_calls": 20,
                                                  "max_subtasks": 5, "max_history_steps": None})
        self.assertEqual(params, {"max_depth": 2, "max_history_steps": None})
        self.assertEqual(validate_params(None)["max_history_steps"], 16)
        for key in validate_params({}):
            for value in (0, -1, True, 1.5, "3"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_params({key: value})
        with self.assertRaises(ValueError):
            validate_params({"max_depht": 3})
        with self.assertRaises(ValueError):
            validate_params([])

    def test_schema_matches_planner_contract(self):
        schema = planner_schema(3)
        self.assertEqual(schema["required"], ["thought", "subtasks", "logic"])
        self.assertEqual(schema["properties"]["subtasks"]["maxItems"], 3)
        self.assertFalse(schema["additionalProperties"])
        with self.assertRaises(ValueError):
            planner_schema(True)


if __name__ == "__main__":
    unittest.main()
