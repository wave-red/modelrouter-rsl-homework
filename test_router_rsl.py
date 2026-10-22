"""router_rsl 的单元测试：用 FakeClient 冒充模型，不调用 Ollama。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_rsl import (ExperienceBank, GateKeeper, Refinery, RSLSystem, Task,
                        VALID_ROUTES, judge, parse_task)


class FakeClient:
    """按脚本返回响应的假模型。"""

    def __init__(self, script):
        self.script = list(script)
        self.model = "fake-model"
        self.calls = 0

    def chat(self, messages, schema=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item, ensure_ascii=False)


def make_system(tmp, script):
    client = FakeClient(script)
    bank = ExperienceBank(Path(tmp) / "bank.sqlite")
    system = RSLSystem(client, bank, Path(tmp) / "traces.jsonl")
    return system, client, bank


def route(route_value):
    return {"reflection": "因为题型合适", "route": route_value}


class TestJudge(unittest.TestCase):
    def test_correct_route_passes(self):
        ok, _ = judge("cloud", "cloud")
        self.assertTrue(ok)

    def test_wrong_route_feedback_hides_gold(self):
        ok, feedback = judge("local", "cloud")
        self.assertFalse(ok)
        self.assertNotIn("cloud", feedback)  # 反馈不许泄露标准答案


class TestTask(unittest.TestCase):
    def test_feature_extraction(self):
        t = parse_task("用 Python 写一个快排函数", "cloud")
        self.assertIn("code", t.features)

    def test_unknown_question_defaults_to_daily(self):
        t = parse_task("嗯嗯啊啊哦哦", "local")
        self.assertEqual(t.features, ("daily",))

    def test_bad_gold_rejected(self):
        with self.assertRaises(ValueError):
            parse_task("你好", "middle")


class TestBank(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_add_and_retrieve(self):
        _, _, bank = make_system(self.tmp, [])
        t = parse_task("证明勾股定理", "cloud")
        bank.add_verified(t, ["math_to_cloud"], source="train", attempts=1)
        self.assertEqual(len(bank.experiences), 1)
        t2 = parse_task("计算 1 加到 100", "cloud")
        self.assertTrue(bank.retrieve(t2))  # 同为 math 题型，经验可检索

    def test_rule_outside_catalog_rejected(self):
        _, _, bank = make_system(self.tmp, [])
        t = parse_task("证明勾股定理", "cloud")
        with self.assertRaises(ValueError):
            bank.add_verified(t, ["invented_rule"], source="train", attempts=1)

    def test_holdout_protection(self):
        system, _, bank = make_system(self.tmp, [])
        ho = parse_task("你好，随便陪我聊聊吧", "local")
        bank.reserve_holdout([ho])
        with self.assertRaises(ValueError):
            bank.check_can_learn(ho)  # 留出题禁止写入经验

    def test_overlap_rejected_on_load(self):
        _, _, bank = make_system(self.tmp, [])
        t = parse_task("证明勾股定理", "cloud")
        bank.add_verified(t, ["math_to_cloud"], source="train", attempts=1)
        with self.assertRaises(ValueError):
            bank.reserve_holdout([t])  # 同一题既是训练经验又是留出题 → 拒绝


class TestRefinery(unittest.TestCase):
    def test_only_catalog_rules_accepted(self):
        tmp = tempfile.mkdtemp()
        system, client, _ = make_system(tmp, [])
        refinery = Refinery(client)
        t = parse_task("证明勾股定理", "cloud")
        client.script.append({"lesson_ids": ["math_to_cloud"]})
        r = refinery.refine(t, [{"number": 1, "route": "cloud"}])
        self.assertEqual(r["status"], "accepted")
        self.assertEqual(r["rule_ids"], ["math_to_cloud"])

    def test_invented_rule_rejected(self):
        tmp = tempfile.mkdtemp()
        system, client, _ = make_system(tmp, [])
        refinery = Refinery(client)
        t = parse_task("证明勾股定理", "cloud")
        client.script.append({"lesson_ids": ["made_up_rule"]})
        r = refinery.refine(t, [{"number": 1, "route": "cloud"}])
        self.assertEqual(r["status"], "rejected")


class TestGate(unittest.TestCase):
    def test_gate_requires_strict_improvement(self):
        tmp = tempfile.mkdtemp()
        # 验证集 2 题：无经验时对 1 题；注入经验后 2 题都对 → 通过
        system, client, _ = make_system(tmp, [
            route("local"),    # val1 无经验：对（gold local）
            route("local"),    # val2 无经验：错（gold cloud，误派本地）
            route("local"),    # val1 有经验：仍对
            route("cloud"),    # val2 有经验：改对
        ])
        val = [parse_task("什么是时钟树综合", "local"),
               parse_task("计算 12345 乘以 6789", "cloud")]
        report = GateKeeper(system).review(["math_to_cloud"], val)
        self.assertTrue(report["accepted"])

    def test_gate_rejects_when_no_gain(self):
        tmp = tempfile.mkdtemp()
        # 有无经验结果相同 → 拒绝
        system, client, _ = make_system(tmp, [
            route("local"), route("cloud"),
            route("local"), route("cloud"),
        ])
        val = [parse_task("什么是时钟树综合", "local"),
               parse_task("计算 12345 乘以 6789", "cloud")]
        report = GateKeeper(system).review(["math_to_cloud"], val)
        self.assertFalse(report["accepted"])

    def test_gate_rejects_on_regression(self):
        tmp = tempfile.mkdtemp()
        # 无经验 2 题都对；有经验后 val1 退步 → 拒绝
        system, client, _ = make_system(tmp, [
            route("local"), route("cloud"),
            route("cloud"), route("cloud"),
        ])
        val = [parse_task("什么是时钟树综合", "local"),
               parse_task("计算 12345 乘以 6789", "cloud")]
        report = GateKeeper(system).review(["math_to_cloud"], val)
        self.assertFalse(report["accepted"])
        self.assertTrue(report["regressed"])


class TestSolveLoop(unittest.TestCase):
    def test_recursive_retry_after_wrong_route(self):
        tmp = tempfile.mkdtemp()
        system, client, _ = make_system(tmp, [
            route("local"),   # 首答：错
            route("cloud"),   # 重试：对
            {"lesson_ids": ["code_to_cloud"]},  # 提炼
        ])
        r = system.solve("用 Python 写一个快排函数", "cloud")
        self.assertTrue(r.success)
        self.assertEqual(len(r.attempts), 2)

    def test_holdout_cannot_learn(self):
        tmp = tempfile.mkdtemp()
        system, client, bank = make_system(tmp, [])
        ho = parse_task("你好，随便陪我聊聊吧", "local")
        bank.reserve_holdout([ho])
        with self.assertRaises(ValueError):
            system.solve("你好，随便陪我聊聊吧", "local")


if __name__ == "__main__":
    unittest.main()
