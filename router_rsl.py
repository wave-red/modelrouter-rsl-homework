"""
作业二原型：ModelRouter + RSL（递归自学习）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3:4b"
MAX_ATTEMPTS = 3

VALID_ROUTES = ("local", "cloud")

# ---------------------------------------------------------------------------
# 第一层：规则目录
# 模型不能自由发明经验，只能从这份审核过的目录里挑规则 ID。
# 每条规则 = 一类题型 → 一个路由决定。
# ---------------------------------------------------------------------------
ROUTE_RULES = {
    "code_to_cloud": "题目要求写代码、调试程序、实现算法或脚本时，派给云端大模型 cloud。",
    "math_to_cloud": "题目要求数学计算、推导、证明或解方程时，派给云端大模型 cloud。",
    "knowledge_to_local": "题目是概念解释、名词问答、原理介绍时，派给本地小模型 local。",
    "translation_to_local": "题目是语言翻译时，派给本地小模型 local。",
    "daily_to_local": "题目是日常寒暄、闲聊或简单常识时，派给本地小模型 local。",
}

# 题型关键词：程序用它们从问题文本里提取特征（不经过模型，保证确定性）
FEATURE_KEYWORDS = {
    "code": ("代码", "编程", "python", "c语言", "c 语言", "脚本", "函数", "算法",
             "程序", "bug", "调试", "单片机", "pid"),
    "math": ("计算", "证明", "推导", "方程", "求导", "积分", "矩阵", "黎曼",
             "伯努利", "勾股", "秩"),
    "knowledge": ("解释", "什么是", "介绍", "原理", "为什么", "区别"),
    "translation": ("翻译", "译成", "translate"),
    "daily": ("你好", "闲聊", "聊聊", "吃了吗", "笑话", "早上好"),
}

# 特征 → 该特征唯一对应的规则 ID。检索经验和提炼规则都走这张表。
FEATURE_TO_RULE = {
    "code": "code_to_cloud",
    "math": "math_to_cloud",
    "knowledge": "knowledge_to_local",
    "translation": "translation_to_local",
    "daily": "daily_to_local",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


# ---------------------------------------------------------------------------
# 第二层：任务与裁判
# Task = 一道路由题。特征由程序提取，gold_route 是人工标注的标准答案。
# Judge 只比对标准答案，绝不把标准答案泄露给模型。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    question: str
    gold_route: str
    task_id: str
    features: tuple[str, ...]


def parse_task(question: str, gold_route: str) -> Task:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空。")
    if gold_route not in VALID_ROUTES:
        raise ValueError(f"标准路由必须是 {VALID_ROUTES} 之一。")
    q = question.strip()
    low = normalize(q)
    features = {name for name, kws in FEATURE_KEYWORDS.items()
                if any(k in low for k in kws)}
    if not features:
        features = {"daily"}  # 没有任何特征时按日常题处理，默认本地
    task_id = hashlib.sha256(low.encode("utf-8")).hexdigest()
    return Task(q, gold_route, task_id, tuple(sorted(features)))


def judge(route: str, gold_route: str) -> tuple[bool, str]:
    """判分。反馈只说'没过'，不说正确答案——防止污染，模型必须真的想。"""
    if route == gold_route:
        return True, ""
    return False, ("路由验证未通过：该问题不适合你选的路由。"
                   "请重新判断题型的复杂度和质量要求，再选一次。")


# ---------------------------------------------------------------------------
# 第三层：经验库（sqlite）
# 只存：题目的指纹、题型特征、规则 ID、来源。不存题目文本和答案。
# holdout 表保存留出题指纹，训练经验永远不许和留出题重叠。
# ---------------------------------------------------------------------------
class ExperienceBank:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._validate()

    def _init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiences(
            task_id    TEXT PRIMARY KEY,
            features   TEXT NOT NULL,
            rule_ids   TEXT NOT NULL,
            source     TEXT NOT NULL,
            attempts   INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS holdout(
            task_id      TEXT PRIMARY KEY,
            reserved_at  TEXT NOT NULL
        );
        """)
        self.conn.commit()

    def _validate(self) -> None:
        for row in self.conn.execute("SELECT * FROM experiences"):
            rules = json.loads(row["rule_ids"])
            feats = json.loads(row["features"])
            if any(r not in ROUTE_RULES for r in rules):
                raise ValueError(f"经验库含未审核规则：{rules}")
            if any(f not in FEATURE_KEYWORDS for f in feats):
                raise ValueError(f"经验库含未知特征：{feats}")
            if row["source"] not in {"solve", "train"}:
                raise ValueError(f"经验库含非法来源：{row['source']}")
        protected = {r["task_id"] for r in self.conn.execute("SELECT task_id FROM holdout")}
        clash = protected & {r["task_id"] for r in
                             self.conn.execute("SELECT task_id FROM experiences")}
        if clash:
            raise ValueError(f"经验库中训练题与受保护留出题重叠：{len(clash)} 条")

    @property
    def experiences(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM experiences ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    @property
    def protected(self) -> set[str]:
        return {r["task_id"] for r in self.conn.execute("SELECT task_id FROM holdout")}

    def check_can_learn(self, task: Task) -> None:
        if task.task_id in self.protected:
            raise ValueError("该题是受保护留出题，禁止写入经验库。")

    def reserve_holdout(self, tasks: list[Task]) -> None:
        for t in tasks:
            self.conn.execute(
                "INSERT OR IGNORE INTO holdout(task_id, reserved_at) VALUES(?,?)",
                (t.task_id, now_iso()))
        self.conn.commit()
        self._validate()

    def retrieve(self, task: Task) -> list[str]:
        """按题型特征找相似经验，取出规则文本，注入给模型。"""
        texts: list[str] = []
        for record in self.experiences:
            feats = set(json.loads(record["features"]))
            overlap = feats & set(task.features)
            if not overlap:
                continue
            for rid in json.loads(record["rule_ids"]):
                if rid in FEATURE_TO_RULE.values() and ROUTE_RULES[rid] not in texts:
                    texts.append(ROUTE_RULES[rid])
        return texts

    def add_verified(self, task: Task, rule_ids: list[str], *, source: str,
                     attempts: int) -> bool:
        self.check_can_learn(task)
        if source not in {"solve", "train"}:
            raise ValueError("只允许 solve 或 train 来源写库。")
        for rid in rule_ids:
            if rid not in ROUTE_RULES:
                raise ValueError(f"规则 {rid} 不在审核目录中。")
        if self.conn.execute("SELECT 1 FROM experiences WHERE task_id=?",
                             (task.task_id,)).fetchone():
            return False
        self.conn.execute(
            "INSERT INTO experiences(task_id, features, rule_ids, source, attempts, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (task.task_id, json.dumps(list(task.features), ensure_ascii=False),
             json.dumps(rule_ids, ensure_ascii=False), source, attempts, now_iso()))
        self.conn.commit()
        return True


# ---------------------------------------------------------------------------
# 第四层：模型客户端
# 只做一件事：把消息发给本地 Ollama，拿回 JSON 文本。
# 配置固定：think=false + temperature=0 + seed=42，保证可复现。
# ---------------------------------------------------------------------------
class RouterAgent:
    def __init__(self, host: str, model: str, timeout: float = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]],
             schema: dict[str, Any] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema or {
                "type": "object",
                "properties": {
                    "reflection": {"type": "string"},
                    "route": {"type": "string", "enum": list(VALID_ROUTES)},
                },
                "required": ["reflection", "route"],
                "additionalProperties": False,
            },
            "options": {"num_ctx": 4096, "temperature": 0, "seed": 42,
                        "num_predict": 256},
        }
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = json.load(resp)["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("模型返回的 content 不是字符串。")
            return content
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read(2048).decode('utf-8', 'replace')}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Ollama 请求失败: {exc}") from exc


# ---------------------------------------------------------------------------
# 第五层：经验提炼器
# 答对之后才调用：让模型从"本题特征对应的规则子集"里挑规则 ID。
# 挑出的规则要过程校验：必须在目录里、必须匹配本题特征。
# ---------------------------------------------------------------------------
class Refinery:
    def __init__(self, client: RouterAgent):
        self.client = client

    def refine(self, task: Task, verified_attempts: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "rejected", "rule_ids": []}
        candidates = {FEATURE_TO_RULE[f]: ROUTE_RULES[FEATURE_TO_RULE[f]]
                      for f in task.features}
        schema = {
            "type": "object",
            "properties": {"lesson_ids": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(candidates)},
                "minItems": 1, "maxItems": len(candidates),
            }},
            "required": ["lesson_ids"],
            "additionalProperties": False,
        }
        messages = [
            {"role": "system", "content": (
                "你是经验提炼器。下面给你一道已验证正确的路由任务，"
                "你只能从提供的规则目录中挑出最该记住的规则ID。"
                "只输出 JSON，如 {\"lesson_ids\":[\"math_to_cloud\"]}；"
                "不许发明目录外规则，不许保存题目文本或答案。")},
            {"role": "user", "content": json.dumps({
                "question_features": list(task.features),
                "verified_attempts": [
                    {k: a.get(k) for k in ("number", "route", "reflection") if k in a}
                    for a in verified_attempts],
                "allowed_rules": candidates,
            }, ensure_ascii=False)},
        ]
        try:
            payload = json.loads(self.client.chat(messages, schema=schema))
            picked = payload["lesson_ids"]
            if not isinstance(picked, list) or not picked:
                raise ValueError("提炼结果必须是非空列表。")
            if any(r not in candidates for r in picked):
                raise ValueError("提炼出目录外或不匹配特征的规则，拒绝。")
            result.update(status="accepted",
                          rule_ids=list(dict.fromkeys(picked)))
        except (json.JSONDecodeError, ValueError, RuntimeError, KeyError) as exc:
            result["feedback"] = str(exc)
        return result


# ---------------------------------------------------------------------------
# 第六层：入库门控（借自 ollama-rsl-homework 的思路）
# 候选经验不是答对就入库：先在验证集上对比"无经验 vs 有候选经验"，
# 完全正确数严格提高、且原来答对的题不许退步，两条都满足才写库。
# ---------------------------------------------------------------------------
class GateKeeper:
    def __init__(self, system: "RSLSystem"):
        self.system = system

    def review(self, candidates: list[str], val_tasks: list[Task]) -> dict[str, Any]:
        report: dict[str, Any] = {"candidates": candidates, "val_tasks": len(val_tasks)}
        if not candidates:
            report.update(accepted=False, reason="训练阶段没有产出候选经验")
            return report
        before = [self.system.solve_once(t, extra_rules=[]) for t in val_tasks]
        after = [self.system.solve_once(t, extra_rules=[ROUTE_RULES[c] for c in candidates])
                 for t in val_tasks]
        b_ok = {t.task_id for t, r in zip(val_tasks, before) if r["success"]}
        a_ok = {t.task_id for t, r in zip(val_tasks, after) if r["success"]}
        regress = b_ok - a_ok
        strictly_better = len(a_ok) > len(b_ok)
        report.update(
            baseline_correct=len(b_ok), with_candidates_correct=len(a_ok),
            regressed=sorted(regress),
            accepted=bool(strictly_better and not regress),
        )
        if not strictly_better:
            report["reason"] = "候选经验没有让验证集严格变好"
        elif regress:
            report["reason"] = "候选经验让原本答对的题退步了"
        else:
            report["reason"] = "通过：严格提高且无退步"
        return report


# ---------------------------------------------------------------------------
# 第七层：RSL 编排 + 评测
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    task_id: str
    question: str
    gold_route: str
    success: bool = False
    route: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    rules_used: list[str] = field(default_factory=list)
    lesson: dict[str, Any] | None = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "solve"
    started_at: str = field(default_factory=now_iso)
    wall_seconds: float = 0.0


class RSLSystem:
    def __init__(self, client: RouterAgent, bank: ExperienceBank,
                 traces: Path | str):
        self.client = client
        self.bank = bank
        self.traces = Path(traces)
        self.refinery = Refinery(client)
        self.gate = GateKeeper(self)

    def _trace(self, result: SolveResult) -> None:
        self.traces.parent.mkdir(parents=True, exist_ok=True)
        with self.traces.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trace_id": result.trace_id, "task_id": result.task_id,
                "question": result.question, "gold_route": result.gold_route,
                "success": result.success, "route": result.route,
                "attempts": result.attempts, "rules_used": result.rules_used,
                "lesson": result.lesson, "source": result.source,
                "wall_seconds": result.wall_seconds,
            }, ensure_ascii=False) + "\n")

    def solve(self, question: str, gold_route: str, *, max_attempts: int = MAX_ATTEMPTS,
              learn: bool = True, use_bank: bool = True,
              source: str = "solve") -> SolveResult:
        """单题闭环：检索经验 → 首答 → 错了就反馈重试 → 对了才提炼经验。"""
        started = time.perf_counter()
        task = parse_task(question, gold_route)
        if learn:
            self.bank.check_can_learn(task)
        rules = self.bank.retrieve(task) if use_bank else []
        result = SolveResult(task_id=task.task_id, question=task.question,
                             gold_route=task.gold_route, rules_used=rules,
                             source=source)
        system_prompt = (
            "你是模型路由器。用户会发来一个问题，你要决定把它派给哪个模型：\n"
            "  local = 本地小模型，快、免费、能力一般\n"
            "  cloud = 云端大模型，慢、按量收费、能力强\n"
            "先判断题型和难度，再选路由。只输出 JSON："
            "{\"reflection\":\"一句话说明你为什么这么派\",\"route\":\"local 或 cloud\"}。"
        )
        if rules:
            system_prompt += "\n历史经验（来自已验证的路由任务）：\n" + "\n".join(
                "- " + r for r in rules)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请路由这个问题：" + task.question},
        ]
        for number in range(1, max_attempts + 1):
            attempt: dict[str, Any] = {"number": number, "verified": False}
            try:
                raw = self.client.chat(messages)
                payload = json.loads(raw)
                route = payload.get("route")
                attempt.update(raw_response=raw, route=route,
                               reflection=str(payload.get("reflection", "")))
                ok, feedback = judge(route, task.gold_route)
                if ok:
                    result.success = True
                    result.route = route
                    attempt["verified"] = True
                else:
                    attempt.update(error_type="wrong_route", feedback=feedback)
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                attempt.update(error_type="bad_output", feedback=str(exc))
            result.attempts.append(attempt)
            if result.success:
                if learn:
                    lesson = self.refinery.refine(task, result.attempts)
                    result.lesson = lesson
                break
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "上一轮结果：" + attempt["feedback"] +
                    "请重新评估题型，只输出 JSON，字段为 reflection 和 route。")},
            ])
        result.wall_seconds = round(time.perf_counter() - started, 6)
        self._trace(result)
        return result

    def solve_once(self, task: Task, *, extra_rules: list[str] | None = None) -> dict[str, Any]:
        """门控专用：单次作答、不写库、可临时注入指定经验文本。"""
        system_prompt = (
            "你是模型路由器。判断问题该派给 local（本地小模型，快免费）"
            "还是 cloud（云端大模型，慢但强）。只输出 JSON："
            "{\"reflection\":\"理由\",\"route\":\"local 或 cloud\"}。"
        )
        if extra_rules:
            system_prompt += "\n历史经验：\n" + "\n".join("- " + r for r in extra_rules)
        try:
            payload = json.loads(self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请路由这个问题：" + task.question},
            ]))
            route = payload.get("route")
            ok, _ = judge(route, task.gold_route)
            return {"success": ok, "route": route}
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            return {"success": False, "route": None, "error": str(exc)}


def check_split(train: list[tuple[str, str]], val: list[tuple[str, str]],
                holdout: list[tuple[str, str]], bank: ExperienceBank):
    """评估隔离：训练/验证/留出三者不许重叠，也不许撞上库里已有的留出指纹。"""
    def to_tasks(pairs):
        return [parse_task(q, g) for q, g in pairs]
    tr, va, ho = to_tasks(train), to_tasks(val), to_tasks(holdout)
    tr_ids = {t.task_id for t in tr}
    va_ids = {t.task_id for t in va}
    ho_ids = {t.task_id for t in ho}
    if not (tr and va and ho):
        raise ValueError("训练、验证、留出三个集合都不能为空。")
    for name, ids, tasks in (("train", tr_ids, tr), ("val", va_ids, va),
                             ("holdout", ho_ids, ho)):
        if len(ids) != len(tasks):
            raise ValueError(f"{name} 集合内有重复题目。")
    if tr_ids & va_ids or tr_ids & ho_ids or va_ids & ho_ids:
        raise ValueError("训练/验证/留出集合之间存在重叠，拒绝评测。")
    if tr_ids & bank.protected or va_ids & bank.protected:
        raise ValueError("训练或验证集包含此前保留的留出题，拒绝评测。")
    if ho_ids & {r["task_id"] for r in bank.experiences}:
        raise ValueError("留出集与已有训练经验重叠，拒绝评测。")
    return tr, va, ho


def summarize(results: list[SolveResult]) -> dict[str, Any]:
    ok = sum(r.success for r in results)
    return {
        "tasks": len(results),
        "correct": ok,
        "accuracy": round(ok / len(results), 4) if results else 0,
        "model_calls": sum(len(r.attempts) + (1 if r.lesson else 0) for r in results),
        "wall_seconds": round(sum(r.wall_seconds for r in results), 3),
    }


def benchmark(system: RSLSystem, train, val, holdout, *, max_attempts: int = MAX_ATTEMPTS):
    """四阶段评测：baseline → 训练+提炼 → 门控 → memory → recursive。"""
    tr, va, ho = check_split(train, val, holdout, system.bank)
    system.bank.reserve_holdout(ho)
    before = len(system.bank.experiences)

    baseline = summarize([system.solve(t.question, t.gold_route, max_attempts=1,
                                       learn=False, use_bank=False,
                                       source="benchmark-baseline") for t in ho])

    training = [system.solve(t.question, t.gold_route, max_attempts=max_attempts,
                             use_bank=True, source="train") for t in tr]
    candidates: list[str] = []
    for r in training:
        if r.lesson and r.lesson.get("status") == "accepted":
            candidates.extend(r.lesson["rule_ids"])
    candidates = sorted(set(candidates))

    gate_report = system.gate.review(candidates, va)
    accepted_rules: list[str] = []
    if gate_report["accepted"]:
        for r in training:  # 门控通过才把训练阶段答对的题写入经验库
            if r.success and r.lesson and r.lesson.get("status") == "accepted":
                task = parse_task(r.question, r.gold_route)
                if system.bank.add_verified(task, r.lesson["rule_ids"],
                                            source="train",
                                            attempts=len(r.attempts)):
                    accepted_rules.extend(r.lesson["rule_ids"])
        accepted_rules = sorted(set(accepted_rules))

    memory = summarize([system.solve(t.question, t.gold_route, max_attempts=1,
                                     learn=False, use_bank=True,
                                     source="benchmark-memory") for t in ho])
    recursive = summarize([system.solve(t.question, t.gold_route,
                                        max_attempts=max_attempts, learn=False,
                                        use_bank=True,
                                        source="benchmark-recursive") for t in ho])
    if len(system.bank.experiences) != before + len(accepted_rules):
        raise RuntimeError("评测阶段意外改写了经验库。")
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "model": system.client.model,
        "learning": "verified_rule_memory_only_no_weight_updates",
        "training": summarize(training),
        "baseline": baseline,
        "memory": memory,
        "recursive": recursive,
        "gate": gate_report,
        "rules_in_bank": accepted_rules,
        "interpretation": ("小样本结果仅描述本次运行；经验记忆和递归修正"
                           "不保证提高准确率。"),
    }


def load_dataset(path: Path) -> dict[str, list[tuple[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) != {"train", "val", "holdout"}:
        raise ValueError("dataset 必须含 train / val / holdout 三个列表。")
    out = {}
    for key in ("train", "val", "holdout"):
        items = data[key]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{key} 必须是非空列表。")
        pairs = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {"question", "gold"}:
                raise ValueError(f"{key} 里每项必须含 question 和 gold 两个字段。")
            pairs.append((item["question"], item["gold"]))
        out[key] = pairs
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ModelRouter + RSL 最小原型")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--bank", type=Path, default=BASE_DIR / "bank.sqlite")
    p.add_argument("--traces", type=Path, default=BASE_DIR / "traces.jsonl")
    p.add_argument("--timeout", type=float, default=120)
    sub = p.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser, suppress: bool = False) -> None:
        """把全局选项也挂到子命令上，两种位置都能写。"""

        def default(value):
            return argparse.SUPPRESS if suppress else value
        target.add_argument("--host", default=default(p.get_default("host")))
        target.add_argument("--model", default=default(p.get_default("model")))
        target.add_argument("--bank", type=Path,
                            default=default(p.get_default("bank")))
        target.add_argument("--traces", type=Path,
                            default=default(p.get_default("traces")))
        target.add_argument("--timeout", type=float,
                            default=default(p.get_default("timeout")))

    s = sub.add_parser("solve", help="路由一道题")
    common(s, suppress=True)
    s.add_argument("--question", required=True)
    s.add_argument("--gold", required=True, choices=VALID_ROUTES,
                   help="人工标准答案，仅用于验证流程")
    s.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                   choices=range(1, MAX_ATTEMPTS + 1))
    s.add_argument("--no-learn", action="store_true")
    s.add_argument("--no-bank", action="store_true")

    b = sub.add_parser("benchmark", help="四阶段评测")
    common(b, suppress=True)
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--metrics", type=Path, default=BASE_DIR / "metrics.json")
    b.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS,
                   choices=range(1, MAX_ATTEMPTS + 1))

    m = sub.add_parser("bank", help="查看经验库（不调模型）")
    common(m, suppress=True)
    r = sub.add_parser("rules", help="查看审核过的规则目录（不调模型）")
    common(r, suppress=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout 必须大于 0。")
        if args.command == "rules":
            print(json.dumps(ROUTE_RULES, ensure_ascii=False, indent=2))
            return 0
        bank = ExperienceBank(args.bank)
        if args.command == "bank":
            print(json.dumps({
                "experiences": bank.experiences,
                "protected_holdout": len(bank.protected),
            }, ensure_ascii=False, indent=2, default=str))
            return 0
        if args.bank.resolve() == args.traces.resolve():
            raise ValueError("bank 和 traces 必须是不同文件。")
        system = RSLSystem(RouterAgent(args.host, args.model, args.timeout),
                           bank, args.traces)
        if args.command == "solve":
            r = system.solve(args.question, args.gold,
                             max_attempts=args.max_attempts,
                             learn=not args.no_learn,
                             use_bank=not args.no_bank)
            print(json.dumps({
                "success": r.success, "route": r.route, "gold": r.gold_route,
                "attempts": len(r.attempts), "rules_used": r.rules_used,
                "lesson": r.lesson,
            }, ensure_ascii=False, indent=2))
            return 0 if r.success else 1
        metrics = benchmark(system, *[load_dataset(args.dataset)[k]
                                      for k in ("train", "val", "holdout")],
                            max_attempts=args.max_attempts)
        args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        for name in ("baseline", "memory", "recursive"):
            g = metrics[name]
            print(f"{name:10s} {g['correct']}/{g['tasks']}  "
                  f"({g['accuracy']:.0%})  调用 {g['model_calls']} 次  "
                  f"{g['wall_seconds']}s")
        g = metrics["gate"]
        print(f"门控：候选 {len(g['candidates'])} 条，验证集 "
              f"{g.get('baseline_correct')} → {g.get('with_candidates_correct')}，"
              f"{'入库' if g['accepted'] else '拒绝'}（{g['reason']}）")
        print(f"指标：{args.metrics.resolve()}")
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
