# ModelRouter + RSL：会学习的模型调度员

课程作业二：在 ModelRouter 基础上添加 RSL（Recursive Self-Learning，递归自学习）。

**方法论**：用户问题来了，调度员决定派给本地小模型（local）还是云端大模型（cloud）；派错了它会反思、把教训提炼成规则存进经验库，下次同类问题派得更准。模型权重全程不更新，学的全是"派活策略"。

## 运行环境

- Python 3.10+
- Ollama + 任一 qwen3 系 4B 模型（`ollama pull qwen3:4b` 或 `qwen3.5:4b`）

## 怎么跑

```bash
# 四阶段评测：baseline → 训练提炼 → 门控 → memory → recursive
python router_rsl.py benchmark --dataset dataset.json --metrics metrics.json

# 换模型对照实验（用全新经验库，避免旧经验污染）
python router_rsl.py benchmark --dataset dataset.json --model qwen3.5:4b \
    --bank bank_qwen35.sqlite --traces traces_qwen35.jsonl --metrics metrics_qwen35.json

# 单题演示：路由一道题
python router_rsl.py solve --question "写一个快速排序函数" --gold cloud

# 查看经验库 / 规则目录（不调模型）
python router_rsl.py bank
python router_rsl.py rules

# 单元测试（16 项，假模型客户端，不依赖 Ollama）
python -m unittest test_router_rsl -v
```

## 实验结果（2026-10-22，本机实测）

同一题库、各自全新经验库，只换模型：

| 组别 | qwen3:4b | qwen3.5:4b |
|---|---|---|
| baseline（无经验，单答） | 3/4 | 3/4 |
| memory（带经验，单答） | 4/4 | 4/4 |
| recursive（带经验，最多 3 答） | 4/4 | 4/4 |
| 训练阶段 | 3/4 | 4/4 |
| 门控验证集 | 0/3 → 2/3，入库 3 条规则 | 1/3 → 3/3，入库 4 条规则 |

经验起作用的直接证据：baseline 组错把"求矩阵的秩"派给了本地，memory 组凭经验规则"数学题派云端"派对。

## 设计要点

1. **精确判分**：路由题有人工标准答案，程序直接比对，不用模型当裁判。
2. **反泄露反馈**：判错只告知"未通过"，绝不透露正确答案，否则递归提升是背答案背出来的。
3. **规则目录**：模型只能从 5 条审核过的路由规则中选 ID 入库，不许自由发明规则。
4. **入库门控**：候选规则必须让验证集完全正确数严格提高、且原正确样本不退步，两条都满足才写库。
5. **留出题指纹保护**：评测题指纹登记进 holdout 表，任何学习路径都无法把评测题写进经验库。

## 文件清单

| 文件 | 用途 |
|---|---|
| `router_rsl.py` | 主程序：Task / Judge / ExperienceBank / Refinery / GateKeeper / RSLSystem 六个类 |
| `test_router_rsl.py` | 16 项单元测试 |
| `dataset.json` | 训练 4 题 / 验证 3 题 / 留出 4 题 |
| `metrics.json` / `metrics_qwen35.json` | 
| `traces.jsonl` / `traces_qwen35.jsonl` | 
| `bank.sqlite` / `bank_qwen35.sqlite` | 经验库（sqlite） |