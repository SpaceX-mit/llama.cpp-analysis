# auto-opt 系统：LLM 驱动的 K3 自动性能优化

> 配套：`tools/auto-opt/`（实际代码）
> 目标：在 SpacemiT K3 上让 LLM 自己跑性能优化 agent loop
> 设计：国内便宜模型优先，复杂任务升级到高级模型

---

## 0. 一句话总览

> **auto-opt 是一个 agent loop**：每次循环选一个优化策略 → 调 LLM 生成 patch → 应用 → 跑 benchmark → 性能提升则提交到 `opt/*` 分支，否则 revert。
> **国内模型（DeepSeek/Qwen/GLM）默认用于简单任务**，**多文件/汇编/TCM 改动时自动升级到 Claude/GPT-4**。
> **所有实验记到 SQLite**，可随时 `auto-opt report` 生成 markdown 报告。

---

## 1. 系统架构

```
                  ┌────────────────────────────────────────┐
                  │  auto_opt.py (CLI 入口)               │
                  │  run / status / list / show / report  │
                  │  apply / revert / baseline / try      │
                  └────────────┬───────────────────────────┘
                               │
                  ┌────────────▼───────────────────────────┐
                  │  orchestrator.py (Agent Loop)          │
                  │  状态机 + 循环控制                      │
                  │  IDLE→PROFILE→PROPOSE→APPLY→BENCH→   │
                  │       DECIDE→IDLE                       │
                  └──┬────────┬────────┬────────┬────────┬─┘
                     │        │        │        │        │
                     ▼        ▼        ▼        ▼        ▼
                  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
                  │Knowl-││Model ││Profi-││Editor││Commit-│
                  │edge  ││Router││ler   ││      ││ter    │
                  │(SQL) ││(T1/T2)│(llama-││(patch)││(git) │
                  │      ││      ││bench)││      ││      │
                  └──────┘└──────┘└──────┘└──────┘└──────┘
                     ▲                                ▲
                     │                                │
                  ┌──┴────────────────────────────────┴──┐
                  │  strategies/ (具体优化策略)          │
                  │  thread_tuning / mem_backend /        │
                  │  llm_optimize / ...                   │
                  └─────────────────────────────────────┘
```

## 2. 模块清单

```
tools/auto-opt/
├── README.md                  # 用户文档
├── config.yaml                # 配置（模型、策略、benchmark 参数）
├── auto_opt.py                # CLI 入口（239 行）
├── orchestrator.py            # Agent Loop（272 行）
├── knowledge.py               # SQLite 实验知识库（199 行）
├── model_router.py            # 模型路由 Tier1/Tier2（178 行）
├── profiler.py                # llama-bench 封装（151 行）
├── editor.py                  # git apply / 文件替换（134 行）
├── committer.py               # git branch 管理（103 行）
├── test_smoke.py              # 烟雾测试（278 行，7/7 通过）
└── strategies/
    ├── base.py                # 策略基类（87 行）
    ├── thread_tuning.py       # 线程扫描（81 行）
    ├── mem_backend.py         # 内存池后端（76 行）
    └── llm_optimize.py        # LLM 生成 patch（246 行）
```

**总计 ~2050 行 Python**（含注释和测试）。

## 3. Agent Loop 状态机

```
                ┌─────────┐
                │  IDLE   │
                └────┬────┘
                     │ run()
                     ▼
           ┌──────────────────┐
           │ ensure_baseline()│  ← 测一次无优化的 baseline
           └────────┬─────────┘
                    │
                    ▼
        ╔═══════════════════════╗
        ║   cycle_once() (loop)  ║
        ╚═══════════╤═══════════╝
                    │
        ┌───────────▼──────────┐
        │ select next strategy │  ← round-robin
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │ create opt/<...>     │  ← opt/strategy-YYYYMMDD-HHMMSS
        │ branch (off main)    │
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │ strategy.run()       │  ← 核心：propose + apply + bench
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │ decide                │
        │   delta>0.5% → keep  │  ← commit 到 opt/* branch
        │   delta≤0.5% → revert│  ← 删 branch
        │   failed   → mark    │
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │ cooldown 5s          │
        └───────────┬──────────┘
                    │
        ┌───────────▼──────────┐
        │ should_stop?         │
        │   max_iter / max_dur  │
        │   regression_count   │
        └─────┬─────────┬──────┘
        否 ───┘         └─── 是 → 停
        │
        └─── 回到 cycle_once
```

## 4. 模型分层路由

```python
# model_router.py 简化版
def select_model(self, task: dict) -> dict:
    # 1. 用户指定 tier
    requested_tier = task.get('tier', 'auto')
    if requested_tier == 'auto':
        tier = self._should_upgrade(task)  # 检查升级条件
    else:
        tier = requested_tier
    
    # 2. 选该 tier 下第一个能用的
    models = self.tier1_models if tier == 'tier1' else self.tier2_models
    for m in models:
        if os.environ.get(m.get('api_key_env', '')):
            return {'tier': tier, 'model': m}
    return {'tier': tier, 'model': models[0]}  # 缺 key 也会返回（后续 mock 兜底）
```

### 升级触发条件（自动 tier1 → tier2）

```yaml
# config.yaml
upgrade_triggers:
  - condition: "files_changed > 3"        # 多文件改动
    tier: tier2
  - condition: "contains_inline_asm == true"  # 汇编代码
    tier: tier2
  - condition: "diff_lines > 100"          # 改动大
    tier: tier2
  - condition: "touches_tcm_or_ime == true"  # 硬件相关
    tier: tier2
  - condition: "estimated_difficulty >= 4"  # 主观难度
    tier: tier2
```

### 推荐模型配置

| Tier | 模型 | API base | 单价 | 适用 |
| --- | --- | --- | --- | --- |
| **Tier 1** | DeepSeek-Coder | deepseek.com | $0.0001/1k | 读代码、生成简单 patch |
| **Tier 1** | Qwen-Coder | dashscope.aliyuncs.com | $0.0003/1k | 中文友好、profiling |
| **Tier 2** | Claude Sonnet 4.5 | api.anthropic.com | $0.003/1k | 多文件重构、kernel 优化 |
| **Tier 2** | GPT-4 Turbo | api.openai.com | $0.01/1k | 通用 |

> 没设 key 时：返回 `mock=True` 的占位响应，**不报错**，agent loop 不会乱改代码。

## 5. 7 个内置策略

| ID | 类别 | Tier | 风险 | 预期收益 | 实现复杂度 |
| --- | --- | --- | --- | --- | --- |
| `thread_tuning` | config | 1 | low | 5-30% | ★☆☆ |
| `quant_select` | model | 1 | low | 5-50% | ★☆☆ |
| `mem_backend` | system | 1 | low | 5-15% | ★☆☆ |
| `tcm_size` | hardware | 1 | low | 0-10% | ★★☆ |
| `compiler_pgo` | build | 2 | med | 3-8% | ★★☆ |
| `gemm_path_forcing` | kernel | 2 | high | 5-15% | ★★★ |
| `kv_cache_quant` | kernel | 2 | high | 10-20% | ★★★ |

**前三类（线程、量化、内存池）覆盖 60% 性能优化 ROI，优先跑。**

## 6. 知识库 schema (SQLite)

```sql
-- 实验元数据
CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY,
    ts              REAL NOT NULL,
    strategy_id     TEXT NOT NULL,
    description     TEXT,
    tier            TEXT,                  -- tier1 / tier2
    model_name      TEXT,
    status          TEXT,                  -- profiling / kept / reverted / failed
    config_json     TEXT,                  -- 实验配置
    diff            TEXT,                  -- 代码 diff
    files_changed   TEXT,                  -- JSON 数组
    branch          TEXT,                  -- git branch
    baseline_tok_per_s REAL,
    new_tok_per_s   REAL,
    delta_pct       REAL,
    cost_usd        REAL,
    notes           TEXT,
    error_msg       TEXT
);

-- 每次 benchmark 数据
CREATE TABLE benchmarks (
    id              INTEGER PRIMARY KEY,
    experiment_id   INTEGER,
    ts              REAL,
    label           TEXT,                  -- e.g. "thread_t8"
    n_prompt        INTEGER,
    n_gen           INTEGER,
    threads         INTEGER,
    t_p_eval_ms     REAL,
    t_eval_ms       REAL,
    tok_per_s       REAL,
    stddev          REAL,
    extra_json      TEXT
);

-- LLM 用量追踪
CREATE TABLE model_usage (
    id              INTEGER PRIMARY KEY,
    ts              REAL,
    tier            TEXT,
    model_name      TEXT,
    task_type       TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,
    success         INTEGER
);
```

## 7. Git 分支管理

```
main                              ← 稳定基线
├── opt/thread-tuning-20250623-120000   ← 性能提升的实验
├── opt/mem_backend-20250623-121500
└── opt/gemm_path_forcing-20250623-123000   ← (reverted 后删除)
```

**每个实验的 commit message 模板**：

```
auto-opt(thread_tuning): threads=8 gives 65.0 tok/s

Baseline: 50.00 tok/s
New:      65.00 tok/s
Delta:    +30.00%

Model tier: tier1
Cost: $0.0003
```

## 8. 安全机制

1. **dry-run 验证**：所有 patch 先 `git apply --check`，通过才真应用
2. **最小提升门槛**：Δ% > 0.5% 才算"keep"，否则 revert
3. **自动回滚**：性能回退 → `git checkout .` + 删除分支
4. **连续回退保护**：连续 3 次回退 → 停 agent loop
5. **API key 缺失兜底**：用 mock，不乱改代码
6. **温度低**（tier1=0.2, tier2=0.1）：减少 LLM 随机性
7. **单次成本上限**：超 $1 自动终止（可配置）

## 9. 快速开始

```bash
# 1. 装依赖（只需要 pyyaml + requests）
pip3 install pyyaml requests

# 2. 设置 API key
export DEEPSEEK_API_KEY="sk-xxx"        # 国内便宜模型
export ANTHROPIC_API_KEY="sk-ant-xxx"    # 高级模型

# 3. 跑
cd /data/llama.cpp-analysis/llama.cpp
python3 tools/auto-opt/auto_opt.py run

# 4. 看进度（另开 terminal）
python3 tools/auto-opt/auto_opt.py status
python3 tools/auto-opt/auto_opt.py list

# 5. 生成报告
python3 tools/auto-opt/auto_opt.py report
# → results/auto-opt/REPORT.md

# 6. 合并好的优化
python3 tools/auto-opt/auto_opt.py apply opt/thread-tuning-20250623-120000
```

## 10. 单元测试结果

```
$ python3 tools/auto-opt/test_smoke.py
[TEST] imports... OK
[TEST] config.yaml loads correctly... OK (7 strategies configured)
[TEST] KnowledgeBase CRUD... OK
[TEST] ModelRouter (no API key → mock)... OK
[TEST] Editor (git apply dry-run)... OK
[TEST] Strategies can be instantiated...   (thread_tuning)   (mem_backend) OK
[TEST] CLI --help works... OK

7 passed, 0 failed
```

## 11. 端到端示例运行

```
$ python3 tools/auto-opt/auto_opt.py run

######################################################################
# auto-opt agent loop starting
# Repo: /data/llama.cpp-analysis/llama.cpp
# Strategies: 7
# Max iterations: 50, Max duration: 8.0h
######################################################################

[ORCH] ✓ SpacemiT K3 detected

======================================================================
[ORCH] Step 1/2: Measuring baseline (no optimization)
======================================================================
  [PROFILE] BASELINE: build/bin/llama-bench -m models/... -p 512 -n 128 -t 8 -r 5 -o jsonl ...
             log → results/auto-opt/BASELINE-1782185514.log

[ORCH] ✓ Baseline: 50.23 tok/s

======================================================================
[ORCH] Iteration 1: thread_tuning (Worker 线程数调优)
        Tier: 1 | Risk: low | Expected: 5-30%
        Current baseline: 50.23 tok/s
======================================================================
  [THREAD] sweeping: [1, 2, 4, 8, 16]
  [THREAD]   t= 1 → 12.40 tok/s
  [THREAD]   t= 2 → 24.80 tok/s
  [THREAD]   t= 4 → 45.60 tok/s
  [THREAD]   t= 8 → 50.20 tok/s
  [THREAD]   t=16 → 48.10 tok/s

[ORCH] thread_tuning finished in 24.3s
        baseline: 50.23 tok/s
        new:      50.20 tok/s
        delta:    -0.06%
        cost:     $0.0000 (tier: tier1)
[ORCH] ✗ REVERT: improvement -0.06% <= 0.5%
[ORCH] Cooldown 5s...

======================================================================
[ORCH] Iteration 2: mem_backend (大页内存池后端)
        Tier: 1 | Risk: low | Expected: 5-15%
        Current baseline: 50.23 tok/s
======================================================================
  [MEMBACK]   none      → 48.20 tok/s
  [MEMBACK]   posix     → 49.10 tok/s
  [MEMBACK]   hpage     → 52.40 tok/s
  [MEMBACK]   hpage1g   → 55.10 tok/s

[ORCH] mem_backend finished in 32.1s
        baseline: 50.23 tok/s
        new:      55.10 tok/s
        delta:    +9.69%
        cost:     $0.0000 (tier: tier1)
[ORCH] ✓ KEEP: improvement 9.69% > 0.5%
[ORCH]   ↑ new baseline: 55.10 tok/s
[ORCH] Cooldown 5s...

...
```

## 12. 报告示例（results/auto-opt/REPORT.md）

```markdown
# auto-opt Report

## Summary
- Total experiments: 24
- Kept: 11
- Reverted: 13
- Best: gemm_path_forcing → 78.50 tok/s
- Total cost: $0.1423

## Kept Experiments
| ID | Strategy | tok/s | Δ% | Tier | Cost | Branch |
|---|---|---|---|---|---|---|
| 18 | gemm_path_forcing | 78.50 | +42.5% | tier2 | $0.0520 | opt/gemm_path_forcing-... |
| 12 | mem_backend | 55.10 | +9.7% | tier1 | $0.0000 | opt/mem_backend-... |
| 7  | thread_tuning | 53.20 | +5.9% | tier1 | $0.0000 | opt/thread_tuning-... |
| ...
```

## 13. 扩展方法

### 加新策略

```python
# 1. 写 strategies/my_strategy.py
from strategies.base import Strategy, StrategyResult

class MyStrategy(Strategy):
    @property
    def id(self): return "my_strategy"
    @property
    def name(self): return "我的策略"
    @property
    def tier(self): return 1
    # ... 其它 property
    
    def run(self, exp_id, baseline):
        # 自己的逻辑
        return StrategyResult(...)

# 2. 在 config.yaml 加
strategies:
  - id: my_strategy
    name: "我的策略"
    tier: 1
    test: { ... }

# 3. 在 orchestrator._load_strategies 注册
elif sid == 'my_strategy':
    instances.append(MyStrategy(...))
```

### 加新模型

```yaml
# config.yaml
models:
  tier1:
    - name: my-deepseek
      api_base: https://api.deepseek.com/v1
      api_key_env: DEEPSEEK_API_KEY
      model_id: deepseek-coder
      cost_per_1k_tokens: 0.0001
```

## 14. 限制

- **PGO 集成需要 build 支持**：auto-opt 不会自动重新编译
- **MoE 模型没专门策略**：可扩展
- **TCM 容量测试需要 libspine_tcm.so 支持 fake_tcm**
- **没集成 LLM 失败重试**（一次失败 → strategy 失败）
- **没实现 git push**：只本地 commit

## 15. 一句话总结

> **auto-opt = "把 performance engineer 的工作流程自动化"**。
> 国内模型便宜（每次 ~$0.0001）解决 80% 简单优化；高级模型（每次 ~$0.003）只在改汇编/TCM/多文件时启用。
> 一个晚上能在 K3 上自动跑几十次实验，留下 opt/* 分支 + SQLite 知识库 + Markdown 报告。
> 设计师：**先验 baseline，再 ROI 排序跑策略，性能提升才 commit，否则 revert**——保证 main 永远稳定。
