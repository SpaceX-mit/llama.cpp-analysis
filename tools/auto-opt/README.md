# auto-opt: LLM 驱动的性能自动优化系统

> ⚠️ **实验性工具**：本工具在 K3 上自动跑 llama.cpp 性能优化。
> 它会改代码、建分支、调 LLM、用 benchmark 验证——**用前请理解它在干什么**。

## 工作原理

```
                    ┌───────────────────────────────┐
                    │  Orchestrator (agent loop)    │
                    │  状态机：                     │
                    │  IDLE→PROFILE→PROPOSE→APPLY  │
                    │     →BENCH→DECIDE→IDLE       │
                    └──────────┬────────────────────┘
                               │
       ┌───────────┬───────────┼───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Knowledge│ │ Model   │ │Profiler │ │ Editor  ││Committer │
  │  Base   │ │ Router  │ │(benchmark│ │(patch)  ││(git)    │
  │(SQLite) │ │(Tier1/2)│ │ llama-  │ │         ││         │
  │         │ │         │ │ bench)  │ │         ││         │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## 模型分层

| Tier | 用途 | 推荐模型 | 单次成本 |
| --- | --- | --- | --- |
| **Tier 1** | 读代码、生成单文件 patch、profile 分析 | DeepSeek-Coder, Qwen-Coder, GLM | ~$0.0001/1k |
| **Tier 2** | 多文件重构、kernel 优化、架构决策、debug | Claude Sonnet, GPT-4 | ~$0.003/1k |

**自动升级触发**（代码改动符合时自动用 Tier 2）：
- 改动 > 3 个文件
- diff > 100 行
- 涉及内联汇编
- 触碰 TCM / IME
- 难度评级 ≥ 4

## 快速开始

```bash
# 1. 设置 API key（至少一个）
export DEEPSEEK_API_KEY="sk-xxx"      # Tier 1
export ANTHROPIC_API_KEY="sk-ant-xxx"  # Tier 2 (可选)

# 2. 准备 build
cd /data/llama.cpp-analysis/llama.cpp
cmake -B build -DGGML_CPU_RISCV64_SPACEMIT=ON
make -j8

# 3. 准备模型（任意 Q4_0 GGUF）
ls models/qwen2.5-0.5b-instruct-q4_0.gguf  # 或改 config.yaml

# 4. 跑 agent loop
python3 tools/auto-opt/auto_opt.py run

# 5. 看状态
python3 tools/auto-opt/auto_opt.py status

# 6. 生成报告
python3 tools/auto-opt/auto_opt.py report
# → results/auto-opt/REPORT.md
```

## 包含的优化策略

| ID | 名字 | Tier | 风险 | 预期收益 |
| --- | --- | --- | --- | --- |
| `thread_tuning` | 线程数扫描 | 1 | low | 5-30% |
| `quant_select` | 量化方案对比 | 1 | low | 5-50% |
| `mem_backend` | 内存池后端 | 1 | low | 5-15% |
| `tcm_size` | TCM 容量测试 | 1 | low | 0-10% |
| `compiler_pgo` | PGO 编译 | 2 | med | 3-8% |
| `gemm_path_forcing` | GEMM 路径强制 | 2 | high | 5-15% |
| `kv_cache_quant` | KV cache 量化 | 2 | high | 10-20% (长 ctx) |

## 数据流

每个实验都记录到 `results/auto-opt/knowledge.db`：

```
experiments 表：
  id, strategy_id, tier, model_name, status (profiling/kept/reverted/failed),
  baseline_tok_per_s, new_tok_per_s, delta_pct, cost_usd, branch, ...

benchmarks 表（每个实验跑多次）：
  experiment_id, label, n_prompt, n_gen, threads,
  t_p_eval_ms, t_eval_ms, tok_per_s, stddev, ...

model_usage 表（追踪 LLM 用量）：
  ts, tier, model_name, task_type, input_tokens, output_tokens, cost_usd, ...
```

## Git 分支管理

```
main                            ← 稳定基线
opt/thread-tuning-20250101-120000  ← 每个实验一个分支
opt/mem_backend-20250101-121500
opt/gemm_path_forcing-20250101-123000 (reverted)
...
```

性能提升 ≥ 0.5% 的 commit **保留**到 opt/* 分支；
性能下降的 **自动 revert** 并删除分支。

## 安全机制

1. **dry-run 验证**：所有 patch 先 `git apply --check`，通过才真应用
2. **baseline 对比**：每个实验必须比 baseline 提升 ≥ 0.5% 才算成功
3. **自动回滚**：性能回退 → 立即 `git checkout .` + 删分支
4. **连续回退保护**：连续 3 次回退 → 自动停 agent loop
5. **API key 缺失兜底**：没 key 时用 heuristic fallback（不会乱改代码）

## 状态机

```
IDLE
  │
  ├── ensure_baseline()    → 测一次 baseline
  │
  ▼
cycle_once()  (loop)
  │
  ├── select strategy (round-robin)
  ├── create opt/<strategy>-<timestamp> branch
  ├── run strategy
  │     ├── propose patch (调 LLM)
  │     ├── apply patch
  │     ├── run benchmark
  │     └── return result
  ├── decide
  │     ├── delta > 0.5%  → keep (commit to opt branch)
  │     ├── delta ≤ 0.5%  → revert (delete branch)
  │     └── failed        → mark failed, delete branch
  ├── cooldown 5s
  │
  └── loop until stop
        ├── max_iterations
        ├── max_duration
        └── regression_count
```

## 限制 / TODO

- **没实现 LLM 调用失败的细粒度重试**
- **没集成 git push（只本地 commit，opt/* branch 不自动 push）**
- **PGO 集成需要 build 支持，没自动重 build**
- **MoE 模型没专门策略**
- **TCM 容量测试需要 libspine_tcm.so 支持** fake_tcm

## 调试

```bash
# 看策略列表
python3 tools/auto-opt/auto_opt.py list

# 看单个实验
python3 tools/auto-opt/auto_opt.py show 5

# 跑单个策略（不进入 loop）
python3 tools/auto-opt/auto_opt.py try thread_tuning

# 测一次 baseline（不写 DB）
python3 tools/auto-opt/auto_opt.py baseline
```

## 扩展

加新策略：
1. 在 `strategies/` 写一个新文件，继承 `Strategy` 基类
2. 在 `config.yaml` 的 `strategies:` 加一项
3. 在 `orchestrator._load_strategies` 加分支（如果是新类别）

## 关联文档

- `docs/07_performance_evaluation_plan.md` — 怎么评估
- `docs/08_performance_optimization_plan.md` — 怎么优化
- `docs/05_qwen25_0_5b_on_k3_full_walkthrough.md` — K3 + Qwen2.5 完整流程
