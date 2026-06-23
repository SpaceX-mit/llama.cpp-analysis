# 性能评估方案：Qwen2.5 0.5B Q4_0 on SpacemiT K3

> 配套：`05_qwen25_0_5b_on_k3_full_walkthrough.md`、`06_what_is_node_and_cgraph.md`
> 目标：给出**可执行、可复现、可对比**的性能评估流程

---

## 0. 总览：评估的"四把尺子"

任何推理性能评估，归根到底回答四个问题：

| 维度 | 关键问题 | 典型指标 |
| --- | --- | --- |
| **延迟** | 单个 token 多久出？第一个 token 多久出？ | TTFT, TPOT, 单 token 延迟 |
| **吞吐** | 1 秒能产多少 token？ | tokens/s (decode), tokens/s (prefill) |
| **资源** | 用了多少内存？多少核？多少能耗？ | 内存峰值, 线程数, 功耗 |
| **质量** | 答案对不对？量化掉多少精度？ | perplexity, MMLU, GSM8K |

SpacemiT K3 上还需要再加两个**硬件特异指标**：

| 维度 | 关键问题 | 典型指标 |
| --- | --- | --- |
| **IME 利用率** | A100 核的 `vmadotsu.hp` 跑得多满？ | IPC, GEMM cycles / total cycles |
| **TCM 效率** | TCM 命中率多少？搬运开销多大？ | TCM hit rate, memcpy / GEMM ratio |

---

## 1. 第一步：明确工作负载（Workload）

任何 benchmark 在跑之前都要先固定输入。Qwen2.5 0.5B 的典型场景：

| 场景 | prompt 长度 | decode 长度 | batch size | 评估目标 |
| --- | --- | --- | --- | --- |
| **单轮对话** | 512 | 128 | 1 | TTFT + TPOT |
| **长上下文** | 4096 | 256 | 1 | 长 ctx 退化曲线 |
| **批量推理** | 256 | 64 | 8 / 32 | 吞吐 |
| **流式多用户** | 128 | 32 | 16 seq | 并发性能 |
| **代码补全** | 1024 | 256 | 1 | TTFT |

**建议标准测试矩阵**（一个都不能少）：

```
n_prompt  = { 32, 128, 512, 1024, 2048, 4096 }
n_gen     = { 1, 16, 64, 128, 256, 512 }
batch     = { 1, 2, 4, 8, 16 }
threads   = { 1, 2, 4, 8 }       # K3 只有 8 个 A100 核
```

---

## 2. 第二步：选评估工具

### 2.1 工具矩阵

| 工具 | 在仓库的路径 | 干啥 | 何时用 |
| --- | --- | --- | --- |
| `llama-bench` | `tools/llama-bench/llama-bench.cpp` | **首选**。批量跑多种配置，给出 mean/stddev/CSV/JSON 输出 | 系统级吞吐/延迟测试 |
| `examples/simple/simple.cpp` | `examples/simple/` | 最小可运行 demo，自带 `llama_perf_context_print` | 调通 + 简单时间统计 |
| `examples/llama-eval` | `examples/llama-eval/` | 用 vLLM-style 评估榜单 | 精度评估 |
| `examples/batched` | `examples/batched/` | 连续批量推理 demo | 批量场景 |
| `examples/parallel` | `examples/parallel/` | 多用户并行 | 并发场景 |
| `examples/passkey` | `examples/passkey/` | 长 ctx 检索 | 长 ctx 正确性 |
| `perf` / `perfetto` | 系统工具 | CPU instruction 级 profiling | 优化前定位瓶颈 |
| `spike` / `oclgrind` | RISC-V 工具链 | 跑 SpacemiT K3 上的指令分布 | 内核级深度分析 |
| `ggml` 自带 `GGML_LOG_DEBUG` | 仓库默认编译开 | 打每个 op 的耗时 | 节点级 profiling |
| 自定义 `ggml_callback` | 仓库 `common.h:ggml_numa_strategies` | 在 op 前/后注入测量代码 | 灵活自定义 |

### 2.2 `llama-bench` 用法（最推荐）

```bash
# 系统级 benchmark，自动测多种 (n_prompt, n_gen, threads) 组合
./llama-bench \
    -m qwen2.5-0.5b-instruct-q4_0.gguf \
    -p 512 -n 128 \
    -t 1,2,4,8 \
    -ngl 99 \
    -r 5 \                    # 每个组合跑 5 次取平均
    -o md \                   # 输出 Markdown 表格
    2>&1 | tee bench_k3.md
```

`llama-bench` 关键输出（`src/llama-context.cpp:4069-4091` 同款指标）：

| 字段 | 含义 | 来自 |
| --- | --- | --- |
| `load time` | 模型加载 + repack 总耗时 | `t_load_ms` |
| `prompt eval time` | prefill 总耗时 | `t_p_eval_ms` |
| `eval time` | decode 总耗时 | `t_eval_ms` |
| `prompt eval tokens` | prefill 处理的 token 数 | `n_p_eval` |
| `eval runs` | decode 调用次数 | `n_eval` |
| `t/s prefill` | prefill 吞吐 | `n_p_eval / t_p_eval_ms × 1000` |
| `t/s decode` | decode 吞吐 | `n_eval / t_eval_ms × 1000` |
| `ms per token` (decode) | **单 token 延迟** | `t_eval_ms / n_eval` |

输出长这样（示例，K3 上 Q4_0）：

```
model             size    params  backend  ngl  n_threads  t/s prefill  t/s decode  ms/tok
qwen2.5-0.5b      330 MB  0.5B    CPU      99   8          2400         65.0        15.4
```

### 2.3 自定义 hook：`ggml_numa_strategies` + `cparams.cb_eval`

llama.cpp 提供了**每节点回调**（`src/llama-context.cpp:1333`）：

```cpp
ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
```

写自己的回调就能拿到每个 node 的 `(node, ith, nth, threadpool_user_data)`：

```cpp
// 伪代码：测量每个 MUL_MAT 节点的真实耗时
static bool my_eval_cb(struct ggml_tensor * node, int ith, int nth, void * user_data) {
    auto * ctx = (MyCtx*) user_data;
    if (node->op == GGML_OP_MUL_MAT && ith == 0) {
        ctx->t_start = ggml_time_us();
        ctx->node_name = node->name;
    }
    if (ith == nth - 1) {  // 最后一根线程退出时打点
        int64_t dt = ggml_time_us() - ctx->t_start;
        fprintf(stderr, "[%d] %s : %lld us\n", ctx->n_called++, node->name, dt);
    }
    return true;  // 继续
}
```

挂上：

```cpp
cparams.cb_eval = my_eval_cb;
cparams.cb_eval_user_data = &my_ctx;
```

→ 跑一次 decode，能看到每个 MUL_MAT 的精确耗时。

---

## 3. 第三步：spacemit 特有的"打开调试日志"

SpacemiT 后端在 `GGML_LOG_DEBUG` 宏下会输出**关键决策点**。要打开它：

```bash
# 编译时：打开 GGML_LOG_DEBUG
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CPU_RISCV64_SPACEMIT=ON \
      -DCMAKE_C_FLAGS="-DGGML_DEBUG=1" ...
```

或者用环境变量（如果 `ggml_log_level` 支持运行时切换）：

```bash
GGML_LOG_LEVEL=debug ./llama-cli -m ...
```

开启后会看到这些信息（节选自 `spacemit/ime_env.cpp:287-291`、`ime.cpp:932`）：

```
[CPU_RISCV64_SPACEMIT] num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064,
                       exclude_main_thread: 0, use_ime1: 0, use_ime2: 1,
                       mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
[CPU_RISCV64_SPACEMIT] tcm is available, blk_size: 4194304, blk_num: 8, is_fake_tcm: 0
[repack] repack tensor blk.0.attn_q.weight with Q4_0_32x256
[repack] repack tensor blk.0.attn_k.weight with Q4_0_32x256
...
```

可以验证：
- ✓ 是否真识别到 8 个 A100 核
- ✓ TCM 是否真可用
- ✓ Repack 是否选了最优 tensor_traits

---

## 4. 第四步：测量 IME 核的硬件指标

### 4.1 用 `perf` 抓 RISC-V 指令分布

```bash
# 静态计数器：统计一段时间内 vmadotsu.hp 的执行次数
perf stat -e instructions,cycles,vmadotsu_hp,vmadotu_hp,L1-icache-load-misses \
    ./llama-bench -m qwen2.5-0.5b-q4_0.gguf -p 128 -n 64 -t 8
```

输出示例（节选）：

```
 Performance counter stats for './llama-bench':
       4,521,834,221      instructions              #    1.42  insn per cycle
       3,182,100,000      cycles                    #    2.10 GHz
         812,448,192      vmadotsu_hp               # ★ 这个数字越大越好
         102,448,100      L1-icache-load-misses     # 越小越好
       2.341              seconds time elapsed
```

> **理想值**：`vmadotsu_hp` 数量 / `instructions` 比例越高越好；`L1-icache-load-misses` 越低越好。

### 4.2 用 `perf record` 抓热点

```bash
perf record -F 4000 -g --output=perf.data \
    ./llama-bench -m qwen2.5-0.5b-q4_0.gguf -p 128 -n 64 -t 8
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

打开 `flamegraph.svg` 能直观看到时间花在：
- `gemm_kernel_i8i4_hp` (期待高占比 60-80%)
- `rvv::quantize_a_4row_i8_hp` (期待 5-10%)
- `rvv::forward_flash_attn_ext_f16` (期待 5-10%)
- `memcpy1d` (期待 5-10%)

### 4.3 RISC-V 专用工具

```bash
# spike 模拟器（如果开发板没接上 K3 实体）：单步执行看真实指令数
spike --varch=vlen=128 ./llama-cli ...

# 看 TCM 占用情况
ls -l /dev/tcm_sync_mem
cat /proc/spacemit_tcm_info    # 假设有这个 proc
```

---

## 5. 第五步：测量内存行为

### 5.1 内存峰值

```bash
# /usr/bin/time 比 GNU time 更准
/usr/bin/time -v ./llama-cli -m ... -n 128
# "Maximum resident set size (kbytes):" 就是 RSS 峰值
```

K3 跑 Qwen2.5 0.5B Q4_0 应该看到：
- 模型权重: 330 MB
- KV cache (ctx=4096): 5.5 MB
- Workspace: 10-30 MB
- TCM: 8 × 4 MB = 32 MB（每核）
- 合计: **~400 MB**

### 5.2 TCM 命中/未命中统计

写一个 hook 数 TCM memcpy 调用：

```cpp
// 在 ime.cpp:410 的 rvv::memcpy1d(tcm_buffer, ...) 前后加计数器
static std::atomic<size_t> tcm_load_count{0};
static std::atomic<size_t> tcm_load_bytes{0};

void * memcpy1d_to_tcm(void * dst, const void * src, int64_t size) {
    tcm_load_count++;
    tcm_load_bytes += size;
    return memcpy(dst, src, size);
}
```

跑完打印 `tcm_load_bytes / total_GEMM_bytes` 算 TCM 利用率。

### 5.3 大页 vs hugetlb 1G

切换内存池后端（`SPACEMIT_MEM_BACKEND`）：

```bash
SPACEMIT_MEM_BACKEND=hpage   ./llama-cli ...    # 默认：2MB transparent hugepage
SPACEMIT_MEM_BACKEND=hpage1g ./llama-cli ...    # 1GB hugetlb（需要内核模块）
SPACEMIT_MEM_BACKEND=posix   ./llama-cli ...    # 普通 4KB 页（最差 baseline）
SPACEMIT_MEM_BACKEND=none    ./llama-cli ...    # 关闭（最最差 baseline）
```

对比 `t/s decode` 差异，预期 5-15% 提升（hugepage → 1G hugetlb）。

---

## 6. 第六步：精度评估

性能再快，答案错了也不行。Q4_0 量化会掉点精度。

### 6.1 量化误差（Quantization Error）

```bash
# 准备 fp16 原始模型（精度上限）
./llama-bench -m qwen2.5-0.5b-f16.gguf -p 0 -n 0  # 只为了加载一次，做 baseline
# Q4_0 vs F16 的 perplexity 差异应 < 0.5
```

### 6.2 Benchmark 榜单

```bash
# examples/llama-eval 提供统一接口
./llama-eval -m qwen2.5-0.5b-q4_0.gguf --task mmlu --limit 200
./llama-eval -m qwen2.5-0.5b-q4_0.gguf --task gsm8k --limit 200
```

对比基线（fp16 / Q8_0 / Q4_K）：

| 量化 | WikiText perplexity | MMLU 5-shot | 期望差距 |
| --- | --- | --- | --- |
| F16 (baseline) | 12.3 | 45.2% | – |
| Q8_0 | 12.4 | 45.0% | < 0.5% |
| Q4_K | 12.8 | 44.5% | < 1% |
| **Q4_0** | 13.5 | 43.8% | < 1.5% |
| Q3_K | 15.2 | 41.5% | < 4% |

---

## 7. 第七步：可视化与对比

### 7.1 生成可对比的表格

```bash
for q in Q4_0 Q4_K Q5_K Q6_K Q8_0; do
    ./llama-bench -m qwen2.5-0.5b-${q}.gguf -p 512 -n 128 -t 8 -r 3 -o jsonl \
        2>&1 | tail -1 >> all_results.jsonl
done
```

`all_results.jsonl` 可以丢给 pandas / Excel 画图。

### 7.2 Chrome trace 火焰图（最直观）

llama.cpp 的 `cparams.cb_eval` 配合 `chrome://tracing` 格式能生成 timeline：

```cpp
// 伪代码
static int64_t t_prev = 0;
static bool my_cb(node, ith, nth, ud) {
    if (ith == 0) {
        int64_t t_now = ggml_time_us();
        // 输出 JSON event: {"name":"Q@layer5","ph":"X","ts":..,"dur":..}
        print_trace_event(node->name, t_prev, t_now - t_prev);
        t_prev = t_now;
    }
    return true;
}
```

打开 chrome → `chrome://tracing` → Load 文件 → 看哪一层最慢。

---

## 8. 评估 Checklist（自检用）

跑一次完整的评估流程，必须确认：

- [ ] **环境稳定**：CPU 频率锁定 (`cpupower frequency-set -d 1.8G`)，关闭 turbo，关掉其他进程
- [ ] **模型已 repack**：第一次跑会有 repack 开销，要 discard 第一次数据
- [ ] **Warmup 充分**：先跑 5-10 次不计时间，让 KV cache 填好
- [ ] **重复足够**：每个组合至少 3-5 次取 mean / stddev
- [ ] **时间维度**：wall clock + CPU time + perf cycles 三者对比
- [ ] **温度记录**：芯片温度（thermal throttling 会让后半段变慢）
- [ ] **TCM 验证**：`GGML_LOG_DEBUG` 输出确认 `tcm is available`
- [ ] **Kern 选择正确**：`repack tensor X with Q4_0_32x256` 而不是 fallback 版本
- [ ] **数值正确性**：Q4_0 输出和 F16 比 perplexity 差距 < 1.5
- [ ] **跨硬件对比**：至少和 K1 / K2 / 通用 RVV baseline 对比

---

## 9. 一句话总结

> **K3 上评估 Qwen2.5 0.5B Q4_0 = `llama-bench` 跑全矩阵 + `perf stat` 抓 `vmadotsu_hp` 指令占比 + `GGML_LOG_DEBUG` 验证路径正确 + 量化精度对比 < 1.5%**。重点不是绝对数字，而是**对比实验**——换量化方案、换线程数、换内存池、换 TCM 配置，找到最佳的 Pareto 曲线。
