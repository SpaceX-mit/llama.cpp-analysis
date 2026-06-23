# llama.cpp 推理与 SpacemiT Backend 深度分析

本仓库对 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的推理链路以及 `ggml-cpu/spacemit/` 后端做了完整的源码级分析。

## 内容

| 文档 | 主题 |
| --- | --- |
| [`docs/01_llama_cpp_inference_overview.md`](docs/01_llama_cpp_inference_overview.md) | llama.cpp 推理全过程：`llama_decode` → `ggml_cgraph` → 后端算子的端到端调用链路 |
| [`docs/02_spacemit_backend_deep_dive.md`](docs/02_spacemit_backend_deep_dive.md) | SpacemiT K1/K2/X60/X100/X200/A60/A100/A200 RISC-V + IME 后端源码剖析：拓扑探测、TCM、双线程 ld/compute 流水线、IME1/IME2 GEMM 内联汇编 |
| [`docs/03_spacemit_file_index.md`](docs/03_spacemit_file_index.md) | 关键文件 / 行号速查表 |

## 复现

```bash
git clone https://github.com/ggml-org/llama.cpp.git
# 配合 docs/ 文档阅读
```

## 关键收获

1. **三层架构**：LLM 前端 → 静态 `ggml_cgraph` → 后端算子。SpacemiT 以 "extra buffer type" 形式嵌入 CPU backend，复用 `ggml_threadpool` + `ggml_compute_forward` 分派链路。

2. **异构 SoC 调度**：`ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity` 把 worker 线程通过 `/proc/set_ai_thread` 迁移到 RISC-V AI 核，并 `pthread_setaffinity_np` 锁核 + 抢占 TCM。

3. **双线程 ld/compute 流水线**：每对线程 `(ith%2==0, ith%2==1)` 共享一个 `spine_barrier_t`，交替做 memcpy 与 `vmadotsu` / `vmadotu.hp` GEMM，掩盖数据搬运延迟。

4. **离线 repack + 在线 quant**：加载期把 GGUF Q4_K/Q6_K/Q8_0/Q2_K/Q3_K/Q5_K/MXFP4 一次性重排成 IME 友好的 N×K 排布；推理期把 fp32 激活在线量化为 int8，统一在 int8 域做 int8×int4→fp32 MAC。

5. **MoE 优化**：`moe_m2_gemm_kernel_*` 一次处理 2 个 token 共享 weight 加载；tile-based 路径把多个 token 的 A 量化数据先聚到 TCM 再批量 GEMM。
