# 关键文件 / 行号速查表

> 所有行号对应：`/data/llama.cpp-analysis/llama.cpp/`

## A. 整体推理流程

| 位置 | 关键函数/宏 | 作用 |
| --- | --- | --- |
| `src/llama-context.cpp:1680` | `llama_context::decode()` | decode 入口 |
| `src/llama-context.cpp:1304` | `llama_context::process_ubatch()` | 构建图 + 调 ggml 计算 |
| `src/llama-context.cpp:2421` | `llama_context::graph_compute()` | 调 `ggml_backend_sched_graph_compute_async` |
| `src/llama-model.cpp:2234` | `llama_model::build_graph()` | 选 arch + 拼图 |
| `src/models/llama.cpp:94` | `llama_model_llama::build_arch_graph()` | 拼 Llama 类模型图 |
| `ggml/src/ggml-backend.cpp:1014` | `ggml_backend_sched_split_graph()` | 把图按 backend 切分 |
| `ggml/src/ggml-backend.cpp:1541` | `ggml_backend_sched_compute_splits()` | 实际执行 |
| `ggml/src/ggml-cpu/ggml-cpu.c:3018` | `ggml_graph_compute_thread()` | CPU backend worker 线程入口 |
| `ggml/src/ggml-cpu/ggml-cpu.c:1245` | `ggml_compute_forward_mul_mat()` | CPU 默认 MUL_MAT |
| `ggml/src/ggml-cpu/ggml-cpu.c:1827` | `GGML_OP_MUL_MAT` 分派 | 进 tensor_traits |

## B. SpacemiT 后端

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/CMakeLists.txt:438-499` | riscv64 + spacemit 编译开关 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp:42-56` | spacemit buffer type 注册到 extra bufts |
| `ggml/src/ggml-cpu/ggml-cpu.c:3026` | 工作线程调 `ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity` |
| `ggml/src/ggml-cpu/spacemit/ime.h` | 后端对外 C API |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1-58` | 头文件 + 编译期检查 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:75-80` | TLSContext (per-thread TCM) |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:84-120` | `get_repacked_block_type_size` / `block_type_has_zp` 模板元函数 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:122-936` | `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>` |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:127-185` | `work_size`（workspace 估算） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:187-232` | `compute_forward`（按 op 分派） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:234-541` | `forward_mul_mat`（含 3 条路径） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:359-389` | A 矩阵量化（4row / 1row 调度） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:391-540` | 3 条 GEMM 路径（TCM / ld-compute pipe / fallback） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:543-929` | `forward_mul_mat_id`（MoE） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:661-697` | mmid_row_mapping 表构造 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:720-927` | MoE 2 条路径 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:931-935` | `repack()` 入口 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:938-1232` | `tensor_traits_common`（RVV 算子集合） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1149-1226` | `forward_flash_attn_ext_f16` 调度 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1235-1251` | 13 个 tensor_traits 具现化 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1257-1393` | `ggml_riscv64_spacemit_get_optimal_repack_type` |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1395-1458` | buffer hooks (init/set/free) |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1496-1574` | `nbytes`（按 repack 后的尺寸算） |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1577-1643` | `extra_buffer_type::supports_op` / `get_tensor_traits` |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1647-1665` | `ggml_backend_cpu_riscv64_spacemit_buffer_type` |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1668-1729` | 线程绑定 + TCM 申请 |
| `ggml/src/ggml-cpu/spacemit/ime.cpp:1731-1739` | TCM 释放 |

## C. SpacemiT 环境与内存

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/spacemit/ime_env.h:12-23` | `spine_core_arch_id` 枚举 |
| `ggml/src/ggml-cpu/spacemit/ime_env.h:25-30` | `spine_core_info` |
| `ggml/src/ggml-cpu/spacemit/ime_env.h:32-51` | `spine_env_info` |
| `ggml/src/ggml-cpu/spacemit/ime_env.cpp:18-83` | `get_spine_core_info`：marchid 解析 |
| `ggml/src/ggml-cpu/spacemit/ime_env.cpp:148-305` | `spine_env_info` 构造 |
| `ggml/src/ggml-cpu/spacemit/ime_env.cpp:259-262` | `use_ime1/2` 决策 |
| `ggml/src/ggml-cpu/spacemit/ime_env.cpp:264-285` | TCM 探测 + `use_tcm` 决策 |
| `ggml/src/ggml-cpu/spacemit/spine_barrier.h` | 两变量 cache-line barrier |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.h` | 内存池后端枚举 + TCM API |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp:106-414` | `spine_mem_pool_manager` 通用分配器 |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp:416-491` | `posix` + `transparent_hugepage` 实现 |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp:493-760` | `hugetlb_1g` 实现 + TCM 包装 |
| `ggml/src/ggml-cpu/spacemit/spine_tcm.h:67-148` | libspine_tcm runtime ABI |
| `ggml/src/ggml-cpu/spacemit/spine_tcm.h:200-409` | dlsym loader 实现 |

## D. Repack

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/spacemit/repack.h:11-13` | `repack<BLOC_TYPE, INTER_SIZE, NB_COLS>` 模板入口 |
| `ggml/src/ggml-cpu/spacemit/repack.cpp:50-78` | IME block 排布（`block<K,N>` / `block_with_zp<K,N>`） |
| `ggml/src/ggml-cpu/spacemit/repack.cpp:80-150` | 16 行 IME1 块构造 |
| `ggml/src/ggml-cpu/spacemit/repack.cpp:152+` | 各种 repack 函数（q4_0/q4_1/q4_K/q2_K/...） |
| `ggml/src/ggml-cpu/spacemit/ime_kernels.h:11-63` | `nrow_block_q2_k/q3_k/q5_0/q5_1/mxfp4` 排布 |

## E. RVV 标量/算子

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.h:18-37` | 量化块大小常量 |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.h:43-91` | 量化 / mem 拷贝 / FlashAttn / Norm / Sum / Repeat / Concat / GetRows 声明 |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:57-88` | `rvv_expf_approx_f32m2`（带 saturated 处理） |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:90-100` | `rvv_tanh_approx_f32m2` |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:102-131` | `softcap` + softmax 内联 |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1121-1303` | `forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16` |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1305-1629` | `forward_flash_attn_ext_f16_tiled_vlen1024_vf16` |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1630-1800+` | `forward_rms_norm_f32` 等 |

## F. IME1 Kernels

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp:34-94` | `QUANTIZEM4ROW_KERNEL` / `STORE` 宏 |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp:97-200+` | `quantize_a_4row_i8` |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp:300-500` | SQ4BIT 宏（scale/zp 加载） |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp:586-1027` | `SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl` |
| `ggml/src/ggml-cpu/spacemit/ime_kernels.h:71-86` | `ime1::gemm_kernel_i8i4` 声明 |

## G. IME2 Kernels

| 位置 | 关键内容 |
| --- | --- |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:36-957` | 各量化类型的 mrow_ref 参考实现 |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:958-2030` | `gemm_kernel_i8i2k_m1/m4` |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:2031-2880` | `gemm_kernel_i8i3k_m1/m4` |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:2430-2880` | `gemm_kernel_i8i4_m1/m4` 详细 asm |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:2883-3360` | `gemm_kernel_i8i4_hp_m1/m4` |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:4773-5000+` | `gemm_kernel_i8i8_m1/m4` |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:5006-5529` | `moe_m2_gemm_kernel_i8i4_impl`（MoE 双行） |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:5530-5768` | 顶层 wrapper（按 count_m 分派 m1/m4 + moe_m2） |
| `ggml/src/ggml-cpu/spacemit/ime_kernels.h:88-188` | `ime2::gemm_kernel_*` 声明 |
