# llama.cpp 推理全过程详解

> 仓库路径：`/data/llama.cpp-analysis/llama.cpp`
> 对应 commit：本分析基于本次 clone 的最新 main 分支（2025-06）
> 本文目标：从用户调用 `llama_decode()` 出发，逐层展开到最终的向量化矩阵乘 kernel，完整梳理一次 token 推理的端到端执行路径。

---

## 0. 顶层架构总览

llama.cpp 的代码物理上分三层，从上到下：

| 层 | 路径 | 职责 |
| --- | --- | --- |
| LLM 前端 | `src/llama-*.cpp` | 模型加载、tokenization、KV cache、采样、循环解码 |
| 计算图 + 张量 | `ggml/src/ggml.cpp` + `ggml/src/ggml-cpu/*` | 静态算子图构建、内存分配、调度、CPU/GPU 算子实现 |
| 后端实现 | `ggml-cpu/spacemit/`、`ggml-cuda/`、`ggml-metal/` … | 真正把图节点跑起来的算子（含 SpacemiT K1/K2/X60/X100/X200 的 RVV+IME 内核） |

一次推理 (`llama_decode`) 的全部工作，最终会下沉为对 `ggml_cgraph` 中若干节点的 `compute_forward` 调用。CPU backend 上，每个节点再分发给最优的特化 kernel（SpacemiT 后端会优先选 IME 内核）。

---

## 1. 用户层调用入口

### 1.1 `llama_decode()` (`src/llama-context.cpp:4054`)

```cpp
int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);   // → llama_context::decode
    ...
}
```

`llama_context::decode()` (`src/llama-context.cpp:1680`) 实际负责：

1. **校验 batch**：`token`/`embd` 至少有一个非空、`n_tokens > 0`、与采样器兼容。
2. **初始化 batch 分配器**：`balloc->init()` 把用户给的 batch 按 `n_seq_max` 拆分成 `ubatch`。
3. **预留 KV cache 槽位**：`memory_update()` + `memory->init_batch()`，如空间不足会做 cache optimization（驱逐/重排）后重试。
4. **循环处理每个 ubatch**：

```cpp
do {
    const auto & ubatch = mctx->get_ubatch();
    ...
    const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(...), mctx.get(), status);
    ...
} while (mctx->next());
```

### 1.2 `process_ubatch()` (`src/llama-context.cpp:1304`)

```cpp
llm_graph_result * llama_context::process_ubatch(...) {
    // 1) 让 KV cache memory context 真正写入 slot
    mctx->apply();

    // 2) 决定是否复用上一次图
    if (gparams 与之前完全一致 且没有显式禁用) {
        n_reused++;          // 命中缓存的 graph 走法
    } else {
        res->reset();
        ggml_backend_sched_reset(...);
        ggml_backend_sched_set_eval_callback(...);
        gf = model.build_graph(gparams);  // ← 关键：构建计算图
        ggml_backend_sched_alloc_graph(sched.get(), gf);  // 分配/映射 buffer
    }

    // 3) 把 ubatch 里的 token/pos/seq_id 拷进图的输入张量
    res->set_inputs(&ubatch);

    // 4) 真·执行
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);

    return res;
}
```

`graph_compute()` 实际上调用 `ggml_backend_sched_graph_compute_async()` 触发 `ggml_cgraph` 的执行。

---

## 2. 计算图构建：`llama_model::build_graph` → `llm_graph_context`

### 2.1 入口 (`src/llama-model.cpp:2234`)

```cpp
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);
    llm->build_pooling(...);                 // 池化层（embedding 模式）
    llm->build_sampling();                   // 后端 sampling（可选）
    llm->build_dense_out(...);               // sentence-transformers dense head
    llm->res->set_outputs(params);
    return llm->res->get_gf();
}
```

`build_arch_graph()` 是 `src/models/<arch>.cpp` 里的虚函数重写，按 `LLM_ARCH_*` 分派（LLAMA / QWEN2 / QWEN3 / DEEPSEEK2 / MISTRAL / GEMMA …）。

### 2.2 典型 Llama 类架构 (`src/models/llama.cpp:94`)

```cpp
std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(
        const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}
```

真正的图构建在 `graph<embed>::graph(...)` 构造里：

```cpp
inpL = build_inp_embd(model.tok_embd);             // tok_embd 张量 + token id 输入
inp_pos = build_inp_pos();                         // 位置 ids
inp_attn = build_attn_inp_kv();                    // KV cache slot / RoPE 缓存

for (int il = 0; il < n_layer; ++il) {
    cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);   // RMSNorm
    auto [Q, K, V] = build_qkv(layer, cur, ...);                        // QKV 投影
    Q = ggml_rope_ext(ctx0, Q, inp_pos, ...);                          // RoPE
    K = ggml_rope_ext(ctx0, K, inp_pos, ...);
    cur = build_attn(inp_attn, layer.wo, ..., Q, K, V, ...);            // self-attn + Wo
    ffn_inp = ggml_add(ctx0, cur, inpSA);                               // 残差
    cur = build_ffn(cur, gate, up, down, LLM_FFN_SILU, LLM_FFN_PAR);    // SwiGLU FFN
    inpL = ggml_add(ctx0, cur, ffn_inp);                                // 残差
}

cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);      // final norm
cur = build_lora_mm(model.output, cur);                                // lm_head
ggml_build_forward_expand(gf, cur);                                    // 固化图
```

> 不同 arch（`src/models/llama.cpp`、`deepseek2.cpp`、`qwen2.cpp` 等）的差异仅在于 norm 类型、激活、attention 类型（GQA / MQA / MLA）、是否存在 MoE、是否存在 MTP hook 等，但**图节点的 opcode 集合**与通用 `ggml_compute_forward` 分发表完全一致。

### 2.3 关键算子（ggml op）枚举

由 `ggml.h` 定义，常见包括：

| Op | 用途 | 关键 src/dst |
| --- | --- | --- |
| `GGML_OP_MUL_MAT` | 线性层 (W·x) | src0=weight, src1=activation |
| `GGML_OP_MUL_MAT_ID` | MoE expert GEMM | src0=expert weights, src1=act, src2=expert ids |
| `GGML_OP_RMS_NORM` | RMSNorm | src |
| `GGML_OP_NORM` | LayerNorm | src |
| `GGML_OP_ADD`/`SUB`/`MUL`/`DIV` | 逐元素 | src0, src1 |
| `GGML_OP_ROPE` / `ROPE_BACK` | 旋转位置编码 | Q/K + positions |
| `GGML_OP_GET_ROWS` | 嵌入查表 | tok_embd + token ids |
| `GGML_OP_FLASH_ATTN_EXT` | FlashAttention | Q, K, V, mask, sinks |
| `GGML_OP_SOFT_MAX` | softmax | attn_scores |
| `GGML_OP_REPEAT` / `CPY` / `CONT` | 形状 / 拷贝 | – |
| `GGML_OP_SET_ROWS` / `GET_ROWS` | 写 KV cache | – |
| `GGML_OP_GLU` (SwiGLU 等) | FFN 门控 | – |
| `GGML_OP_SUM_ROWS` | 序列求和 | – |
| `GGML_OP_CONCAT` | 拼接 | – |

---

## 3. 后端调度：`ggml_backend_sched_*`

文件：`ggml/src/ggml-backend.cpp` + `ggml/src/ggml-backend-sched.cpp`

### 3.1 注册阶段

- `ggml_backend_cpu_get_extra_buffer_types()` (`ggml-cpu/ggml-cpu.cpp:42`) 在启动时把 **`spacemit`**、AMX、KleidiAI、CPU-repack 等可选 buffer type 注入到 backend registry。SpacemiT 后端由此获得自己的 `buffer_type`，可以被分配到 `src0` weight tensor。

### 3.2 图切分（`ggml_backend_sched_split_graph`，`ggml-backend.cpp:1014`）

对一个 `ggml_cgraph`：

1. 维护一个 `backend_buf_exp` 列表跟踪每个 tensor 当前归属哪个 backend。
2. 遍历每个节点：尝试让所有 `src[i]` 落在同一 backend；若不行，就**插入 `GGML_OP_CPY` 节点**做 cross-backend 搬运。
3. 记录到节点的 `backend_id`，并设置 `GGML_TENSOR_FLAG_COMPUTE`。

### 3.3 图分配（`ggml_backend_sched_alloc_graph`）

对每个 backend：
- 把它负责的 tensor 用 `ggml_backend_alloc_ctx_tensors` 真正分配。
- 对 SpacemiT 而言，weight tensor 会触发 `extra_buffer_type::init_tensor()` → `ggml_riscv64_spacemit_get_optimal_repack_type()`，把 32/16 行打包的 Q4_0/Q4_K/Q6_K… 一次性 repack 成 IME 友好的 N×K 排布。

### 3.4 图执行（`ggml_backend_sched_compute_splits`）

按 backend 分组（pipeline），每组是连续的同 backend 节点段：
- 若该 backend 实现了 `graph_compute_async`，则后台线程异步执行；
- 否则 fallback 到 `ggml_backend_cpu_graph_compute` (`ggml-cpu/ggml-cpu.cpp`)。

---

## 4. CPU 后端线程模型与算子分派

文件：`ggml/src/ggml-cpu/ggml-cpu.c` 和 `ggml-cpu/ggml-cpu.cpp`

### 4.1 线程入口 `ggml_graph_compute_thread` (`ggml-cpu.c:3018`)

```c
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif
    ...
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }
        ggml_barrier(state->threadpool);   // 节点之间隐式 barrier
    }
```

→ 在 SpacemiT 上**每个 worker 线程启动时**就会调用 `set_numa_thread_affinity`，把线程绑到 RISC-V 的 AI 核 + 拿一块 TCM。

### 4.2 `ggml_compute_forward` → 大 switch 分发 (`ggml-cpu.c:~1500`)

对每个 op 跳到对应 `ggml_compute_forward_*` 函数。SpacemiT 后端通过 `tensor_traits` 机制 hook 走了高优路径：

- `tensor_traits::compute_forward()` 优先于默认 `ggml_compute_forward_*`。
- 任何 `MUL_MAT` / `MUL_MAT_ID` / `NORM` / `RMS_NORM` / `ADD`/`SUB`/`MUL`/`DIV` / `FLASH_ATTN_EXT` / `CONT` / `CPY` / `REPEAT` / `SUM_ROWS` / `GET_ROWS` / `CONCAT` 都会进 `tensor_traits_common`（RVV kernel）或 `tensor_traits<BLOC_TYPE,…>`（IME kernel）。

### 4.3 fused ops

`ggml_cpu_try_fuse_ops` 实现了一些 op 融合加速：
- `RMS_NORM + MUL` → `ggml_compute_forward_rms_norm_mul_fused`
- …

---

## 5. 一条 `MUL_MAT` 节点到底在 SpacemiT 上跑了什么

`GGML_OP_MUL_MAT` 的语义：`dst = src0 @ src1`，其中：

| 维度 | 含义 | 大小 |
| --- | --- | --- |
| `ne01` × `ne00` | 权重矩阵 W（量化后） | `[out_features, in_features]` |
| `ne11` × `ne10` | 激活矩阵 X（fp32，必要时在线 quant） | `[batch_tokens, in_features]` |
| `ne1` × `ne0` | 输出 = X·Wᵀ | `[batch_tokens, out_features]` |

### 5.1 前置条件检查

`tensor_traits_common::compute_forward()` 在调度时已经走 RVV；进 MUL_MAT 的路径是 `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>`，要做两件事：

1. **从 `src1` (fp32) 量化到 int8**：
   - 路径一：4-row 量化（`quantize_a_4row_i8` / `quantize_a_4row_i8_hp`），一次扫 4 行，输出按 32/256 元素一组，附带 fp32 scale。
   - 路径二：单行量化（`quantize_a_row_i8*`）。
2. **调用 `gemm_kernel_*`**：`C += A_int8 × B_int4`（IME2 Q4_0 行 repack），或更通用的 Q2_K/Q3_K/Q6_K/Q8_0/MXFP4/Q5_K。

### 5.2 workspace 与 TCM

`work_size()` 计算：

```cpp
size = nbytes(src1_ne) * Q8_blk_size;     // 量化 src1 的 buffer
for (MUL_MAT_ID) size += MMID_row_mapping 表;
```

实际计算时：
- 默认走 `params->wdata`（CPU DRAM）。
- 若 SpacemiT AI 核启用 TCM 且 `per_mb_rows_wsize <= tcm_buffer_size`，把 repack 后的 `B` 列（或 `A` 行）放进每核的 **TCM（Tightly-Coupled Memory）**。
- 双 buffer 乒乓：奇/偶 thread（`ith % 2`）交替做 memcpy 与 GEMM，被 `spine_barrier_wait()` 同步。

### 5.3 IME kernel 调用约定

```cpp
size_t rows_handled = gemm_kernel(
    blk_len,                // K block length: 32 (IME1/Q4_0) / 256 (IME2)
    a_row_ptr,              // int8 量化 src1 行（已放 TCM 或 DRAM）
    b_col,                  // int4 repacked weight
    b_col_zp,               // weight zero-point (q4_1/q4_K/q5_1 等)
    c_blk,                  // output (fp32)
    rows_remaining,         // M
    n_blk_real,             // N
    b_k_blks,               // K blocks
    gemm_n);                // ldc
```

不同数据类型的 kernel 选型见 `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>::compute_forward`（`ime.cpp:264-310`）：

| BLOC_TYPE | INTER_SIZE | NB_COLS | 选哪个 gemm kernel |
| --- | --- | --- | --- |
| `block_q4_0`/`q4_1`/`q4_K` (IME1) | 32 | 16 | `ime1::gemm_kernel_i8i4` |
| `block_q4_0`/`q4_1`/`q4_K` (IME2) | 32 | 32 | `ime2::gemm_kernel_i8i4` |
| `block_q4_0`/`q4_1`/`q4_K` (IME2 HP) | 256 | 32 | `ime2::gemm_kernel_i8i4_hp` |
| `block_q2_K` (IME2) | 256 | 32 | `ime2::gemm_kernel_i8i2k` |
| `block_q3_K` (IME2) | 256 | 32 | `ime2::gemm_kernel_i8i3k` |
| `block_q6_K` / `block_q8_0` (IME2) | 32 | 32 | `ime2::gemm_kernel_i8i8` |
| `block_mxfp4` (IME2) | 32 | 32 | `ime2::gemm_kernel_i8mxfp4` |
| `block_q5_K`/`q5_1`/`q5_0` (IME2) | 32 | 32 | `ime2::gemm_kernel_i8i5` |

### 5.4 `MUL_MAT_ID`（MoE）

`tensor_traits::forward_mul_mat_id()`（`ime.cpp:543-929`）做三件事：

1. **Quantize src1**（同 MUL_MAT），按 `[i12][i11][ak_blk]` 三维写入 workspace。
2. **构造 `mmid_row_mapping` 表**：ith==0 线程扫描 `ids` 把每个 token 路由到选中的 expert，并填充 `valid_ep_count` / `valid_act_count`。
3. **遍历每个 expert**（`valid_id`）：
   - 若所有 expert 都有人用、且 TCM 足够 → 走"全专家并行"路径（每个 thread 负责一个 expert 内的 1 个 token）。
   - 否则走"普通"路径：按 `[expert][M 块 1~2 行][N 列块]` 切片。
   - 提供 `moe_m2_gemm_kernel_i8i4` / `moe_m2_gemm_kernel_i8mxfp4` / `moe_m2_gemm_kernel_i8i5`（`ime2_kernels.cpp:5006-5529`），一次处理 2 个 token × 1 行的矩阵乘，减少地址准备开销。

---

## 6. Attention 路径

`build_attn(...)` 在 `llama-graph.cpp` 内部会根据模型类型选择：

- **GQA/MHA**：标准的 `GGML_OP_FLASH_ATTN_EXT` op（自 llama.cpp 起所有路径都走 flash-attn 风格，无 KV 拼接）。
- **MLA / Linear / Recurrent (Mamba, RWKV, DeltaNet)**：直接展开为一系列 `MUL_MAT` + 自定义 op。

`tensor_traits_common::compute_forward()` 命中 `GGML_OP_FLASH_ATTN_EXT` 时调用 `forward_flash_attn_ext_f16()`（`ime.cpp:1149`）：

1. 形状/精度合法性检查（Q 必须 fp32，K/V 必须 fp16，DK,DV ≤ 128，`__riscv_vlenb()==128`）。
2. Q 行分片给 worker：`nchunk = nth`，每行用 `ggml_threadpool_chunk_add` 动态认领。
3. **Tiled 路径** (`Q_TILE_SZ=128`) 优先用 `forward_flash_attn_ext_f16_tiled_vlen1024_vf16`：
   - 把 Q 切成 Q_TILE 行的小块，在 TCM 里用 `vfwmul`/`vfredsum` 做 K·Qᵀ、softmax、PV 累加；
   - `m_tiling` 和 `s_tiling` 双缓冲做 online softmax；
4. 退路用单 chunk 全 K/V 在 DRAM 跑的 `forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16`，里面还会视 ir_step∈{1,2,4} 自动展开成 `_mrow<2>` / `_mrow<4>` 多行批量。

若类型 / 形状不满足回退到默认 `ggml_compute_forward_flash_attn_ext`。

---

## 7. 其它算子的 RVV/IME 加速点

`tensor_traits_common::compute_forward()` 里接管了：

| Op | 实现 | 备注 |
| --- | --- | --- |
| `GGML_OP_NORM` (f32) | `spacemit_kernels::rvv::forward_norm_f32` | RVV |
| `GGML_OP_RMS_NORM` (f32) | `rvv::forward_rms_norm_f32` | RVV |
| `GGML_OP_ADD/SUB/MUL/DIV` | `rvv::forward_binary<op, T>` (T=f32/f16) | RVV |
| `GGML_OP_FLASH_ATTN_EXT` | `forward_flash_attn_ext_f16` | RVV + TCM |
| `GGML_OP_CONT` (with permute) | `rvv::forward_cont_with_permute` | RVV |
| `GGML_OP_CPY` (with permute) | `rvv::forward_cpy_with_permute` | RVV |
| `GGML_OP_REPEAT` (rows=equal / dim1=1) | `rvv::forward_repeat_nrows<T>` / `forward_repeat_dim1<T>` | RVV |
| `GGML_OP_SUM_ROWS` | `rvv::forward_sum_rows<T>` | RVV |
| `GGML_OP_GET_ROWS` | `rvv::forward_get_rows<T>` | RVV |
| `GGML_OP_CONCAT` (dim=0) | `rvv::forward_concat<T>` | RVV |

未列出的 op（`GGML_OP_SOFT_MAX`、`GGML_OP_ROPE`、`GGML_OP_GLU` 等）直接落到默认 `ggml_compute_forward_*`（RVV intrinsics 版本，由 `ggml-cpu/vec.cpp` / `vec.h` 提供）。

---

## 8. 一条 token 的端到端时间线

```
用户 prompt + KV 初始空
        │
        ▼
llama_decode()  ─┐
                 │  while (ubatch):
                 ▼
        process_ubatch()
        ├─ model.build_graph()         ← 构造 ggml_cgraph (token→RMSNorm→QKV→RoPE→FlashAttn→Wo→+残差→RMSNorm→FFN→+残差)
        ├─ ggml_backend_sched_alloc_graph()
        │      ├─ 选 backend:  weight → SpacemiT IME backend (Q4_0/Q4_K…)
        │      │            activation → host backend (fp32)
        │      └─ 插入 GGML_OP_CPY 把 src1 从 host 拷到 SpacemiT buffer（如果需要）
        ├─ res->set_inputs(&ubatch)   ← 写 token id / position / seq_id
        └─ ggml_backend_sched_graph_compute_async()
              │
              ▼
        for each backend: compute_splits()
        for each node: ggml_compute_forward()
              ├─ SpacemiT worker 线程启动: set_numa_thread_affinity
              │     ├─ 绑定到 RISC-V AI 核
              │     ├─ 拿一块 TCM (spine_tcm)
              │     └─ pthread_setaffinity_np 锁核
              │
              ├─ MUL_MAT
              │     ├─ quantize_a_4row_i8_hp (RVV) — 4 行 int8 + scale
              │     ├─ memcpy b_col → TCM (ith%2==0 偶线程)
              │     ├─ spine_barrier_wait
              │     └─ gemm_kernel_i8i4_hp (IME2 vmadotsu/vmadotu.hp)
              │
              ├─ RMS_NORM / ADD / MUL / ROPE / FLASH_ATTN_EXT / GLU … (RVV)
              │
              └─ 节点结束: ggml_barrier(threadpool)
        │
        ▼
        logits 已写回 host buffer (dst->type == F32)
        │
        ▼
llama_sampler_sample() → token id
        │
        ▼
        loop next llama_decode
```

---

## 9. 关键文件地图

| 路径 | 行数 | 作用 |
| --- | --- | --- |
| `src/llama.cpp` | 581 | 公共 C API、模型加载入口 |
| `src/llama-context.cpp` | 4140 | `decode()`、`encode()`、`process_ubatch()`、`graph_compute()` |
| `src/llama-graph.cpp` | 3169 | `llm_graph_context`：norm/attn/ffn/moe 算子图节点拼装 |
| `src/llama-model.cpp` | 2713 | `llama_model::build_graph()`：调度到具体架构 |
| `src/models/*.cpp` | ~50 文件 | 每种 LLM_ARCH 自己的图构建 |
| `ggml/src/ggml.c` | – | 张量、算子定义、内存分配 |
| `ggml/src/ggml-backend.cpp` | – | `ggml_backend_sched_*`：图切分与调度 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 3840 | CPU 线程池、算子分派、work_size 计算 |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | 703 | CPU backend 框架、buffer type 注册 |
| `ggml/src/ggml-cpu/ops.cpp` | ~7000 | 默认算子 (RVV intrinsics) |
| `ggml/src/ggml-cpu/spacemit/ime.cpp` | 1740 | SpacemiT 后端入口、算子选择、TCM/Barrier 调度 |
| `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp` | 1027 | IME1: vmadot 内联汇编 GEMM |
| `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` | 5768 | IME2: vmadotsu / vmadotu.hp 内联汇编 GEMM |
| `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp` | 3178 | 标量/FlashAttn 的 RVV intrinsics 实现 |
| `ggml/src/ggml-cpu/spacemit/repack.cpp` | 1795 | 把 GGUF 量化权重重排成 IME 友好 N×K |
| `ggml/src/ggml-cpu/spacemit/spine_mem_pool.cpp` | 760 | 透明大页 / 1G hugetlb 内存池 |
| `ggml/src/ggml-cpu/spacemit/spine_tcm.h` | 409 | TCM 共享库 dlopen 加载头文件 |

---

## 10. 关键设计要点小结

1. **三层结构**：前端 LLM → 静态算子图 → 后端算子实现。后端只需要感知 ggml tensor / op，**不感知 LLM 架构**。
2. **图复用**：`res->can_reuse(gparams)` 命中时直接重跑同一图，省去重建 + 重分配的开销。这是 decode 阶段单 token 高吞吐的关键。
3. **后端无关性**：`tensor_traits` 抽象让任何 backend 可以 hook 进任意 op，并以比默认实现更优的方式完成（SpacemiT 通过它接走了几乎所有热点 op）。
4. **线程 + TCM 协同**：SpacemiT 走 `set_numa_thread_affinity` 把 worker 线程绑到 AI 核，并抢占式申请 TCM；`spine_barrier_t` 实现两线程（load/compute）的同步。
5. **量化分层**：
   - 加载时：`repack.cpp` 把通用 GGUF Q4_K/Q6_K/Q8_0… **一次性重排**为 N×K 排布。
   - 推理时：`quantize_a_*_i8*` 把 fp32 激活**在线量化**为 int8。
   - 计算时：IME 内核用 RISC-V V 扩展的 `vmadot` / `vmadotsu` / `vmadotu.hp` 完成 int8×int4→int32 的乘加。
6. **可观测性**：`tensor->extra` 字段在 `init_tensor` 时被设为 `tensor_traits*`，`repack` 时被调用并完成一次性重排；`repack` 失败时 `ggml_assert` 阻断。
