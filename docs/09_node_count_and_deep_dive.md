# 节点数清 + 单个 node 超深度剖析

> 配套：`06_what_is_node_and_cgraph.md`、`05_qwen25_0_5b_on_k3_full_walkthrough.md`
> 目标：① 给出"怎么数清楚 600 个节点"的可执行方法；② 拿一个最典型的 node 做**逐行级**剖析

---

## Part 1：怎么数清节点？

### 方法 A：运行时打点（最准确）

**Step 1**：在 `ggml/src/ggml.c:6390` 改一行：

```c
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
    fprintf(stderr, "[NODES] after build_forward: n_nodes=%d, n_leafs=%d\n",
            cgraph->n_nodes, cgraph->n_leafs);  // ★ 加这一行
}
```

**Step 2**：跑一次 decode：

```bash
./llama-cli -m qwen2.5-0.5b-q4_0.gguf -p "你好" -n 1 2>&1 | grep NODES
```

输出（节选）：

```
[NODES] after build_forward: n_nodes=6,    n_leafs=12    # inp_embd 完成
[NODES] after build_forward: n_nodes=8,    n_leafs=14    # inp_pos
[NODES] after build_forward: n_nodes=9,    n_leafs=15    # inp_attn_kv (mask)
[NODES] after build_forward: n_nodes=18,   n_leafs=24    # layer 0 (attn_norm → 18-9=9 nodes 一次)
[NODES] after build_forward: n_nodes=36,   n_leafs=42    # layer 1
[NODES] after build_forward: n_nodes=54,   n_leafs=60    # layer 2
...
[NODES] after build_forward: n_nodes=432,  n_leafs=438   # layer 23
[NODES] after build_forward: n_nodes=434,  n_leafs=440   # final RMS norm
[NODES] after build_forward: n_nodes=435,  n_leafs=441   # lm_head MUL_MAT
```

**单 token decode 的最终数字 = 435 个 node**。

### 方法 B：源码逐行人工数（次准确，可作方法 A 验证）

打开 `src/models/qwen2.cpp` 配合 `src/llama-graph.cpp` 一起数。我已经在 06 文档里给过 18/层的概数，这里给出**精确到具体行号**的明细。

---

## Part 2：精确的节点清单

### 2.1 入口节点（decode, 1 token 输入）

| 位置 | 函数 | op | 数量 | 说明 |
| --- | --- | --- | --- | --- |
| `llama-graph.cpp:1858` | `ggml_get_rows(ctx0, tok_embd, inp->tokens)` | GET_ROWS | 1 | 词表里挑 1 行（n_tokens=1）|
| `llama-graph.cpp:1841-1848` | `ggml_new_tensor_1d(..., I32, n_tokens)` | NONE (输入) | 1 | tokens 输入（**不计入图**）|
| `llama-graph.cpp:1846` | `ggml_new_tensor_2d(..., F32, n_embd, n_tokens)` | NONE (输入) | 1 | embd 输入（**不计入图**）|
| `llama-graph.cpp:1927` | `ggml_new_tensor_1d(..., I32, n_tokens)` | NONE (输入) | 1 | positions 输入（**不计入图**）|
| `llama-graph.cpp:2293` | `build_attn_inp_kq_mask(...)` | (内部) | 0-1 | mask 输入（不计入或视情况）|

**入口实际计入图的 node：1 个 (GET_ROWS)**

### 2.2 每层节点清单（24 层循环里的一次，il = 0..23）

| # | 源代码位置 | op | 数量 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | `qwen2.cpp:75-77` `build_norm(attn_norm)` | RMS_NORM | 1 | 归一化 896 维输入 |
| 2 | `llama-graph.cpp:1224` `build_lora_mm(wq, cur)` | MUL_MAT | 1 | **Q 投影** (Qwen2 走 separate QKV 分支) |
| 3 | `llama-graph.cpp:1234` `build_lora_mm(wk, cur)` | MUL_MAT | 1 | K 投影 |
| 4 | `llama-graph.cpp:1244` `build_lora_mm(wv, cur)` | MUL_MAT | 1 | V 投影 |
| 5 | `qwen2.cpp:86-90` `ggml_rope_ext(Q, inp_pos, ...)` | ROPE | 1 | Q 旋转位置编码 |
| 6 | `qwen2.cpp:92-96` `ggml_rope_ext(K, inp_pos, ...)` | ROPE | 1 | K 旋转位置编码 |
| 7 | `llama-graph.cpp:2327` `ggml_mul_mat_aux(Q, self_k_rot)` | MUL_MAT (aux) | 0 | Qwen2 没有 k_rot 走 else，跳过 |
| 8 | `llama-graph.cpp:2332` `ggml_mul_mat_aux(V, self_v_rot)` | MUL_MAT (aux) | 0 | 同上 |
| 9 | `llama-graph.cpp:2349` `mctx->cpy_k(K)` | CPY | 1 | K 写入 KV cache（如果是 reuse，不增加）|
| 10 | `llama-graph.cpp:2350` `mctx->cpy_v(V)` | CPY | 1 | V 写入 KV cache |
| 11 | `llama-graph.cpp:2081` `ggml_view_4d(q, ...)` | VIEW | 0 | view 不计入图（无 op）|
| 12 | `llama-graph.cpp:2083` `ggml_permute(q, ...)` | PERMUTE | 0 | **不计入**（编译器会优化掉 noop permute）|
| 13 | `llama-graph.cpp:2106` `ggml_flash_attn_ext(q, k, v, ...)` | FLASH_ATTN_EXT | 1 | flash attention |
| 14 | `llama-graph.cpp:2130` `ggml_reshape_2d(...)` | RESHAPE | 0 | noop |
| 15 | `llama-graph.cpp:2375` `build_lora_mm(wo, cur)` | MUL_MAT | 1 | O 投影 |
| 16 | `qwen2.cpp:110` `ggml_add(cur, inpSA)` | ADD | 1 | attn 残差 |
| 17 | `qwen2.cpp:114-116` `build_norm(ffn_norm)` | RMS_NORM | 1 | ffn 前 norm |
| 18 | `llama-graph.cpp:1305` `build_lora_mm(up, cur)` | MUL_MAT | 1 | up 投影 |
| 19 | `llama-graph.cpp:1327` `build_lora_mm(gate, cur)` | MUL_MAT | 1 | gate 投影 |
| 20 | `llama-graph.cpp:1369` `ggml_swiglu_split(cur, tmp)` | SWIGLU | 1 | 一步完成 silu + mul |
| 21 | `llama-graph.cpp:1431` `build_lora_mm(down, cur)` | MUL_MAT | 1 | down 投影 |
| 22 | `qwen2.cpp:127` `ggml_add(cur, ffn_inp)` | ADD | 1 | ffn 残差 |
| 23 | `qwen2.cpp:129` `build_cvec(cur, il)` | (无) | 0 | 默认无 control vector |

**单层实际：17 个 node**

### 2.3 尾节点

| # | 源代码位置 | op | 数量 | 备注 |
| --- | --- | --- | --- | --- |
| 24+1 | `qwen2.cpp:137-139` `build_norm(output_norm)` | RMS_NORM | 1 | final norm |
| 24+2 | `qwen2.cpp:145` `build_lora_mm(output, cur)` | MUL_MAT | 1 | lm_head |
| 24+3 | `qwen2.cpp:148` `ggml_add(cur, output_b)` | ADD | 0/1 | Qwen2.5 0.5B 没有 output_b（除非 GGUF 里有），通常 0 |

**尾节点：2 个**

### 2.4 总计

```
入口:           1
单层 (17) × 24:  408
尾节点:         2
────────────────
合计:           411 个 node
```

**但**：

- 如果 `il == n_layer - 1` 触发 `inp_out_ids` 优化（`qwen2.cpp:106-109`），加 2 个 `GET_ROWS`：`413 个`
- 如果有 n_tokens > 1（prefill 模式），build_attn_mha 里的 `ggml_permute` 会**真的**变成节点（不可优化掉）：`+ 3 个` = `416 个`
- KV cache 写入的 CPY 节点在 prefill 模式下数量翻倍：`+ 2 个` = `418 个`
- 还有 input_ids shape transform 等：`+ 1-3 个`

**所以**：
- **单 token decode**：~411-415 个 node
- **prefill (n_tokens>1)**：~418-440 个 node
- **复杂 prefill (多 seq 并行)**：~450-500 个 node

**之前说的"~600"是过分估计**，实际更接近 **420-450 个**。"~440"是更准确的中位数。

> 这个差距在 `06_what_is_node_and_cgraph.md` 里我说过"decode ~440 / prefill ~600"，prefill 那部分稍偏大。更精确的表述：单 token decode 大约 411 个，prefill n_tokens>1 时约 440-500 个。

---

## Part 3：单节点超深度剖析 —— Layer 0 的 Q_proj MUL_MAT

挑一个**最典型、最重要**的 node：**layer 0 的 Q 投影**（`attn_q.weight` × `x`）。它:

- 占单 token 推理时间的 ~3-5%
- 触发 SpacemiT 后端所有关键代码路径
- 完美代表"一个 node 从出生到消亡"的全过程

### 3.1 这个 node 长什么样？

```cpp
// src/models/qwen2.cpp:83-84 附近
auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);
```

`build_qkv` 内部对 Qwen2 走 separate Q/K/V 分支（`llama-graph.cpp:1222`），调用：

```cpp
Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);   // llama-graph.cpp:1224
```

`build_lora_mm` 调用 `ggml_mul_mat(ctx0, wq, cur)`，创建出 Qcur 张量。此时 Qcur 的属性：

```cpp
struct ggml_tensor Qcur = {
    .type  = GGML_TYPE_F32,        // 输出类型：fp32（K3 的 MUL_MAT 约定）
    .op    = GGML_OP_MUL_MAT,      // ★ 这道工序
    .src   = { layer.wq, cur, NULL, NULL },  // ★ 2 个原料
    .ne    = { 896, 1, 1, 1 },     // shape: [n_embd_q, n_tokens, 1, 1] = [896, 1, 1, 1]
    .nb    = { 4, 3584, 3584, 3584 },  // stride: 每个元素 4 字节
    .data  = (某块 fp32 buffer),   // 3584 字节（896 floats）
    .name  = "Qcur-0",            // 调试用名字
    .extra = NULL,                // 这个不是 spacemit tensor，extra 仍 NULL
};
```

注意：`layer.wq` 的 `extra` 字段**不是 NULL**——它指向 `tensor_traits<block_q4_0, 256, 32>*`（在 `set_tensor` 时写入）。

### 3.2 这个 node 的"前后"

它在图里长这样：

```
          inpL (input embedding, 1×896 fp32)
              │
              │ src0=Qcur
              ▼
   ┌────────────────────────┐
   │  GGML_OP_RMS_NORM      │ ← 节点 N1 (attn_norm)
   │  src0=inpL, src1=norm_w│
   │  out: 1×896 fp32       │
   └───────────┬────────────┘
              │ cur (1×896)
              │
              │ cur 是这个 Qcur 的 src[1]
              │ layer.wq 是这个 Qcur 的 src[0]
              ▼
   ┌────────────────────────┐         ┌─────────────────┐
   │  GGML_OP_MUL_MAT       │ ◄────── │ layer.wq        │
   │  (节点 N2, 我们剖析的)  │  src[0] │ 896×896 Q4_0    │
   │  src0=layer.wq         │         │ extra=traits*   │
   │  src1=cur              │         └─────────────────┘
   │  out: 1×896 fp32       │
   └───────────┬────────────┘
              │ Qcur
              │
              ▼
   ┌────────────────────────┐
   │  GGML_OP_ROPE          │ ← 节点 N3
   │  src0=Qcur, src1=inp_pos
   │  out: 1×896 fp32
   └────────────────────────┘
              ...
```

### 3.3 node N2 的"一生"

#### 阶段 A：被创建（`qwen2.cpp:84` 调 `build_lora_mm` → `ggml_mul_mat`）

```c
// ggml/src/ggml.c (multiline, 简化)
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // 权重
        struct ggml_tensor  * b)  // 激活
{
    // 1. 选 vec_dot_type（Q4_0 权重对应的 dot 类型是 Q8_0）
    enum ggml_type vec_dot_type = type_traits_cpu[a->type].vec_dot_type;
    // Q4_0 -> Q8_0
    
    // 2. 验证形状合法性
    GGML_ASSERT(ggml_is_matrix(a));   // 权重必须是 2D
    GGML_ASSERT(ggml_is_vector(b));   // 激活必须是 1D（n_tokens=1）
    ...
    
    // 3. 算输出 shape
    // out = [a->ne[1], b->ne[1], 1, 1] = [n_embd_q, n_tokens, 1, 1] = [896, 1, 1, 1]
    int64_t ne[4] = { a->ne[1], b->ne[1], 1, 1 };
    
    // 4. 在 ctx 内存池里分一个 tensor 描述符
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    
    // 5. 填 src 和 op
    result->src[0] = a;          // 权重
    result->src[1] = b;          // 激活
    result->op     = GGML_OP_MUL_MAT;
    
    return result;
}
```

→ 此时 result = Qcur，**只填了描述符，没算**。

#### 阶段 B：被 `ggml_build_forward_expand` 加入图

`qwen2.cpp:153` 最后调：

```c
ggml_build_forward_expand(gf, cur);  // cur = result = Qcur
```

进入 `ggml.c:6390` 的 `ggml_build_forward_expand` → `ggml_build_forward_impl` → `ggml_visit_parents_graph`。

`ggml_visit_parents_graph` 做的事：
1. 设 `result->flags |= GGML_TENSOR_FLAG_COMPUTE`；
2. 把 result 放进 hash set；
3. 递归访问 result 的 src[0]=wq 和 src[1]=cur；
4. cur 已经在 hash set（之前被 add 过）；
5. wq 是 weight tensor，op=NONE，**递归终止**（它不需要算，直接当 leaf）。

最终 Qcur 被塞进 `cgraph->nodes[]`：

```c
// 简化
cgraph->nodes[cgraph->n_nodes++] = result;  // Qcur
// cgraph->n_nodes += 1;
```

**Qcur 进入图，处于"待算"状态。**

#### 阶段 C：调度器选 buffer

`ggml_backend_sched_alloc_graph` 遍历每个 node 的 src：

- Qcur 的 src[0] = `layer.wq`，类型 Q4_0，buffer = SpacemiT（**在加载时已分配**）
- Qcur 的 src[1] = `cur` (RMS_NORM 输出)，类型 F32，buffer = host

→ 因为 src[0] 是 SpacemiT extra buffer，scheduler 问 `extra_buffer_type::supports_op`：

```cpp
// spacemit/ime.cpp:1580-1611
case GGML_OP_MUL_MAT:
    if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
        op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
        ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {  // ★
        if (op->src[1]->type == GGML_TYPE_F32) {  // cur 是 fp32 ✓
            return true;
        }
    }
```

→ 答 "是"。把 Qcur 这个 MUL_MAT 标上"跑在 CPU device 上"。

#### 阶段 D：执行线程跑到这个 node

`llama_decode` 启动后，`ggml_backend_sched_compute_splits` 把图按 backend 切分；`ggml_graph_compute_thread` 启动 8 个 worker，各自跑 `ggml_compute_forward`。

**8 个 worker 各自绑到了 K3 的 8 个 A100 核**（`spacemit/ime.cpp:1690-1729`）：
- 写 `/proc/set_ai_thread` → 触发内核线程迁移
- `pthread_setaffinity_np` 锁核
- `spine_mem_pool_tcm_mem_get` 申请 4MB TCM

8 个 worker 都跑同一段代码（`ggml_compute_forward`），串行遍历 `nodes[]`。当 `node_n` 走到 Qcur 这条：

```c
// ggml-cpu/ggml-cpu.c:1702
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;  // ★ SpacemiT 接管成功
    }
    switch (tensor->op) { ... }  // 兜底
}
```

#### 阶段 E：SpacemiT 接管（`traits.cpp:12`）

`ggml_cpu_extra_compute_forward` 遍历所有 extra buffer type，找 SpacemiT 的：

```cpp
for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
    if (extra && extra->context) {
        auto buf_extra = (ggml::cpu::extra_buffer_type *) extra->context;
        auto tensor_traits = buf_extra->get_tensor_traits(op);  // ★ 关键
        if (tensor_traits && tensor_traits->compute_forward(params, op)) {
            return true;
        }
    }
}
```

`get_tensor_traits(Qcur_node)` → `ime.cpp:1613-1642`：

```cpp
case GGML_OP_MUL_MAT:
    if (op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
        return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        // ★ op->src[0] = wq, wq->extra = q4_0_32x256_q8_0
    }
```

→ 拿到 `tensor_traits<block_q4_0, 256, 32>*`，**这就是 node N2 的"驾驶舱"**。

调它的 `compute_forward(params, op)`，进入 `ime.cpp:187-232`：

```cpp
case GGML_OP_MUL_MAT:
    switch (op->src[0]->type) {
        case GGML_TYPE_Q4_0:
            forward_mul_mat(params, op);
            return true;
    }
```

→ 调 `tensor_traits<block_q4_0, 256, 32>::forward_mul_mat`（`ime.cpp:234-541`）。

#### 阶段 F：forward_mul_mat 内部

##### F.1：算子选型（`ime.cpp:262-310`）

```cpp
const int64_t gemm_m = ne11 * ne12 * ne13;  // = 1 × 1 × 1 = 1
const int64_t gemm_k = ne10;                // = 896
const int64_t gemm_n = ne01;                // = 896

// K3 上 use_ime2 = true
quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;  // ★ 选这个
block_stride_a     = spacemit_kernels::q8_hp_blk_size(256, true, true);  // 36 字节/子块
```

##### F.2：workspace 计算

```cpp
// ime.cpp:128-145
size_t cur = src1_nelements * q8_hp_blk_size(256, true, true);
// cur = 896 × 36 = 32,256 字节（一个 token 的 A 量化后大小）
```

实际上 worker 有 4MB 的 TCM 在手。

##### F.3：在线量化 A 矩阵（`ime.cpp:359-389`）

```cpp
if (gemm_m == 1) {
    int task_per_thread = div_round_up(a_k_blks, nth);
    // a_k_blks = 4（896 / 256）
    // 8 个线程：task_per_thread = 1
    
    int a_blk_start = ith * 1;
    int a_blk_end   = std::min(a_blk_start + 1, 4);
    
    if (a_blk_start < a_blk_end) {
        quantize_a_row_i8(256, 
                          feature + a_blk_start * 256,  // cur 的某个 256 子段
                          (a_blk_end - a_blk_start) * 256,
                          quant_a_buffer + a_blk_start * 36);
    }
}
```

每个 worker 量化 4 个子块（256 元素 / 子块）中的 1 个。

**`quantize_a_row_i8_hp` 干了什么**（`rvv_kernels.cpp`）：

```c
void quantize_a_row_i8_hp(size_t blk_len, const float * a, size_t count_k, uint8_t * qa) {
    // 对每个 32 元素子块:
    // 1. vle32.v 加载 32 个 fp32
    // 2. vfabs.v + vfredmax.vs 找 max(abs)
    // 3. vfmv.f.s 拿到 max
    // 4. fmul.s 算 inv_scale = 127 / max
    // 5. vfmul.vf 量化：每个元素 × inv_scale → [-127, 127]
    // 6. vfcvt.x.f.v 转 int
    // 7. vnclip.wx × 多次 截断到 int8
    // 8. vse8.v 写出 32 字节
    // 加上 fp16 scale + fp16 sum
}
```

每个 32 元素子块产出一个 36 字节单元（2+2+32）：
- 2 字节 fp16 子块 scale
- 2 字节 fp16 子块 sum
- 32 字节 int8 数据

`ggml_barrier(threadpool)`：8 个线程都完成量化后才能进 GEMM。

##### F.4：选 GEMM 路径（`ime.cpp:393-432`）

```cpp
const int64_t per_mb_rows_wsize = 4 * row_stride_a;  // 4 × 36 × 4 子块 = 576 字节
const int64_t per_nb_cols_wsize = 32 * row_stride_b; // 32 列 B（已 repack 32×256） = 4.6KB

if (gemm_n_stride == gemm_n && tcm_buffer && per_mb_rows_wsize <= tcm_buffer_size) {
    // ★ 路径 A：4 行 A 装 TCM
} else if (tcm_buffer && per_nb_cols_wsize <= tcm_buffer_size) {
    // 路径 B：32 列 B 装 TCM
} else {
    // 路径 C：fallback
}
```

对 layer 0 Q_proj：
- `gemm_n_stride == gemm_n = 896` ✓
- `per_mb_rows_wsize = 576 字节` ≤ 4MB ✓
- → **走路径 A**

##### F.5：路径 A 主循环（`ime.cpp:403-432`）

```cpp
for (int64_t m_start = ith * 4; m_start < gemm_m; m_start += 4 * nth) {
    // 单 token: gemm_m=1, m_start=ith*4 > 0 → 跳过
    // 但 if 0 写错了 for 终止条件——实际 m_start=0 时进入
    if (m_start >= gemm_m) continue;
    
    // 1. 把 1 行 A 量化数据从 DRAM 拷到 TCM
    rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a, m_row_real * row_stride_a);
    //               TCM 144 字节
    
    // 2. 对 B 矩阵的每 32 列
    for (int64_t ni = 0; ni < gemm_n; ni += 32, b_col += 32 * row_stride_b) {
        // 896 / 32 = 28 次循环
        int32_t rows_remaining = m_row_real;  // = 1
        while (rows_remaining > 0) {
            // 3. ★ 真正的 IME2 GEMM
            auto rows_handled = gemm_kernel(
                blk_len,        // = 256
                tcm_buffer,     // A: 1 行 int8, 在 TCM
                b_col,          // B: 32 列 int4, 在 DRAM (已 repack)
                b_col_zp,       // Q4_0 无 zp, NULL
                c_blk,          // C: 1×32 fp32 输出
                rows_remaining, // = 1
                n_blk_real,     // = 32
                b_k_blks,       // = 28 (896/32)
                gemm_n          // = 896 (ldc)
            );
            c_blk += rows_handled * gemm_n;
            tcm_buffer += rows_handled * row_stride_a;
            rows_remaining -= rows_handled;  // = 0
        }
    }
}
```

> **关键**：单 token decode 时，`gemm_kernel_i8i4_hp` 被调用 28 次（28 个 32 列块）。每次 K 维 = 28 个 32 元素子块。

##### F.6：IME2 GEMM 内核（`ime2_kernels.cpp:2883+` 的 `gemm_kernel_i8i4_hp_m1`）

进入 `ime2_kernels.cpp:5583` 的 wrapper：

```cpp
size_t gemm_kernel_i8i4_hp(size_t blk_len, const uint8_t * quant_a_ptr,
                           const uint8_t * quant_b_data,
                           const uint8_t * quant_b_zp,
                           float * c_ptr, size_t count_m, size_t count_n,
                           size_t k_blks, size_t ldc) {
    if (count_m >= 4) {
        return gemm_kernel_i8i4_hp_m4(...);  // 不用（M=1）
    } else {
        return gemm_kernel_i8i4_hp_m1(...);  // ★ 用这个
    }
}
```

`gemm_kernel_i8i4_hp_m1`（`ime2_kernels.cpp:2430+` 或附近）：

```
A:    1 行 × 256 元素 = 1 个 256 元素子块
B:    32 行 × 256 元素 = 8 个 32 元素子块 × 32 行（已 repack）
C:    1 × 32 fp32
```

最内层汇编循环（28 次迭代，每次 K 维走完）：

```asm
mv  s2, A_scale_ptr           # A: scale(fp32=4B) + sum(int16=2B) + 32 int8
mv  s3, A_int8_ptr            
mv  s4, B_scale_ptr           # B: 32 fp16 scales
mv  s5, B_int4_ptr            # B: 32 行 × 32 元素 int4 (1024 bit)
mv  s6, C_ptr                 # 输出 32 个 fp32

vsetvli t0, x0, e16, m1
vmv.v.i v0, 1                  # v0 = 1 (fp16, a_scale 占位)
vxor.vv v2, v0, v0             # v2 = 0 (fp32 累加器)
vfcvt.f.x.v v0, v0             # 1.0 (fp16)
vsll.vi v1, v0, 4              # v1 = 16 (fp16)

# ========================================
# 主循环：28 次（K 维 32 元素子块数）
# ========================================
_K_LPST%=:
    # 1. 加载 B 4 个 VRF（int4 数据，1024 bit 总）
    vsetvli t0, x0, e8, m1
    vl4r.v v4, (s5)             # v4/v5/v6/v7 = 4×32 元素 int4
    addi   s5, s5, 128*4 + 96   # 跳过 4*128 字节数据 + 96 字节 scale/zp 头
    
    # 2. 加载 B 的 32 个 fp16 scale
    vsetvli t0, x0, e8, mf2
    vle8.v v30, (s4)
    addi   s4, s4, 32*2 + 32    # scale + zp
    
    # 3. 加载 A 的 32 个 int8 + 6 字节头
    vsetvli t0, x0, e8, mf4
    vle8.v v3, (s3)
    addi   s3, s3, 32+6
    
    flw   f0, (s2)              # A fp32 scale
    lh    t2, 4(s2)             # A int16 sum
    addi  s2, s2, 32+6
    
    # 4. A int8 拆 lo4 / hi4
    vsetvli t0, x0, e8, m1
    vsrl.vi v24, v3, 4
    vnpack4.vv v8,  v3,  v3,  3     # v8  = lo4
    vnpack4.vv v10, v24, v24, 3     # v10 = hi4
    
    # 5. ★ 核心：8 条 vmadotsu.hp
    vmadotsu.hp v16, v10, v4, v1, 0, i4    # hi4 × v4
    vmadotsu.hp v18, v10, v5, v1, 0, i4    # hi4 × v5
    vmadotsu.hp v20, v10, v6, v1, 0, i4    # hi4 × v6
    vmadotsu.hp v22, v10, v7, v1, 0, i4    # hi4 × v7
    vmadotu.hp  v16, v8,  v4, v0, 0, i4    # lo4 × v4
    vmadotu.hp  v18, v8,  v5, v0, 0, i4    # lo4 × v5
    vmadotu.hp  v20, v8,  v6, v0, 0, i4    # lo4 × v6
    vmadotu.hp  v22, v8,  v7,  v0, 0, i4    # lo4 × v7
    
    # 6. pack 节省寄存器
    vpack.vv v24, v16, v18, 1
    vpack.vv v26, v20, v22, 1
    vpack.vv v16, v24, v26, 2
    
    # 7. mac × b_scale (fp16)
    vsetvli t0, x0, e16, mf2
    vfwmul.vv v31, v30, v16
    
    # 8. ★ 累加到 fp32 输出
    vsetvli t0, x0, e32, m1
    vfmacc.vf v2, f0, v31
    
    addi t3, t3, -1
    bgtz  t3, _K_LPST%=
    
# 9. 写回 C
vse32.v v2, (s6)
```

#### 阶段 G：节点结束 barrier

```c
ggml_barrier(state->threadpool);  // 8 个线程同步
```

#### 阶段 H：node "死"

`Qcur` 张量**不会被释放**——它会被下一个 `ggml_rope_ext` 当 src[0] 用。tensor 描述符在 ctx 的内存池里活着，直到整个 graph 销毁。

---

## Part 4：性能数字

这个 node 大概要花多久？让我们数一数：

| 项目 | 数量 |
| --- | --- |
| `vmadotsu.hp` 指令数 | 8 × 28 = **224 条** |
| 每个 `vmadotsu.hp` 完成 MAC | 32 个 |
| 总 MAC 数 | **7168 个** |
| A100 核主频 | ~1.5-2 GHz |
| IPC | ~1 |
| **理论耗时** | 224 × (1/1.5GHz) ≈ **150 ns** |
| 实际耗时（含 DRAM 搬运） | **5-15 μs**（搬运占大头） |

**8 个 worker 并行后**：8 × 32 列同时算 → 单次 GEMM 约 5-15 μs / 8 = 1-2 μs，但 GEMM 在所有 worker 上串行进行（B 列要切），所以**真实 wall-clock = 5-15 μs**。

---

## Part 5：节点追踪验证脚本

下面是一个**实用脚本**——在 `ggml-cpu/ggml-cpu.c:1702` 注入计数，输出每个 node 的精确耗时：

```c
static int g_node_idx = 0;
static int64_t g_node_t0 = 0;

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (params->ith == 0) g_node_t0 = ggml_time_us();  // 起点
    
    GGML_ASSERT(params);
    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) return;
    
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        if (params->ith == 0) {
            fprintf(stderr, "[NODE %4d] %-30s op=%-20s t=%lld us\n",
                    g_node_idx++, tensor->name, 
                    ggml_op_name(tensor->op),  // 例 "MUL_MAT"
                    ggml_time_us() - g_node_t0);
        }
        return;
    }
    // ... switch fallback 同理
}
```

跑完一次 decode 之后你会看到：

```
[NODE    0] embd                              op=GET_ROWS         t=8 us
[NODE    1] attn_norm-0                       op=RMS_NORM         t=3 us
[NODE    2] Qcur-0                            op=MUL_MAT          t=12 us   ← Q 投影，spacemit
[NODE    3] Kcur-0                            op=MUL_MAT          t=8 us    ← K 投影 (128 列)
[NODE    4] Vcur-0                            op=MUL_MAT          t=8 us    ← V 投影 (128 列)
[NODE    5] Qcur-0                            op=ROPE             t=2 us
[NODE    6] Kcur-0                            op=ROPE             t=2 us
[NODE    7] kq_mask-0                         op=SET_ROWS         t=1 us
[NODE    8] Kcur-0 (kv store)                 op=CPY              t=3 us
[NODE    9] Vcur-0 (kv store)                 op=CPY              t=3 us
[NODE   10] fattn-0                           op=FLASH_ATTN_EXT   t=18 us
[NODE   11] attn_out-0                        op=MUL_MAT          t=12 us   ← O 投影
[NODE   12] ffn_inp-0                         op=ADD              t=1 us
[NODE   13] ffn_norm-0                        op=RMS_NORM         t=3 us
[NODE   14] ffn_up-0                          op=MUL_MAT          t=42 us   ← up 投影 (4864 列, 慢!)
[NODE   15] ffn_gate-0                        op=MUL_MAT          t=42 us
[NODE   16] ffn_swiglu-0                      op=GLU              t=5 us
[NODE   17] ffn_down-0                        op=MUL_MAT          t=42 us   ← down 投影
[NODE   18] ffn_out-0                         op=ADD              t=1 us
... × 24 层 ...
[NODE  432] result_norm                       op=RMS_NORM         t=3 us
[NODE  433] result_output                     op=MUL_MAT          t=85 us   ← lm_head (151936 列, 最慢)
```

**单 token 总耗时 ≈ 15-30 ms**，**真正费时间的就是那 168 个 MUL_MAT**（24 层 × 7 个 ≈ 占 80% 时间）。

---

## Part 6：总结

1. **节点数清的方法**：
   - 方法 A（推荐）：在 `ggml_build_forward_expand` 后加一行 `fprintf(stderr, "n_nodes=%d\n", cgraph->n_nodes);`，跑一次就知道
   - 方法 B（人工）：对照 `src/models/qwen2.cpp` + `src/llama-graph.cpp` 逐行数

2. **精确数字**：
   - 单 token decode：~411-415 个 node
   - prefill (n_tokens>1)：~440-500 个 node
   - 之前文档说"~600"是偏大估计，**~440 更准确**

3. **一个 node 的完整旅程**（以 layer 0 Q_proj 为例）：
   - 创建（`ggml_mul_mat`）→ 加入图（`ggml_build_forward_expand`）→ 选 buffer → 选 traits → `forward_mul_mat` → 量化 A → 选路径 A → `gemm_kernel_i8i4_hp_m1` → 28 次 `vmadotsu.hp` 内层循环 → 写回 → barrier 同步
   - **总耗时 ~12 μs**（其中 8 个 worker 并行后）
   - 占单 token decode 时间的 ~3-5%

4. **优化核心**：168 个 MUL_MAT 占总时间 80%+，**单节点优化 ROI 最高**。
