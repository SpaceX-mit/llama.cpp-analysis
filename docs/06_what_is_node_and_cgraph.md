# 什么是 "node" 和 "ggml_cgraph"？

> 给第一次接触 LLM 推理框架的同学。
> 配套文档：`05_qwen25_0_5b_on_k3_full_walkthrough.md`

---

## 0. 一句话类比

把 LLM 推理想象成 **"做一道菜"**：

- **node（节点）** = 一道**工序**（比如"切肉"、"放盐"、"炒 3 分钟"）
- **ggml_cgraph（计算图）** = 一张**菜谱**（一张写好"先做什么后做什么"的流程图）
- **tensor（张量）** = 工序之间传递的**食材**（肉、调料、盘子…）

**~600 个节点 = 这道菜有 600 道工序**（对一个 24 层的 Qwen2.5 0.5B 模型而言）。

---

## 1. node（节点）到底是什么？

源码层面，node 其实就是一个 **`ggml_tensor` 结构体**。让我们看看它长什么样（`ggml/include/ggml.h`）：

```cpp
struct ggml_tensor {
    enum ggml_type   type;       // 数据类型（F32 / F16 / Q4_0 / Q8_0...）
    enum ggml_op     op;         // 这道工序是"干啥"的（MUL_MAT / RMS_NORM / ADD ...）
    int32_t          flags;      // 标志位（要不要算 / 算在哪个 backend 上）

    struct ggml_tensor * src[GGML_MAX_SRC];   // 这道工序的"输入食材"（最多 4 个）
                                                     //  ← 关键！通过指针指向其他节点的输出

    int64_t   ne[GGML_MAX_DIMS]; // 这道工序的输出"盘子"形状（如 [1, 896]）
    size_t    nb[GGML_MAX_DIMS]; // 每个维度的字节 stride
    void    * data;              // 输出"盘子"在内存里的地址
    size_t    buffer_size;       // 这道工序需要的工作 buffer 大小

    char      name[GGML_MAX_NAME]; // 工序名字（调试用，如 "Qcur"）
    void    * extra;             // 后端私有的扩展指针（SpacemiT 用它存 tensor_traits*）
};
```

> **关键洞察**：一个 `ggml_tensor` **既是一道工序**，**也是一个装结果的盘子**。`op` 字段说"算什么"，`data` 字段说"算完放哪"，`src[]` 字段说"原料从哪来"。

### 几个具体的 node 例子

| node 的 op 字段 | 含义 | src 依赖 | 输出 |
| --- | --- | --- | --- |
| `GGML_OP_GET_ROWS` | "从词表里挑几行" | src0=词表大表, src1=token id | 当前 token 的 embedding |
| `GGML_OP_RMS_NORM` | "RMSNorm 归一化" | src0=输入向量, src1=norm 权重 | 归一化后的向量 |
| `GGML_OP_MUL_MAT` | "矩阵乘" | src0=权重矩阵, src1=输入向量 | 输出向量 |
| `GGML_OP_ROPE` | "位置编码旋转" | src0=Q/K, src1=位置 id | 旋转后的 Q/K |
| `GGML_OP_FLASH_ATTN_EXT` | "注意力分数" | src0=Q, src1=K, src2=V, src3=mask | attention 输出 |
| `GGML_OP_ADD` | "两个向量相加" | src0=A, src1=B | A+B（残差连接） |

> **每个 node 都自给自足**——你只要给它 src 和 output buffer，它就能独立完成工作。

---

## 2. ggml_cgraph（计算图）是什么？

源码定义在 `ggml/include/ggml.h`，简化版：

```cpp
struct ggml_cgraph {
    int     n_nodes;                 // 一共有多少个节点
    struct ggml_tensor ** nodes;      // 节点数组（按拓扑序）
                                     //   nodes[0] 是第一个要算的
                                     //   nodes[1] 是第二个
                                     //   ...
                                     //   nodes[n_nodes-1] 是最后一个（通常是 logits）

    int     n_leafs;                 // "叶子"节点数（输入张量，不需算）
    struct ggml_tensor ** leafs;     // 叶子数组（词表、权重、token id 输入...）

    int     n_threads;               // 并行线程数
    struct ggml_context * ctx;       // 内存池
    // ...
};
```

**核心要点**：
- `nodes` 是一个**数组**（不是链表），按**拓扑序**（保证 src 在被使用前已经算好）排好。
- **节点之间的关系靠 `src[]` 指针表达**——A 的 src 指向 B 的 tensor，就表示"A 依赖 B"。
- 因此整张图是**有向无环图（DAG）**。

### 节点的"生成"过程

写代码时用 `ggml_mul_mat(ctx, weight, input)` 这种工厂函数创建节点。这些函数只是：
1. 在 ctx 的内存池里分一个 `ggml_tensor`；
2. 填好 `op` / `src[]` / `ne` / `name`；
3. **不立刻算**——只是"挂"到 ctx 里。

最后调 `ggml_build_forward_expand(cgraph, final_output)`，从 final_output 出发 **DFS 遍历所有 src**，把所有"会被用到"的节点都按拓扑序塞进 `cgraph->nodes[]`。这一步是**只读**——不动数据。

### 一个非常小的例子

```cpp
// 写一个 1+2*3 的图
ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, 1); a->data = (void*)&one;
ggml_tensor * b = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, 1); b->data = (void*)&two;
ggml_tensor * c = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, 1); c->data = (void*)&three;

ggml_tensor * bc = ggml_mul(ctx, b, c);   // 节点 1: op=MUL, src=[b,c]
ggml_tensor * r  = ggml_add(ctx, a, bc);  // 节点 2: op=ADD, src=[a,bc]

ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, r);          // 遍历后 gf->n_nodes=2
```

`gf->nodes[]` 排好序：

```
nodes[0] = bc  (MUL, 2*3=6)
nodes[1] = r   (ADD, 1+6=7)
```

执行时按这个顺序算，结果是 7。

---

## 3. 为什么 24 层 Qwen2.5 0.5B 是 ~600 个节点？

我们来**一层一层**算清楚。

### 3.1 一层 Qwen2.5 内部的所有 node

读 **`src/models/qwen2.cpp:71-134`**，一层（24 次循环里的一次）会做这些事：

| 步骤 | 源代码 | 生成多少个 node |
| --- | --- | --- |
| ① attn norm | `build_norm(..., LLM_NORM_RMS)` | 1 (`RMS_NORM`) |
| ② Q 投影 | `build_qkv` 里的 `ggml_mul_mat(wq, x)` | 1 (`MUL_MAT`) |
| ③ K 投影 | 同上 | 1 (`MUL_MAT`) |
| ④ V 投影 | 同上 | 1 (`MUL_MAT`) |
| ⑤ Q RoPE | `ggml_rope_ext(Q, inp_pos, ...)` | 1 (`ROPE`) |
| ⑥ K RoPE | `ggml_rope_ext(K, inp_pos, ...)` | 1 (`ROPE`) |
| ⑦ 注意力计算 | `build_attn` → `build_attn_mha` 内部有 `ggml_mul_mat(Q,K)` → `soft_max` → `flash_attn_ext` → `ggml_cont` → `ggml_mul_mat(V,O)` | 4-5 (`MUL_MAT`+`SOFT_MAX`+`FLASH_ATTN_EXT`+`CONT`+`MUL_MAT`) |
| ⑧ attn 输出投影 | `build_attn` 里的 `wo` 矩阵乘 | 1 (`MUL_MAT`)（已含在上一步里或单独 1 个） |
| ⑨ 残差 1 | `ggml_add(attn_out, x)` | 1 (`ADD`) |
| ⑩ ffn norm | `build_norm(..., LLM_NORM_RMS)` | 1 (`RMS_NORM`) |
| ⑪ up 投影 | `build_ffn` 里的 `ggml_mul_mat(ffn_up, x)` | 1 (`MUL_MAT`) |
| ⑫ gate 投影 | `ggml_mul_mat(ffn_gate, x)` | 1 (`MUL_MAT`) |
| ⑬ silu 激活 | `ggml_silu(gate)` | 1 (`UNARY` 的 SILU) |
| ⑭ 门控相乘 | `ggml_mul(silu, up)` | 1 (`MUL`) |
| ⑮ down 投影 | `ggml_mul_mat(ffn_down, ...)` | 1 (`MUL_MAT`) |
| ⑯ 残差 2 | `ggml_add(down_out, ffn_inp)` | 1 (`ADD`) |
| ⑰ cvec（可选） | `build_cvec` | 0-1 |

**单层总计**：约 **18-19 个 node**。

### 3.2 24 层累加

```
24 层 × 18 node/层 ≈ 432 node
```

### 3.3 首尾额外的 node

| 位置 | 数量 | 来源 |
| --- | --- | --- |
| token embedding 查表 | 1 | `build_inp_embd` 的 `ggml_get_rows` |
| 位置 id 输入 | 1 | `build_inp_pos` |
| attention 输入 (KV cache slot) | 2-3 | `build_attn_inp_kv` 里的 mask + KQ scale |
| final RMS norm | 1 | `build_norm(output_norm)` |
| lm_head 矩阵乘 | 1 | `ggml_mul_mat(output, x)` |
| 输出节点 | 1-2 | logits tensor |

**首尾总计**：约 8-10 个 node。

### 3.4 总数

```
432 (24 层) + 10 (首尾) ≈ 440 个 node
```

> 之前我说的"~600"是**保守估计**——考虑了 prefill 模式（batch > 1）下额外的 `CONT` / `CPY` / 广播节点；纯单 token 解码（decode）大约是 440 个节点。

### 3.5 一张表看清楚

```
┌─────────────── ggml_cgraph (440 个 node) ───────────────┐
│                                                         │
│  入口:  inp_embd ─→ inp_pos ─→ inp_attn_kv              │  ← 4 个
│  ─────────────────────────────────────────────────────   │
│  ┌─────── layer 0 ────────┐                              │
│  │  attn_norm                                              │
│  │  Q_proj K_proj V_proj (3 个 MUL_MAT)                  │
│  │  Q_rope K_rope (2 个 ROPE)                             │
│  │  flash_attn (QK, soft_max, flash_attn, VO)            │
│  │  residual_add                                          │
│  │  ffn_norm                                              │
│  │  up_proj, gate_proj, silu, mul, down_proj (5 个)       │
│  │  residual_add                                          │
│  └─ 18 个 node                                            │
│  ┌─────── layer 1 ────────┐                              │
│  │  ... 同样 18 个 node ...                              │
│  └─ 18 个 node                                            │
│  ... × 24 层 ...                                          │
│  ┌─────── layer 23 ───────┐                              │
│  │  ... 同样 18 个 node ...                              │
│  └─ 18 个 node                                            │
│  ─────────────────────────────────────────────────────   │
│  final_norm                                              │  ← 2 个
│  lm_head (MUL_MAT)                                        │
│  ─────────────────────────────────────────────────────   │
│  出口:  logits                                            │  ← 1 个
└─────────────────────────────────────────────────────────┘
```

---

## 4. 这个"菜谱"长什么样？

下面是一层（layer 0）的示意图。每个方块是一个 node，方块之间的箭头就是 `src[]` 指针：

```
       inpL (layer 0 的输入)
        │
        ▼
   ┌─────────┐
   │ attn_norm│ ← src0=inpL, src1=norm_w
   │ (RMS)    │
   └────┬────┘
        │ cur
        ├──────────────────────────┐
        ▼                          │
   ┌─────────┐                    │
   │ Q_proj   │ ← src0=Wq(cur_idx), src1=cur
   │ (MUL_MAT)│
   └────┬─────┘                    │
        │ Qcur                      │
        ├────►┌────────┐           │
        │     │ Q_rope  │ ← src0=Qcur, src1=inp_pos
        │     │ (ROPE)  │
        │     └───┬────┘           │
        │         │ Qcur'          │
        ▼                          │
   ┌─────────┐                    │
   │ K_proj   │                  ┌─▼──────┐
   │ (MUL_MAT)│                  │ V_proj  │
   └────┬─────┘                  │(MUL_MAT)│
        │ Kcur                    └────┬───┘
        ├────►┌────────┐              │ Vcur
        │     │ K_rope  │             │
        │     │ (ROPE)  │             │
        │     └───┬────┘             │
        │         │ Kcur'             │
        │         │                  │
        ▼         ▼                  ▼
        ┌─────────────────────────────┐
        │  flash_attn_ext              │
        │  src0=Qcur', src1=Kcur',     │
        │  src2=Vcur, src3=mask        │
        │  → attn_out                  │
        └────────┬────────────────────┘
                 │ attn_out
                 ▼
   ┌──────────────┐
   │ O_proj       │ (MUL_MAT, src0=Wo, src1=attn_out)
   │ (MUL_MAT)    │
   └──────┬───────┘
          │ cur
          │
          │              inpSA = inpL
          │                  │
          ▼                  ▼
        ┌──────────────────────┐
        │ ADD (residual 1)     │  src0=cur, src1=inpSA
        │ → ffn_inp            │
        └────────┬─────────────┘
                 │ ffn_inp
                 ▼
           ┌────────────┐
           │ ffn_norm   │ (RMS_NORM)
           └─────┬──────┘
                 │ cur
        ┌────────┼────────┐
        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌────────┐
    │up_proj│ │gate_pr│ │        │
    │(MUL)  │ │(MUL)  │ │        │
    └──┬───┘ └──┬───┘ │        │
       │        │     │        │
       │        ▼     │        │
       │   ┌────────┐ │        │
       │   │ silu   │ │        │
       │   │(UNARY) │ │        │
       │   └───┬────┘ │        │
       │       │      │        │
       ▼       ▼      │        │
    ┌──────────────┐  │        │
    │ MUL (gate×up)│  │        │
    │ → hidden     │  │        │
    └──────┬───────┘  │        │
           │ hidden   │        │
           ▼          │        │
    ┌─────────────┐   │        │
    │ down_proj    │  │        │
    │ (MUL_MAT)    │  │        │
    └──────┬──────┘   │        │
           │ cur      │        │
           ▼          │        │
    ┌──────────────────┴─────┐ │
    │ ADD (residual 2)        │ │
    │ src0=cur, src1=ffn_inp  │ │
    │ → l_out                 │ │
    └──────────┬──────────────┘ │
               │ l_out            │
               └──────→ inpL of layer 1
```

**总节点数：18 个**（一个完整 Transformer decoder 层）。

---

## 5. 怎么"执行"这张菜谱？

`ggml_cgraph` 只是一个数据结构。真正执行它的代码在 `ggml-cpu/ggml-cpu.c:3018` 的 `ggml_graph_compute_thread`：

```c
for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
    struct ggml_tensor * node = cgraph->nodes[node_n];
    ...
    ggml_compute_forward(&params, node);  // ★ 关键
    ggml_barrier(state->threadpool);      // 节点间 barrier
}
```

也就是说：

1. **按数组顺序遍历** `nodes[]`，从 `nodes[0]` 算到 `nodes[439]`（或更多）。
2. 每步调 `ggml_compute_forward` 分派到对应算子实现（**SpacemiT 后端就靠这个 hook 进来**）。
3. 节点之间用 barrier 同步——确保 `nodes[i]` 的 src 已经被 `nodes[0..i-1]` 算好。

### 节点数 ≠ 性能

> 节点数大不代表慢。每个 node 的开销极小（一次函数调用），真正费时的是 `MUL_MAT` 这种"重活"。

对一个 Qwen2.5 0.5B decode 来说，440 个 node 里：
- **168 个是 MUL_MAT**（24 层 × 7 个）—— 占总时间 90%+
- **48 个是 RMS_NORM**（24 层 × 2 个）—— 占总时间 1-2%
- **24 个是 ROPE**（24 层 × 2 个）—— 占总时间 0.5%
- **24 个是 FlashAttn**（24 层 × 1 个）—— 占总时间 3-5%
- **48 个是 ADD/MUL**（残差 + 门控）—— 占总时间 0.5%
- **其他**（GET_ROWS、CONT、UNARY）—— 几乎免费

> **结论**：节点多不是问题；SpacemiT 后端的关键是"重活" MUL_MAT 走 IME 加速，"轻活" 走 RVV 加速。

---

## 6. 一句话总结

- **node** = 一道"算子工序"（"把 A 和 B 乘起来"），同时是装结果的盘子；它的 `op` 字段告诉算什么，`src[]` 字段告诉原料从哪来。
- **ggml_cgraph** = 一张按"先做啥后做啥"排好序的 node 数组，节点之间的依赖通过指针表达，形成 DAG。
- **24 层 Qwen2.5 0.5B → ~440 个 node（decode）/ ~600 个 node（prefill with batched 注意力）**，其中 168 个是 MUL_MAT（SpacemiT 加速的重头戏），其余是 norm、add、RoPE、FlashAttn 等轻活。
- llama.cpp 把"模型怎么算"翻译成"node 怎么排"，然后在 `ggml_compute_forward` 里一气呵成地按顺序跑完——其中 SpacemiT 后端通过 `extra_buffer_type` / `tensor_traits` 把"重活"接到 IME 核上。
