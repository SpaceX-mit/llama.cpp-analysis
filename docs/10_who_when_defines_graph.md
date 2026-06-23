# "图"是谁定的？什么时候知道？

> 配套：`06_what_is_node_and_cgraph.md`、`09_node_count_and_deep_dive.md`
> 目标：彻底讲清楚 llama.cpp **静态计算图**的来龙去脉

---

## 0. 一句话答案

> **"图"由 4 方协作确定**：
> 1. **模型架构代码**（`src/models/qwen2.cpp`）—— 定"算哪些 op、按什么顺序"
> 2. **用户参数**（`cparams`）—— 定"batch 多大、ctx 多长"
> 3. **模型文件**（GGUF）—— 定"模型多宽（n_embd）、多少层（n_layer）"
> 4. **scheduler** —— 定"每个 op 跑在哪个后端"
>
> **"图"的生命周期**：
> - **编译期**：C++ 源码**就是**图模板
> - **加载期**：解析 GGUF，得到具体形状
> - **第一次 decode**：`process_ubatch` → `build_graph(gparams)` **真正构造**出 `ggml_cgraph`
> - **后续 decode**：通过 `can_reuse(gparams)` 判断**复用**还是**重建**

下面逐项展开。

---

## 1. "图"是谁定的？4 个层面

### 1.1 第一层：C++ 源码决定 op 类型和顺序

**位置**：`src/models/<arch>.cpp` 的 `llama_model_<arch>::graph::graph()` 构造函数。

**`src/models/qwen2.cpp:53-153`** 的 `qwen2::graph` 构造时**挨个创建**算子：

```cpp
llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    ...
    inpL = build_inp_embd(model.tok_embd);                       // 1. 词嵌入查表
    ggml_tensor * inp_pos = build_inp_pos();                     // 2. 位置输入
    auto * inp_attn = build_attn_inp_kv();                        // 3. KV cache 输入
    
    for (int il = 0; il < n_layer; ++il) {                       // 4. 24 层循环
        cur = build_norm(inpL, ..., LLM_NORM_RMS, il);           //    4a. attn norm
        auto [Q, K, V] = build_qkv(...);                         //    4b. Q/K/V 投影
        Q = ggml_rope_ext(ctx0, Q, inp_pos, ...);                //    4c. Q RoPE
        K = ggml_rope_ext(ctx0, K, inp_pos, ...);                //    4d. K RoPE
        cur = build_attn(inp_attn, ..., Q, K, V, ...);           //    4e. flash attn
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);      //    4f. 残差
        cur = build_norm(ffn_inp, ..., LLM_NORM_RMS, il);        //    4g. ffn norm
        cur = build_ffn(cur, ..., LLM_FFN_SILU, LLM_FFN_PAR);    //    4h. FFN
        cur = ggml_add(ctx0, cur, ffn_inp);                       //    4i. ffn 残差
    }
    cur = build_norm(cur, model.output_norm, ..., LLM_NORM_RMS); // 5. final norm
    cur = build_lora_mm(model.output, cur);                       // 6. lm_head
    ggml_build_forward_expand(gf, cur);                            // 7. 固化图
}
```

> **关键事实**：这段 C++ 代码就是 Qwen2 架构的"菜谱"，**编译后**它就是图模板。改它就改图，**无需重新训练**。

### 1.2 第二层：GGUF 模型文件决定张量形状

**`qwen2.5-0.5b-instruct-q4_0.gguf`** 里写的元数据：

| 元数据 | 值 | 决定图的什么 |
| --- | --- | --- |
| `qwen2.block_count` | 24 | `n_layer` → 决定 for 循环 24 次 |
| `qwen2.embedding_length` | 896 | `n_embd` → 决定 Q/K/V/O 权重 shape |
| `qwen2.attention.head_count` | 14 | `n_head` |
| `qwen2.attention.head_count_kv` | 2 | `n_head_kv`（GQA）|
| `qwen2.feed_forward_length` | 4864 | `n_ff` → 决定 FFN 权重 shape |
| `qwen2.rope.freq_base` | 1000000 | RoPE 频率基数 |
| `tokenizer.ggml.tokens` | ... | `n_vocab` → lm_head 输出维度 |

**加载时**（`llama.cpp:llama_model_load_from_file`）这些 KV 写进 `hparams`：

```cpp
// src/llama-model.cpp 的某个地方
hparams.n_layer = gguf_get_val_u32(gguf_ctx, "qwen2.block_count");  // = 24
hparams.n_embd  = gguf_get_val_u32(gguf_ctx, "qwen2.embedding_length");  // = 896
// ...
```

之后 `hparams.n_layer` 决定 `for (int il = 0; il < n_layer; ++il)` 跑几次。

### 1.3 第三层：用户参数 `cparams` 决定运行维度

`llama_context_params ctx_params`（用户传）：

| 字段 | 含义 | 决定图的什么 |
| --- | --- | --- |
| `n_ctx` | 上下文长度 | KV cache 容量 |
| `n_batch` | 单次最大 token 数 | 输入张量最大 shape |
| `n_ubatch` | 实际单次处理的 token 数 | **本轮 ubatch 的 `n_tokens`** |
| `n_seq_max` | 最大并发序列数 | 多 seq 时的 batch 维度 |
| `n_threads` | worker 线程数 | 调度粒度（不是图的形状）|

**这些是"运行期形状"**——它们和"模型架构形状"（`n_embd` 等）正交。

> 重要：**`n_ubatch` 决定 ubatch.n_tokens，进而决定本轮图的输入节点 shape**。

### 1.4 第四层：scheduler 决定每个 op 跑在哪个后端

`ggml_backend_sched_alloc_graph`（`ggml/src/ggml-backend.cpp:1864`）遍历每个 node，问每个 backend：

```cpp
// 对 node = MUL_MAT, src0 是 Q4_0 weight:
ggml_backend_cpu_device_supports_op(dev, op) {
    // 走到 ggml-cpu/ggml-cpu.cpp:423-439
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);  // ← SpacemiT 答"是"
        }
    }
}
```

→ 把这个 MUL_MAT 标"跑在 CPU device 上"，**也就是 SpacemiT 后端**（因为它是 CPU extra buffer）。

### 1.5 总结：4 层共同决定一张图

```
┌─────────────────────────────────────────────┐
│  1. C++ 源码        → op 类型与顺序        │  ← 编译期固定
│  2. GGUF 文件        → 张量具体 shape        │  ← 加载时读
│  3. 用户 cparams    → 运行期输入 shape     │  ← 用户传
│  4. scheduler       → 每个 op 跑哪个后端    │  ← alloc_graph 时定
└─────────────────────────────────────────────┘
                  ↓
            一张具体的 ggml_cgraph
```

---

## 2. "图"什么时候知道？5 个时间点

### 2.1 时间线总览

```
T0  编译期       C++ 源码成为"图模板"
   ↓
T1  启动期       ggml_backend_load_all() ───→ 注册 SpacemiT 等 backend
   ↓
T2  加载期       llama_model_load_from_file() ───→ 解析 GGUF，建 hparams
   ↓                触发 weight tensor 分配 → SpacemiT 大页池 + repack
   ↓
T3  上下文初始化  llama_init_from_model() ───→ 建 KV cache + scheduler
   ↓                sched_reserve()  ───→ 分配 gf_res_prev 缓冲区
   ↓
T4  第一次 decode  process_ubatch()
   ↓                ├─ can_reuse(gparams) → false（第一次）
   ↓                └─ model.build_graph(gparams)  ← 真正的图在这里构造！
   ↓                └─ ggml_backend_sched_alloc_graph(gf)  ← 分配 buffer
   ↓                └─ res->set_inputs(&ubatch)  ← 写 token id
   ↓                └─ graph_compute(gf)  ← 第一次跑
   ↓
T5  第二次 decode  process_ubatch()
   ↓                ├─ can_reuse(gparams) → true（如果 ubatch 形状一样）
   ↓                └─ 复用图，**不再构造**！只 set_inputs + graph_compute
   ↓
T6  第三次 decode  同 T5 ...
   ↓
T7  N 次 decode   一旦 batch 变化 → 重新走 T4 路径
```

### 2.2 关键时间点详解

#### T0：编译期

**用户**写（或修改）`src/models/qwen2.cpp` 里的 `graph::graph()` 构造函数，编译器把它编进二进制。

> 此时图还**不存在**，但图的"骨架"已经编进程序。

#### T1+T2：启动 + 加载

- `main()` 调 `ggml_backend_load_all()` → 注册 CPU + SpacemiT extra buffer type
- `llama_model_load_from_file("qwen2.5-0.5b-q4_0.gguf")`：
  - 解析 GGUF，填充 `hparams.n_layer=24, n_embd=896, ...`
  - 给每个 weight tensor 分配 SpacemiT buffer + 一次性 repack
- 此时**图还没建**，但所有"原料"都备好了

#### T3：上下文初始化

- `llama_init_from_model(model, ctx_params)`：
  - 建 KV cache（容量由 `ctx_params.n_ctx` 决定）
  - 建 `ggml_backend_sched`（多 backend 调度器）
  - 调 `sched_reserve()`（`llama-context.cpp:439`）—— **预分配 `gf_res_prev` 缓冲区**

`sched_reserve` 关键代码（`llama-context.cpp:439-470`）：

```cpp
void llama_context::sched_reserve() {
    if (!sched_need_reserve) return;
    sched_need_reserve = false;
    
    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
    const size_t max_nodes = this->graph_max_nodes(n_tokens);  // 算最大节点数
    
    gf_res_prev.reset(new llm_graph_result(max_nodes));  // 预分配能装 max_nodes 的 graph
    
    // 调 sched_alloc_graph 测出"worst case" 的 split 数
    ...
}
```

> 此时 `gf_res_prev` 是个**空的** `llm_graph_result`，**等待**被填。

#### T4：第一次 `llama_decode` —— 真正的图在这里构造

```cpp
// 用户代码
llama_decode(ctx, batch);  // batch.n_tokens = 4 (4 个 prompt token)
```

→ `llama_context::decode()` (1680) → `process_ubatch()` (1304)

```cpp
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, ...) {
    auto * res = gf_res_prev.get();       // 拿预分配的容器
    auto * gf  = res->get_gf();           // 当前是空 graph
    
    const auto gparams = graph_params(res, ubatch, mctx, gtype);  // ★ 准备参数
    
    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        // 第一次 can_reuse 必返回 false
    } else {
        res->reset();
        ggml_backend_sched_reset(sched.get());
        // ...
        gf = model.build_graph(gparams);   // ★ 真正构造图！
        if (!gf) return nullptr;
        ggml_backend_sched_alloc_graph(sched.get(), gf);  // 给每个 tensor 分配 buffer
    }
    
    res->set_inputs(&ubatch);             // 写 token id 到 inp_tokens
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    return res;
}
```

`model.build_graph(gparams)` 干了什么？调到 `src/llama-model.cpp:2234`：

```cpp
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);  // 多态
    llm->build_pooling(...);
    llm->build_sampling();
    llm->res->set_outputs(params);
    return llm->res->get_gf();
}
```

对 Qwen2：`build_arch_graph` → `qwen2.cpp:49` 的 `qwen2::graph` 构造：

```cpp
std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(...) const {
    return std::make_unique<graph>(*this, params);   // ★ 调 graph 构造
}
```

→ 构造里依次调 `ggml_*` 创建节点，最后 `ggml_build_forward_expand(gf, cur)` 把所有节点 DFS 收集到 `gf->nodes[]`。

**至此，`gf->n_nodes = 411-415`（Qwen2.5 0.5B decode）或 440-500（prefill），图就建好了。**

#### T5：第二次 `llama_decode` —— 复用图

```cpp
const auto gparams = graph_params(res, ubatch, mctx, gtype);  // 新一轮的 ubatch
if (res->can_reuse(gparams)) {   // ★ 关键：判断能否复用
    n_reused++;
} else {
    // 重建图（同 T4）
}
```

`can_reuse` 检查**输入形状是否一致**：

```cpp
// src/llama-graph.h:98 (简化)
virtual bool can_reuse(const llm_graph_params & params) {
    return this->params.equal(params);  // 形状完全一样才复用
}
```

**两个 ubatch 形状一样 = 图可复用**。比如：
- 都是 batch=1 decode → 形状一样 → **复用** ✓
- 一个 batch=1, 一个 batch=4 → 形状不一样 → **重建**

#### T6：后续 decode

只要 ubatch 形状不变，**图永远复用**。`gf_res_prev` 一直保存，**省去 build_graph 的开销**。

#### T7：什么时候必须重建图？

| 变化 | 后果 |
| --- | --- |
| `n_ubatch` 变化（如从 batch=1 切到 batch=8） | ubatch.n_tokens 变 → 重建 |
| 进入 prefill（n_tokens=512）后切回 decode | n_tokens 变 → 重建 |
| 多 sequence 并发场景 | `n_seqs` 变 → 重建 |
| `ctx_type` 变（encode vs decode） | gtype 变 → 重建 |
| KV cache 策略变（如 swa/full 切换）| 重建 |

---

## 3. 为什么是"静态图"（不是 PyTorch 那种动态图）？

### 3.1 静态 vs 动态对比

| 框架 | 图方式 | 模型代码 |
| --- | --- | --- |
| **PyTorch** | 动态图 | `def forward(x): return self.linear(x)` 每次前向**都跑 Python 重新建图** |
| **TensorFlow Graph** | 静态图 | `@tf.function` 装饰的 Python 函数编译成静态图 |
| **llama.cpp** | 静态图 | C++ 构造函数**直接**就是"建图"，运行期只复用 |

### 3.2 静态图的 4 大好处

1. **构建开销只付一次**
   - 第一次 decode 100 ms 建图，后续 411 个 node 的执行只花 15 ms
   - 动态图每次都要付 100 ms

2. **图优化**（constant folding、op fusion）
   - 编译器/调度器能看到完整图，能做优化
   - llama.cpp 有 `ggml_cpu_try_fuse_ops` 融合 RMSNorm + MUL 等

3. **内存预先规划**
   - scheduler 在 build_graph 之后**立刻**算好每个 tensor 的 buffer
   - 不会有"算到一半 OOM"

4. **跨节点优化**（比如 TCM 复用、KV cache 直接写到固定地址）
   - scheduler 知道整张图的 tensor 生命周期
   - 一次分配，所有节点复用

### 3.3 llama.cpp 的特殊设计：can_reuse

- **大多数静态图框架**（TF1.x）一旦 batch shape 变就重建
- **llama.cpp 走得更远**：
  - 它能识别"两次 ubatch 形状完全一样" → **直接复用整张图**，连重新分配都省了
  - `n_reused` 计数器（`llama-context.cpp:1328`）会统计复用次数
  - 跑长 prompt 时你能看到 `n_reused` 增长很快

---

## 4. 图的"全生命周期"代码地图

| 时间点 | 文件 | 行 | 关键函数 |
| --- | --- | --- | --- |
| 编译期 | `src/models/qwen2.cpp` | 53-153 | `graph::graph()` 构造函数 |
| T1 启动 | `ggml/src/ggml-backend-reg.cpp` | 555-586 | `ggml_backend_load_all()` |
| T2 加载 | `src/llama.cpp` | 426 | `llama_model_load_from_file()` |
| T2 repack | `ggml/src/ggml-cpu/spacemit/ime.cpp` | 1443 | `ggml_backend_riscv64_spacemit_buffer_set_tensor()` |
| T3 上下文 | `src/llama-context.cpp` | 439 | `llama_context::sched_reserve()` |
| T4 第一次 decode | `src/llama-context.cpp` | 1304-1374 | `process_ubatch()` |
| T4 建图 | `src/llama-model.cpp` | 2234 | `llama_model::build_graph()` |
| T4 选 backend | `ggml/src/ggml-cpu/ggml-cpu.cpp` | 423-439 | `ggml_backend_cpu_device_supports_op()` |
| T4 节点分发 | `ggml/src/ggml-cpu/ggml-cpu.c` | 1702-1712 | `ggml_compute_forward()` |
| T4 SpacemiT 接管 | `ggml/src/ggml-cpu/traits.cpp` | 12-23 | `ggml_cpu_extra_compute_forward()` |
| T5 复用判断 | `src/llama-context.cpp` | 1318 | `res->can_reuse(gparams)` |
| T5 复用实现 | `src/llama-graph.h` | 98 | `virtual bool can_reuse()` |

---

## 5. 一个具体的例子

跑 `llama-cli -m qwen2.5-0.5b-q4_0.gguf -p "Hello" -n 32`，背后实际发生：

```
T0: 编译 → 我的二进制里有 qwen2.cpp 的 graph 构造函数
T1: ggml_backend_load_all() → SpacemiT extra buffer type 已注册
T2: 加载 GGUF → 330 MB 权重已在大页池里，已 repack
T3: 初始化 ctx → KV cache 建好，gf_res_prev 缓冲区预分配

T4: llama_decode(batch=[Hello])           # 5 tokens
    ├─ can_reuse → false（首次）
    ├─ build_graph(gparams) → 415 个 node 的 ggml_cgraph 诞生
    ├─ alloc_graph → 每个 tensor 绑定到 buffer
    ├─ set_inputs → 把 5 个 token id 写进 inp_tokens
    └─ graph_compute → 跑 415 个 node，得到 5 个 logits
        → sampler 采样 → 第一个输出 token

T5: llama_decode(batch=[new_token])       # 1 token
    ├─ can_reuse → true（形状一样）
    ├─ ★ 跳过 build_graph！★
    ├─ set_inputs → 把 1 个 token id 写进 inp_tokens
    └─ graph_compute → 跑 415 个 node，得到 1 个 logits
        → 第二个输出 token

T6: 同 T5 30 次 → 32 个输出 token 总共只花了 ~500ms

如果中途切到 batch=4：
T7: llama_decode(batch=[t1, t2, t3, t4])  # 4 tokens
    ├─ can_reuse → false（n_tokens 变了）
    ├─ build_graph(gparams) → 新的图
    ├─ alloc_graph → 重新分配
    └─ graph_compute
```

---

## 6. 一句话总结

> **图是 4 方（架构代码 / GGUF / 用户参数 / scheduler）协作的产物，**
> **在第一次 `llama_decode` 时被 `process_ubatch` 真正构造出来，**
> **之后只要 ubatch 形状不变就 `can_reuse` 直接复用——这是 llama.cpp 跑得快（不重新建图）+ 跑得稳（不重新规划内存）的关键设计**。
>
> 改成 dynamic graph（每次重建）也能跑，但每秒会多花几十毫秒建图；这就是为什么 llama.cpp 选择 static graph + can_reuse 这个看似简单的设计。
