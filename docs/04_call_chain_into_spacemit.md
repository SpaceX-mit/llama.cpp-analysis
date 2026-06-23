# 调用链详解：main → ggml-cpu → spacemit

> 配套文档：`01_llama_cpp_inference_overview.md`、`02_spacemit_backend_deep_dive.md`
> 本文回答两个具体问题：
> 1. 通用 `ggml-cpu/` 代码如何跳进 `ggml-cpu/spacemit/`？
> 2. llama 可执行程序的 `main` 如何一步步进入 `ggml-cpu`？

---

## 1️⃣ `ggml-cpu/` 怎么跳进 `spacemit/`？

核心机制是 **"extra buffer type + tensor_traits" 模式**。`spacemit/` 不直接 hook 任何 op，而是注册一个伪装成 CPU buffer 的特殊 buffer type，CPU 后端在分配 weight tensor 时会优先选它，**算子分派时再通过 `tensor->extra` 字段拿到 spacemit 的 `tensor_traits` 实例**。

整条链路只有 **3 个 hook 点 + 2 个辅助 hook**，全部带编译期 `#ifdef GGML_USE_CPU_RISCV64_SPACEMIT`。

### Hook 1：buffer type 注册（启动期一次）

**`ggml/src/ggml-cpu/ggml-cpu.cpp:21-23`**
```cpp
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"        // ← 跨进 spacemit 目录
#endif
```

**`ggml/src/ggml-cpu/ggml-cpu.cpp:42-56`** — 把 SpacemiT 自己的 buffer type 推到一个全局 vector：
```cpp
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;
        ...
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif
        ...
        return bufts;
    }();
    return bufts;
}
```

`ggml_backend_cpu_riscv64_spacemit_buffer_type()` 自身在 **`ggml/src/ggml-cpu/spacemit/ime.cpp:1647-1665`** 定义：构造一个 `ggml_backend_buffer_type`，`.context` 字段装入一个 `extra_buffer_type` 子类对象。

> 进去方式 ①：**作为 CPU backend 的"额外 buffer type"被加到注册表**，由 CPU backend 的 `extra_bufts` 链表暴露给上层 scheduler。

### Hook 2：调度期 `supports_op` 查询

**`ggml/src/ggml-cpu/ggml-cpu.cpp:423-439`** — CPU backend 判断 op 是否能跑：
```cpp
static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ...
    // 检查 extra buffer types —— 关键！
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);   // ← 跳进 spacemit
        }
    }
    ...
}
```

`extra_buffer_type::supports_op` 的实现在 **`ggml/src/ggml-cpu/spacemit/ime.cpp:1580-1611`**。它说"我可以跑 `MUL_MAT` 当 src0 是 spacemit buffer 且 2D 且 src1 是 fp32 host buffer"。

> 进去方式 ②：**通过 `extra_buffer_type` 多态接口回答"是否支持这个 op"**。

### Hook 3：算子分派（最关键 — 每次 op 都会走）

**`ggml/src/ggml-cpu/ggml-cpu.c:1702-1712`**：
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ...
    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;       // ← SpacemiT 接管成功，直接 return
    }

    switch (tensor->op) {
        case GGML_OP_DUP: ggml_compute_forward_dup(...); break;
        case GGML_OP_MUL_MAT: ggml_compute_forward_mul_mat(...); break;
        ...
    }
}
```

`ggml_cpu_extra_compute_forward` 在 **`ggml/src/ggml-cpu/traits.cpp:12-23`**：
```cpp
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {     // 遍历所有 extra buft
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);     // 调 spacemit 的方法
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;     // ← 接管
            }
        }
    }
    return false;  // 走默认 switch
}
```

链上的关键类 `tensor_traits` / `extra_buffer_type` 声明在 **`ggml/src/ggml-cpu/traits.h:20-32`**（纯虚 C++ 类）；spacemit 的实现在 **`ggml/src/ggml-cpu/spacemit/ime.cpp:122-1232`** —— 13 个 `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>` 模板具现化 + 1 个 `tensor_traits_common`，分别接管 MUL_MAT/MUL_MAT_ID 与 13 类 RVV 算子。

**`extra_buffer_type::get_tensor_traits`** 在 **`spacemit/ime.cpp:1613-1642`**：
```cpp
ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;  // ← 关键
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CONT: case GGML_OP_CPY:
        case GGML_OP_REPEAT: case GGML_OP_SUM_ROWS: case GGML_OP_GET_ROWS: case GGML_OP_CONCAT:
            return (ggml::cpu::tensor_traits *) (&rvv_impl);  // → tensor_traits_common
    }
    return nullptr;
}
```

注意 `(ggml::cpu::tensor_traits *) op->src[0]->extra` —— 这个 `extra` 字段是 `ggml_backend_riscv64_spacemit_buffer_init_tensor` 在 **`ime.cpp:1395-1403`** 写入的：
```cpp
static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(buffer, t) {
    t->extra = (void *) ggml_riscv64_spacemit_get_optimal_repack_type(t);
    return GGML_STATUS_SUCCESS;
}
```

> 进去方式 ③：**通过 `tensor->extra` 指针回查——在 tensor 分配 buffer 时写入对应的 `tensor_traits*`，算子分派时取出。**

### 辅助 hook A：work_size 估算

**`ggml-cpu.c:2786`**：
```c
if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {  // 先问 spacemit
    switch (node->op) {
        case GGML_OP_MUL_MAT:  ...  // 默认 fallback
    }
}
```
→ **`traits.cpp:25-36`** → **`spacemit/ime.cpp:128-185`** 的 `tensor_traits<...>::work_size` 计算量化 src1 用的 workspace。

### 辅助 hook B：线程入口/出口

**`ggml-cpu.c:3025-3029` 和 `3086-3088`**：
```c
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);  // 线程启动：绑 AI 核 + 申请 TCM
#endif
    ...
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);  // 线程退出：释放 TCM
#endif
```
→ 都在 **`spacemit/ime.cpp:1690-1739`**。

---

## 2️⃣ llama main 怎么一步步进到 ggml-cpu？

### Step 0：可执行入口

`examples/simple/simple.cpp:14` —— 极简 demo 入口（生产上是 `llama-cli`，结构相同）：
```cpp
int main(int argc, char ** argv) {
    ...
    ggml_backend_load_all();              // ← 注册所有 backend（含 CPU）
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;     // 99 = 全部 layer 走加速器
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);  // 加载 GGUF
    ...
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    ...
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) { ... }                // ← 单次推理
        new_token_id = llama_sampler_sample(smpl, ctx, -1); // 采样
        batch = llama_batch_get_one(&new_token_id, 1);
    }
}
```

### Step 1：`ggml_backend_load_all()` — 静态拉起所有 backend

`ggml/src/ggml-backend-reg.cpp:555-557`：
```cpp
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}
```

`ggml/src/ggml-backend-reg.cpp:559-586`：
```cpp
void ggml_backend_load_all_from_path(const char * dir_path) {
    ...
    ggml_backend_load_best("cpu", silent, dir_path);   // ← CPU backend 关键
    ...
}
```

→ 静态链接版直接调 `ggml_backend_cpu_reg()`（**`ggml-cpu/ggml-cpu.cpp:690`**）：
```cpp
ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    ggml_cpu_init();                                  // 探测 CPU 特性
    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,  // 含 get_extra_bufts 等
        /* .context     = */ NULL,
    };
    return &ggml_backend_cpu_reg;
}
```

→ `ggml_backend_cpu_reg_i.get_extra_bufts` → **`ggml-cpu/ggml-cpu.cpp:76-86`**：
```cpp
static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();
    return extra_bufts.data();
}
```

→ 调 **`ggml-cpu/ggml-cpu.cpp:42-74`** 的 `ggml_backend_cpu_get_extra_buffer_types()` —— **SpacemiT buffer type 在这一刻被 push 进 vector**（Hook 1）。

### Step 2：`llama_model_load_from_file()` — 加载模型 + 选 buffer

`src/llama.cpp:426`：
```cpp
struct llama_model * llama_model_load_from_file(...) {
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, nullptr, params);
}
```

`src/llama-context.cpp` 上层在加载完 model 后会构造 scheduler（`sched_reserve`）。scheduler 在做 `ggml_backend_sched_alloc_graph` 时会**遍历** CPU backend 的 `get_extra_bufts`，发现 SpacemiT 后端"我可以跑 Q4_0 权重的 MUL_MAT"（Hook 2 命中），于是把 weight tensor 分配到 SpacemiT buffer。

→ 分配时调 `ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer`（**`spacemit/ime.cpp:1480-1488`**）：
```cpp
static ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(buft, size) {
    void * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);  // 大页池
    return ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}
```

→ 写数据时调 `ggml_backend_riscv64_spacemit_buffer_set_tensor`（**`ime.cpp:1443-1458`**）触发**一次性 repack**：
```cpp
static void ggml_backend_riscv64_spacemit_buffer_set_tensor(buffer, t, data, offset, size) {
    auto traits = (tensor_traits_base *) t->extra;       // Hook 1 写入的
    if (traits) {
        auto OK = traits->repack(t, data, size);          // → spacemit/repack.cpp
    }
}
```

### Step 3：`llama_decode()` — 进入推理主循环

`src/llama-context.cpp:4054-4063` → `llama_context::decode()` (`src/llama-context.cpp:1680`) → **`process_ubatch()`** (`src/llama-context.cpp:1304-1374`)：

```cpp
res->set_inputs(&ubatch);                                       // 1. 写入 token id / position
const auto status = graph_compute(res->get_gf(), ...);          // 2. 真·执行
```

`graph_compute` (`src/llama-context.cpp:2421-2442`)：
```cpp
ggml_status llama_context::graph_compute(ggml_cgraph * gf, ...) {
    ...
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    ...
}
```

### Step 4：`ggml_backend_sched_graph_compute_async` — 调度器

`ggml/src/ggml-backend.cpp:1889`：
```cpp
enum ggml_status ggml_backend_sched_graph_compute_async(sched, graph) {
    ...
    return ggml_backend_sched_compute_splits(sched);   // → 计算每段 split
}
```

`ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1541`) 内部：按 backend 切成多个 split，每个 split 内部所有节点都跑在同一个 backend 上。

SpacemiT 因为是 CPU 的 extra buffer type，**对应 backend 就是 CPU backend**（不是独立 device），所以 SpacemiT 算子最终会走到 `ggml_backend_cpu_graph_compute`。

### Step 5：`ggml_backend_cpu_graph_compute` — CPU backend 执行

`ggml-cpu/ggml-cpu.cpp:130+` 的 `ggml_backend_cpu_graph_plan_create` 和 `graph_plan_compute` 会构造 `ggml_cplan`，最终在 `ggml-cpu.c:3018` 启动线程：

```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    ...

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);  // ← 绑 AI 核 + 抢 TCM
#endif

    struct ggml_compute_params params = { state->ith, nth, wsize, wdata, threadpool, use_ref };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;

        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);       // ← 节点计算
        }

        if (state->ith == 0 && cplan->abort_callback(...)) {
            ...                                        // 失败中止
        }
        ggml_barrier(state->threadpool);               // 节点间 barrier
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);  // 释放 TCM
#endif
    return 0;
}
```

### Step 6：`ggml_compute_forward` — 节点分派

`ggml-cpu/ggml-cpu.c:1702-1712`：
```c
static void ggml_compute_forward(params, tensor) {
    ...
    if (ggml_cpu_extra_compute_forward(params, tensor)) {  // ← SpacemiT 接管？
        return;                                            //    成功 → 直接返回
    }
    switch (tensor->op) {
        case GGML_OP_MUL_MAT: ggml_compute_forward_mul_mat(params, tensor); break;
        case GGML_OP_RMS_NORM: ggml_compute_forward_rms_norm(params, tensor); break;
        ...
    }
}
```

### Step 7：进入 `spacemit/`

`ggml_cpu_extra_compute_forward`（**`traits.cpp:12-23`**） →
1. 遍历 `ggml_backend_cpu_get_extra_buffer_types()` 找到 SpacemiT buffer type；
2. 强转 `context` 为 `ggml::cpu::extra_buffer_type*`；
3. 调 `get_tensor_traits(op)`（**`spacemit/ime.cpp:1613-1642`**）拿到 `tensor_traits*`：
   - MUL_MAT/MUL_MAT_ID → `op->src[0]->extra`（Hook 1 写入的具现化）；
   - 其它 → 静态 `rvv_impl`（`tensor_traits_common`）。
4. 调 `tensor_traits->compute_forward(params, op)`（**`spacemit/ime.cpp:187-232`**）→ `forward_mul_mat` / `forward_mul_mat_id` / `tensor_traits_common::compute_forward`（RVV 算子）。

---

## 3️⃣ 完整链路图

```
用户
  ↓ ./llama-cli -m model.gguf
examples/llama-cli/.../main.cpp::main()
  ├─ ggml_backend_load_all()                                [ggml-backend-reg.cpp:555]
  │   └─ ggml_backend_load_best("cpu", ...)                 [ggml-backend-reg.cpp:580]
  │       └─ ggml_backend_cpu_reg()                         [ggml-cpu/ggml-cpu.cpp:690]
  │           ├─ ggml_cpu_init()                            // CPU 特性探测
  │           └─ 注册 ggml_backend_cpu_reg
  │                (CPU device 的 get_extra_bufts →
  │                 ggml_backend_cpu_device_get_extra_buffers_type [ggml-cpu.cpp:76]
  │                  → ggml_backend_cpu_get_extra_buffer_types  [ggml-cpu.cpp:42]
  │                    → ggml_backend_cpu_riscv64_spacemit_buffer_type()  [spacemit/ime.cpp:1647]   ★ 第①次进入 spacemit
  │                )
  │
  ├─ llama_model_load_from_file()                          [src/llama.cpp:426]
  │   └─ 触发 weight tensor 分配 →
  │        ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer  [ime.cpp:1480]   ★ 第②次进入
  │           → spine_mem_pool_alloc (大页池)
  │        → set_tensor 时触发 repack
  │           → tensor_traits::repack()                      [ime.cpp:931]
  │              → spacemit/repack.cpp::repack_q4_0_to_q4_0_16_bl  ★ 第③次进入
  │
  └─ for (each token) llama_decode(ctx, batch)              [src/llama-context.cpp:4054]
      └─ llama_context::decode()                            [src/llama-context.cpp:1680]
         └─ process_ubatch()                                [src/llama-context.cpp:1304]
            ├─ model.build_graph()                          // 拼 ggml_cgraph
            │   └─ src/models/llama.cpp::graph<>::graph()  // 调 ggml_* API
            ├─ ggml_backend_sched_alloc_graph()             // 给 weight 选 spacemit buft
            ├─ set_inputs(&ubatch)                          // 写 token/pos
            └─ graph_compute() → ggml_backend_sched_graph_compute_async()
                └─ ggml_backend_sched_compute_splits()      [ggml-backend.cpp:1541]
                   └─ 对每段 split 调 backend.graph_compute()
                      └─ ggml_backend_cpu_graph_compute()
                         └─ ggml_graph_compute_thread()    [ggml-cpu.c:3018]   ← CPU worker 线程入口
                            ├─ ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity() [ime.cpp:1690]  ★ 第④次进入（绑核）
                            │
                            └─ for (node in cgraph):
                                ggml_compute_forward()      [ggml-cpu.c:1702]
                                  ├─ ggml_cpu_extra_compute_forward()   [traits.cpp:12]
                                  │    └─ extra_buffer_type::get_tensor_traits(op)  [ime.cpp:1613]   ★ 第⑤次进入（按 op 拿 traits）
                                  │       └─ tensor_traits::compute_forward()    [ime.cpp:187]
                                  │          ├─ forward_mul_mat()         [ime.cpp:234]           // MUL_MAT
                                  │          │   ├─ quantize_a_4row_i8*() [ime1_kernels.cpp / rvv_kernels.cpp]
                                  │          │   └─ gemm_kernel_i8i4_hp() [ime2_kernels.cpp:2883]  // vmadotsu.hp
                                  │          ├─ forward_mul_mat_id()     [ime.cpp:543]           // MoE
                                  │          └─ tensor_traits_common::compute_forward()          // RVV 算子
                                  │                ├─ rvv::forward_rms_norm_f32 [rvv_kernels.cpp:1630]
                                  │                └─ rvv::forward_flash_attn_ext_f16  [ime.cpp:1149]
                                  └─ switch (tensor->op)  // 兜底默认实现
                            └─ ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded() [ime.cpp:1731]  ★ 释放 TCM
```

---

## 4️⃣ 总结

1. **`ggml-cpu → spacemit` 的入口有 3 个静态 + 2 个动态 hook**，全部通过 `extra_buffer_type` / `tensor_traits` 抽象与多态完成，spacemit **不需要改 `ggml-cpu.c` / `ops.cpp` 一行代码**。
2. **`main → ggml-cpu` 是常规 5 跳**：`main → ggml_backend_load_all → ggml_backend_cpu_reg → llama_decode → graph_compute → ggml_backend_sched → ggml_backend_cpu_graph_compute_thread → ggml_compute_forward`。
3. **唯一一处"显式侵入"** 是 `ggml-cpu.c:3026` 和 `:3087` 的两条 `#ifdef GGML_USE_CPU_RISCV64_SPACEMIT` —— 用来在线程入口处触发线程迁移和 TCM 申请，因为 CPU 默认的 `set_numa_thread_affinity` 不知道 SpacemiT 异构 SoC 上还要去 `/proc/set_ai_thread`。
