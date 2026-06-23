# 完整流程：llama.cpp 跑 Qwen2.5 0.5B (Q4_0) on SpacemiT K3

> **面向对象**：第一次接触 LLM 推理栈、想了解"一行 `-m model.gguf` 背后究竟发生了什么"的读者
> **目标硬件**：SpacemiT K3 (Banbu K3, 8× X100 + 8× A100) — SpacemiT 迄今最强的 RISC-V + 自研 IME NPU SoC
> **目标模型**：Qwen2.5 0.5B Instruct，量化方案 **Q4_0**（4-bit weight per 32 weight block，1 fp16 scale）
> **关键工具链**：llama.cpp (本仓库) + libspine_tcm.so + 内核的 `/proc/set_ai_thread`

---

## 0. 准备知识：Qwen2.5 0.5B 是什么？

Qwen2.5 0.5B 是阿里通义千问 2.5 代的小模型。它的"骨架"长这样：

| 项目 | 数值 | 含义 |
| --- | --- | --- |
| `n_layer` | 24 | Transformer 层数（"深度"） |
| `n_embd` | 896 | 每层 token 隐向量维度 |
| `n_head` | 14 | 注意力头数 |
| `n_head_kv` | 2 | KV 共享头数（GQA：每 7 个 Q 头共享 1 对 KV） |
| `head_dim` | 64 | 每个头的维度 (n_embd/n_head) |
| `n_ff` | 4864 | FFN 中间层维度 |
| `n_vocab` | 151936 | 词表大小 |
| `rope_theta` | 1000000 | RoPE 频率基数 |
| 总参数量 | ~5 亿 | (24 层 × 各种线性层) ≈ 0.5B |

每层有两个主要模块：
1. **Self-Attention**（自注意力）
2. **FFN**（前馈网络，SwiGLU 激活）

每个模块都是 `x → Linear → 激活 → Linear → +残差` 的模式。

### Q4_0 量化是什么？

把 32 个连续的 fp32 权重压成 18 字节：

```
[fp16 scale d] [16 bytes = 32 个 4-bit]
                ↑ 把 -8..7 范围的 int4 数装进 nibble
```

恢复公式：`w[i] = (q[i] - 8) * d`，其中 d 是这 32 个权重共用的缩放因子。压缩比 ≈ 7 倍（128 bit → 18 byte）。

---

## 1. 准备知识：SpacemiT K3 是什么？

SpacemiT K3（Banbu K3）是进迭时空（SpacemiT）最新一代 RISC-V 异构 AI SoC。**异构**是关键——一颗芯片里装两类核，**总计 16 核**：

```
┌──────────────────────────────────────────────────────────┐
│  SpacemiT K3 SoC (16 核异构)                              │
│                                                          │
│  ┌──────────────────┐   ┌────────────────────────────┐  │
│  │  8× X100 主控核    │   │  8× A100 AI 协处理器核       │  │
│  │  (X-class)        │   │  (A-class, 自研)             │  │
│  │  跑 OS / llama.cpp │   │  跑矩阵乘 / 量化 / softmax  │  │
│  │  marchid=0x5064   │   │  marchid=0xA064           │  │
│  │                   │   │  + 4MB TCM/核（共 32MB）    │  │
│  │  RVV 1.0          │   │  RVV 1.0 + VMADOT/IME2    │  │
│  │  NEON/SVE 无      │   │  fp16 MAC 累加             │  │
│  └──────────────────┘   └────────────────────────────┘  │
│           │                       ▲                       │
│           └────── 共享 DDR ────────┘                       │
└──────────────────────────────────────────────────────────┘
```

### 16 核分工

| 核 | 数量 | 角色 | marchid | 跑的活 |
| --- | --- | --- | --- | --- |
| **X100** | 8 | 主控 | `0x5064` | 操作系统、llama.cpp 主循环、prompt 处理、token 采样、I/O |
| **A100** | 8 | AI 协处理器 | `0xA064` | 矩阵乘（MUL_MAT）、量化、FlashAttention tile |

> 用户态的 worker 线程只绑到 **8 个 A100 核**；X100 留给系统调度和 llama 主循环。
> 8 核并行矩阵乘，对 Qwen2.5 0.5B 这种小模型有充裕的算力冗余。

### A100 核的"秘密武器"指令

A100 核在标准 RISC-V V 扩展基础上加了**自研 IME2 指令**：

| 指令 | 含义 | 干啥的 |
| --- | --- | --- |
| `vmadotsu` | unsigned × signed → int32 | 一次算 32 个 int4×int8 的乘加 |
| `vmadotu.hp` / `vmadotsu.hp` | unsigned × signed → **fp16 累加器** | 同上但累加器是 fp16，吞吐翻倍 |

> `hp` = half-precision partial sum，即"半精度部分和"。

每个 A100 核有自己私有的 **4MB TCM（Tightly Coupled Memory，紧耦合内存）**——比 DDR 快 5-10 倍的低延迟 SRAM。**K3 一共 8 × 4MB = 32MB TCM**，是 SpacemiT 后端把"正在算的那一片权重"搬过去的最大资本。

### 启动时如何识别 K3？

`ggml/src/ggml-cpu/spacemit/ime_env.cpp:18-83` 读 `/proc/cpuinfo` 拿 `marchid`：

| marchid | 映射到 | 含义 | 出现在哪颗芯片 |
| --- | --- | --- | --- |
| `0x5064` | `core_arch_x100` | X-class 核（主控） | K1/K2/K3 的主控 |
| `0xA03C` | `core_arch_a60`  | A-class 核（AI） | K1/K1-bis 的 AI |
| `0xA064` | `core_arch_a100` | A-class 核（AI） | **K3 的 AI** ← 8 个 |
| `0x50C8` | `core_arch_x200` | X-class 核（主控） | 下一代 |
| `0xA0C8` | `core_arch_a200` | A-class 核（AI） | 下一代 |

K3 上跑：

- 8 个核的 `marchid=0x5064` → `core_arch_x100`（主控）
- 8 个核的 `marchid=0xA064` → `core_arch_a100`（AI）

`ime_env.cpp:198-209` 启动时按"高 4 bit = 0xA"自动把 8 个 A100 核选为 prefer：

```cpp
for (auto & core_info : core_info_list) {
    auto core_arch_id   = core_info.arch_id;
    auto core_arch_head = (uint16_t) (core_arch_id) >> 12;
    if (core_arch_head == 0xA) {     // 0xA*** = A-class
        num_perfer_cores++;          // K3: += 8
        perfer_core_arch_id = core_arch_id;   // core_arch_a100
        cpu_mask |= (1ULL << core_info.core_id);
        perfer_core_ids.push_back(core_info.core_id);
    }
}
```

`ime_env.cpp:259-262` 根据 `perfer_core_arch_id` 决定用哪条 IME 路径：

```cpp
use_ime1 = perfer_core_arch_id == core_arch_a60 || perfer_core_arch_id == core_arch_x100;
use_ime2 = perfer_core_arch_id == core_arch_a100;
```

K3：`perfer_core_arch_id = core_arch_a100` → `use_ime2 = true`，走 **IME2 路径**。

---

## 2. 准备知识：llama.cpp 是什么？

一个**纯 CPU/GPU 推理框架**，但支持各种"伪后端"——只要这个后端能吐"张量 + 算子"的 API 就能接入。**SpacemiT 后端就是借 CPU 后端的光挂上去的**。

整个仓库的核心思想：
1. 把模型描述成**一张静态计算图**（节点=算子，边=张量）；
2. 把图分到不同**后端**上跑（CPU、CUDA、Metal、SpacemiT…）；
3. 每个后端有自己的**算子实现**。

---

## 3. 整体流程：用户视角的 5 个阶段

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ ① 启动    │→│ ② 加载    │→│ ③ 准备    │→│ ④ 推理    │→│ ⑤ 退出    │
│ backend   │  │ 模型      │  │ 上下文    │  │ 循环      │  │ 释放      │
│ 注册      │  │ repack   │  │ 分配图    │  │ decode   │  │           │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
   0.1 s         0.5-2 s         0.1 s         每 token 30ms
```

下面用具体数字（Q4_0 Qwen2.5 0.5B + K3）讲清每一阶段。

---

## 4. 阶段 ①：启动 + Backend 注册

### 用户敲命令

```bash
./llama-cli -m qwen2.5-0.5b-instruct-q4_0.gguf -n 32 -p "你好"
```

### `main()` 干了什么

`examples/llama-cli/.../cli.cpp` 大致：

```cpp
int main(int argc, char **argv) {
    // 解析参数
    ...
    ggml_backend_load_all();          // ← 关键一步：拉起所有 backend
    ...
    llama_model * model = llama_model_load_from_file(argv_model, mparams);  // ②
    llama_context * ctx  = llama_init_from_model(model, cparams);          // ③
    ...
    for (...) {
        llama_decode(ctx, batch);     // ← ④
        llama_sampler_sample(smpl, ctx, -1);
    }
    ...
}
```

### `ggml_backend_load_all()` 内部

**`ggml/src/ggml-backend-reg.cpp:555-586`** —— 静态链接时调 `ggml_backend_cpu_reg()`：

```cpp
void ggml_backend_load_all_from_path(const char * dir_path) {
    ggml_backend_load_best("cpu", silent, dir_path);  // ← CPU backend
    ...
}
```

**`ggml/src/ggml-cpu/ggml-cpu.cpp:690-703`** —— `ggml_backend_cpu_reg()`：

```cpp
ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    ggml_cpu_init();                                  // ①.1 探测 CPU 特性
    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,  // ①.2 关键接口表
        ...
    };
    return &ggml_backend_cpu_reg;
}
```

`ggml_backend_cpu_reg_i.get_extra_bufts` 字段指向 **`ggml-cpu/ggml-cpu.cpp:76-86`**：

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

`ggml_backend_cpu_get_extra_buffer_types` 在 **`ggml-cpu/ggml-cpu.cpp:42-56`**：

```cpp
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());  // ★ SpacemiT 进来了
        }
#endif
        ...
        return bufts;
    }();
    return bufts;
}
```

`ggml_backend_cpu_riscv64_spacemit_buffer_type()` 自身在 **`spacemit/ime.cpp:1647-1665`**：

```cpp
ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
        .iface = {
            .get_name      = ... "CPU_RISCV64_SPACEMIT",
            .alloc_buffer  = ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
            .get_alignment = ... 64,
            .get_alloc_size= ggml_backend_riscv64_spacemit_nbytes,
            ...
        },
        .device  = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        .context = new ggml::cpu::riscv64_spacemit::extra_buffer_type(),  // ★ 关键
    };
    return &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}
```

**`extra_buffer_type` 子类对象** 被装到 `.context` —— 它就是后续"是否能跑这个 op"的判断依据。

### 同时，`spine_env_info` 也在静默初始化

`ime.cpp:1255` 末尾有：

```cpp
}  // namespace ggml::cpu::riscv64_spacemit
spine_env_info global_spine_env_info;   // ★ 全局对象
```

这个全局对象在第一次被引用时构造——`spine_env_info::spine_env_info()` 在 **`ime_env.cpp:148-305`** 干这些事：

1. 读 `/proc/cpuinfo` → 发现 8 个 `a100` 核 → `num_perfer_cores=8`, `cpu_mask=0xFF00`（假设 a100 在 core 8-15）
2. 设 `use_ime2 = true`（因为 K3 的 prefer 是 A100），`use_tcm = true`
3. `madvise(MADV_HUGEPAGE)` 申请 2MB 大页做内存池
4. `dlopen("libspine_tcm.so")` 拿到 TCM 入口 → 探测到 **每核 4MB TCM**（共 32MB）
5. 在共享内存里建 16 个 `spine_barrier_t`（线程间同步用，最多支持 32 线程 = 8 a100 核 × 4 thread/核）

> **到本阶段结束，llama.cpp 在系统里"挂"上了一个特殊 buffer type**（伪装成 CPU buffer，但实际走 SpacemiT 路径）。

---

## 5. 阶段 ②：加载模型 + 一次性 Repack

### 加载 GGUF 文件

`src/llama.cpp:426` → `llama_model_load_from_file` 读 GGUF（GGML Universal Format，llama.cpp 的模型文件格式）。

GGUF 头里写着：
- 魔数 `GGUF`
- 元数据 KV 对：`general.architecture=qwen2`, `qwen2.block_count=24`, `qwen2.embedding_length=896`, `qwen2.attention.head_count=14` …
- 一堆 tensor 数据，每个张量都标了 shape + dtype（Q4_0 = 12）

加载完会得到 24×3+1+1 = 74 个权重 tensor：

| Tensor 名 | shape | dtype | 大小 |
| --- | --- | --- | --- |
| `token_embd.weight` | `[896, 151936]` | Q4_0 | ≈ 261 MB |
| `blk.0.attn_q.weight` | `[896, 896]` | Q4_0 | ≈ 1.54 MB |
| `blk.0.attn_k.weight` | `[128, 896]` | Q4_0 | ≈ 220 KB |
| `blk.0.attn_v.weight` | `[128, 896]` | Q4_0 | ≈ 220 KB |
| `blk.0.attn_output.weight` | `[896, 896]` | Q4_0 | ≈ 1.54 MB |
| `blk.0.ffn_gate.weight` | `[896, 4864]` | Q4_0 | ≈ 8.4 MB |
| `blk.0.ffn_up.weight` | `[896, 4864]` | Q4_0 | ≈ 8.4 MB |
| `blk.0.ffn_down.weight` | `[4864, 896]` | Q4_0 | ≈ 8.4 MB |
| `output.weight` | `[896, 151936]` | Q4_0 | ≈ 261 MB |
| `output_norm.weight` | `[896]` | F32 | 3.5 KB |
| ... | （每层 9 个 tensor，× 24 层）| | |

总权重 ≈ **330 MB**（Q4_0 后）。

### 给 weight tensor 选 buffer type

scheduler (`ggml_backend_sched_alloc_graph`) 给每个 tensor 问："你能跑在哪儿？"

对 `blk.0.attn_q.weight` 这种 Q4_0 量化 weight，**`ggml_backend_cpu_device_supports_op`** 在 **`ggml-cpu/ggml-cpu.cpp:423-439`** 会被调到：

```cpp
for (int i = 0; i < 4; i++) {
    if (op->src[i] && op->src[i]->buffer &&
        ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
        auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
        return buf_extra->supports_op(dev, op);   // ← 问 SpacemiT
    }
}
```

`buf_extra->supports_op` 走到 **`spacemit/ime.cpp:1580-1611`**：

```cpp
bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {  // ★ 关键检查
                if (op->src[1]->type == GGML_TYPE_F32) {
                    return true;   // ★ 答："我能跑"
                }
            }
            break;
        ...
    }
    return false;
}
```

`ggml_riscv64_spacemit_get_optimal_repack_type` 在 **`ime.cpp:1257-1393`**——对 Q4_0 weight，它选：

```cpp
case GGML_TYPE_Q4_0:
    if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 && (use_ime2)) {
        return &q4_0_32x256_q8_0;          // ★ 32 列 × 256 K 块的高吞吐版
    } else if (cur->ne[1] % 32 == 0 && (use_ime2)) {
        return &q4_0_32x32_q8_0;           // 32 列 × 32 K 块的标准版
    } else if (use_ime1) {
        return &q4_0_16x32_q8_0;           // 16 列 × 32 K 块 (IME1)
    }
```

`q4_0_32x256_q8_0` 是 **`tensor_traits<block_q4_0, 256, 32>`** 的静态实例（在 `ime.cpp:1243`），它的意思是：
- BLOC_TYPE = `block_q4_0`（每 32 元素一个 Q4_0 块）
- INTER_SIZE = 256（一次 GEMM 内 K 维并行 = 8 个 Q4_0 块）
- NB_COLS = 32（一次 GEMM N 维输出 = 32 行）

注意这一步 scheduler 选完 buffer type 后，**会立刻把 tensor 的地址用 `ggml_backend_riscv64_spacemit_buffer_type_alloc_buffer` 分到大页池里**（`ime.cpp:1480-1488`）：

```cpp
void * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);  // 64 字节对齐
```

### 一次性 Repack（重排）

**`spacemit/ime.cpp:1443-1458`** —— `set_tensor` 时调：

```cpp
static void ggml_backend_riscv64_spacemit_buffer_set_tensor(buffer, t, data, offset, size) {
    auto traits = (tensor_traits_base *) t->extra;    // ← 上面选好的
    if (traits) {
        auto OK = traits->repack(t, data, size);     // ★ repack
    }
}
```

`traits->repack` → **`spacemit/ime.cpp:931`** → **`spacemit/repack.cpp`** 的某个 `repack_q4_0_to_q4_0_*_bl` 函数。

**Q4_0 的 repack 实质**——把 GGUF 默认的"逐行 32 元素块"重排成 IME 友好的 **"32 行 × 256 列" 块**。

GGUF 原始排布（针对 `attn_q.weight`, shape `[896, 896]`）：

```
行 0: [Q4_0 block: d0 + 32 nibbles] [Q4_0 block: d1 + 32 nibbles] ... [Q4_0 block: d27 + 32 nibbles]
行 1: [Q4_0 block] ... [Q4_0 block]
...
行 895: ...
```
每行 28 个 Q4_0 块（896/32）。

IME 喜欢的排布（`q4_0_32x256_q8_0`）：

```
列块 0: 32 行 × 8 个 32 元素子块 × (16 scales + 16*32 nibbles)  ← 完整 32×256 块
       = 32 × (8×(2+16)) = 32×144 = 4608 字节
列块 1: 同样
...
列块 27: 同样
```

`repack.cpp:152-185` 的 `repack_q4_0_to_q4_0_16_bl`（IME1 用）展示了最简版的 16 行重排思路：

```cpp
static block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x16 out;
    // 1. 16 个 scale 一起放在前面
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    // 2. 把 int4 nibbles 交叉：把同一行的低 4bit 和高 4bit 拆开重新打包
    //    src: [b0 b16][b1 b17]...[b15 b31]
    //    dst: [b0 b1]...[b15 b16]   ← 32 列连续
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < QK4_0 / 4; j++) {
            out.qs[i*QK4_0/4 + j] = (in[i].qs[j] & 0x0F) | ((in[i].qs[j+QK4_0/4] & 0x0F) << 4);
        }
    }
    return out;
}
```

> **这一步只发生一次**，加载时把 330 MB 数据重排成 IME 喜欢的 N×K 排布。代价 ~几百毫秒，但**之后每次推理都受益**。

> **本阶段结束，330 MB 的 Q4_0 权重已经躺在 2MB 大页里，格式是 IME 友好的 32×256 块**。

---

## 6. 阶段 ③：准备上下文 + 构造计算图

### `llama_init_from_model` 干啥

1. 建 KV cache：24 层 × 2（K/V）× 14 head × 64 dim × 1024 上下文 = 几 MB
2. 创 scheduler：包含 CPU device（带 SpacemiT extra buffer）
3. 准备好 `llama_batch` 数据结构

### `llama_decode` 第一次被调

`src/llama-context.cpp:4054` → `llama_context::decode()` (`src/llama-context.cpp:1680`) → `process_ubatch()` (`src/llama-context.cpp:1304`)：

```cpp
gf = model.build_graph(gparams);                          // ★ 关键
ggml_backend_sched_alloc_graph(sched.get(), gf);          // 给每个张量分配 buffer
res->set_inputs(&ubatch);                                 // 写入 prompt tokens
const auto status = graph_compute(res->get_gf(), ...);    // 跑图
```

### `model.build_graph(gparams)` 干啥

调到 **`src/models/qwen2.cpp:49-150`** 的 `llama_model_qwen2::graph::graph()` 构造。

对 24 层 Qwen2.5 0.5B，它拼出**一个 ~600 个节点的 `ggml_cgraph`**：

```
[ token_embd ]   (embedding lookup)
   ↓
┌──── for il=0..23 ─────────────────────────┐
│ [ rms_norm ]   (attn_norm, 896 个 fp16)     │
│ [ q_proj ]     (MUL_MAT: 896×896 权重 @ 1×896 输入)
│ [ k_proj ]     (MUL_MAT: 896×128 权重 @ 1×896 输入)
│ [ v_proj ]     (MUL_MAT: 896×128 权重 @ 1×896 输入)
│ [ q_rope ]     (GGML_OP_ROPE)
│ [ k_rope ]     (GGML_OP_ROPE)
│ [ flash_attn ] (GGML_OP_FLASH_ATTN_EXT)
│ [ o_proj ]     (MUL_MAT: 896×896 权重 @ 1×896 输出)
│ [ +残差 ]      (GGML_OP_ADD)
│ [ rms_norm ]   (ffn_norm)
│ [ gate_proj ]  (MUL_MAT: 896×4864 权重 @ 1×896 输入)
│ [ up_proj ]    (MUL_MAT: 896×4864 权重 @ 1×896 输入)
│ [ silu*up ]    (GGML_OP_MUL 即 SwiGLU 门控)
│ [ down_proj ]  (MUL_MAT: 4864×896 权重 @ 1×4864 输入)
│ [ +残差 ]      (GGML_OP_ADD)
└───────────────────────────────────────────┘
   ↓
[ final_rms_norm ]
[ lm_head:  MUL_MAT: 151936×896 权重 @ 1×896 ]    ← 分类
   ↓
[ logits (1×151936) ]
```

`ggml_build_forward_expand(gf, logits)` 把这张图固化。

### 给 weight 张量分配 SpacemiT buffer

`ggml_backend_sched_alloc_graph` 遍历每个 weight tensor：
- 对 Q4_0 类型 → 选 SpacemiT buffer
- 对 F32 类型（norm 权重）→ 选普通 host buffer
- 调 `ggml_backend_riscv64_spacemit_buffer_type_alloc_buffer` 分配 → **返回的 tensor 都已经过 repack**

**`t->extra` 字段写入**：

```cpp
static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(buffer, t) {
    t->extra = (void *) ggml_riscv64_spacemit_get_optimal_repack_type(t);  // ★
    return GGML_STATUS_SUCCESS;
}
```

> 之后每次 `ggml_compute_forward` 走到 MUL_MAT 时，spacemit 通过 `t->extra` 找回 `tensor_traits*` 来执行。

### `set_inputs(&ubatch)` 写入

把 4 个 prompt token 的 id 写到 `token_embd` 的索引输入 `inp_embd` 里，position 写到 `inp_pos` 里。

> **本阶段结束，图已经拼好，330 MB 权重躺在 SpacemiT 大页池里**，只等一声令下。

---

## 7. 阶段 ④：推理循环（每个 token）

### `llama_decode` 主循环

`process_ubatch` → `graph_compute` → `ggml_backend_sched_graph_compute_async` → `ggml_backend_sched_compute_splits` → 调 `ggml_backend_cpu_graph_compute`。

`ggml_backend_cpu_graph_compute` 内部（`ggml-cpu/ggml-cpu.cpp:130+`）起**多个 worker 线程**（K3 上默认 **8 个**，对应 8 个 A100 AI 核）执行 `ggml_graph_compute_thread`：

### 线程入口（关键！）

**`ggml-cpu/ggml-cpu.c:3018-3091`**：

```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    ...
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);  // ★ 绑核
#endif

    struct ggml_compute_params params = {
        .ith = state->ith, .nth = total_threads, .wdata = work_data, ...
    };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        ...
        ggml_compute_forward(&params, node);    // ★ 节点分派
        ggml_barrier(state->threadpool);
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);  // 释放 TCM
#endif
    return 0;
}
```

`ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith)` 在 **`spacemit/ime.cpp:1690-1729`** 干这些事：

```cpp
void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n) {
    int cpu_id = sched_getcpu();
    // 1. 如果当前不在 a* 核，通过 /proc/set_ai_thread 触发内核迁移
    if (use_ime2 && !((1 << cpu_id) & cpu_mask)) {
        bind_ai_thread();  // write "0" to /proc/set_ai_thread
    }
    
    // 2. 申请本核的 TCM（仅第一次）
    if (use_tcm && tls_context.cpu_id == -1) {
        CPU_ZERO(&cpuset);
        CPU_SET(perfer_core_ids[thread_n], &cpuset);    // 绑到 a* 核
        pthread_setaffinity_np(main_thread, sizeof(cpuset), &cpuset);
        tls_context.cpu_id = perfer_core_ids[thread_n] - aicpu_id_offset;
        tls_context.tcm_buffer = spine_mem_pool_tcm_mem_get(ai_cpu_id);  // ★ 拿 4MB TCM
        tls_context.tcm_buffer_size = global_spine_env_info.tcm_blk_size;
    }
    
    // 3. 等 TCM 就绪
    if (tls_context.tcm_buffer != nullptr) {
        void * rt = spine_mem_pool_tcm_mem_wait(tls_context.cpu_id);
    }
}
```

> **到此，8 个 worker 线程被绑到 K3 的 8 个 a100 AI 核上，每个线程各持 4MB TCM**。这就是"异步计算加速"的基础。

---

## 8. 一个 MUL_MAT 节点的完整旅程（核心中的核心）

假设当前正在算第 1 层的 `q_proj`：
- weight `attn_q.weight`, shape `[896, 896]`, Q4_0, **已 repack** 为 32×256 块
- input 是当前 token 的 `x` (fp32), shape `[1, 896]`
- output 形状 `[1, 896]`

### Step 8.1：`ggml_compute_forward` 分派

`ggml-cpu/ggml-cpu.c:1702-1712`：

```c
static void ggml_compute_forward(params, tensor) {  // tensor = MUL_MAT 节点
    if (ggml_cpu_extra_compute_forward(params, tensor)) {  // ★ 优先问 SpacemiT
        return;
    }
    switch (tensor->op) { ... }  // 兜底默认
}
```

### Step 8.2：`ggml_cpu_extra_compute_forward` 路由

**`ggml-cpu/traits.cpp:12-23`**：

```cpp
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {  // 找 SpacemiT buft
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);   // ★ 拿 traits
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}
```

`buf_extra->get_tensor_traits` 走到 **`spacemit/ime.cpp:1613-1642`**：

```cpp
case GGML_OP_MUL_MAT:
    if (op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
        return (ggml::cpu::tensor_traits *) op->src[0]->extra;   // ★ 拿回初始化时写的 traits
    }
```

对 Q4_0 这就是 `q4_0_32x256_q8_0` = `tensor_traits<block_q4_0, 256, 32>` 静态实例。

### Step 8.3：`compute_forward` 进入 IME 路径

`tensor_traits<block_q4_0, 256, 32>::compute_forward` 在 **`spacemit/ime.cpp:187-232`**：

```cpp
bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            switch (op->src[0]->type) {
                case GGML_TYPE_Q4_0:
                    forward_mul_mat(params, op);   // ★ 调成员函数
                    return true;
            }
    }
    return false;
}
```

### Step 8.4：`forward_mul_mat` 三段式

`spacemit/ime.cpp:234-541` 干三件事：

#### 子步骤 A：选 kernel

```cpp
const int64_t gemm_m = ne11 * ne12 * ne13;   // 1 (单 token 解码)
const int64_t gemm_k = ne10;                  // 896
const int64_t gemm_n = ne01;                  // 896

spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;

#if defined(RISCV64_SPACEMIT_IME2)
    if (use_ime2) {
        quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
        quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
        if constexpr (BLOC_TYPE == block_q4_0) {
            if constexpr (INTER_SIZE == 256) {
                gemm_kernel = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;  // ★ 选 IME2 高吞吐 Q4_0
                block_stride_a = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
            }
        }
    }
#endif
```

对 K3（A100，IME2 路径）+ Q4_0，**最终选 `ime2::gemm_kernel_i8i4_hp`**（高吞吐 fp16 partial sum 版）。

#### 子步骤 B：在线量化 A 矩阵

`spacemit/ime.cpp:368-389`：

```cpp
} else {  // gemm_m > 1
    int task_per_thread = div_round_up(row_blks, nth);
    int m_row_blk_start = ith * task_per_thread;
    for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
        int m_idx = m_row_blk * 4;
        if (rows_tobe_handled == 4 && quantize_a_4row_i8) {
            quantize_a_4row_i8(a_blk_len, feature + m_idx * gemm_k, gemm_k,
                              quant_a_buffer + m_idx * row_stride_a);  // ★ 量化 4 行
        } else {
            quantize_a_row_i8(...);  // 1 行 fallback
        }
    }
}
```

对单 token (`gemm_m=1`)，走 `quantize_a_row_i8_hp`（`rvv_kernels.cpp` 里的 RVV intrinsic 实现）：

```cpp
void quantize_a_row_i8_hp(size_t blk_len, const float * a, size_t count_k, uint8_t * qa) {
    // 1. 把 256 个 fp32 元素切成 8 个 32 元素子块
    // 2. 每个子块：找 max(abs) → scale = max/127 → 量化到 int8
    // 3. 记录子块的 sum（用于 zero-point correction）
    // 4. 输出排布：|| scale(fp16) | sum(fp16) | 32 int8 || ...
}
```

> 输出 A 排布（`q8_hp_blk_size(256, true, true)`）：
> - 1 个 fp16 子块 scale (2 字节)
> - 1 个 fp16 子块 sum (2 字节)
> - 32 个 int8 (32 字节)
> - 总计每个 256 元素子块 = 36 字节
> - 896 元素 = 3.5 个子块 → 实际占 4 个子块 × 36 = 144 字节
> - 加 padding 到 8 字节对齐

A 量化后大约 896/256 × 36 ≈ 130 字节（单 token）。

`ggml_barrier(threadpool)` —— **8 个线程都完成量化后**，才能开始 GEMM。

#### 子步骤 C：选 GEMM 路径

`spacemit/ime.cpp:393-540`——根据 A 量化 buffer 能不能塞进 TCM 选 3 条路径之一：

```cpp
const int64_t per_mb_rows_wsize = 4 * row_stride_a;   // 4 行 A 量化 buffer 大小
const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;  // 32 列 B 大小

if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
    // ★ 路径 A：4 行 A 装进 TCM
} else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
    // ★ 路径 B：32 列 B 装进 TCM（双线程 ld/compute 流水线）
} else {
    // ★ 路径 C：fallback 任务分块
}
```

对 K3 + Qwen2.5 0.5B 的 `q_proj`：
- TCM = 4 MB
- per_mb_rows_wsize = 4 × 144 = 576 字节 ✓
- → **路径 A**（4 行 A 装进 TCM）

但单 token `gemm_m=1` 会走 fallback：选 **路径 C**（任务分块）或者路径 A 的 m=1 退化（实际还是把 1 行 A 装进 TCM）。

### Step 8.5：进入真正的 GEMM kernel

对路径 A (`spacemit/ime.cpp:403-432`)：

```cpp
for (int64_t m_start = ith * 4; m_start < gemm_m; m_start += 4 * nth) {
    // 1. 把 4 行 A 量化数据从 DRAM 拷到 TCM
    spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a, ...);

    // 2. 对 B 矩阵的每 32 列
    for (int64_t ni = 0; ni < gemm_n; ni += 32, b_col += 32 * row_stride_b) {
        int32_t rows_remaining = m_row_real;
        while (rows_remaining > 0) {
            // 3. ★ 真正的 IME2 GEMM
            auto rows_handled = gemm_kernel(blk_len, tcm_buffer, b_col, b_col_zp, c_blk,
                                            rows_remaining, n_blk_real, b_k_blks, gemm_n);
            c_blk += rows_handled * gemm_n;
            tcm_buffer += rows_handled * row_stride_a;
            rows_remaining -= rows_handled;
        }
    }
}
```

### Step 8.6：IME2 GEMM 内部（**这是最精彩的部分**）

`spacemit/ime2_kernels.cpp:2883+` 的 `gemm_kernel_i8i4_hp_m1` 是单 token 路径：

```
A（输入，已量化到 TCM）: 1 行 × 256 元素 = 1 个 "256 元素" 子块
B（权重，repack 后）:    32 行 × 256 元素 = 8 个 32 元素子块 × 32 行
C（输出）:              1 × 32 个 fp32
```

汇编核心（简化版）：

```asm
# 初始：v1 = B scale 累加器基址（fp16 × 16）
#      v0 = 1.0（fp16，用于 a_lo 路径的 a_scale 占位）
#      v2 = 0（fp32 累加器）

mv  s2, A_scale_ptr         # A 的 4 字节 scale (fp32) + 2 字节 sum (int16)
mv  s3, A_int8_ptr          # A 的 32 个 int8
mv  s4, B_scale_ptr         # B 的 32 个 fp16 scale
mv  s5, B_int4_ptr          # B 的 32 行 × 32 元素 int4 (1024 bit)
mv  s6, C_ptr               # 输出 32 个 fp32

vsetvli t0, x0, e16, m1
vmv.v.i v0, 1                # v0 = 1 (fp16, 当作 a_scale 16-bit 近似)
vxor.vv v2, v0, v0           # v2 = 0 (fp32 累加器)

K_LPST:
    # === 1. 加载 B 的 4 VRF (vl4r.v 一次取 4×256 bit = 1024 bit + 512 bit scale) ===
    vsetvli t0, x0, e8, m1
    vl4r.v v4, (s5)                  # v4/v5/v6/v7 = 4×32 元素 int4
    addi   s5, s5, 128*4 + 96        # 跳到下一组

    # === 2. 加载 B 的 32 个 fp16 scale ===
    vsetvli t0, x0, e8, mf2
    vle8.v v30, (s4)
    addi   s4, s4, 32*2 + 32          # scale (32B) + zp (32B, 即使没有也占位)

    # === 3. 加载 A 的 32 个 int8 + scale + sum ===
    vsetvli t0, x0, e8, mf4
    vle8.v v3, (s3)
    addi   s3, s3, 32+6               # 32 int8 + 4 字节 fp32 scale + 2 字节 int16 sum

    flw   f0, (s2)                    # A fp32 scale
    lh    t2, 4(s2)                   # A int16 sum
    addi  s2, s2, 32+6

    # === 4. 把 int8 拆成 lo4 (v8) 和 hi4 (v10) ===
    vsetvli t0, x0, e8, m1
    vsrl.vi v24, v3, 4                # 右移 4 位得 hi4
    vnpack4.vv v8,  v3,  v3,  3       # v8  = lo4 扩展到 8 个 i8 (signed)
    vnpack4.vv v10, v24, v24, 3       # v10 = hi4 扩展到 8 个 i8 (unsigned)

    # === 5. ★ 核心：vmadotsu.hp ===
    # 8 个 int8 元素 × 32 个 int4 weight = 8 × 32 个 i32 partial sums
    # 但 .hp 版本是 fp16 partial sums
    vmadotsu.hp v16, v10, v4, v1, 0, i4    # v16 = high 4 bits × v4
    vmadotsu.hp v18, v10, v5, v1, 0, i4    # v18 = high 4 bits × v5
    vmadotsu.hp v20, v10, v6, v1, 0, i4    # v20 = high 4 bits × v6
    vmadotsu.hp v22, v10, v7, v1, 0, i4    # v22 = high 4 bits × v7

    # === 6. ★ vmadotu.hp（low 4 bits 累加到同一 fp16 累加器）===
    vmadotu.hp  v16, v8,  v4, v0, 0, i4
    vmadotu.hp  v18, v8,  v5, v0, 0, i4
    vmadotu.hp  v20, v8,  v6, v0, 0, i4
    vmadotu.hp  v22, v8,  v7,  v0, 0, i4

    # === 7. 把 fp16 partial sums pack 回 int16 视图（节省寄存器） ===
    vpack.vv v24, v16, v18, 1
    vpack.vv v26, v20, v22, 1
    vpack.vv v16, v24, v26, 2

    # === 8. mac × b_scale (fp16 * fp16 → fp32) ===
    vsetvli t0, x0, e16, mf2
    vfwmul.vv v31, v30, v16

    # === 9. ★ 最终累加：acc += a_scale * (mac * b_scale) ===
    vsetvli t0, x0, e32, m1
    vfmacc.vf v2, f0, v31

    addi t3, t3, -1                  # K 块计数
    bgtz  t3, K_LPST                  # 循环

# === 10. 写回 C ===
vse32.v v2, (s6)
```

**这条汇编干了什么**——一次内层循环就完成了：
- 32 个 int4 weight × 32 个 int8 activation = 1024 个乘
- 全部加到 fp16 partial sum
- 再乘 b_scale (fp16)
- 再加 a_scale (fp32) 累加到 fp32 输出寄存器

A100 核的 `vmadotsu.hp` 一个时钟周期能完成 32 个 int4×int8 的乘加（4 VRF × 8 lane = 32 个）。**比通用 RVV `vmadot`（int32 累加）快约 2 倍**，因为 fp16 partial sum 的吞吐是 fp32 的 2 倍。

### Step 8.7：节点结束 barrier

```c
ggml_barrier(state->threadpool);   // 8 个线程同步
```

### Step 8.8：所有节点跑完

回到 24 层循环 → 跑完所有层 → final norm → lm_head（这是个 151936×896 的 MUL_MAT）→ 输出 logits (1×151936) → sampler → 下一个 token id。

---

## 9. 双线程 ld/compute 流水线（路径 B，K3 上更常用）

对更大的 token 批次（如 batch_size=8），A 量化 buffer 装不下 TCM（8×144=1.1KB 还行，但 m=64 就 9KB 装不下），改走 **路径 B**（**`spacemit/ime.cpp:433-492`**）：

```cpp
// 8 个线程两两配对：(0,1) (2,3) (4,5) (6,7)，共享 spine_barrier[0..3]
// 偶数线程 (ith%2==0) 负责 memcpy
// 奇数线程 (ith%2==1) 负责 GEMM
// 用 barrier 同步

if (ith % 2 == 0) {
    rvv::memcpy1d(b_col, w_data + ni * row_stride_b, nb_real * row_stride_b);  // ld B 到 TCM
    if (a_row != quant_a_buffer) rvv::memcpy1d(a_row, quant_a_buffer, ws_size);
}
spine_barrier_wait(cur_barrier);  // ★ 等奇数线程到位

if (ith % 2 != 0) {
    rvv::memcpy1d(a_row, quant_a_buffer, ws_size);  // ld A
    rvv::memcpy1d(b_col, w_data + ni * row_stride_b, nb_real * row_stride_b);
}

for (; ni < gemm_n; ni += NB_COLS * nth) {
    if (ith % 2 != 0) spine_barrier_wait(cur_barrier);  // 上一轮 ld 完成
    gemm_kernel(...);                                    // ★ 偶数线程算 GEMM
    if (ith % 2 == 0) spine_barrier_wait(cur_barrier);   // 同步给奇数线程 ld 下一块
    // 奇数线程开始 ld 下一块
    if (next_ni < gemm_n) {
        rvv::memcpy1d(b_col, w_data + next_ni * row_stride_b, ...);
    }
}
```

> **双发射思想**：K3 的 A100 核是双发射的——一条指令做 memcpy，下一条指令做 GEMM，两条流水线并行跑。SpacemiT 后端用"双线程 + barrier"在软件层面实现了这个双发射：
> - 偶数线程这一轮 memcpy，奇数线程上一轮 GEMM；
> - 偶数线程下一轮 GEMM，奇数线程这一轮 memcpy。
> - `spine_barrier_t`（cache-line 对齐的双变量 barrier）保证数据依赖正确。
> - 8 个 a100 核 × 2 线程/核 = 4 个 barrier pair（`spine_init_barrier_count=16`，够用）。

---

## 10. 一张总图：从 CLI 到电信号

```
$ ./llama-cli -m qwen2.5-0.5b-q4_0.gguf -p "你好"
   │
   ▼
[main]  →  ggml_backend_load_all()                     [backend-reg.cpp:555]
             └─→ ggml_backend_cpu_reg()                [ggml-cpu.cpp:690]
                 └─→ 注册 SpacemiT extra buffer type   [ggml-cpu.cpp:42 + spacemit/ime.cpp:1647]

   │
   ▼
[main]  →  llama_model_load_from_file()                [src/llama.cpp:426]
             └─→ 读 GGUF 文件
                 └─→ 给每个 Q4_0 weight 选 SpacemiT buft   [ime.cpp:1257+]
                     └─→ 调 alloc_buffer() → 2MB 大页     [ime.cpp:1480]
                         └─→ 调 set_tensor() → repack()  [ime.cpp:1443 + repack.cpp]
                              Q4_0 [896×896] → 32×256 块排布

   │
   ▼
[main]  →  llama_init_from_model()                     [src/llama-context.cpp]
             └─→ 建 KV cache (24 层 × 14 头 × 64 dim × 1024 ctx)
             └─→ 建 scheduler（含 SpacemiT extra buft）

   │
   ▼
[main]  →  for each prompt token: llama_decode()       [src/llama-context.cpp:4054]
             └─→ process_ubatch()                       [src/llama-context.cpp:1304]
                 ├─→ model.build_graph()                [src/models/qwen2.cpp]
                 │   拼出 24 层 × (QKV + Attn + FFN) + lm_head = 600+ 节点
                 ├─→ ggml_backend_sched_alloc_graph()   分配张量
                 ├─→ res->set_inputs(&ubatch)           写 token id
                 └─→ graph_compute()                     [src/llama-context.cpp:2421]
                     └─→ ggml_backend_sched_graph_compute_async()
                         └─→ ggml_backend_sched_compute_splits()
                             └─→ 调 CPU backend 跑每个 split
                                 └─→ ggml_graph_compute_thread()    [ggml-cpu.c:3018]
                                     ├─→ set_numa_thread_affinity()   [ime.cpp:1690]
                                     │   ① sched_getcpu()
                                     │   ② bind_ai_thread (/proc/set_ai_thread)
                                     │   ③ pthread_setaffinity_np → 绑到 a100 核
                                     │   ④ spine_mem_pool_tcm_mem_get() → 拿 4MB TCM
                                     │   ⑤ spine_mem_pool_tcm_mem_wait() → 等 TCM 就绪
                                     │
                                     └─→ for each node in graph:
                                         ggml_compute_forward()         [ggml-cpu.c:1702]
                                           ├─→ ggml_cpu_extra_compute_forward()  [traits.cpp:12]
                                           │    └─→ extra_buffer_type::get_tensor_traits()  [ime.cpp:1613]
                                           │        └─→ tensor_traits::compute_forward()      [ime.cpp:187]
                                           │            └─→ forward_mul_mat()                  [ime.cpp:234]
                                           │                ├─→ quantize_a_row_i8_hp()         [rvv_kernels.cpp]
                                           │                │   fp32[896] → int8 + scale + sum
                                           │                ├─→ rvv::memcpy1d() → 把 A/B 装到 TCM
                                           │                └─→ gemm_kernel_i8i4_hp()         [ime2_kernels.cpp:2883]
                                           │                    ★ vmadotsu.hp × 8 条
                                           │                      32×32 int4×int8 → fp16 partial
                                           │                    ★ vmadotu.hp × 8 条
                                           │                    ★ vfwmul (×b_scale)
                                           │                    ★ vfmacc (×a_scale)
                                           │                    → 32 个 fp32 输出
                                           └─→ (兜底 switch) 其它 op
                                         ggml_barrier(threadpool)        节点间同步

   │
   ▼
[main]  →  llama_sampler_sample() → 下一个 token id
   │
   ▼  循环
[main]  →  ... 继续生成直到 n_predict 个 token
```

---

## 11. 关键数字汇总（Qwen2.5 0.5B Q4_0 on K3）

| 项目 | 数值 | 备注 |
| --- | --- | --- |
| K3 总核数 | 16 | 8× X100 + 8× A100 |
| AI 协处理器 | 8× A100 (marchid=0xA064) | llama.cpp worker 线程目标 |
| 模型大小（Q4_0） | ≈ 330 MB | 24 层 + embedding + lm_head |
| KV cache（ctx=1024） | ≈ 1.4 MB | 24 层 × 2 × 14 头 × 64 dim × 1024 × 2 bytes |
| 单 token 推理时间（K3 8 核满载） | ≈ 15-30 ms | 8 个 a100 核并行 |
| Pre-fill（128 token） | ≈ 0.5-1.5 s | 取决于 batch 大小 |
| 单层 MUL_MAT 数量 | 7 个 | Q/K/V/O/Up/Gate/Down |
| 单 token 调 `gemm_kernel_i8i4_hp` 次数 | 7 × 24 = 168 次 | 24 层 × 7 个 MUL_MAT |
| 每次 `gemm_kernel` 内 `vmadotsu.hp` 数 | 8 条 | 4 VRF × lo/hi = 8 |
| 每次 `vmadotsu.hp` 完成 MAC | 32 个 | 4 VRF × 8 lane |
| K3 TCM 总容量 | 8 × 4 = 32 MB | 每核 4MB |
| 加载时 repack 耗时 | 几百 ms | 一次性 |
| A 量化（单 token） | ≈ 5-10 μs | RVV vfabs/vfredmax |
| A100 核主频 | ~1.5-2 GHz | SpacemiT 公开资料 |

---

## 12. 一句话总结（给小白）

> **llama.cpp 把 Qwen2.5 0.5B 描述成一张 600 个节点的"算子菜谱"**。
> **SpacemiT 后端在加载时把 Q4_0 权重一次性重新摆盘**（repack），让 K3 的 8 个 A100 AI 核能用专用指令 `vmadotsu.hp` 一次完成 32×32=1024 个 int4×int8 乘加。
> **推理时，8 个 worker 线程被分别绑到 8 个 A100 核并各抢占 4MB TCM**（共 32MB TCM），在线把 fp32 激活量化成 int8 后，从 TCM 喂给 GEMM 内核。
> **路径 A 直接单线程跑 GEMM，路径 B 用双线程 + cache-line barrier 实现 ld/compute 流水线**，靠 A100 的双发射能力把数据搬运和计算重叠。
> **24 层 × 7 个 MUL_MAT × 8 条 vmadotsu.hp × 32 个 MAC = 43 万次乘加每 token**，全部跑在 A100 核上，8 核并起来 ≈ K3 实测每 token 15-30 ms。
