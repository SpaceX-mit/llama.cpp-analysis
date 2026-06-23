# `ggml-cpu/spacemit/` 深度分析

> 对应目录：`/data/llama.cpp-analysis/llama.cpp/ggml/src/ggml-cpu/spacemit/`
> 适用平台：SpacemiT K1 (X60)、K2/X60、X100、X200、K1-bis (A60)、Banbu K2 (A100)、Banbu K3 (A200) 等基于 RISC-V 64 + 自研 IME NPU 的 SoC。
> 编译开关：`GGML_CPU_RISCV64_SPACEMIT=ON`，并 `RISCV64_SPACEMIT_IME_SPEC` 在 cmake 中决定 `IME1` 或 `IME2`（FindSMTIME.cmake）。
> 必备编译宏：`RISCV64_SPACEMIT_IME1` 或 `RISCV64_SPACEMIT_IME2` 二选一（`ime.cpp:49-52` `#error` 强制）。
> 必备 RISC-V 扩展：`V`（向量扩展）、`Zfh` / `Zvfh`（半精度浮点）、`Zba`（地址生成加速）。IME2 还需要 `Xsmtvdotii`（GCC≥15 自动加）或 `-march=rv64gc_xtheadvector`。

---

## 1. 目录全景

```
spacemit/
├── ime.h / ime.cpp           ← 后端入口 / buffer type / tensor_traits / 主算子分发
├── ime_kernels.h             ← GEMM 函数声明 + 量化的 block layout
├── ime1_kernels.cpp          ← IME1 (A60/X100) 的内联汇编 GEMM（vmadot）
├── ime2_kernels.cpp          ← IME2 (A100/A200) 的内联汇编 GEMM（vmadotsu / vmadotu.hp）
├── repack.h / repack.cpp     ← 加载时把通用 Q4_K/Q6_K/Q8_0... 重排成 IME 友好的 N×K
├── rvv_kernels.h / .cpp      ← 非 GEMM 算子的 RVV intrinsics 实现
│                                (RMSNorm / FlashAttn / Softmax / Concat / SumRows / Repeat / GetRows)
├── ime_env.h / ime_env.cpp   ← 启动期硬件拓扑探测、TCM 初始化、内存池选择
├── spine_barrier.h           ← 轻量级自旋 barrier (cache-line padded)
├── spine_mem_pool.h / .cpp   ← 大页内存池 (transparent hugepage / 1G hugetlb / posix)
└── spine_tcm.h               ← libspine_tcm.so 动态加载头（双模式：直接链接 / dlopen）
```

总计 **~15k 行 C++**。下文按数据流顺序逐模块展开。

---

## 2. 启动期：`ime_env` + `spine_mem_pool` + `spine_tcm`

### 2.1 拓扑探测 `spine_core_info::get_spine_core_info` (`ime_env.cpp:18`)

通过 `/proc/cpuinfo` 读取 `processor` + `marchid`：

| marchid | 映射到的 arch id | 芯片 |
| --- | --- | --- |
| `0x8000000058000001` | `core_arch_x60` | SpacemiT K1 (X60) |
| `0x8000000041000001` | `core_arch_a60` | SpacemiT K1-bis (A60) — AI 协处理器核 |
| `0x8000000058000002` | `core_arch_x100` | SpacemiT K2 (X100) |
| `0x8000000041000002` | `core_arch_a100` | SpacemiT Banbu K2 (A100) — AI 协处理器核 |
| `0x5064` 内置 | `core_arch_x100` | QEMU 模拟（无 `/proc/cpuinfo` 兜底） |
| `0xA0C8` | `core_arch_a200` | Banbu K3 (A200) |

> 关键经验：SpacemiT SoC 是 **big.LITTLE** 异构，`x*` 是主控 A55，`a*` 是 AI 协处理器核。SpacemiT 后端的核心是只把 `ggml` 的 worker 线程绑到 `a*` 核上跑矩阵乘，让 `x*` 核做归一化、采样、循环控制等。

`spine_env_info::spine_env_info()` 构造时还做了：

- **`SPACEMIT_CORE_ARCH` / `SPACEMIT_PERFER_CORE_ARCH` / `SPACEMIT_PERFER_CORE_ID` 环境变量** 强制指定哪些核被用。
- **`SPACEMIT_MEM_BACKEND` 环境变量**：`none` / `posix` / `hpage` / `hpage1gb`，选择内存池后端。
- **`SPACEMIT_DISABLE_TCM` 环境变量**：非 0 时禁用 TCM 回退到纯 DRAM。
- **X60 K1 8 核** 特例：前 4 核被识别为 `a60`（实际是异构 SoC 的简化假设）。
- **aicpu_id_offset**：AI 核的 `core_id` 在 TCM API 里从 0 开始编号，需要减掉 A55 核的偏移。
- **`use_ime1` / `use_ime2` / `use_tcm`** 三个 boolean 决定了 kernel 派发路径。
- **init_barrier 分配**：用 `spine_mem_pool_shared_mem_alloc` 申请（fallback heap），且只在两个线程之间（`thread_count=2`）同步 — 这是为 IME 内部 ld/compute 流水线用的。

### 2.2 内存池 `spine_mem_pool_manager` (`spine_mem_pool.cpp:106-414`)

抽象类 + 三种后端实现：

| 后端 | 用途 | 实现 |
| --- | --- | --- |
| `posix_memalign` | 兜底，普通对齐 | `spine_mem_pool_posix` |
| `transparent_hugepage` | 默认，2MB 大页，TLB miss 显著下降 | `spine_mem_pool_transparent_hugepage`，`madvise(MADV_HUGEPAGE)` |
| `hugetlb_1g` | 高端，1GB 大页（需要 `/dev/hugetlb_1g` 内核模块） | `spine_mem_pool_hugetlb_1g`，通过 ioctl `HUGETLB_1G_IOC_ALLOC` 申请 |

核心数据结构：
- `pool_chunk`：每次大页 mmap 出来的一段（默认 512 MiB）。
- `free_block`：chunk 内的空闲段（按 offset 升序排），free 时合并相邻段。
- `pool_allocation`：用户拿到的 `<base, size, chunk_base>`，通过 `allocations_` 哈希表反查。
- `try_alloc_locked` 选 **base 地址最小** 的可对齐 block 切出来。

`spine_mem_pool_tcm_*` 单独管理 TCM（绕过普通 pool），用于 AI 核的紧耦合内存。详见 §2.4。

### 2.3 `spine_barrier_t` (`spine_barrier.h`)

```cpp
struct spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};
```

经典的两变量 barrier，cache-line 对齐避免 false sharing：

```cpp
inline void spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(acquire);
    auto cnt       = --b->pending_;
    if (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } else {
        while (cur_round == b->rounds_.load(relaxed)) {
            __asm__ volatile("pause " ::: "memory");
        }
    }
}
```

> 整个 `global_spine_env_info.init_barrier` 是个 16 元素的 `spine_barrier_t[]`（`spine_init_barrier_count=16`），每两个 thread pair 共用一对 barrier。

### 2.4 TCM：`spine_tcm.h` + `spine_mem_pool_tcm_*` (`spine_mem_pool.cpp:560+`)

TCM（Tightly Coupled Memory）是 SpacemiT AI 核上的低延迟 on-chip SRAM，**直接 map 到物理地址**，单核独享。

`spine_tcm.h` 提供两种使用模式：

- **直接链接** (`SPINE_TCM_DIRECT_LINK`)：编译期链接 `libspine_tcm.so`，`spine_tcm_is_available()` 直接调 runtime 桩。
- **头文件 loader 模式**（默认）：用 `dlfcn.h` 做延迟绑定，结构 `spine_tcm_handle_t` 装满了 dlsym 出的函数指针。

runtime ABI 共有 10 个函数：

```cpp
spine_tcm_runtime_is_available()         // 是否真的有 TCM（无 TCM 设备 → fake TCM）
spine_tcm_runtime_layout_info(info)      // blk_size, blk_num, is_fake_tcm
spine_tcm_runtime_mem_info(id, info)     // 单 block 的元数据
spine_tcm_runtime_mem_get(id)            // 拿 cache 住的 buffer
spine_tcm_runtime_mem_free(id)
spine_tcm_runtime_mem_try_wait(id, us)   // 等 block handoff
spine_tcm_runtime_mem_release(id)
spine_tcm_runtime_mem_force_release(id)
spine_tcm_runtime_mem_query(id)
spine_tcm_runtime_version()              // 调试用
```

应用层（`spine_mem_pool.cpp`）只暴露 5 个 wrapper：

```cpp
spine_mem_pool_tcm_init(&tcm_info);                          // 初始化：拿 layout
spine_mem_pool_tcm_mem_get(cpu_id)     → 立刻返回 cache 块
spine_mem_pool_tcm_mem_wait(cpu_id)    → 阻塞等
spine_mem_pool_tcm_mem_release(cpu_id) → 释放引用
spine_mem_pool_tcm_mem_get cpu_id   → tls_context.tcm_buffer
```

`tls_context`（`ime.cpp:75-80`）保存了 per-thread 的 `tcm_buffer` + `tcm_buffer_size`，由 `set_numa_thread_affinity` 在线程启动时通过：

```cpp
ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer = spine_mem_pool_tcm_mem_get(ai_cpu_id);
ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size = global_spine_env_info.tcm_blk_size;
```

固化下来，全算子共享。

---

## 3. 后端注册与 buffer type

### 3.1 CMake 钩子 (`ggml-cpu/CMakeLists.txt:438-499`)

```cmake
elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
    if (GGML_CPU_RISCV64_SPACEMIT)
        include(ggml-cpu/cmake/FindSMTIME.cmake)        # 检测 IME 编译器/库
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_RISCV64_SPACEMIT ${RISCV64_SPACEMIT_IME_SPEC})
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/spacemit/ime.cpp
            ggml-cpu/spacemit/ime1_kernels.cpp
            ggml-cpu/spacemit/ime2_kernels.cpp
            ggml-cpu/spacemit/rvv_kernels.cpp
            ggml-cpu/spacemit/repack.cpp
            ggml-cpu/spacemit/ime_env.cpp
            ggml-cpu/spacemit/spine_mem_pool.cpp
        )
    endif()
    ...
    if (GGML_CPU_RISCV64_SPACEMIT)
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
            string(APPEND MARCH_STR "_xsmtvdotii")     # VMADOT 扩展
        endif()
    endif()
```

`FindSMTIME.cmake` 检测 `marchid` → `IME1` / `IME2` 选择并设 `RISCV64_SPACEMIT_IME_SPEC` 为 `-DRISCV64_SPACEMIT_IME1` 或 `-DRISCV64_SPACEMIT_IME2`。

### 3.2 backend 注册 (`ggml-cpu/ggml-cpu.cpp:42-56`)

```cpp
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;
#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif
        ...
    }();
    return bufts;
}
```

注意是 **CPU backend 的 extra buffer type**——SpacemiT 后端在 backend 框架里被视作"同一颗 CPU 的一种 buffer 形式"，这意味着：
- 它复用 `ggml_threadpool`、复用 `ggml_backend_sched`；
- 但因为 IME 算子只跑在 `a*` 核上，CPU 上的 `set_numa_thread_affinity` 是不够的，所以多了一个 `ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith)` 在线程入口（`ggml-cpu.c:3026`）被显式调用。

### 3.3 `ggml_backend_cpu_riscv64_spacemit_buffer_type` (`ime.cpp:1647`)

构造 `ggml_backend_buffer_type`，关键 hook：

| 函数 | 实现 | 备注 |
| --- | --- | --- |
| `get_name` | `"CPU_RISCV64_SPACEMIT"` |  |
| `alloc_buffer` | `spine_mem_pool_alloc(size, 64)` | 64 字节对齐（cache line） |
| `get_alignment` | `64` |  |
| `get_alloc_size` | `ggml_backend_riscv64_spacemit_nbytes` | 关键！按目标布局算大小 |
| `iface.free_buffer` | `spine_mem_pool_free` |  |

### 3.4 `init_tensor` (`ime.cpp:1395`)

```cpp
static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(buffer, t) {
    t->extra = (void *) ggml_riscv64_spacemit_get_optimal_repack_type(t);
    return GGML_STATUS_SUCCESS;
}
```

`get_optimal_repack_type`（`ime.cpp:1257`）按 `tensor->type` 和 `ne[1] % 32`、`ne[0] % 256` 等约束选最佳特化：

| 量化类型 | 选哪个 tensor_traits |
| --- | --- |
| `Q4_0`  (ne1%32=0, ne0%256=0, IME2) | `q4_0_32x256_q8_0` (256 K 块) |
| `Q4_0`  (ne1%32=0, IME2) | `q4_0_32x32_q8_0` (32 K 块) |
| `Q4_0`  (ne1%16=0, IME1) | `q4_0_16x32_q8_0` (16 K 块) |
| `Q4_1` / `Q4_K` / `Q2_K` / `Q3_K` / `Q5_K` / `Q5_0` / `Q5_1` / `Q6_K` / `Q8_0` / `MXFP4` | 对应 `NNxK_q8_0` 特化 |

这些特化在文件末尾以 `static const` 实例化（`ime.cpp:1235-1251`），它们是模板 `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>` 的具现化，每个对应一个具体 repack + GEMM 实现。

### 3.5 `set_tensor`（一次性 repack）(`ime.cpp:1443`)

```cpp
static void ggml_backend_riscv64_spacemit_buffer_set_tensor(buffer, t, data, offset, size) {
    auto traits = (tensor_traits_base *) t->extra;
    if (traits) {
        auto OK = traits->repack(t, data, size);
        GGML_ASSERT(OK == 0);
    }
}
```

当 `ggml_backend_sched_alloc_graph` 把 weight tensor 真正写入时调用 `repack()`（→ `repack.cpp`），把通用 GGUF 排布转成 IME 友好的 N×K。

### 3.6 `nbytes` 重写 (`ime.cpp:1496`)

为每种量化类型在加载前就预计算 repack 后的大小，方便 mmap / 分配：

```cpp
case GGML_TYPE_Q4_K:  nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
case GGML_TYPE_Q6_K:  nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
case GGML_TYPE_Q2_K:  nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(nrow_block_q2_k<1>));
case GGML_TYPE_Q3_K:  nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(nrow_block_q3_k<1>));
...
```

> 注意：`Q5_K` 的目标 size 是 `nrow_block_q5_1<1> * 8`，意思是把 8 个 K-quant 子块压成 1 个 row block（`nrow_block_q5_1` 模板 N=8 的尺寸）。

### 3.7 `extra_buffer_type::supports_op` (`ime.cpp:1580`)

`ggml_backend_sched` 用它判断某个 op 是否能跑在该 buft 上。SpacemiT 后端只声明支持：

- `MUL_MAT` 当 `src0` 是 spacemit buffer，2D，且能匹配到一个最优 `repack_type`。
- `MUL_MAT_ID` 同上，但 `src0` 是 3D（多 expert）。
- `src1` 必须是 host buffer 且 `type==F32`（不在别的 backend 上）。

任何不符合的 op 全部 fall through 到默认 CPU backend（仍然是 RVV intrinsics 但走普通路径）。

---

## 4. Repack：把通用量化排布转成 IME N×K 排布

### 4.1 目的

IME 的 `vmadotsu` / `vmadotu.hp` 指令要求 **B 矩阵按 4 个虚拟寄存器 segment 排布**（N8K32，1024-bit + 512-bit scale），并要求 **行内做 interleave**：

- IME1 (Q4_0/Q4_1/Q4_K)：`16 rows × 32 cols`，每行的低 4bit 与高 4bit 交叉存放。
- IME2 (Q4_0/Q4_1/Q4_K/Q6_K/Q8_0/Q2_K/Q3_K/Q5_K/Q5_0/Q5_1/MXFP4)：`32 rows × 32 cols`（或 `32 rows × 256 cols` 的"高吞吐"模式）。

`repack.cpp` 把 GGUF 的逐行 `block_q4_K` 排布，转换为：

```cpp
struct block_q4_0x16 {                // 16 行 × 32 列 的 IME1 块
    ggml_half d[16];                  // 16 个 fp16 scale
    uint8_t   qs[16*QK4_0/4];         // i4 紧致交叉
};
static_assert(sizeof(block_q4_0x16) == 16*sizeof(ggml_half) + QK4_0*8);
```

对 IME2：

```cpp
struct block_q4_0x32x256 {           // 32 行 × 256 列
    block_q4_0x32 blocks[8];          // 8 个 32×32 子块连续排列
};
```

`Q2_K` / `Q3_K` / `Q5_K` / `Q5_1` / `Q5_0` / `MXFP4` 还有**专用的 nrow_block layout**（`ime_kernels.h:11-63`），按 N 行交错排，方便一次循环里 N 行同时取 scale 和量化值。

### 4.2 主要 repack 函数

`repack.cpp` 包含：
- `repack_q4_0_to_q4_0_16_bl` → IME1 16×32
- `repack_q4_0_to_q4_0_32_bl` → IME2 32×32
- `repack_q4_0_to_q4_0_32x256` → IME2 32×256
- `repack_q4_1_to_q4_1_*`  → 类似但带 zero-point
- `repack_q4_K_to_q4_K_*`  → 把 super-block scale/min 重排
- `repack_q2_K_to_q2_K_*`  → nrow_block_q2_k 排布
- `repack_q3_K_to_q3_K_*`  → nrow_block_q3_k 排布
- `repack_q5_K_to_q5_1_*`  → nrow_block_q5_1 排布
- `repack_q5_1_to_q5_1_*`  → 同上
- `repack_q5_0_to_q5_0_*`  → nrow_block_q5_0 排布
- `repack_mxfp4_to_mxfp4_*`→ nrow_block_mxfp4 排布
- `repack_q6_K_to_q6_K_*`  → 重排成 `block_q8_0 * 8`
- `repack_q8_0_to_q8_0_*`  → 重排成 32 行对齐

每个都做一遍 C++ 的标量重排，CPU 上跑一遍，但**只在加载模型时执行一次**，代价可接受。

### 4.3 模板入口 `repack<BLOC_TYPE, INTER_SIZE, NB_COLS>()` (`ime.cpp:931`)

```cpp
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size) {
    ...
}
```

`repack.cpp` 给每个 (BLOC_TYPE, INTER_SIZE, NB_COLS) 特化做 if-else 派发到对应函数。

---

## 5. 算子分派：`tensor_traits`

`ime.cpp:122-936` 定义了两个 `tensor_traits` 子类：

| 类 | 负责 op | 备注 |
| --- | --- | --- |
| `tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>` | `MUL_MAT` / `MUL_MAT_ID` | 一个模板类，13 个具现化（见 §3.4） |
| `tensor_traits_common` | `NORM` / `RMS_NORM` / `ADD`/`SUB`/`MUL`/`DIV` / `FLASH_ATTN_EXT` / `CONT` / `CPY` / `REPEAT` / `SUM_ROWS` / `GET_ROWS` / `CONCAT` | 全部走 RVV intrinsics，1 个具现化 `rvv_impl` |

### 5.1 接口

```cpp
class tensor_traits_base : public ggml::cpu::tensor_traits {
    virtual int repack(ggml_tensor * t, const void * data, size_t data_size) = 0;
};
class tensor_traits<...> : public tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override;  // 算 workspace
    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override; // 真·算
    void forward_mul_mat(params, op);   // 主算子
    void forward_mul_mat_id(params, op);// MoE
    int repack(...) override;           // → ggml::cpu::riscv64_spacemit::repack<...>
};
```

`ggml::cpu::tensor_traits` 在 `ggml-cpu/traits.h` 中定义，是 CPU backend 用来"接管"标准 op 的 hook。

### 5.2 `work_size`（workspace 估算）(`ime.cpp:128`)

对 MUL_MAT：

```cpp
case GGML_OP_MUL_MAT:
    if constexpr (BLOC_TYPE == Q2_K || Q3_K) {
        size = src1_ne * q8k_blk_size(QK_K);   // 256 + 2
    } else if constexpr (INTER_SIZE == QK4_0) {
        size = src1_ne * q8_blk_size(QK4_0, with_blk_sum=true);  // 4 + 32
    } else if constexpr (INTER_SIZE == 256) {
        size = src1_ne * q8_hp_blk_size(256, with_blk_sum=true, with_blk_scale=true);
    }
    size = GGML_PAD(size, sizeof(int64_t));
    return true;
```

对 MUL_MAT_ID：额外加 `mmid_row_mapping` 表的大小。

> `q8_blk_size` / `q8_hp_blk_size` / `q8k_blk_size` 在 `rvv_kernels.h:18-37` 定义，对应三种 A 矩阵量化排布：32 元素块、32 元素子块、256 元素 K-quant 块。

---

## 6. MUL_MAT 主算子：`forward_mul_mat` (`ime.cpp:234-541`)

### 6.1 算子三段式

```cpp
void forward_mul_mat(params, op) {
    // [1] 算子选型
    if (use_ime2) 选择 IME2 的 gemm_kernel + quantize 函数;
    else if (use_ime1) 选择 IME1 的;
    else GGML_ABORT;

    // [2] A 矩阵量化（fp32 → int8）
    for (m_row_blk in 自己的范围) {
        if (rows_remaining >= 4)
            quantize_a_4row_i8(...);
        else
            quantize_a_row_i8(...);
    }
    ggml_barrier(threadpool);

    // [3] GEMM 路径三选一
    if (gemm_m <= 全 N / M * 64)  // 全 N 在一列里
        走 "TCM 复用 a_row" 路径
    else if (per_nb_cols_wsize <= tcm_buffer_size)
        走 "双线程 ld/compute 流水线" 路径
    else
        走 "fallback 任务分块" 路径
}
```

### 6.2 路径 A：M 全在同列（TCM 装 A）

```cpp
for (int64_t m_start = ith * 4; m_start < gemm_m; m_start += 4*nth) {
    rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a, m_row_real * row_stride_a);
    for (n_blk) {
        gemm_kernel(blk_len, tcm_buffer, b_col, b_col_zp, c_blk, m_row_real, n_blk_real, b_k_blks, gemm_n);
    }
}
```

`gemm_m_stride = gemm_n / gemm_m > 64 ? gemm_m : 16`：根据 m/n 比例选 m 维 stride，避免 TCM 抖动。

### 6.3 路径 B：双线程 ld/compute 流水线

```cpp
if (ith % 2 == 0) {
    rvv::memcpy1d(b_col, w_data + ni * row_stride_b, nb_real * row_stride_b);   // ld
    if (a_row != quant_a_buffer) rvv::memcpy1d(a_row, quant_a_buffer, ws_size);
}
spine_barrier_wait(cur_barrier);
if (ith % 2 != 0) {
    if (a_row != quant_a_buffer) rvv::memcpy1d(a_row, quant_a_buffer, ws_size);
    rvv::memcpy1d(b_col, w_data + ni * row_stride_b, nb_real * row_stride_b);   // ld
}
for (; ni < gemm_n; ni += NB_COLS * nth) {
    if (ith % 2 != 0) spine_barrier_wait(cur_barrier);
    gemm_kernel(...);                                                          // compute
    if (ith % 2 == 0) spine_barrier_wait(cur_barrier);
    // ld next block
}
```

> 这是 SpacemiT 的精华：**用 RISC-V 双发射能力让一个线程搬运数据、一个线程跑 MAC，靠 barrier 同步**。每对 `(2k, 2k+1)` 线程共享一个 `spine_barrier_t`（`init_barrier[ith/2]`），所以 `spine_init_barrier_count=16` 最多支持 32 线程。

### 6.4 路径 C：fallback 任务分块

如果 TCM 装不下任何切块，按 1D 任务切 `task_count_m * task_count_n` 任务：

```cpp
int64_t task_per_thread = (task_count + nth - 1) / nth;
int64_t start = ith * task_per_thread;
int64_t end   = min((ith + 1) * task_per_thread, task_count);
for (compute_idx = start; compute_idx < end; ++compute_idx) {
    tid_n = compute_idx / task_count_m;
    tid_m = compute_idx % task_count_m;
    m_start = tid_m * gemm_m_stride;
    n_start = tid_n * gemm_n_stride;
    for (ni < n_count) {
        gemm_kernel(...);
    }
}
```

走 DRAM，`b_col` 直接 `w_data + n_start * row_stride_b`，无 TCM 也无 barrier。

---

## 7. MUL_MAT_ID (MoE) 主算子：`forward_mul_mat_id` (`ime.cpp:543-929`)

### 7.1 mmid_row_mapping 表构造

`mmid_row_mapping { int32_t i1; int32_t i2; }`：每条记录告诉 GEMM **"expert a 第几个 token 来自原始 batch 的哪一行"**。

```cpp
for (iid1 in ids->ne[1])
    for (id in ids->ne[0])              // 每个 token 选中的 expert
        MMID_MATRIX_ROW(expert, count++) = { id, iid1 };

valid_ep_count = 去重后有效 expert 数;
valid_act_count = 总 token 数;
```

### 7.2 两条路径

#### 路径 X（"完美 MoE"）：所有 expert 都被命中、TCM 足够大

```cpp
if (valid_ep_count_t % nth == 0 && tcm_buffer && per_nb_cols_wsize <= tcm_buffer_size
    && valid_ep_count_t == n_as && valid_act_count_t == n_as) {
    for (valid_id = ith; valid_id < n_expert; valid_id += nth) {
        // 单 expert 内 1 token × NB_COLS
        // 双线程 ld/compute 流水线
    }
}
```

#### 路径 Y（"通用 MoE"）：按 expert 拆分

```cpp
for (valid_id = ith_es; valid_id < n_expert; valid_id += nth_es) {
    cne1 = matrix_row_counts[expert];                 // 该 expert 接收的 token 数
    src0_n_start = (ith_n * ne01) / nth_n;            // 按 N 维分片
    ...
    if (tcm 足够 装 src0_N)
        memcpy b_col → tcm;
    if (extra_tcm >= nbw1) {
        // 整块 tile 处理
        do {
            // 一次把 4 个 token 的 src1 拷到 TCM (单 thread)
            // 然后调 moe_m2_gemm_kernel_i8i4 一次处理 2 个 token (双 thread)
            // 剩下不足 2 的部分走单 token gemm_kernel
        } while (...);
    } else {
        // 走 DRAM，2 个 token 同时跑 (moe_m2) + 1 个 token (gemm)
    }
}
```

> `moe_m2_gemm_kernel_*`（`ime2_kernels.cpp:5006-5529`）一次跑 2 个不同 dst 的 GEMM 共享 weight 加载，节省带宽。剩下的奇数 token 再走普通 `gemm_kernel_i8i4`。

### 7.3 加载 src1 量化

```cpp
if (ne11 == 1) {
    // prefill：每行 batch 内独立
    for (ii = ith; ii < ne12 * a_k_blks; ii += nth) {
        quantize_a_row_i8(...);
    }
} else {
    // decode：每个 expert 接收多个 token
    for (ii = ith; ii < ne12 * ne11; ii += nth) {
        quantize_a_row_i8(...);
    }
}
```

---

## 8. RVV Kernels (`rvv_kernels.cpp`) — 非 GEMM 算子

### 8.1 总览

`tensor_traits_common` 把这些 op 全部接走，避免落到 `ggml-cpu/ops.cpp` 的默认路径上：

| 函数 | 行 | 实现要点 |
| --- | --- | --- |
| `memcpy1d` / `memcpy2d` | 43-45 | RVV `vle8.v` / `vse8.v` 32-byte block copy |
| `forward_rms_norm_f32` | 1630 | `vfwmul.vv` 平方求和 → `vfredsum` → rsqrt 近似 |
| `forward_norm_f32` | – | mean + var + gamma*(x-mean)/σ |
| `forward_cont_with_permute` | – | 特殊步长 strided copy |
| `forward_cpy_with_permute` | – | 同上 |
| `forward_get_rows<T>` | – | 模板化，按行号取 embd |
| `forward_concat<T>` | – | 沿 dim=0 拼接两个同类型张量 |
| `forward_binary<op,T>` | – | ADD/SUB/MUL/DIV 对 f32/f16 特化 |
| `forward_sum_rows<T>` | – | 行求和 |
| `forward_repeat_nrows<T>` / `forward_repeat_dim1<T>` | – | 复制 row / 复制 dim1 |
| `forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16` | 1121 | FlashAttn 单 chunk，多行 m∈{1,2,4} |
| `forward_flash_attn_ext_f16_tiled_vlen1024_vf16` | 1305 | Q_TILE 行一组的分块版本，online softmax |
| `quantize_a_row_i8` / `_i8_hp` / `_i8k` | – | 行内 32 元素 / 256 元素 K-blk int8 量化 |
| `quantize_a_4row_i8` / `_i8_hp` / `_i8k` | – | 4 行并行量化（RVV vsetvl + vfredmax） |

### 8.2 `quantize_a_4row_i8` 拆解（IME1，asm 写法）(`ime1_kernels.cpp:97`)

`QUANTIZEM4ROW_KERNEL` 宏：

```asm
vsetvli  t0, zero, e32, m8          # 每个 thread 一次处理 4 行 × 8 个 f32 = 32 个 f32
vle32.v  v0, (SRC)                  # 加载 1 个 f32 寄存器的 row
vfabs.v  v8, v0
vfredmax v16, v8, v16               # 求绝对值最大
vfmv.f.s f10, v16                   # 拿到 max(abs)
fmul.s   f10, f10, 1/127            # 算 scale = max / 127
fsw      f10, (a1)                  # 写 scale
fdiv.s   f11, 1, f10                # inv_scale
vfmul.vf v16, v0, f11               # 量化
vfcvt.x.f v16, v16                  # fp → int
vsetvli  t0, zero, e16, mf2
vnclip   v16, v16, zero × 8 次      # saturate to int16 然后 narrow 到 int8
vsetvli  t0, zero, e8, mf4
vnclip   v24..v31, v16..v23, zero   # int8 输出
```

`QUANTIZEM4ROW_STORE` 宏把 v24..v31 写出 32 字节。整体是 **饱和 + 截断**，符合 int8 量化要求。

> `_hp` 版本是 IME2 的 "high precision" 量化，把 K 拆成 32 元素子块，每个子块带 fp16 scale + fp16 sum（用于融合 zero-point correction）。

### 8.3 FlashAttention 单 chunk (`rvv_kernels.cpp:1121`)

```cpp
// 1. 索引计算：把 (ir0..ir1) 的 ir 拆成 (iq3, iq2, iq1) → head index h
// 2. 计算 broadcast factor rk2, rk3, rv2, rv3
// 3. 算 scale = 1/sqrt(D) or f_attention_scale
// 4. 算 ALiBi slope = m0/m1（如果 max_bias > 0）
// 5. 选择 ir_step ∈ {1, 2, 4}：连续的 ir 是否 share mask / K head / V head
// 6. 跳到 flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<mr>(...)
```

`_mrow<mr>` 模板（mr=1/2/4）一次算 mr 个 Q 行的 attention。

支持的特性：
- **logit softcap**：`tanh(QK/cap) * cap`
- **sinks**：attention sink term（在普通 softmax 上加一个不依赖 K 的偏置）
- **ALiBi 斜率**
- **可变 head 维数 DK, DV ≤ 128**（要求 VLENB=128，i.e. SpacemiT A100+）
- **mask**：fp16 mask

> 不支持 `DK > 128` / `DV > 128` 的情况——`supported_shape` 检查会 fall through 到默认 `ggml_compute_forward_flash_attn_ext`。

### 8.4 FlashAttention 分块版本（`forward_flash_attn_ext_f16_tiled_vlen1024_vf16`）

把 Q 切成 Q_TILE_SZ 行（`Q_TILE_SZ=ggml_fa_tile_config::Q`，在 `common.h` 里配置）的小组，在 TCM 里完成整个 K·Qᵀ / softmax / PV 工作集：

- 工作缓冲：
  ```
  sizeof(float) * (
      GGML_FA_TILE_Q * DK              # Q_q
    + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV   # KQ + KQ2
    + GGML_FA_TILE_Q * DV              # mask
    + GGML_FA_TILE_KV * DV             # V
    + GGML_FA_TILE_KV * DK             # K_f32
  ) * n_tasks
  ```

  也就是用 TCM 做分块 attention 的"软件 tiling"。

- online softmax：`m` 跟踪 max，`s` 跟踪 sum，每处理一个新 K/V 块按 FlashAttention 公式更新 `acc`/`m`/`s`。

> 这是把 FlashAttention2/3 论文里的 tile 思路直接移植到 RISC-V V 扩展上。`forward_flash_attn_ext_f16`（`ime.cpp:1149`）只在 `neq1 >= Q_TILE_SZ` 且 `use_ref==false` 时选 tiled 路径。

---

## 9. IME GEMM 内核细节

### 9.1 数据排布（以 IME2 Q4_0 为例，`ime2_kernels.cpp:2430+`）

A（int8，量化后 src1）：
```
A: M1K32 int8 (256-bit)
   Ascale: fp32 × 1
Ascale[0] fp32 | Asum int16 | 32 int8 values | Ascale[1] fp32 | Asum int16 | 32 int8 values | ...
```

B（int4，repacked weight）：
```
B: N8K32 int4 (1024-bit)
   4 VRF（vl4r.v 一次加载 4 个 256-bit 段）
   Bscale: fp16 × N32 (512-bit)
   Bzp:    uint8 × N32 (256-bit)  // 仅 Q4_1/Q4_K/Q5_1/Q5_K 有
```

C：fp32 × N32（1024-bit = 32 floats）。

### 9.2 K loop asm 模板（IME2 Q4_0 + zp 路径，宏写在注释里）

```asm
# t3 = k_blocks
mv  s2, %[pA]                # s2 = A scale ptr
add s3, s2, 4+2              # s3 = A int8 ptr (skip 4B scale + 2B sum)
mv  s4, %[pB]                # s4 = B scale ptr
add s5, s4, 32*3             # s5 = B int4 ptr (skip 32B scale + 32B zp)
mv  s6, %[pC]                # s6 = C ptr
vsetvli t0, x0, e32, m1
vxor.vv v2, v0, v0           # v2 = acc (M=1, N=32)

align 4
_K_LPST%=:
  vsetvli t0, x0, e8, m1
  vl4r.v v4, (s5)            # 4 VRF segment 加载 B int4
  add s5, s5, 128*4 + 96     # 跳过 4*128 字节数据 + 96 字节 scale/zp 头

  vsetvli t0, x0, e8, mf2
  vle8.v v30, (s4)           # 加载 B scale (32 个 fp16)
  add s4, s4, 32*2 + 32      # scale + zp

  vsetvli t0, x0, e8, mf4
  vle8.v v3, (s3)            # 加载 A int8
  add s3, s3, 32 + 6         # 32 int8 + 6 字节 scale/sum

  flw  f0, (s2)              # A scale fp32
  lh   t2, 4(s2)             # A sum int16
  add  s2, s2, 32+6

  vsetvli t0, x0, e16, m1
  vmv.v.i v28, 8             # zp multiplier (16-bit 8)
  vsrl.vi v24, v3, 4         # A int8 的高 4 bit

  vmul.vx v26, v28, t2       # zp * asum (i16 * i16)
  vnpack4.vv v8, v3, v3, 3   # A lo4 (signed)
  vnpack4.vv v10, v24, v24, 3# A hi4 (unsigned, for vmadotsu)
  vfcvt.f.x.v v16, v26       # fp16

  # hi4 unsigned * B int4 → v16..v22
  vmadotsu.hp v16, v10, v4, v1, 0, i4
  vmadotsu.hp v18, v10, v5, v1, 0, i4
  vmadotsu.hp v20, v10, v6, v1, 0, i4
  vmadotsu.hp v22, v10, v7, v1, 0, i4
  # lo4 unsigned * B int4 → v16..v22
  vmadotu.hp  v16, v8, v4, v0, 0, i4
  vmadotu.hp  v18, v8, v5, v0, 0, i4
  vmadotu.hp  v20, v8, v6, v0, 0, i4
  vmadotu.hp  v22, v8, v7, v0, 0, i4

  vpack.vv    v24, v16, v18, 1
  vpack.vv    v26, v20, v22, 1
  vpack.vv    v16, v24, v26, 2     # 把 hi/lo 拼回 i16

  vsetvli t0, x0, e16, mf2
  vfwmul.vv v31, v30, v16          # mac * b_scale (f16 * f16 -> f32)

  vsetvli t0, x0, e32, m1
  vfmacc.vf v2, f0, v31            # acc += a_scale * (mac * b_scale)

  addi t3, t3, -1
  bgtz  t3, _K_LPST%=
```

关键点：

1. **`vmadotsu`**：unsigned × signed → int32，结果按 4 bit 移位以补偿 i4 的位置权重（`vsll.vi 4`）。
2. **`vmadotu.hp`** / **`vmadotsu.hp`**：加 .hp 后缀意味着 **乘积直接累加到 fp16 累加器**（半精度 MAC，2x 吞吐）。
3. **`vl4r.v`**：用 4 个虚拟寄存器做 interleaved load，是 RISC-V 扩展的 segment load，等价于一次 4×256-bit 传输。
4. **`vnpack4.vv`**：把 i8 unpack 成 4 个 i4，存到 4 个虚拟寄存器（lo/hi pair）。
5. **vfmacc**：fp32 累加器的最终步骤，先把 fp16 mac 结果 × b_scale（fp16 升 fp32）再 × a_scale 累加。

> 非 zp 路径几乎一样，差别只是 `vfwmul` 之前没有 `vmul.vx` 也没有 `vmv.v.i v28, 8`。

### 9.3 M4 版本（`ime2_kernels.cpp:2883+` for Q4_0_hp, `:3007+` for Q4_0）

`M4 kernel` 一次算 4 个 token 行 × 32 列输出，用 4 个独立的 acc 寄存器 `v8..v15`，共享一次 weight 加载。SpacemiT 在 `forward_mul_mat` 中根据 `count_m >= 4` 自动选 M4，否则用 M1（行数 = 1）。

### 9.4 MoE M2 kernel（`ime2_kernels.cpp:5006`）

`moe_m2_gemm_kernel_i8i4_impl` 一次处理 2 个不同 token 的 GEMM，共享 weight 和 B 加载。每个 token 用自己的 A 量化 buffer，输出到不同 `c_ptr`：

```cpp
moe_gemm_kernel_m2(blk_len,
                   [a_ptr0, a_ptr1],         // 两个 int8 行指针
                   b_ptr, b_zp,              // shared weight
                   [c_ptr0, c_ptr1],         // 两个 output
                   1, src0_n, b_k_blks, ne01);
```

它用 v24..v27 作为第一行 acc、v28..v31 作为第二行 acc，节省 1 次 weight load 的带宽。

### 9.5 IME1 vs IME2 关键差异

| 项 | IME1 (A60/X100) | IME2 (A100/A200) |
| --- | --- | --- |
| 主指令 | `vmadot` (i8×i4→i32 累加) | `vmadotsu` / `vmadotu` + `.hp` (fp16 累加) |
| 累加器 | int32 | fp16 + fp32 |
| 支持量化 | Q4_0 / Q4_1 / Q4_K | Q4_0 / Q4_1 / Q4_K / Q6_K / Q8_0 / Q2_K / Q3_K / Q5_K / Q5_1 / Q5_0 / MXFP4 |
| N tile | 16 | 32（或 32×256 高吞吐版） |
| K block | 32 | 32 或 256 |
| `_hp` 变体 | ❌ | ✅（fp16 partial sum） |
| MoE M2 | ❌ | ✅ |

---

## 10. 线程绑定：`ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity` (`ime.cpp:1690`)

```cpp
void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n) {
    int cpu_id = sched_getcpu();
    // 如果当前 cpu 不是 a* 核，把线程移到第一个 a* 核（通过写 /proc/set_ai_thread）
    if (use_ime2 && !((1 << cpu_id) & cpu_mask)) {
        bind_ai_thread();   // write "0" to /proc/set_ai_thread
    }

    // 申请 TCM（仅第一次）
    if (use_tcm && tls_context.cpu_id == -1) {
        CPU_ZERO(&cpuset);
        CPU_SET(perfer_core_ids[thread_n], &cpuset);
        pthread_setaffinity_np(main_thread, sizeof(cpuset), &cpuset);

        int ai_cpu_id = perfer_core_ids[thread_n] - aicpu_id_offset;
        tls_context.cpu_id = ai_cpu_id;
        tls_context.tcm_buffer = spine_mem_pool_tcm_mem_get(ai_cpu_id);
        tls_context.tcm_buffer_size = global_spine_env_info.tcm_blk_size;
    }

    // 阻塞等 TCM 就绪
    if (tls_context.tcm_buffer != nullptr) {
        void * rt = spine_mem_pool_tcm_mem_wait(tls_context.cpu_id);
        GGML_ABORT_IF(rt == nullptr);
    }
}
```

`clear_numa_thread_affinity_threaded` 在线程退出时调 `spine_mem_pool_tcm_mem_release` 释放 TCM 引用。

> 这是 SpacemiT 后端**唯一需要主动 hook 到线程入口**的原因：CPU backend 的默认 `set_numa_thread_affinity` 只调 `sched_setaffinity` 到一组 NUMA node，不动 TCM、不动 AI 核迁移。

---

## 11. 总结：SpacemiT 后端的设计哲学

1. **异构 SoC 上的库调用方**：以 CPU backend 框架的"extra buffer type"形式存在，与 CUDA/Metal/SYCL 平级，但复用 `ggml_threadpool` + `ggml_cgraph` + `ggml_compute_forward` 分派链路 → 大幅减少代码改动面。

2. **RISC-V V 扩展 + 自研 IME**：
   - 一切非 GEMM 算子用标准 RISC-V V intrinsics（RVV 1.0）；
   - 矩阵乘用 SpacemiT 自研的 VMADOT 扩展（`vmadot` / `vmadotsu` / `vmadotu.hp`）拿到 int8×int4 高吞吐；
   - 通过 IME1/IME2 两条 compile-time 路径支持不同代次的 AI 核。

3. **离线 repack + 在线 quant 的两段式**：
   - 加载期把 GGUF 量化权重重排成 N×K，零运行时开销；
   - 推理期把 fp32 激活在线量化为 int8（+ 同步 int4 weight 的 scale/zp），统一到 int8 域做 MAC。

4. **TCM + 双线程 ld/compute 流水线**：
   - 用 RISC-V AI 核的 on-chip SRAM 装下 A 或 B 的一个 tile；
   - 让两条线程（`ith%2==0` 和 `ith%2==1`）共享一个 barrier，交替做 memcpy 和 GEMM，掩盖数据搬运延迟；
   - 没用 TCM 时退化到普通 1D 任务分块。

5. **MoE 友好**：
   - 提供了 `moe_m2_gemm_kernel_*` 让 2 个 token 共享 weight 加载；
   - tile-based 的 expert 内循环把 4 个 token 的 A 量化数据先聚到 TCM，再批量跑 GEMM。

6. **可观测性**：
   - 启动期打印 `num_cores, num_perfer_cores, perfer_core_arch_id, use_ime1/2, mem_backend, cpu_mask, aicpu_id_offset`；
   - repack 阶段打 `repack tensor <name> with <type>_NBxINTER`；
   - TCM 初始化打 `tcm is available, blk_size, blk_num, is_fake_tcm`。

7. **失败路径明确**：
   - `GGML_ABORT` 写在所有"必须有 TCM" / "必须 IME1/IME2" / "线程不在 prefer 核" 的地方；
   - `repack` 失败（ne1 不被整除）→ 返回 `-1` → `set_tensor` 抛 `GGML_ASSERT`，整图回退到默认 CPU backend。
