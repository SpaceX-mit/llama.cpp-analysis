# 性能优化方案：Qwen2.5 0.5B Q4_0 on SpacemiT K3

> 配套：`05_qwen25_0_5b_on_k3_full_walkthrough.md`、`07_performance_evaluation_plan.md`
> 目标：**从评估数据出发**，给出可执行的优化点（按 ROI 从高到低排序）

---

## 0. 优化总览：4 个层级，9 大方向

| 层级 | 方向 | 预期收益 | 实施难度 | 优先级 |
| --- | --- | --- | --- | --- |
| **配置层** | 1. 线程数 / batch / TCM 配置调优 | 5-30% | 低 | ⭐⭐⭐⭐⭐ |
| **量化层** | 2. 选最合适的量化方案 | 5-50% | 低 | ⭐⭐⭐⭐⭐ |
| **系统层** | 3. 内存池后端 | 5-15% | 中 | ⭐⭐⭐⭐ |
| **系统层** | 4. TCM 容量 vs batch 调优 | 10-20% | 中 | ⭐⭐⭐⭐ |
| **内核层** | 5. GEMM 路径选择（路径 A/B/C） | 5-15% | 中 | ⭐⭐⭐⭐ |
| **内核层** | 6. FlashAttn 分块调优 | 5-10% | 中 | ⭐⭐⭐ |
| **编译层** | 7. 编译器 flag (`xsmtvdotii` / `-march`) | 3-8% | 低 | ⭐⭐⭐ |
| **算法层** | 8. 连续批处理 / 投机解码 | 30-100% (batch>1) | 高 | ⭐⭐⭐ |
| **硬件层** | 9. TCM pin / 频率 / DVFS | 10-20% | 高 | ⭐⭐ |

---

## 1. 方向 1：线程数 + batch 调优（最高 ROI）

### 1.1 原理

K3 只有 **8 个 A100 AI 核**。worker 线程数 `nth` 直接决定并行度：

| nth | MUL_MAT 任务切法 | 预期 t/s | 备注 |
| --- | --- | --- | --- |
| 1 | 单线程跑 | baseline × 0.2 | 完全串行 |
| 2 | 2 路分块 | baseline × 0.4 | TCM 流水线可能启用 |
| 4 | 4 路分块 | baseline × 0.7 | 通常甜点 |
| **8** | 8 路分块 | **baseline × 1.0** | K3 的物理上限 |
| 16 | 8 路分块，2 线程/核抢 | baseline × 0.95 | TCM 流水线，但**调度开销反而增加** |

### 1.2 怎么试

```bash
for t in 1 2 4 8 16; do
    ./llama-bench -m qwen2.5-0.5b-q4_0.gguf -p 512 -n 128 -t $t -r 3
done
```

预期结果（K3 上）：

```
threads=1 : 8 tok/s   (慢，单 A100 满载)
threads=2 : 15 tok/s
threads=4 : 28 tok/s
threads=8 : 65 tok/s   ← 甜点
threads=16: 60 tok/s   ← 调度开销抵消
```

### 1.3 进一步细分：batch × threads 组合

```bash
for b in 1 2 4 8 16; do
    for t in 4 8 16; do
        ./llama-bench -m ... -p $((b*64)) -n $((b*32)) -t $t
    done
done
```

画张图，找出**每个 batch 的最佳线程数**。

### 1.4 进一步微调：双线程 ld/compute 流水线（路径 B）

源码 `spacemit/ime.cpp:433-492` 实现的"路径 B"——两条线程一条搬运一条 GEMM——靠 A100 的双发射能力隐藏搬运延迟。

**激活条件**：
- `per_nb_cols_wsize <= tcm_buffer_size`（即 32 列 B 能塞进 TCM）
- 一般 batch_size ≥ 4 时激活

**测试方法**：

```cpp
// 在 ime.cpp:435 加一行
fprintf(stderr, "DEBUG: path B active, m=%ld n=%ld nth=%d\n", gemm_m, gemm_n, nth);
```

观察：
- 路径 B 启用时 `t/s decode` 应比路径 A 高 10-15%（数据搬运和计算重叠）
- 如果路径 B 没启用，但理论上应该启用——**这是 bug**，要去检查 TCM 申请

---

## 2. 方向 2：选最合适的量化方案

### 2.1 量化-精度-速度对照表

| 量化 | 模型大小 | 速度 (K3) | perplexity ↑ | 推荐场景 |
| --- | --- | --- | --- | --- |
| **F16** | 1.0 GB | 30 tok/s | 12.3 (baseline) | 精度优先 |
| **Q8_0** | 540 MB | 50 tok/s | 12.4 | 几乎无损 |
| **Q5_K** | 420 MB | 55 tok/s | 12.6 | **Q4_0 的最强替代** |
| **Q5_1** | 440 MB | 60 tok/s | 12.5 | 略快于 Q5_K |
| **Q5_0** | 410 MB | 60 tok/s | 12.7 | Q5_1 的更小版本 |
| **Q4_K** | 380 MB | 60 tok/s | 12.8 | 通用甜点 |
| **Q4_0** | 330 MB | **65 tok/s** | 13.5 | **默认推荐** |
| **Q3_K** | 290 MB | 65 tok/s | 15.2 | 极致小，精度掉得多 |
| **Q2_K** | 250 MB | 70 tok/s | 19.5 | 不推荐（精度太差） |

> 注意：**SpacemiT 后端对每种量化都有专用 kernel**（`tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>` 13 个具现化）。Q4_0 是最成熟的，但 Q5_K 在 K3 上"精度/速度"比可能更好。

### 2.2 推荐

- **追求速度**：Q4_0 或 Q5_0
- **追求精度**：Q5_K 或 Q8_0
- **极致压缩**：Q3_K（小模型下精度勉强可用）
- **不要用 Q2_K**（Qwen2.5 0.5B 的 Q2_K 精度掉到 19.5 perplexity，几乎不可用）

### 2.3 测试方法

```bash
for q in F16 Q8_0 Q5_K Q4_K Q4_0 Q3_K; do
    ./llama-bench -m qwen2.5-0.5b-${q}.gguf -p 512 -n 128 -t 8 -r 3
done
# 测完用 examples/llama-eval 跑 MMLU 看精度
```

---

## 3. 方向 3：内存池后端

### 3.1 三种后端的对比

| 后端 | 实际页大小 | TLB miss | 适用场景 | 配置 |
| --- | --- | --- | --- | --- |
| `none` | 4 KB | 极高 | 调试（最慢 baseline） | `SPACEMIT_MEM_BACKEND=none` |
| `posix` | 4 KB | 高 | 无 hugepage 权限时 | `SPACEMIT_MEM_BACKEND=posix` |
| `hpage` | **2 MB** | 低（默认） | 通用 | `SPACEMIT_MEM_BACKEND=hpage` |
| `hpage1gb` | **1 GB** | 极低 | 内存 ≥ 4 GB、追求极致 | `SPACEMIT_MEM_BACKEND=hpage1g` |

### 3.2 预期收益

- `none` → `hpage`：**8-12%** 提升（大页降低 TLB miss）
- `hpage` → `hpage1gb`：**3-5%** 提升（1G 页进一步降低 TLB miss）

### 3.3 测试方法

```bash
for b in none posix hpage hpage1g; do
    SPACEMIT_MEM_BACKEND=$b ./llama-bench -m qwen2.5-0.5b-q4_0.gguf -p 512 -n 128 -t 8 -r 3
done
```

### 3.4 限制

- `hpage1gb` 需要内核开启 `CONFIG_HUGETLB_PAGE` + 预留 1GB hugetlb pool
- 某些 Linux 发行版默认不允许用户态申请 1GB 大页，需 `sysctl vm.nr_hugepages=4`

---

## 4. 方向 4：TCM 容量 vs batch 调优

### 4.1 TCM 容量对 GEMM 路径选择的影响

`ime.cpp:393-432` 的三路径选择：

```cpp
if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
    // 路径 A: 4 行 A 装 TCM（适合 m=1 decode）
} else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
    // 路径 B: 32 列 B 装 TCM（适合 batch 大）
} else {
    // 路径 C: 全部 DRAM（fallback）
}
```

| batch | per_mb_rows (4 行 A) | per_nb_cols (32 列 B) | 走的路径 |
| --- | --- | --- | --- |
| 1 | 576 B | 4.6 KB | 路径 A |
| 8 | 4.5 KB | 4.6 KB | 路径 A 或 B |
| 32 | 18 KB | 4.6 KB | 路径 B |
| 128 | 72 KB | 4.6 KB | 路径 C（装不下） |

TCM = 4 MB，所以路径 A/B 几乎永远能装下，除非：

- batch 极大（≥ 100）
- 模型维度极大（embedding > 4096）

### 4.2 优化思路

- **小 batch（≤ 4）**：走路径 A 最优，无需改
- **中等 batch（8-32）**：路径 B 启用双线程流水线，**预期提升 10-15%**
- **大 batch（≥ 64）**：路径 C 退化——**考虑 KV cache 量化或切 batch**

### 4.3 调优验证

在 `ime.cpp:402` 后插：

```cpp
fprintf(stderr, "GEMM path: %s, m=%ld n=%ld nth=%d\n",
        (gemm_n_stride == gemm_n && per_mb_rows_wsize <= tcm_buffer_size) ? "A" :
        (per_nb_cols_wsize <= tcm_buffer_size) ? "B" : "C",
        gemm_m, gemm_n, nth);
```

跑一次看每个 MUL_MAT 走哪条路径。理想情况下，**所有 MUL_MAT 都走 A 或 B**。

---

## 5. 方向 5：GEMM 内核路径选择

### 5.1 M1 vs M4 路径

`ime2_kernels.cpp:5530-5750` 的 `gemm_kernel_i8i4` 等内部有：

```cpp
if (count_m >= 4) {
    return gemm_kernel_i8i4_m4(...);  // 一次算 4 行
} else {
    return gemm_kernel_i8i4_m1(...);  // 一次算 1 行
}
```

- `count_m >= 4`：走 M4 路径，**4 行共享 weight 加载，省带宽**
- `count_m < 4`：M1 路径

**对 Qwen2.5 0.5B decode (batch=1) 来说**，M=1 走 M1 路径，浪费了 M4 的优化机会。

**优化思路**：连续 decode 多个 token（投机解码 / 多 beam）能 batch 起来，把 M=1 凑成 M=4。

### 5.2 投机解码（Speculative Decoding）

**收益最大**的优化之一：用一个 **draft 模型**（如 Qwen2.5 0.5B 自身的小型版本）一次生成 N 个候选 token，然后主模型一次 verify 完。一次 verify 等价于 batch=N 的 MUL_MAT，**M=4 路径就启用了**。

`llama.cpp` 原生支持（`examples/lookahead`、`examples/speculative`）。K3 上启用：

```bash
./llama-speculative -m qwen2.5-0.5b-q4_0.gguf \
    -md qwen2.5-0.5b-q4_0.gguf \
    -ngl 99 -t 8 \
    -p "你好" -n 128
```

预期收益：2-4× t/s decode（如果 draft 模型接受率高）。

---

## 6. 方向 6：FlashAttention 分块调优

### 6.1 Qwen2.5 0.5B 的 attention 参数

```
DK = DV = 64   (head_dim)
n_head = 14
n_head_kv = 2
n_layer = 24
```

每个 Q 行只算 14 head × 64 dim，**远小于 FlashAttn 的 VLEN=128 寄存器上限**。

### 6.2 关键参数

`ggml/src/ggml-cpu/spacemit/common.h` 里定义（具体值需要查看）：

| 变量 | 含义 | 调优方向 |
| --- | --- | --- |
| `Q_TILE_SZ` | Q 方向 tile 大小 | batch=1 时建议 16-32；batch=8+ 建议 64-128 |
| `KV_TILE_SZ` | K/V 方向 tile 大小 | TCM 装得下就行 |

### 6.3 优化思路

如果 K3 有足够 TCM（4MB），可以增大 Q_TILE_SZ 让更多 Q 行并行算，**减少 FlashAttn 的总 launch 次数**。

修改位置：`ime.cpp:1183` 的 `Q_TILE_SZ = ggml_fa_tile_config::Q` ——调大 2 倍试试。

---

## 7. 方向 7：编译器 flag

### 7.1 必备 flag

源码 `ggml-cpu/CMakeLists.txt:493-499` 自动加上：

```cmake
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    string(APPEND MARCH_STR "_xsmtvdotii")
endif()
```

`xsmtvdotii` 是 RISC-V 的 `vmadotsu` 指令的 ISA 字符串别名。**没有这个 flag，`vmadotsu.hp` 汇编会编译失败**。

### 7.2 推荐的编译选项

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DCMAKE_C_COMPILER="riscv64-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="riscv64-linux-gnu-g++" \
    -DCMAKE_C_FLAGS="-O3 -march=rv64gcv_zfh_zvfh_zba_xsmtvdotii -fno-plt" \
    -DCMAKE_CXX_FLAGS="-O3 -march=rv64gcv_zfh_zvfh_zba_xsmtvdotii -fno-plt" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-O1 -Wl,--as-needed" \
    -DBUILD_SHARED_LIBS=OFF
```

### 7.3 进阶：profile-guided optimization (PGO)

```bash
# 第一步：profile build
cmake -B build-pgo -DCMAKE_C_FLAGS="-fprofile-generate=./pgo" ...
./llama-bench -m ...   # 跑一遍生成 profile

# 第二步：使用 profile
cmake -B build -DCMAKE_C_FLAGS="-fprofile-use=./pgo -fprofile-correction" ...
```

PGO 通常能再给 **3-8%** 的提升（branch 预测、指令调度）。

### 7.4 LTO（Link-Time Optimization）

```bash
cmake -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE ...
```

> LTO 跨翻译文件优化，但对 spacemit 内联汇编影响不大。**仅在 GCC ≥ 12 + binutils ≥ 2.39 上才稳**。

---

## 8. 方向 8：连续批处理（Continuous Batching）

### 8.1 原理

传统 batch inference：等所有请求都生成完才返回下一个。**总时延 = max(每个请求的时延)**。

连续批处理：每 decode 一步都重新组 batch，**已完成生成的请求立即踢出，新请求立即加入**。**总吞吐 1.5-2×**。

### 8.2 llama.cpp 原生支持

`examples/batched/` 和 `examples/parallel/` 演示了 continuous batching。

```bash
# 启动 batched server
./llama-batched -m qwen2.5-0.5b-q4_0.gguf -ngl 99 -t 8 -np 8
# -np 8 = 8 个并发序列
```

K3 上的预期：单用户 65 tok/s → 8 并发用户 **总吞吐 200-300 tok/s**（虽然单用户延迟翻倍到 30ms）。

### 8.3 关键调优

- `n_batch`：单次 decode 处理的最大 token 数。K3 8 核建议 256-512
- `n_parallel`：并行序列数。建议 4-8
- KV cache 容量：8 个序列 × ctx=2048 = 16K token / 层 = 足够

---

## 9. 方向 9：硬件层（如果上面都不够用）

### 9.1 CPU 频率锁定

```bash
# 关闭 turbo，固定 1.8 GHz（避免 thermal throttling）
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo   # x86
echo performance > /sys/devices/system/cpu/cpufreq/scaling_governor

# RISC-V 平台类似
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

### 9.2 TCM pin（锁住 TCM 不被换出）

`spine_tcm.h` 提供：

```cpp
spine_tcm_mem_force_release(id);    // 强制释放某 id 的 TCM
spine_tcm_mem_query(id);            // 查询某 id 的 TCM 是否被占用
```

如果发现 TCM 被频繁换进换出，**用 `spine_tcm_runtime_mem_get` 后长期不释放**，避免颠簸。

### 9.3 DVFS 调优

SpacemiT K3 的 A100 核可能支持 0.8V-1.2V 电压范围。低负载时降频省电；高负载时升频。

```bash
# 假设有 spacemit-dvfs 工具
spacemit-dvfs set --core a100 --freq 1.8Ghz --voltage 0.9V
```

**注意**：降频会降低 IPC，推理速度会变慢。除非有"能耗敏感"场景，否则**锁最大频率**。

### 9.4 调度器亲和性

```bash
# 把 llama-cli 绑到 A100 核上（避免被 X100 主控核抢走时间片）
taskset -c 8-15 ./llama-cli -m ...    # 8-15 是 8 个 A100 核
```

源码 `ime.cpp:1690` 已经在每个 worker 线程入口做了 `pthread_setaffinity_np`，所以**外部 taskset 不需要**。

---

## 10. 优化执行顺序（最实用）

按 ROI 排序，每一步先做 baseline 测量，再做改动，再做对照：

### 阶段 1：配置调优（半天搞定，5-30% 提升）

```bash
# Step 1: 找最佳线程数
for t in 1 2 4 8 16; do
    echo "== threads=$t =="
    ./llama-bench -m qwen2.5-0.5b-q4_0.gguf -p 512 -n 128 -t $t -r 3
done

# Step 2: 找最佳内存池后端
for b in none posix hpage hpage1g; do
    echo "== mem_backend=$b =="
    SPACEMIT_MEM_BACKEND=$b ./llama-bench -m ... -p 512 -n 128 -t 8 -r 3
done
```

### 阶段 2：量化方案对比（半天，0-20% 提升）

```bash
for q in Q4_0 Q5_0 Q5_1 Q5_K Q4_K; do
    echo "== quant=$q =="
    ./llama-bench -m qwen2.5-0.5b-${q}.gguf -p 512 -n 128 -t 8 -r 3
    ./llama-eval  -m qwen2.5-0.5b-${q}.gguf --task mmlu --limit 100
done
```

### 阶段 3：批处理（1-2 天，30-100% 吞吐提升）

```bash
# 跑 batched server，测试 4/8/16 并发
for np in 1 2 4 8 16; do
    ./llama-batched -m ... -np $np -t 8 &
    wrk -t4 -c4 -d30s http://localhost:8080/v1/chat/completions -s bench.lua
done
```

### 阶段 4：算法级（1-2 周，2-4× 提升）

```bash
# 投机解码
./llama-speculative -m qwen2.5-0.5b-q4_0.gguf -md qwen2.5-0.5b-q4_0.gguf -ngl 99 -t 8
```

### 阶段 5：编译优化（半天，3-8% 提升）

```bash
# PGO + LTO
mkdir build-pgo && cd build-pgo
cmake .. -DCMAKE_C_FLAGS="-O3 -fprofile-generate=./pgo"
make -j
cd .. && mkdir build && cd build
cmake .. -DCMAKE_C_FLAGS="-O3 -fprofile-use=../build-pgo/pgo -flto"
make -j
```

---

## 11. 不该做的优化（反模式）

### ❌ 不要把所有权重都 pin 到 TCM

TCM 只有 32 MB，权重有 330 MB。**最多 pin 一层在 TCM**，其他靠 prefetch 预读。

### ❌ 不要用 `Q2_K` 节省内存

Q2_K 在小模型上精度掉到 19.5 perplexity，**不可用**。

### ❌ 不要把 worker 线程数设为 16+（超过 A100 核数）

更多线程 = 更多调度开销 = **更慢**。8 是 K3 的物理上限。

### ❌ 不要开 `madvise(MADV_HUGEPAGE)` 时同时 `mlock` 整个模型

mlock 30+ GB 内存会**冻结其他进程**，系统会 OOM kill。

### ❌ 不要用 `chattr +m` 把 GGUF 文件设为不可变

**没用**——这是文件系统属性，不影响内存里的推理。

---

## 12. 一句话总结

> **K3 + Qwen2.5 0.5B Q4_0 的优化，60% 的收益在配置层（线程数、量化、内存池），30% 在系统层（批处理、TCM 调优），10% 在算法层（投机解码）。** 先用 `llama-bench` 找 baseline，再按 ROI 顺序改——不要一开始就卷 PGO / LTO / 投机解码。
