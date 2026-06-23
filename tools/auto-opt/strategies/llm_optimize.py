"""
strategies/llm_optimize.py - 用 LLM 生成代码 patch 的策略

这种策略通过调 LLM 来:
  1. 读相关代码
  2. 提出优化方案
  3. 生成 unified diff
  4. 测试是否提升

适用: kernel 优化、PGO 集成、新 GEMM 路径等
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from strategies.base import Strategy, StrategyResult


class LLMOptimizeStrategy(Strategy):
    """
    通用 LLM 优化策略。
    接受一个 target_area（"gemm_kernel" / "flash_attn" / "compiler_flags"），
    让 LLM 阅读相关代码 + benchmark 结果，提出 patch。
    """
    target_area: str = "general"

    @property
    def id(self) -> str:
        return f"llm_optimize_{self.target_area}"

    @property
    def name(self) -> str:
        return f"LLM-driven {self.target_area} 优化"

    @property
    def category(self) -> str:
        return "kernel"

    @property
    def risk(self) -> str:
        return "high"

    @property
    def expected_gain(self) -> str:
        return "5-20%"

    @property
    def tier(self) -> int:
        return 2  # 默认 tier2（复杂代码）

    def run(self, exp_id: int, baseline_tok_per_s: float) -> StrategyResult:
        # 1. 读相关代码（限制在 200KB 以内避免 token 爆炸）
        relevant_files = self._find_relevant_files()
        code_snippets = self._read_files(relevant_files, max_bytes=200_000)

        # 2. 构造 prompt
        prompt = f"""You are optimizing llama.cpp for SpacemiT K3 (RISC-V, 8x X100 + 8x A100 AI cores).
The target area is: **{self.target_area}**.

Current performance: {baseline_tok_per_s:.2f} tok/s for Qwen2.5 0.5B Q4_0 on K3.
Your goal: propose a code optimization that improves this.

Here are the relevant source files:
```
{code_snippets}
```

Constraints:
- Don't break compilation (must compile with GCC on RISC-V)
- Don't change public APIs
- Keep correctness (small perf loss for accuracy is OK if <1%)
- Use RISC-V V extensions (VLEN=128) and SpacemiT IME instructions (vmadotsu.hp etc.)

Output format (MUST follow exactly):
1. A short description of the optimization (1-2 sentences)
2. A unified diff in ```diff ... ``` block

Example:
```
This adds a manual prefetch hint for the weight tensor.

```diff
--- a/ggml/src/ggml-cpu/spacemit/ime.cpp
+++ b/ggml/src/ggml-cpu/spacemit/ime.cpp
@@ -340,1 +340,3 @@
 void forward_mul_mat(...) {{
+    __builtin_prefetch(w_data, 0, 0);
     ...
 }}
```
```"""

        # 3. 选模型（根据代码复杂度自动升级）
        # 简化：看文件数和行数
        task = {
            'files_changed': 1,
            'diff_lines': 50,
            'contains_inline_asm': 'asm' in code_snippets.lower(),
            'touches_tcm_or_ime': 'tcm' in self.target_area or 'ime' in self.target_area,
            'estimated_difficulty': 4,
        }
        selection = self.model_router.select_model(task)

        # 4. 调 LLM
        system = "You are a senior C++ systems programmer specializing in HPC and RISC-V optimization."
        result = self.model_router.call(selection, prompt, system)

        if result.get('mock'):
            print(f"  [LLM-OPT] ⚠ MOCK response (no API key) — using heuristic instead")
            return self._heuristic_fallback(exp_id, baseline_tok_per_s)

        # 5. 提取 diff
        diff_text = self.editor.extract_diff_from_llm_response(result['content'])
        if not diff_text:
            return StrategyResult(
                success=False, baseline_tok_per_s=baseline_tok_per_s,
                new_tok_per_s=0, delta_pct=0,
                description=result['content'][:200],
                error_msg="LLM didn't return valid diff",
                cost_usd=result['cost_usd'],
                tier_used=result['tier'],
            )

        # 6. 应用 patch（dry-run 先）
        ok, err = self.editor.apply_unified_diff(diff_text, dry_run=True)
        if not ok:
            return StrategyResult(
                success=False, baseline_tok_per_s=baseline_tok_per_s,
                new_tok_per_s=0, delta_pct=0,
                description=result['content'][:200],
                diff=diff_text, error_msg=f"patch failed dry-run: {err}",
                cost_usd=result['cost_usd'],
                tier_used=result['tier'],
            )

        # 7. 真应用
        ok, err = self.editor.apply_unified_diff(diff_text, dry_run=False)
        if not ok:
            return StrategyResult(
                success=False, baseline_tok_per_s=baseline_tok_per_s,
                new_tok_per_s=0, delta_pct=0,
                description=result['content'][:200],
                diff=diff_text, error_msg=f"patch failed: {err}",
                cost_usd=result['cost_usd'],
                tier_used=result['tier'],
            )

        # 8. 编译（如果有 build script）— 假设要跑 build
        # 这部分要调 cmake/make；简化：跳过
        # （实际部署时可以加：subprocess.run(['make', '-j'], ...)）

        # 9. 跑 benchmark
        b = self.profiler.run_benchmark(f"llm_opt_{self.target_area}", exp_id=exp_id)
        if not b['success']:
            self.editor.revert_all()
            return StrategyResult(
                success=False, baseline_tok_per_s=baseline_tok_per_s,
                new_tok_per_s=0, delta_pct=0,
                description=result['content'][:200], diff=diff_text,
                error_msg=f"benchmark failed: {b.get('error')}",
                cost_usd=result['cost_usd'],
                tier_used=result['tier'],
            )

        delta_pct = (b['tok_per_s'] - baseline_tok_per_s) / baseline_tok_per_s * 100 if baseline_tok_per_s else 0

        # 10. 决策
        should_keep = delta_pct > 0.5  # 至少 0.5% 提升
        if not should_keep:
            self.editor.revert_all()
            print(f"  [LLM-OPT] reverted (delta {delta_pct:+.2f}% < 0.5%)")

        return StrategyResult(
            success=True,
            baseline_tok_per_s=baseline_tok_per_s,
            new_tok_per_s=b['tok_per_s'],
            delta_pct=delta_pct,
            description=result['content'][:200],
            diff=diff_text,
            files_changed=self.editor.list_changed_files(),
            should_keep=should_keep,
            cost_usd=result['cost_usd'],
            tier_used=result['tier'],
        )

    def _find_relevant_files(self) -> list:
        """根据 target_area 找相关文件"""
        if 'gemm' in self.target_area:
            return [
                'ggml/src/ggml-cpu/spacemit/ime.cpp',
                'ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp',
                'ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp',
            ]
        if 'flash_attn' in self.target_area or 'attn' in self.target_area:
            return [
                'ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp',
                'ggml/src/ggml-cpu/spacemit/ime.cpp',
            ]
        if 'compiler' in self.target_area or 'flag' in self.target_area:
            return [
                'ggml/src/ggml-cpu/CMakeLists.txt',
                'CMakeLists.txt',
            ]
        # 通用：spacemit 整个目录
        import os
        spacemit_dir = 'ggml/src/ggml-cpu/spacemit'
        if os.path.isdir(spacemit_dir):
            return [os.path.join(spacemit_dir, f) for f in os.listdir(spacemit_dir)
                    if f.endswith(('.cpp', '.h'))]
        return []

    def _read_files(self, files: list, max_bytes: int = 200_000) -> str:
        out = []
        total = 0
        for f in files:
            try:
                content = Path_safe_read(f)
                if total + len(content) > max_bytes:
                    content = content[:max_bytes - total] + "\n... (truncated)"
                out.append(f"=== {f} ===\n{content}\n")
                total += len(content)
                if total >= max_bytes:
                    break
            except Exception:
                continue
        return '\n'.join(out)

    def _heuristic_fallback(self, exp_id: int, baseline_tok_per_s: float) -> StrategyResult:
        """没 API key 时的兜底：跑一次 baseline 报告"未实施优化" """
        print(f"  [LLM-OPT] heuristic: testing default config for {self.target_area}")
        b = self.profiler.run_benchmark(f"llm_opt_{self.target_area}_fallback", exp_id=exp_id)
        return StrategyResult(
            success=True,
            baseline_tok_per_s=baseline_tok_per_s,
            new_tok_per_s=b.get('tok_per_s', 0),
            delta_pct=0,
            description=f"heuristic fallback (no API key) for {self.target_area}",
            should_keep=False,
            tier_used="none",
        )


def Path_safe_read(path: str) -> str:
    from pathlib import Path
    return Path(path).read_text(errors='ignore')
