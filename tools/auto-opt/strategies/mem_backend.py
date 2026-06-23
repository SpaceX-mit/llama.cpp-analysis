"""
strategies/mem_backend.py - 内存池后端对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Strategy, StrategyResult


class MemBackendStrategy(Strategy):
    @property
    def id(self) -> str:
        return "mem_backend"

    @property
    def name(self) -> str:
        return "大页内存池后端"

    @property
    def category(self) -> str:
        return "system"

    @property
    def risk(self) -> str:
        return "low"

    @property
    def expected_gain(self) -> str:
        return "5-15%"

    @property
    def tier(self) -> int:
        return 1

    def run(self, exp_id: int, baseline_tok_per_s: float) -> StrategyResult:
        backends = self.config.get('test', {}).get('backends', ['none', 'posix', 'hpage', 'hpage1g'])
        n_prompt = self.config.get('test', {}).get('n_prompt', 512)
        n_gen = self.config.get('test', {}).get('n_gen', 128)
        threads = self.config.get('test', {}).get('threads', 8)
        reps = self.config.get('test', {}).get('reps', 3)

        results = []
        for b in backends:
            r = self.profiler.run_benchmark(
                f"memback_{b}", exp_id=exp_id,
                n_prompt=n_prompt, n_gen=n_gen, threads=threads, reps=reps,
                extra_env={'SPACEMIT_MEM_BACKEND': b},
            )
            if r['success']:
                results.append((b, r['tok_per_s']))
                print(f"  [MEMBACK]   {b:10s} → {r['tok_per_s']:.2f} tok/s")

        if not results:
            return StrategyResult(success=False, baseline_tok_per_s=baseline_tok_per_s,
                                  new_tok_per_s=0, delta_pct=0, description="all failed",
                                  error_msg="benchmark failed")

        best_b, best_tps = max(results, key=lambda x: x[1])
        delta_pct = (best_tps - baseline_tok_per_s) / baseline_tok_per_s * 100 if baseline_tok_per_s else 0

        description = f"SPACEMIT_MEM_BACKEND={best_b} gives {best_tps:.2f} tok/s"

        diff = f"""# Recommended: export SPACEMIT_MEM_BACKEND={best_b} before running
# Best: {best_b} → {best_tps:.2f} tok/s (baseline: {baseline_tok_per_s:.2f}, delta: {delta_pct:+.2f}%)
"""
        return StrategyResult(
            success=True,
            baseline_tok_per_s=baseline_tok_per_s,
            new_tok_per_s=best_tps,
            delta_pct=delta_pct,
            description=description,
            diff=diff,
            files_changed=["tools/auto-opt/results/mem_backend_recommendation.md"],
            should_keep=delta_pct > 0.5,
        )
