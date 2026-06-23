"""
strategies/thread_tuning.py - 线程数调优
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import Strategy, StrategyResult


class ThreadTuningStrategy(Strategy):
    @property
    def id(self) -> str:
        return "thread_tuning"

    @property
    def name(self) -> str:
        return "Worker 线程数调优"

    @property
    def category(self) -> str:
        return "config"

    @property
    def risk(self) -> str:
        return "low"

    @property
    def expected_gain(self) -> str:
        return "5-30%"

    @property
    def tier(self) -> int:
        return 1

    def run(self, exp_id: int, baseline_tok_per_s: float) -> StrategyResult:
        """扫多个 threads 值，找最佳"""
        threads_options = self.config.get('test', {}).get('threads', [1, 2, 4, 8, 16])
        n_prompt = self.config.get('test', {}).get('n_prompt', 512)
        n_gen = self.config.get('test', {}).get('n_gen', 128)
        reps = self.config.get('test', {}).get('reps', 3)

        print(f"  [THREAD] sweeping: {threads_options}")
        results = []
        for t in threads_options:
            r = self.profiler.run_benchmark(
                f"thread_t{t}", exp_id=exp_id,
                n_prompt=n_prompt, n_gen=n_gen, threads=t, reps=reps,
            )
            if r['success']:
                results.append((t, r['tok_per_s']))
                print(f"  [THREAD]   t={t:2d} → {r['tok_per_s']:.2f} tok/s")

        if not results:
            return StrategyResult(
                success=False, baseline_tok_per_s=baseline_tok_per_s,
                new_tok_per_s=0, delta_pct=0, description="all threads failed",
                error_msg="benchmark failed"
            )

        # 找最佳
        best_t, best_tps = max(results, key=lambda x: x[1])
        delta_pct = (best_tps - baseline_tok_per_s) / baseline_tok_per_s * 100 if baseline_tok_per_s else 0

        description = f"threads={best_t} gives {best_tps:.2f} tok/s"

        # 这种策略不用改代码（threads 是 env var 控制的）
        # 但我们可以写一个建议文档
        env_var_diff = f"""# Recommended: export OMP_NUM_THREADS={best_t} before running
# Best result: threads={best_t} → {best_tps:.2f} tok/s (baseline: {baseline_tok_per_s:.2f}, delta: {delta_pct:+.2f}%)
"""
        return StrategyResult(
            success=True,
            baseline_tok_per_s=baseline_tok_per_s,
            new_tok_per_s=best_tps,
            delta_pct=delta_pct,
            description=description,
            diff=env_var_diff,
            files_changed=["tools/auto-opt/results/thread_recommendation.md"],
            should_keep=delta_pct > 0.5,
        )
