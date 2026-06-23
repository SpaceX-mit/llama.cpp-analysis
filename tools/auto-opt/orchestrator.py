"""
orchestrator.py - Agent Loop 主控制器

状态机:
  IDLE → SELECT_STRATEGY → BASELINE → PROPOSE → APPLY → BENCHMARK → DECIDE → IDLE
                                                                          ↓
                                                                  (loop 终止条件)
"""
import time
import signal
import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge import KnowledgeBase
from model_router import ModelRouter
from profiler import Profiler
from editor import Editor
from committer import Committer
from strategies.thread_tuning import ThreadTuningStrategy
from strategies.mem_backend import MemBackendStrategy
from strategies.llm_optimize import LLMOptimizeStrategy


class Orchestrator:
    def __init__(self, config: dict, repo_root: Path):
        self.config = config
        self.repo_root = Path(repo_root).resolve()

        # 初始化各组件
        kb_path = config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db')
        self.knowledge = KnowledgeBase(kb_path)
        self.model_router = ModelRouter(config, self.knowledge)
        self.profiler = Profiler(config, self.knowledge)
        self.editor = Editor(self.repo_root)
        self.committer = Committer(self.repo_root, config.get('git', {}))

        # 加载所有策略
        self.strategies = self._load_strategies(config)

        # 状态
        self.baseline_tok_per_s: Optional[float] = None
        self.regression_count = 0
        self.iteration = 0
        self.start_time = time.time()
        self.running = True
        self.max_duration_s = config.get('loop', {}).get('max_duration_hours', 8) * 3600
        self.max_iterations = config.get('loop', {}).get('max_iterations', 50)
        self.stop_on_regression = config.get('loop', {}).get('stop_on_regression_count', 3)
        self.min_improvement = config.get('loop', {}).get('min_improvement_pct', 0.5)

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _load_strategies(self, config: dict) -> list:
        strategies_cfg = config.get('strategies', [])
        instances = []
        for s in strategies_cfg:
            sid = s.get('id', '')
            if sid == 'thread_tuning':
                instances.append(ThreadTuningStrategy(s, self.knowledge, self.profiler,
                                                     self.editor, self.committer, self.model_router))
            elif sid == 'mem_backend':
                instances.append(MemBackendStrategy(s, self.knowledge, self.profiler,
                                                   self.editor, self.committer, self.model_router))
            elif sid.startswith('llm_optimize') or sid in ('gemm_path_forcing', 'kv_cache_quant'):
                # 用 LLM 优化策略
                target = sid.replace('llm_optimize_', '').replace('gemm_path_forcing', 'gemm_kernel').replace('kv_cache_quant', 'kv_cache')
                cls = type(f"LLMOpt_{target}", (LLMOptimizeStrategy,),
                          {'target_area': target})
                instances.append(cls(s, self.knowledge, self.profiler,
                                    self.editor, self.committer, self.model_router))
        return instances

    def _on_signal(self, signum, frame):
        print("\n[ORCH] Caught signal, stopping...")
        self.running = False

    def _should_stop(self) -> bool:
        if not self.running:
            return True
        if self.iteration >= self.max_iterations:
            print(f"[ORCH] Max iterations reached ({self.max_iterations})")
            return True
        if time.time() - self.start_time > self.max_duration_s:
            print(f"[ORCH] Max duration reached ({self.max_duration_s/3600:.1f}h)")
            return True
        if self.regression_count >= self.stop_on_regression:
            print(f"[ORCH] Too many regressions ({self.regression_count}), stopping")
            return True
        return False

    def ensure_baseline(self):
        """第一次跑：测 baseline（无任何优化）"""
        if self.baseline_tok_per_s is not None:
            return
        if self.knowledge.get_baseline() is not None:
            self.baseline_tok_per_s = self.knowledge.get_baseline()
            print(f"[ORCH] Resumed baseline from DB: {self.baseline_tok_per_s:.2f} tok/s")
            return

        print(f"\n{'='*70}\n[ORCH] Step 1/2: Measuring baseline (no optimization)\n{'='*70}")
        # 先回到 main
        self.committer.checkout_branch(self.committer.main_branch)
        self.editor.revert_all()

        r = self.profiler.run_benchmark("BASELINE", exp_id=0,  # 0 = baseline
                                        n_prompt=512, n_gen=128, threads=8, reps=5)
        if r['success']:
            self.baseline_tok_per_s = r['tok_per_s']
            print(f"\n[ORCH] ✓ Baseline: {self.baseline_tok_per_s:.2f} tok/s")
            # 记录到 DB
            exp_id = self.knowledge.create_experiment(
                strategy_id="BASELINE", tier="none", model_name="none",
                description="baseline measurement", config={},
            )
            self.knowledge.update_experiment(
                exp_id, status='kept', new_tok_per_s=self.baseline_tok_per_s,
                notes="initial baseline"
            )
        else:
            raise RuntimeError(f"Failed to measure baseline: {r.get('error')}")

    def cycle_once(self) -> bool:
        """跑一次 agent 循环。返回 False 表示应停止。"""
        self.iteration += 1

        # 选策略
        strategy = self._select_next_strategy()
        if strategy is None:
            print("[ORCH] No more strategies to try")
            return False

        print(f"\n{'='*70}\n[ORCH] Iteration {self.iteration}: {strategy.id} ({strategy.name})")
        print(f"        Tier: {strategy.tier} | Risk: {strategy.risk} | Expected: {strategy.expected_gain}")
        print(f"        Current baseline: {self.baseline_tok_per_s:.2f} tok/s\n{'='*70}")

        # 创建 opt/ 分支
        branch = self.committer.create_opt_branch(strategy.id, strategy.name)

        # 创建实验记录
        exp_id = self.knowledge.create_experiment(
            strategy_id=strategy.id, tier=f"tier{strategy.tier}",
            model_name="auto", description=strategy.name,
            config=strategy.config,
        )

        # 跑策略
        t0 = time.time()
        result = strategy.run(exp_id, self.baseline_tok_per_s)
        elapsed = time.time() - t0

        print(f"\n[ORCH] {strategy.id} finished in {elapsed:.1f}s")
        print(f"        baseline: {result.baseline_tok_per_s:.2f} tok/s")
        print(f"        new:      {result.new_tok_per_s:.2f} tok/s")
        print(f"        delta:    {result.delta_pct:+.2f}%")
        if result.cost_usd:
            print(f"        cost:     ${result.cost_usd:.4f} (tier: {result.tier_used})")

        # 更新实验记录
        self.knowledge.update_experiment(
            exp_id,
            status='kept' if result.should_keep else ('reverted' if result.success else 'failed'),
            new_tok_per_s=result.new_tok_per_s if result.success else None,
            baseline_tok_per_s=result.baseline_tok_per_s,
            delta_pct=result.delta_pct if result.success else None,
            diff=result.diff[:5000] if result.diff else None,
            files_changed=result.files_changed,
            branch=branch,
            cost_usd=result.cost_usd,
            tier=result.tier_used or f"tier{strategy.tier}",
            notes=result.description,
            error_msg=result.error_msg or None,
        )

        # 决策
        if not result.success:
            print(f"[ORCH] ✗ FAILED: {result.error_msg}")
            self.committer.checkout_branch(self.committer.main_branch)
            self.committer.delete_branch(branch, force=True)
            return True

        if result.should_keep and result.delta_pct > self.min_improvement:
            print(f"[ORCH] ✓ KEEP: improvement {result.delta_pct:+.2f}% > {self.min_improvement}%")
            # 提交到 opt/ 分支
            msg = self.committer.commit_message(
                strategy_id=strategy.id, description=result.description,
                baseline=result.baseline_tok_per_s, new=result.new_tok_per_s,
                delta_pct=result.delta_pct, tier=result.tier_used or f"tier{strategy.tier}",
                cost=result.cost_usd,
            )
            self.committer.commit(msg)
            # 更新 baseline（如果显著提升）
            if result.delta_pct > 1.0:
                self.baseline_tok_per_s = result.new_tok_per_s
                print(f"[ORCH]   ↑ new baseline: {self.baseline_tok_per_s:.2f} tok/s")
            # 推送到远程（如果设了）
            if self.config.get('git', {}).get('auto_push', False):
                self.committer.push(branch)
        else:
            print(f"[ORCH] ✗ REVERT: improvement {result.delta_pct:+.2f}% <= {self.min_improvement}%")
            self.committer.checkout_branch(self.committer.main_branch)
            self.committer.delete_branch(branch, force=True)
            self.regression_count += 1 if result.delta_pct < 0 else 0

        # 散热
        cooldown = self.config.get('loop', {}).get('experiment_cooldown_seconds', 5)
        print(f"[ORCH] Cooldown {cooldown}s...")
        time.sleep(cooldown)
        return True

    def _select_next_strategy(self):
        """选择下一个策略（简单 round-robin）"""
        if not self.strategies:
            return None
        idx = (self.iteration - 1) % len(self.strategies)
        return self.strategies[idx]

    def run(self):
        """主循环"""
        print(f"\n{'#'*70}")
        print(f"# auto-opt agent loop starting")
        print(f"# Repo: {self.repo_root}")
        print(f"# Strategies: {len(self.strategies)}")
        print(f"# Max iterations: {self.max_iterations}, Max duration: {self.max_duration_s/3600:.1f}h")
        print(f"{'#'*70}\n")

        # 检查 K3
        self._check_k3()

        # 第一步：测 baseline
        self.ensure_baseline()

        # 第二步：循环跑策略
        while not self._should_stop():
            if not self.cycle_once():
                break

        # 总结
        self._print_summary()

    def _check_k3(self):
        """检测是否在 K3 上运行"""
        try:
            with open('/proc/cpuinfo') as f:
                content = f.read()
            if 'SpacemiT' in content or 'spacemit' in content:
                print("[ORCH] ✓ SpacemiT K3 detected")
            else:
                print("[ORCH] ⚠ Not on SpacemiT K3 (may still work but optimizations are K3-specific)")
        except Exception:
            print("[ORCH] ⚠ Could not read /proc/cpuinfo")

    def _print_summary(self):
        print(f"\n{'#'*70}\n# auto-opt finished\n{'#'*70}\n")
        s = self.knowledge.summary()
        print(f"Total experiments: {s['total_experiments']}")
        print(f"  Kept:     {s['kept']}")
        print(f"  Reverted: {s['reverted']}")
        print(f"Best: {s['best_strategy']} → {s['best_tok_per_s']:.2f} tok/s" if s['best_strategy'] else "No best found")
        print(f"Total cost: ${s['total_cost_usd']:.4f}")
        print(f"\nBranches:")
        for b in self.committer.branch_list()[:10]:
            print(f"  {b}")
