#!/usr/bin/env python3
"""
auto_opt.py - CLI 入口

用法:
    python3 auto_opt.py run                    # 跑 agent loop
    python3 auto_opt.py status                 # 看当前状态
    python3 auto_opt.py list [--strategy X]    # 列实验
    python3 auto_opt.py show <id>              # 看单个实验
    python3 auto_opt.py report                 # 生成 markdown 报告
    python3 auto_opt.py apply <branch>         # 把一个 opt 分支合并到 main
    python3 auto_opt.py revert <branch>        # 删一个 opt 分支
    python3 auto_opt.py baseline               # 测一次 baseline
    python3 auto_opt.py try <strategy>         # 跑单个策略
"""
import argparse
import sys
import os
from pathlib import Path

import yaml

# 让 Python 找到同目录模块
sys.path.insert(0, str(Path(__file__).parent))

import knowledge as _kb_module
import orchestrator as _orch_module
import committer as _commit_module
import editor as _editor_module

KnowledgeBase = _kb_module.KnowledgeBase
Orchestrator = _orch_module.Orchestrator
Committer = _commit_module.Committer
Editor = _editor_module.Editor


REPO_ROOT = Path(__file__).resolve().parents[2]  # tools/auto-opt → repo root
DEFAULT_CONFIG = REPO_ROOT / "tools" / "auto-opt" / "config.yaml"


def load_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        # 兜底用最小化配置
        print(f"[WARN] Config {path} not found, using minimal config")
        return {
            'strategies': [],
            'benchmark': {
                'binary': str(REPO_ROOT / 'build' / 'bin' / 'llama-bench'),
                'model': str(REPO_ROOT / 'models' / 'qwen2.5-0.5b-instruct-q4_0.gguf'),
                'default_reps': 3,
                'results_dir': str(REPO_ROOT / 'results' / 'auto-opt'),
            },
            'git': {'main_branch': 'main', 'opt_branch_prefix': 'opt/'},
            'loop': {'max_iterations': 5, 'max_duration_hours': 1},
            'knowledge': {'db_path': str(REPO_ROOT / 'results' / 'auto-opt' / 'knowledge.db')},
        }
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_run(args):
    config = load_config(args.config)
    orch = Orchestrator(config, REPO_ROOT)
    orch.run()


def cmd_status(args):
    config = load_config(args.config)
    kb = KnowledgeBase(config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db'))
    s = kb.summary()
    print("=== auto-opt status ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\n=== Recent experiments ===")
    for exp in kb.list_experiments(limit=10):
        delta = f"{exp['delta_pct']:+.2f}%" if exp['delta_pct'] is not None else "N/A"
        print(f"  #{exp['id']:3d} {exp['strategy_id']:25s} {exp['status']:12s} "
              f"{exp['new_tok_per_s'] or 0:6.2f} tok/s ({delta})")


def cmd_list(args):
    config = load_config(args.config)
    kb = KnowledgeBase(config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db'))
    exps = kb.list_experiments(strategy_id=args.strategy, limit=100)
    print(f"{'ID':>4} {'Strategy':<25} {'Status':<12} {'Tok/s':>8} {'Δ%':>8} {'Tier':<6} {'Branch'}")
    print('-' * 100)
    for e in exps:
        delta = f"{e['delta_pct']:+.2f}%" if e['delta_pct'] is not None else "-"
        print(f"{e['id']:>4} {e['strategy_id']:<25} {e['status']:<12} "
              f"{e['new_tok_per_s'] or 0:>8.2f} {delta:>8} {e['tier']:<6} {e['branch'] or '-'}")


def cmd_show(args):
    config = load_config(args.config)
    kb = KnowledgeBase(config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db'))
    exp = kb.get_experiment(args.id)
    if not exp:
        print(f"Experiment #{args.id} not found")
        return
    print(f"=== Experiment #{exp['id']} ===")
    for k in exp.keys():
        v = exp[k]
        if k == 'config_json' or k == 'files_changed' or k == 'diff':
            v = v[:500] + '...' if v and len(v) > 500 else v
        print(f"  {k}: {v}")


def cmd_report(args):
    """生成 markdown 报告"""
    config = load_config(args.config)
    kb = KnowledgeBase(config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db'))

    exps = kb.list_experiments(limit=1000)
    kept = [e for e in exps if e['status'] == 'kept']
    reverted = [e for e in exps if e['status'] == 'reverted']

    md = ["# auto-opt Report\n"]
    s = kb.summary()
    md.append("## Summary\n")
    md.append(f"- Total experiments: {s['total_experiments']}\n")
    md.append(f"- Kept: {s['kept']}\n")
    md.append(f"- Reverted: {s['reverted']}\n")
    md.append(f"- Best: {s['best_strategy']} → {s['best_tok_per_s']:.2f} tok/s\n"
              if s['best_strategy'] else "- No best\n")
    md.append(f"- Total cost: ${s['total_cost_usd']:.4f}\n\n")

    md.append("## Kept Experiments (Performance Improvements)\n")
    md.append("| ID | Strategy | tok/s | Δ% | Tier | Cost | Branch |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for e in sorted(kept, key=lambda x: -(x['new_tok_per_s'] or 0)):
        md.append(f"| {e['id']} | {e['strategy_id']} | {e['new_tok_per_s']:.2f} | "
                  f"{e['delta_pct']:+.2f}% | {e['tier']} | ${e['cost_usd']:.4f} | {e['branch']} |\n")

    md.append("\n## Reverted Experiments (No Improvement)\n")
    md.append("| ID | Strategy | tok/s | Δ% | Reason |\n")
    md.append("|---|---|---|---|---|\n")
    for e in sorted(reverted, key=lambda x: -(x['delta_pct'] or 0))[:20]:
        reason = (e['notes'] or '')[:80]
        md.append(f"| {e['id']} | {e['strategy_id']} | {e['new_tok_per_s'] or 0:.2f} | "
                  f"{e['delta_pct'] or 0:+.2f}% | {reason} |\n")

    out = REPO_ROOT / 'results' / 'auto-opt' / 'REPORT.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(md))
    print(f"Report written to: {out}")


def cmd_apply(args):
    config = load_config(args.config)
    commit = Committer(REPO_ROOT, config.get('git', {}))
    main = commit.main_branch
    commit.checkout_branch(main)
    import subprocess
    r = subprocess.run(['git', 'merge', '--no-ff', args.branch],
                      cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✓ Merged {args.branch} into {main}")
    else:
        print(f"✗ Merge failed: {r.stderr}")


def cmd_revert(args):
    config = load_config(args.config)
    commit = Committer(REPO_ROOT, config.get('git', {}))
    commit.checkout_branch(commit.main_branch)
    commit.delete_branch(args.branch, force=True)
    print(f"✓ Deleted branch {args.branch}")


def cmd_baseline(args):
    """测一次 baseline"""
    config = load_config(args.config)
    from profiler import Profiler
    kb = KnowledgeBase(config.get('knowledge', {}).get('db_path', 'results/auto-opt/knowledge.db'))
    profiler = Profiler(config, kb)
    r = profiler.run_benchmark("manual_baseline", n_prompt=512, n_gen=128, threads=8, reps=5)
    if r['success']:
        print(f"\n✓ Baseline: {r['tok_per_s']:.2f} tok/s")
    else:
        print(f"\n✗ Failed: {r.get('error')}")


def cmd_try(args):
    """跑单个策略一次"""
    config = load_config(args.config)
    orch = Orchestrator(config, REPO_ROOT)
    if orch.baseline_tok_per_s is None:
        orch.ensure_baseline()
    # 找到这个策略
    for s in orch.strategies:
        if s.id == args.strategy:
            orch.cycle_once()
            return
    print(f"Strategy {args.strategy} not found")
    print("Available:", [s.id for s in orch.strategies])


def main():
    p = argparse.ArgumentParser(description='auto-opt: LLM-driven performance optimization')
    p.add_argument('-c', '--config', help='config file path')
    sub = p.add_subparsers(dest='cmd')

    sub.add_parser('run', help='run agent loop').set_defaults(func=cmd_run)
    sub.add_parser('status', help='show status').set_defaults(func=cmd_status)

    p_list = sub.add_parser('list', help='list experiments')
    p_list.add_argument('--strategy', help='filter by strategy')
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser('show', help='show experiment')
    p_show.add_argument('id', type=int)
    p_show.set_defaults(func=cmd_show)

    sub.add_parser('report', help='generate markdown report').set_defaults(func=cmd_report)

    p_apply = sub.add_parser('apply', help='merge opt branch into main')
    p_apply.add_argument('branch')
    p_apply.set_defaults(func=cmd_apply)

    p_revert = sub.add_parser('revert', help='delete opt branch')
    p_revert.add_argument('branch')
    p_revert.set_defaults(func=cmd_revert)

    sub.add_parser('baseline', help='measure baseline once').set_defaults(func=cmd_baseline)

    p_try = sub.add_parser('try', help='try single strategy')
    p_try.add_argument('strategy')
    p_try.set_defaults(func=cmd_try)

    args = p.parse_args()
    if not hasattr(args, 'func'):
        p.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
