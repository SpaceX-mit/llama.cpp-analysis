"""
committer.py - Git 分支管理

策略:
  - main: 稳定基线
  - opt/<strategy-id>-<timestamp>: 每个实验一个分支
  - 性能提升的 commit 才保留
  - 性能下降的 revert
"""
import subprocess
import time
from pathlib import Path
from typing import Optional


class Committer:
    def __init__(self, repo_root: Path, config: dict):
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.main_branch = config.get('main_branch', 'main')
        self.prefix = config.get('opt_branch_prefix', 'opt/')

    def current_branch(self) -> str:
        r = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                          capture_output=True, text=True, cwd=str(self.repo_root))
        return r.stdout.strip()

    def create_opt_branch(self, strategy_id: str, description: str = '') -> str:
        """创建 opt/<strategy>-<timestamp> 分支（基于当前 main）"""
        ts = time.strftime('%Y%m%d-%H%M%S')
        slug = description[:30].replace(' ', '-').replace('/', '-').lower() if description else ''
        slug = ''.join(c for c in slug if c.isalnum() or c == '-')
        branch_name = f"{self.prefix}{strategy_id}-{ts}"
        if slug:
            branch_name += f"-{slug}"

        # 先回到 main
        subprocess.run(['git', 'checkout', self.main_branch], cwd=str(self.repo_root), check=True)
        # 拉最新（如果设了 auto_fetch）
        # 创建新分支
        subprocess.run(['git', 'checkout', '-b', branch_name], cwd=str(self.repo_root), check=True)
        return branch_name

    def checkout_branch(self, branch: str):
        subprocess.run(['git', 'checkout', branch], cwd=str(self.repo_root), check=True)

    def has_changes(self) -> bool:
        r = subprocess.run(['git', 'status', '--porcelain'],
                          capture_output=True, text=True, cwd=str(self.repo_root))
        return bool(r.stdout.strip())

    def commit(self, message: str) -> bool:
        if not self.has_changes():
            return False
        subprocess.run(['git', 'add', '-A'], cwd=str(self.repo_root), check=True)
        r = subprocess.run(['git', 'commit', '-m', message],
                          capture_output=True, text=True, cwd=str(self.repo_root))
        return r.returncode == 0

    def push(self, branch: str) -> bool:
        remote = self.config.get('remote', 'origin')
        r = subprocess.run(['git', 'push', remote, branch],
                          capture_output=True, text=True, cwd=str(self.repo_root))
        if r.returncode != 0:
            print(f"  [WARN] push failed: {r.stderr}")
            return False
        return True

    def commit_message(self, strategy_id: str, description: str,
                      baseline: float, new: float, delta_pct: float,
                      tier: str, cost: float) -> str:
        template = self.config.get('commit_message_template', '').strip()
        if not template:
            return f"auto-opt({strategy_id}): {description}"
        try:
            return template.format(
                strategy_id=strategy_id,
                description=description,
                baseline_metric=f"{baseline:.2f} tok/s" if baseline else "N/A",
                new_metric=f"{new:.2f} tok/s" if new else "N/A",
                delta_pct=f"{delta_pct:+.2f}" if delta_pct is not None else "N/A",
                tier=tier,
                cost=f"{cost:.4f}",
            )
        except Exception:
            return f"auto-opt({strategy_id}): {description}"

    def branch_list(self, prefix: Optional[str] = None) -> list:
        """列出所有 opt/* 分支（按时间倒序）"""
        if prefix is None:
            prefix = self.prefix
        r = subprocess.run(
            ['git', 'branch', '-a', '--sort=-committerdate', f'--list', f'{prefix}*'],
            capture_output=True, text=True, cwd=str(self.repo_root),
        )
        return [b.strip().lstrip('* ').removeprefix('remotes/origin/')
                for b in r.stdout.strip().split('\n') if b.strip()]

    def delete_branch(self, branch: str, force: bool = False):
        """删除分支（性能回退时清理）"""
        flag = '-D' if force else '-d'
        subprocess.run(['git', 'branch', flag, branch], cwd=str(self.repo_root),
                      capture_output=True, text=True)
