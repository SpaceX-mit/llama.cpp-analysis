"""
editor.py - 代码编辑（应用 patch）

支持的 patch 方式:
  1. 完整文件替换 (replace_file)
  2. 搜索替换 (search_replace)
  3. 应用 unified diff (apply_diff)
  4. 运行 shell 命令 (shell_command)
"""
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import re


class Editor:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    def apply_unified_diff(self, diff_text: str, dry_run: bool = False) -> Tuple[bool, str]:
        """用 `git apply` 应用 unified diff，返回 (success, error_msg)"""
        if not diff_text.strip():
            return False, "empty diff"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(diff_text)
            patch_path = f.name

        try:
            args = ['apply', '--check', patch_path]
            if not dry_run:
                args = ['apply', patch_path]

            result = subprocess.run(
                ['git'] + args,
                capture_output=True, text=True,
                cwd=str(self.repo_root),
            )
            if result.returncode == 0:
                return True, ""
            return False, f"git apply failed: {result.stderr}"
        finally:
            Path(patch_path).unlink(missing_ok=True)

    def apply_file_replacement(self, file_path: str, new_content: str) -> Tuple[bool, str]:
        """完整替换文件内容"""
        path = self.repo_root / file_path
        if not path.exists():
            return False, f"file not found: {file_path}"
        old_content = path.read_text()
        if old_content == new_content:
            return True, "no change"
        path.write_text(new_content)
        return True, ""

    def apply_env_var_change(self, env_var: str, value: str) -> dict:
        """不修改文件，env var 改动通过 extra_env 传给 profiler"""
        return {'SPACEMIT_' + env_var: value}

    def get_diff(self, file_path: Optional[str] = None) -> str:
        """获取当前未提交 diff"""
        args = ['diff']
        if file_path:
            args.append(file_path)
        result = subprocess.run(['git'] + args, capture_output=True, text=True,
                                cwd=str(self.repo_root))
        return result.stdout

    def list_changed_files(self) -> list:
        """列出 git status 里的所有改动文件（不包含 untracked）"""
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=str(self.repo_root),
        )
        files = []
        for line in result.stdout.strip().split('\n'):
            if not line or len(line) < 3:
                continue
            # XY format: X=index status, Y=worktree status
            # ?? = untracked, !! = ignored
            # 只关心 M/A/D/R/C (modified/added/deleted/renamed/copied)
            xy = line[:2]
            if '?' in xy or '!' in xy:
                continue
            # 提取文件名
            rest = line[3:]
            f = rest.split(' -> ')[-1]
            files.append(f)
        return files

    def revert_all(self):
        """丢弃所有未提交改动"""
        subprocess.run(['git', 'checkout', '.'], cwd=str(self.repo_root), check=False)
        subprocess.run(['git', 'clean', '-fd'], cwd=str(self.repo_root), check=False)

    def extract_diff_from_llm_response(self, response: str) -> Optional[str]:
        """
        从 LLM 响应里提取 unified diff（```diff ... ``` 块）
        """
        # 找 ```diff ... ``` 块
        m = re.search(r'```diff\s*\n(.*?)\n```', response, re.DOTALL)
        if m:
            return m.group(1)
        # 兜底：找 ``` ... ``` 块
        m = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if m and ('+++' in m.group(1) or '---' in m.group(1)):
            return m.group(1)
        return None

    def count_diff_metrics(self, diff: str) -> dict:
        """统计 diff 的指标"""
        files = set()
        diff_lines = 0
        for line in diff.split('\n'):
            if line.startswith('+++') or line.startswith('---'):
                if '/' in line:
                    files.add(line.split('/')[-1])
            elif line.startswith('+') or line.startswith('-'):
                if not line.startswith('+++') and not line.startswith('---'):
                    diff_lines += 1

        contains_asm = bool(re.search(r'__asm__|asm volatile|"vmadot|"vl4r|"vfwmul|"vfmacc',
                                       diff))
        touches_tcm = 'tcm' in diff.lower() or 'spine_mem_pool' in diff.lower() or \
                       'ime_env' in diff.lower() or 'spine_barrier' in diff.lower()

        return {
            'files_changed': len(files),
            'diff_lines': diff_lines,
            'contains_inline_asm': contains_asm,
            'touches_tcm_or_ime': touches_tcm,
        }
