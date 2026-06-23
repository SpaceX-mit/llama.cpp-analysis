"""
profiler.py - Benchmark 执行和结果解析

封装 llama-bench 调用，解析 JSON/JSONL 输出，记录到 knowledge base。
"""
import subprocess
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Any


class Profiler:
    def __init__(self, config: dict, knowledge):
        self.config = config['benchmark']
        self.knowledge = knowledge
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_args(self, n_prompt: int, n_gen: int, threads: int, reps: int = None,
                    extra_env: Optional[dict] = None) -> list:
        args = [
            self.config['binary'],
            '-m', self.config['model'],
            '-p', str(n_prompt),
            '-n', str(n_gen),
            '-t', str(threads),
            '-r', str(reps or self.config['default_reps']),
            '-o', 'jsonl',  # 易于解析
        ]
        args.extend(self.config.get('extra_args', []))
        return args

    def run_benchmark(self, label: str, exp_id: int = None,
                      n_prompt: int = 512, n_gen: int = 128,
                      threads: int = 8, reps: int = None,
                      extra_env: Optional[dict] = None) -> Dict[str, Any]:
        """
        跑一次 benchmark，返回:
          {
            'tok_per_s': decode t/s,
            't_p_eval_ms': prefill ms,
            't_eval_ms': decode ms,
            'stddev': 标准差,
            'success': bool,
            'raw_output': str,
          }
        """
        args = self._build_args(n_prompt, n_gen, threads, reps, extra_env)
        env = {**__import__('os').environ, **(extra_env or {})}
        timeout = self.config.get('timeout_seconds', 300)

        # 写 log
        log_path = self.results_dir / f"{label}-{int(time.time())}.log"

        print(f"  [PROFILE] {label}: {' '.join(args)}")
        print(f"             log → {log_path}")

        t0 = time.time()
        try:
            result = subprocess.run(
                args, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            duration = time.time() - t0
            log_path.write_text(f"CMD: {' '.join(args)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr, 'raw_output': result.stdout}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f'Timeout after {timeout}s'}
        except FileNotFoundError as e:
            return {'success': False, 'error': f'llama-bench not found: {e}'}

        # 解析 JSONL 输出
        parsed = self._parse_jsonl_output(result.stdout)
        if not parsed:
            return {'success': False, 'error': 'Failed to parse output', 'raw_output': result.stdout}

        # 记录到 knowledge base
        if exp_id is not None:
            self.knowledge.record_benchmark(
                exp_id, label,
                n_prompt=n_prompt, n_gen=n_gen, threads=threads,
                t_p_eval_ms=parsed.get('t_p_eval_ms'),
                t_eval_ms=parsed.get('t_eval_ms'),
                tok_per_s=parsed.get('tok_per_s'),
                stddev=parsed.get('stddev'),
            )

        return {
            'success': True,
            'tok_per_s': parsed.get('tok_per_s', 0.0),
            't_p_eval_ms': parsed.get('t_p_eval_ms', 0.0),
            't_eval_ms': parsed.get('t_eval_ms', 0.0),
            'stddev': parsed.get('stddev', 0.0),
            'duration': duration,
            'raw_output': result.stdout,
        }

    def _parse_jsonl_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """llama-bench 的 -o jsonl 输出每行一个 JSON 记录"""
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                # 找到 tg (text generation / decode) 的 t/s
                if 'tg_avg_ts' in data or 'avg_ts' in data:
                    return {
                        'tok_per_s': float(data.get('tg_avg_ts', data.get('avg_ts', 0))),
                        't_p_eval_ms': float(data.get('pp_avg_ms', 0)),
                        't_eval_ms': float(data.get('tg_avg_ms', 0)),
                        'stddev': float(data.get('tg_avg_ts_stdev', 0)),
                    }
            except json.JSONDecodeError:
                continue

        # Fallback: 用 markdown 输出解析
        return self._parse_markdown_fallback(stdout)

    def _parse_markdown_fallback(self, stdout: str) -> Optional[Dict[str, Any]]:
        """从 markdown 表格里找 tg 列"""
        for line in stdout.split('\n'):
            if 'ms/tok' in line or 't/s' in line:
                cols = line.split('|')
                for i, c in enumerate(cols):
                    if 'ms/tok' in c and i+1 < len(cols):
                        try:
                            ms_per_tok = float(cols[i+1].strip())
                            return {'tok_per_s': 1000.0 / ms_per_tok, 't_eval_ms': ms_per_tok * 128}
                        except ValueError:
                            pass
        return None

    def get_best_threads(self, exp_id: int, model: str, n_prompt: int, n_gen: int,
                        thread_options: List[int]) -> int:
        """扫多个 threads 值，返回 t/s 最高的那个"""
        results = {}
        for t in thread_options:
            r = self.run_benchmark(f"thread_sweep_t{t}", exp_id=exp_id,
                                    n_prompt=n_prompt, n_gen=n_gen, threads=t)
            if r['success']:
                results[t] = r['tok_per_s']
                print(f"  [SWEEP] threads={t} → {r['tok_per_s']:.2f} tok/s")

        if not results:
            return -1
        best = max(results, key=results.get)
        return best
