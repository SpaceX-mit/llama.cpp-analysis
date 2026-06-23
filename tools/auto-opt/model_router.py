"""
model_router.py - 模型路由器

负责:
  1. 任务分级（tier1 / tier2）
  2. 选择合适的模型
  3. 调 API
  4. 记录使用统计到 knowledge base
"""
import os
import time
import json
import requests
from typing import Optional, Dict, Any, List


class ModelRouter:
    def __init__(self, config: dict, knowledge):
        self.config = config
        self.knowledge = knowledge
        self.tier1_models = config.get('models', {}).get('tier1', [])
        self.tier2_models = config.get('models', {}).get('tier2', [])
        self.upgrade_triggers = config.get('upgrade_triggers', [])

    def select_model(self, task: dict) -> dict:
        """
        决定用哪个 tier 和哪个 model.

        task 字段:
          - tier: 期望 tier ('tier1' / 'tier2' / 'auto')
          - files_changed: 改动文件数
          - contains_inline_asm: 是否涉及汇编
          - diff_lines: diff 行数
          - touches_tcm_or_ime: 是否触碰 TCM/IME
          - estimated_difficulty: 1-5 难度评级
        """
        requested_tier = task.get('tier', 'auto')
        if requested_tier == 'auto':
            tier = self._should_upgrade(task)
        else:
            tier = requested_tier

        models = self.tier1_models if tier == 'tier1' else self.tier2_models
        if not models:
            # fallback 到另一边
            models = self.tier2_models if tier == 'tier1' else self.tier1_models
            if not models:
                raise RuntimeError("No models configured!")

        # 选第一个能用的（看 API key 是否在 env）
        for m in models:
            env_var = m.get('api_key_env', '')
            if not env_var or os.environ.get(env_var):
                return {'tier': tier, 'model': m}
        # 都缺 key，强行选第一个
        return {'tier': tier, 'model': models[0]}

    def _should_upgrade(self, task: dict) -> str:
        """检查是否需要用 tier2"""
        for trigger in self.upgrade_triggers:
            cond = trigger.get('condition', '')
            if self._eval_condition(cond, task):
                return trigger.get('tier', 'tier2')
        return 'tier1'

    def _eval_condition(self, cond: str, task: dict) -> bool:
        """极简条件求值器，支持：
            files_changed > 3
            diff_lines > 100
            contains_inline_asm == true
            touches_tcm_or_ime == true
            estimated_difficulty >= 4
        """
        try:
            # 替换字段引用（bool 用 Python 的 True/False）
            for k, v in task.items():
                if isinstance(v, bool):
                    cond = cond.replace(k, 'True' if v else 'False')
                else:
                    cond = cond.replace(k, str(v))
            # 'true' → 'True', 'false' → 'False' 兼容 YAML 风格
            cond = cond.replace('== true', '== True').replace('== false', '== False')
            return eval(cond, {"__builtins__": {}}, {})
        except Exception:
            return False

    def call(self, selection: dict, prompt: str, system: str = "") -> dict:
        """
        调 LLM API，返回 dict:
          {
            'content': str,
            'input_tokens': int,
            'output_tokens': int,
            'cost_usd': float,
            'latency_ms': int,
            'tier': str,
            'model_name': str,
          }
        """
        m = selection['model']
        tier = selection['tier']
        api_key = os.environ.get(m.get('api_key_env', ''), '')
        if not api_key:
            return {
                'content': self._mock_response(prompt),
                'input_tokens': len(prompt) // 4,
                'output_tokens': 200,
                'cost_usd': 0.0,
                'latency_ms': 0,
                'tier': tier,
                'model_name': m['name'],
                'mock': True,
            }

        api_base = m['api_base']
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": m['model_id'],
            "max_tokens": m.get('max_tokens', 4096),
            "temperature": m.get('temperature', 0.2),
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
        }
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            in_tok = usage.get('prompt_tokens', 0)
            out_tok = usage.get('completion_tokens', 0)
            cost = (in_tok + out_tok) / 1000.0 * m.get('cost_per_1k_tokens', 0)
            latency = int((time.time() - t0) * 1000)
            success = True
        except Exception as e:
            content = f"[ERROR] {e}"
            in_tok, out_tok, cost, latency = 0, 0, 0, 0
            success = False

        # 记录到 knowledge base
        self.knowledge.record_model_usage(
            tier=tier, model_name=m['name'],
            task_type=self._infer_task_type(prompt),
            input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, latency_ms=latency, success=success,
        )

        return {
            'content': content,
            'input_tokens': in_tok,
            'output_tokens': out_tok,
            'cost_usd': cost,
            'latency_ms': latency,
            'tier': tier,
            'model_name': m['name'],
            'mock': False,
        }

    def _mock_response(self, prompt: str) -> str:
        """无 API key 时返回 mock 响应"""
        return f"[MOCK] 我没 API key，返回占位响应。\n\nPrompt 长度: {len(prompt)} 字符\n\n示例 patch:\n```\n--- a/file.cpp\n+++ b/file.cpp\n@@ -1,1 +1,1 @@\n-old\n+new\n```"

    def _infer_task_type(self, prompt: str) -> str:
        p = prompt.lower()
        if 'patch' in p or '```' in p:
            return 'patch_gen'
        if 'profile' in p or 'benchmark' in p:
            return 'profile_analysis'
        if 'summarize' in p or 'explain' in p:
            return 'code_read'
        return 'other'
