"""
strategies/base.py - 优化策略基类

每个策略是一个 class，定义:
  - id / name / category / risk
  - tier (1 或 2)
  - test() - 跑 benchmark
  - propose_patch() - 提议代码改动（调 LLM）
  - apply() - 应用改动
  - revert() - 撤销
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class StrategyResult:
    """单次策略执行结果"""
    success: bool
    baseline_tok_per_s: float
    new_tok_per_s: float
    delta_pct: float
    description: str
    diff: str = ""
    files_changed: list = None
    should_keep: bool = False   # 是否保留（性能提升）
    cost_usd: float = 0.0
    tier_used: str = ""
    error_msg: str = ""

    def __post_init__(self):
        if self.files_changed is None:
            self.files_changed = []


class Strategy(ABC):
    def __init__(self, config: dict, knowledge, profiler, editor, committer, model_router):
        self.config = config
        self.knowledge = knowledge
        self.profiler = profiler
        self.editor = editor
        self.committer = committer
        self.model_router = model_router

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        pass

    @property
    @abstractmethod
    def risk(self) -> str:
        pass

    @property
    @abstractmethod
    def expected_gain(self) -> str:
        pass

    @property
    @abstractmethod
    def tier(self) -> int:
        pass

    @abstractmethod
    def run(self, exp_id: int, baseline_tok_per_s: float) -> StrategyResult:
        """
        跑一次这个策略，返回结果。
        orchestrator 会根据结果决定 keep / revert。
        """
        pass

    def _parse_diff_metrics(self, diff: str) -> dict:
        if not diff:
            return {'files_changed': 0, 'diff_lines': 0, 'contains_inline_asm': False, 'touches_tcm_or_ime': False}
        return self.editor.count_diff_metrics(diff)
