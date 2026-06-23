"""
knowledge.py - 实验知识库 (SQLite)

每个实验（experiment）记录：
  - 策略 ID
  - 改动文件列表 + diff
  - baseline 性能 + 新性能
  - 使用的 model tier
  - cost
  - 状态 (profiling/impl/benchmarked/kept/reverted)
"""
import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path


class KnowledgeBase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL NOT NULL,
                strategy_id     TEXT NOT NULL,
                description     TEXT,
                tier            TEXT NOT NULL,        -- tier1 / tier2
                model_name      TEXT,
                status          TEXT NOT NULL,        -- profiling / implemented / benchmarked / kept / reverted / failed
                config_json     TEXT,                -- 实验的具体配置
                diff            TEXT,                -- 代码 diff
                files_changed   TEXT,                -- JSON 数组
                branch          TEXT,                -- git branch 名
                baseline_tok_per_s REAL,
                new_tok_per_s   REAL,
                delta_pct       REAL,
                cost_usd        REAL,
                notes           TEXT,
                error_msg       TEXT
            );

            CREATE TABLE IF NOT EXISTS benchmarks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   INTEGER,
                ts              REAL NOT NULL,
                label           TEXT,                 -- e.g. "baseline", "after_patch_threads=8"
                n_prompt        INTEGER,
                n_gen           INTEGER,
                threads         INTEGER,
                t_p_eval_ms     REAL,
                t_eval_ms       REAL,
                tok_per_s       REAL,
                stddev          REAL,
                extra_json      TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS model_usage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL NOT NULL,
                tier            TEXT,
                model_name      TEXT,
                task_type       TEXT,                 -- code_read / patch_gen / profile_analysis / etc
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                cost_usd        REAL,
                latency_ms      INTEGER,
                success         INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_exp_strategy ON experiments(strategy_id);
            CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_bench_exp ON benchmarks(experiment_id);
            """)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def create_experiment(self, strategy_id, tier, model_name, description, config):
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO experiments (ts, strategy_id, description, tier, model_name,
                                         status, config_json)
                VALUES (?, ?, ?, ?, ?, 'profiling', ?)
            """, (time.time(), strategy_id, description, tier, model_name, json.dumps(config)))
            return cur.lastrowid

    def update_experiment(self, exp_id, **kwargs):
        sets, vals = [], []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v if not isinstance(v, (dict, list)) else json.dumps(v))
        if not sets:
            return
        with self._conn() as c:
            c.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE id = ?", (*vals, exp_id))

    def record_benchmark(self, exp_id, label, **kwargs):
        with self._conn() as c:
            c.execute("""
                INSERT INTO benchmarks (experiment_id, ts, label, n_prompt, n_gen, threads,
                                        t_p_eval_ms, t_eval_ms, tok_per_s, stddev, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (exp_id, time.time(), label,
                  kwargs.get('n_prompt'), kwargs.get('n_gen'), kwargs.get('threads'),
                  kwargs.get('t_p_eval_ms'), kwargs.get('t_eval_ms'),
                  kwargs.get('tok_per_s'), kwargs.get('stddev'),
                  json.dumps({k: v for k, v in kwargs.items()
                              if k not in ('n_prompt', 'n_gen', 'threads',
                                           't_p_eval_ms', 't_eval_ms', 'tok_per_s', 'stddev')})
                  ))

    def record_model_usage(self, tier, model_name, task_type, input_tokens,
                          output_tokens, cost_usd, latency_ms, success):
        with self._conn() as c:
            c.execute("""
                INSERT INTO model_usage (ts, tier, model_name, task_type, input_tokens,
                                        output_tokens, cost_usd, latency_ms, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), tier, model_name, task_type, input_tokens,
                  output_tokens, cost_usd, latency_ms, 1 if success else 0))

    def get_experiment(self, exp_id):
        with self._conn() as c:
            return c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()

    def list_experiments(self, strategy_id=None, status=None, limit=50):
        sql = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if strategy_id:
            sql += " AND strategy_id = ?"
            params.append(strategy_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            return c.execute(sql, params).fetchall()

    def get_best_for_strategy(self, strategy_id):
        """获取某策略下性能最好的实验"""
        with self._conn() as c:
            return c.execute("""
                SELECT * FROM experiments
                WHERE strategy_id = ? AND status IN ('kept', 'benchmarked')
                  AND new_tok_per_s IS NOT NULL
                ORDER BY new_tok_per_s DESC
                LIMIT 1
            """, (strategy_id,)).fetchone()

    def get_baseline(self):
        """获取当前 baseline tok_per_s（来自上次成功的 'kept' 实验）"""
        with self._conn() as c:
            row = c.execute("""
                SELECT new_tok_per_s FROM experiments
                WHERE status = 'kept' AND new_tok_per_s IS NOT NULL
                ORDER BY id DESC LIMIT 1
            """).fetchone()
            return row['new_tok_per_s'] if row else None

    def total_cost(self):
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(cost_usd), 0) AS c FROM model_usage").fetchone()
            return row['c']

    def summary(self):
        """返回总览信息（用于 status 命令）"""
        with self._conn() as c:
            exp = c.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
            kept = c.execute("SELECT COUNT(*) AS n FROM experiments WHERE status='kept'").fetchone()
            reverted = c.execute("SELECT COUNT(*) AS n FROM experiments WHERE status='reverted'").fetchone()
            best = c.execute("""
                SELECT strategy_id, new_tok_per_s FROM experiments
                WHERE status='kept' AND new_tok_per_s IS NOT NULL
                ORDER BY new_tok_per_s DESC LIMIT 1
            """).fetchone()
            cost = self.total_cost()
            return {
                'total_experiments': exp['n'],
                'kept': kept['n'],
                'reverted': reverted['n'],
                'best_strategy': best['strategy_id'] if best else None,
                'best_tok_per_s': best['new_tok_per_s'] if best else None,
                'total_cost_usd': cost,
            }
