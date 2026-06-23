"""
test_smoke.py - 烟雾测试

不调 LLM、不真跑 benchmark（避免依赖），
只验证：
  1. 各模块可以 import
  2. KnowledgeBase 能正常创建/读写
  3. Strategy 基类能实例化
  4. Mock 模式能跑完整 cycle
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    print("[TEST] imports... ", end='')
    import knowledge
    import orchestrator
    import committer
    import editor
    import model_router
    import profiler
    print("OK")
    return True


def test_knowledge_base():
    print("[TEST] KnowledgeBase CRUD... ", end='')
    sys.path.insert(0, str(Path(__file__).parent))
    from knowledge import KnowledgeBase

    tmp = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmp, 'test.db')
        kb = KnowledgeBase(db_path)

        # create
        eid = kb.create_experiment('test_strategy', 'tier1', 'deepseek',
                                    'test desc', {'foo': 'bar'})
        assert eid > 0

        # update
        kb.update_experiment(eid, status='kept', new_tok_per_s=100.0,
                              baseline_tok_per_s=80.0, delta_pct=25.0)
        exp = kb.get_experiment(eid)
        assert exp['status'] == 'kept'
        assert exp['new_tok_per_s'] == 100.0
        assert exp['delta_pct'] == 25.0

        # record benchmark
        kb.record_benchmark(eid, 'test_label', n_prompt=512, n_gen=128,
                            threads=8, tok_per_s=100.0)
        # record model usage
        kb.record_model_usage('tier1', 'deepseek', 'patch_gen', 1000, 500, 0.001, 1000, True)

        # list
        exps = kb.list_experiments()
        assert len(exps) == 1

        # summary
        s = kb.summary()
        assert s['total_experiments'] == 1
        assert s['best_strategy'] == 'test_strategy'

        # baseline
        assert kb.get_baseline() == 100.0

        print("OK")
        return True
    finally:
        shutil.rmtree(tmp)


def test_model_router_mock():
    print("[TEST] ModelRouter (no API key → mock)... ", end='')
    sys.path.insert(0, str(Path(__file__).parent))
    from knowledge import KnowledgeBase
    from model_router import ModelRouter

    sys.path.insert(0, str(Path(__file__).parent))
    tmp = tempfile.mkdtemp()
    try:
        from knowledge import KnowledgeBase
        kb = KnowledgeBase(os.path.join(tmp, 'test.db'))
        config = {
            'models': {
                'tier1': [{'name': 'test', 'model_id': 'test', 'api_key_env': 'NONEXISTENT_KEY_12345',
                           'cost_per_1k_tokens': 0.0001, 'max_tokens': 100, 'temperature': 0.2}],
                'tier2': [{'name': 'test2', 'model_id': 'test2', 'api_key_env': 'NONEXISTENT_KEY_67890',
                           'cost_per_1k_tokens': 0.003, 'max_tokens': 100, 'temperature': 0.1}],
            },
            'upgrade_triggers': [
                {'condition': 'files_changed > 3', 'tier': 'tier2'},
                {'condition': 'contains_inline_asm == true', 'tier': 'tier2'},
            ],
        }
        router = ModelRouter(config, kb)

        # 简单任务 → tier1
        s1 = router.select_model({'tier': 'auto', 'files_changed': 1, 'contains_inline_asm': False})
        assert s1['tier'] == 'tier1'

        # 复杂任务（多文件） → tier2
        s2 = router.select_model({'tier': 'auto', 'files_changed': 5, 'contains_inline_asm': False})
        assert s2['tier'] == 'tier2'

        # 复杂任务（汇编） → tier2
        s3 = router.select_model({'tier': 'auto', 'files_changed': 1, 'contains_inline_asm': True})
        assert s3['tier'] == 'tier2'

        # Mock call
        r = router.call(s1, 'test prompt', 'test system')
        assert r['mock'] is True
        assert 'MOCK' in r['content']
        assert r['cost_usd'] == 0.0
        assert r['tier'] == 'tier1'

        print("OK")
        return True
    finally:
        shutil.rmtree(tmp)


def test_editor():
    print("[TEST] Editor (git apply dry-run)... ", end='')
    import subprocess
    sys.path.insert(0, str(Path(__file__).parent))
    from editor import Editor

    # 用一个临时 git repo
    tmp = tempfile.mkdtemp()
    try:
        # init git
        subprocess.run(['git', 'init', '-q'], cwd=tmp, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp, check=True)
        subprocess.run(['git', 'config', 'user.name', 'test'], cwd=tmp, check=True)

        # 写个文件
        f = Path(tmp) / 'test.txt'
        f.write_text("hello\nworld\n")

        # dry-run patch（合法）
        valid_diff = """--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
-hello
-world
+HELLO
+WORLD
"""
        e = Editor(tmp)
        ok, err = e.apply_unified_diff(valid_diff, dry_run=True)
        assert ok, f"valid diff should pass dry-run: {err}"

        # 不合法 patch
        invalid_diff = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1,1 +1,1 @@
-old
+new
"""
        ok, err = e.apply_unified_diff(invalid_diff, dry_run=True)
        assert not ok, "invalid diff should fail"

        # list changed
        files = e.list_changed_files()
        # 没有改动，应该返回空
        assert files == [], f"expected no changes, got: {files}"

        print("OK")
        return True
    finally:
        shutil.rmtree(tmp)


def test_strategies_instantiate():
    print("[TEST] Strategies can be instantiated... ", end='')
    sys.path.insert(0, str(Path(__file__).parent))
    from knowledge import KnowledgeBase
    from model_router import ModelRouter
    from profiler import Profiler
    from editor import Editor
    from committer import Committer
    from strategies.thread_tuning import ThreadTuningStrategy
    from strategies.mem_backend import MemBackendStrategy

    sys.path.insert(0, str(Path(__file__).parent))
    tmp = tempfile.mkdtemp()
    try:
        from knowledge import KnowledgeBase
        kb = KnowledgeBase(os.path.join(tmp, 'test.db'))
        config = {
            'models': {'tier1': [], 'tier2': []},
            'upgrade_triggers': [],
            'benchmark': {'binary': '/bin/echo', 'model': '/dev/null', 'default_reps': 1,
                          'results_dir': tmp},
            'git': {'main_branch': 'main', 'opt_branch_prefix': 'opt/'},
        }
        router = ModelRouter(config, kb)
        profiler = Profiler(config, kb)
        editor = Editor(tmp)
        committer = Committer(Path(tmp), config['git'])

        for cls in [ThreadTuningStrategy, MemBackendStrategy]:
            s = cls({'test': {}}, kb, profiler, editor, committer, router)
            assert s.id
            assert s.name
            assert s.tier
            print(f"  ({s.id})", end=' ')

        print("OK")
        return True
    finally:
        shutil.rmtree(tmp)


def test_config_loads():
    print("[TEST] config.yaml loads correctly... ", end='')
    import yaml
    cfg_path = Path(__file__).parent / 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert 'strategies' in cfg
    assert 'benchmark' in cfg
    assert 'git' in cfg
    assert 'models' in cfg
    assert 'tier1' in cfg['models']
    assert 'tier2' in cfg['models']
    print(f"OK ({len(cfg['strategies'])} strategies configured)")
    return True


def test_cli_help():
    print("[TEST] CLI --help works... ", end='')
    import subprocess
    r = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'auto_opt.py'), '--help'],
        capture_output=True, text=True
    )
    assert r.returncode == 0
    assert 'auto-opt' in r.stdout
    assert 'run' in r.stdout and 'status' in r.stdout and 'list' in r.stdout
    print("OK")
    return True


if __name__ == '__main__':
    tests = [
        test_imports,
        test_config_loads,
        test_knowledge_base,
        test_model_router_mock,
        test_editor,
        test_strategies_instantiate,
        test_cli_help,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
            else:
                failed += 1
                print(f"FAILED")
        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*50}\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
