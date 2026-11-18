"""Parallel input preparation must preserve shards and clean up failed launches."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'inference/self_route_version/modules/qwen_subprocess.py'
SPEC = importlib.util.spec_from_file_location('qwen_launch_scheduling', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_concurrent_job_directories_are_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE.time, 'time', lambda: 123456)
    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = list(pool.map(lambda _: MODULE._make_job_dir(tmp_path, 'triage'), range(64)))
    assert len(set(paths)) == 64
    assert all(path.is_dir() for path in paths)


def test_launches_overlap_and_preserve_assignment_and_output_order(tmp_path, monkeypatch):
    barrier = threading.Barrier(4, timeout=5)
    calls = []

    def launch(prepared, indices, args, tokens, kind, batch, group):
        barrier.wait()  # A serial launcher fails here.
        assert batch == 8 and tokens == 1536 and kind == 'triage'
        return dict(indices=list(indices), cuda_visible_devices=group,
                    proc=SimpleNamespace(returncode=0), job_dir=tmp_path/group)

    def wait(jobs, raw, delivered, callback):
        calls.extend((j['cuda_visible_devices'],j['indices']) for j in jobs)
        for job in reversed(jobs):
            for i in reversed(job['indices']):
                raw[i] = f'output-{i}'

    monkeypatch.setattr(MODULE, '_launch_worker_job', launch)
    monkeypatch.setattr(MODULE, '_wait_worker_jobs', wait)
    raw = {}
    failures, recovered = MODULE._run_worker_strategy(
        [{}]*10, SimpleNamespace(qwen_keep_job_files=True),1536,8,'triage',list(range(10)),
        ['0','1','2','3'],raw,set(),None,'primary')
    assert calls == [('0',[0,4,8]),('1',[1,5,9]),('2',[2,6]),('3',[3,7])]
    assert not failures and recovered == 10
    assert [raw[i] for i in range(10)] == [f'output-{i}' for i in range(10)]


def test_failed_launch_cleans_up_other_started_workers(tmp_path, monkeypatch):
    barrier = threading.Barrier(3, timeout=5)
    cleaned = []

    def launch(prepared, indices, args, tokens, kind, batch, group):
        barrier.wait()
        if group == '1':
            raise OSError('launch failed')
        return dict(cuda_visible_devices=group)

    monkeypatch.setattr(MODULE, '_launch_worker_job', launch)
    monkeypatch.setattr(MODULE, '_terminate_worker_jobs', lambda jobs: cleaned.extend(jobs))
    with pytest.raises(OSError, match='launch failed'):
        MODULE._run_worker_strategy([{}]*3,SimpleNamespace(),1536,8,'triage',[0,1,2],
                                    ['0','1','2'],{},set(),None,'primary')
    assert {j['cuda_visible_devices'] for j in cleaned} == {'0','2'}
