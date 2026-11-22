"""Resource ownership, budget, artifact and single-submission queue contracts."""

from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu
import gpu_queue as queue


class Hub:
    id = 'bound-machine'
    config = dict(power_conf_id='spec', ssh_id='key', startup_type=0)

    def __init__(self, *states, timer=True):
        self.states = deque(states)
        self.calls = []
        self.deadline = None
        self.timer = timer

    def machine(self):
        state = self.states.popleft() if len(self.states) > 1 else self.states[0]
        job = 'old' if state == 'Stopped' else 'new'
        if isinstance(state, tuple):
            state, job = state
        return dict(id=self.id, state=state, jobId=job, powerConfId='spec',
                    sshId='key', startupType=0, autoCloseTime=self.deadline)

    def ssh_files(self):
        pass

    def gpu(self, machine):
        return dict(compute_processes=[])

    def call(self, *args):
        self.calls.append(args)
        if args[1] == 'auto-shutdown':
            if not self.timer:
                raise gpu.GPUError('timer rejected')
            self.deadline = int(datetime.fromisoformat(args[-1]).timestamp() * 1000)
            return dict(id=self.id, closeTime=self.deadline)
        return {}


def test_queue_waits_on_same_platform_job_without_reopening(monkeypatch):
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    hub = Hub('Stopped', 'Queuing', 'Queuing', 'Queuing', 'Running')
    events = []
    machine, deadline = queue.queue(hub, 300, lambda **v: events.append(v))
    assert machine['state'] == 'Running'
    assert machine['jobId'] == 'new'
    assert deadline > datetime.now(timezone.utc)
    assert [c[1] for c in hub.calls] == ['open', 'auto-shutdown']
    assert events[0]['previous_job_id'] == 'old'
    assert any(e.get('job_id') == 'new' for e in events)


def test_timer_failure_cancels_only_the_new_job(monkeypatch):
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    hub = Hub('Stopped', 'Queuing', 'Queuing', 'Stopped', timer=False)
    with pytest.raises(gpu.GPUError, match='timer rejected'):
        queue.queue(hub, 300, lambda **_: None)
    assert [c[1] for c in hub.calls] == ['open', 'auto-shutdown', 'close']


def test_changed_job_is_never_stopped_or_launched():
    hub = Hub('Stopped', 'Queuing', ('Running', 'someone-else'), ('Running', 'someone-else'))
    with pytest.raises(gpu.GPUError, match='job changed'):
        queue.queue(hub, 300, lambda **_: None)
    assert [c[1] for c in hub.calls] == ['open', 'auto-shutdown']


def test_existing_running_machine_is_not_adopted():
    hub = Hub('Running')
    with pytest.raises(gpu.GPUError, match='requires Stopped'):
        queue.queue(hub, 300, lambda **_: None)
    assert hub.calls == []


def test_queue_cancels_when_the_remaining_window_cannot_fit_the_work(monkeypatch):
    real = datetime
    t0 = real(2026,11,20,tzinfo=timezone.utc)
    ticks = iter((t0,t0+timedelta(minutes=133)))

    class Clock(real):
        @classmethod
        def now(cls, tz=None):
            return next(ticks)

    hub = Hub('Stopped', 'Queuing', 'Queuing', 'Queuing', 'Stopped')
    monkeypatch.setattr(queue, 'datetime', Clock)
    with pytest.raises(gpu.GPUError, match='Remaining window cannot fit'):
        queue.queue(hub, 300, lambda **_: None, min_remaining_minutes=168)
    assert [c[1] for c in hub.calls] == ['open','auto-shutdown','close']


@pytest.mark.parametrize(('minutes', 'rate', 'budget', 'cash'), [
    (301, 1.63, 8.6, 8.63), (300, 1.63, 8.6, 8), (300, 1.63, 8, 8.63),
    (0, 1.63, 8.6, 8.63), (300, float('nan'), 8.6, 8.63), (300, 1.63, float('inf'), 8.63),
])
def test_budget_or_window_violation_is_rejected(minutes, rate, budget, cash):
    with pytest.raises(gpu.GPUError):
        queue.budget_limit(minutes, rate, budget, cash)


def test_budget_subtracts_existing_reservations():
    balance = dict(balances=[dict(productId='heros', cash=dict(postedBalance=8.63, reservedAmount=1))])
    assert queue.available_cash(balance) == pytest.approx(7.63)
    assert queue.budget_limit(300, 1.63, 8.6, 8.63) == 8.15


def test_result_paths_and_contents_are_verified(tmp_path):
    path = tmp_path/'last.pt'
    path.write_bytes(b'checkpoint')
    manifest = dict(files={'last.pt': dict(bytes=10, sha256=queue.digest(path))})
    queue.verify_inventory(tmp_path, manifest)
    path.write_bytes(b'corruption')
    with pytest.raises(gpu.GPUError, match='mismatch'):
        queue.verify_inventory(tmp_path, manifest)
    for name in ('../outside', '/tmp/outside', 'C:/outside', 'foo\\bar'):
        with pytest.raises(gpu.GPUError, match='Unsafe'):
            queue.verify_inventory(tmp_path, dict(files={name: dict(bytes=0, sha256='')}))


def test_existing_receipt_prevents_repeated_submission(tmp_path, monkeypatch):
    receipt = tmp_path/'queue.json'
    receipt.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(queue, 'RECEIPT', receipt)
    with pytest.raises(gpu.GPUError, match='existing queue receipt'):
        queue.run(None, None, None, None)


def test_remote_helper_compiles_without_deploying():
    compile(queue.REMOTE, '<remote-queue>', 'exec')
