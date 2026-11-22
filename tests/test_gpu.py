from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu


class Hub:
    id = 'verified-machine'
    config = {'power_conf_id': 'original-spec', 'ssh_id': 'bound-key', 'startup_type': 0}

    def __init__(self, *states, apps=(), timer_ok=True, spec='original-spec'):
        self.states = deque(states)
        self.apps = apps
        self.timer_ok = timer_ok
        self.spec = spec
        self.calls = []
        self.deadline = None

    def machine(self):
        state = self.states.popleft() if len(self.states) > 1 else self.states[0]
        return {'id': self.id, 'state': state, 'powerConfId': self.spec, 'sshId': 'bound-key',
                'startupType': 0, 'jobId': 'job', 'autoCloseTime': self.deadline}

    def ssh_files(self):
        pass

    def gpu(self, machine):
        return {'compute_processes': self.apps}

    def call(self, *args):
        self.calls.append(args)
        if args[1] == 'auto-shutdown':
            if not self.timer_ok:
                raise gpu.GPUError('provider rejected timer')
            self.deadline = int(datetime.fromisoformat(args[-1]).timestamp() * 1000)
            return {'id': self.id, 'closeTime': self.deadline}
        return {}


def test_start_waits_for_running_with_confirmed_provider_deadline(monkeypatch):
    monkeypatch.setattr(gpu.time, 'sleep', lambda seconds: None)
    hub = Hub('Stopped', 'Starting', 'Queuing', 'Running')
    begun = datetime.now(timezone.utc)
    result = gpu.start(hub, minutes=30, timeout=10)
    deadline = datetime.fromisoformat(result['confirmed_shutdown_at_utc'])
    assert 1798 < (deadline - begun).total_seconds() <= 1800
    assert result['state'] == 'Running'
    assert [call[1:3] for call in hub.calls] == [('open', hub.id), ('auto-shutdown', hub.id)]


def test_timer_rejection_rolls_back_new_boot():
    hub = Hub('Stopped', 'Running', 'Running', 'Stopped', timer_ok=False)
    with pytest.raises(gpu.GPUError, match='provider rejected timer.*state: Stopped'):
        gpu.start(hub, minutes=30, timeout=10)
    assert hub.calls[-1] == ('devmachine', 'close', hub.id)


def test_timer_ack_without_platform_readback_does_not_release_machine_for_work(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(gpu.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(gpu.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    hub = Hub('Stopped', 'Running')
    machine = hub.machine
    call = hub.call
    hub.machine = lambda: {**machine(), 'autoCloseTime': None}

    def api(*args):
        if args[1] == 'close':
            hub.states = deque(['Stopped'])
        return call(*args)

    hub.call = api
    with pytest.raises(gpu.GPUError, match='did not confirm.*state: Stopped'):
        gpu.start(hub, minutes=30, timeout=10)
    assert [args[1] for args in hub.calls] == ['open', 'auto-shutdown', 'close']


@pytest.mark.parametrize(('state', 'spec'), [('Running', 'original-spec'), ('Stopped', 'changed-spec')])
def test_start_refuses_existing_workload_or_changed_spec(state, spec):
    hub = Hub(state, spec=spec)
    with pytest.raises(gpu.GPUError):
        gpu.start(hub, minutes=30, timeout=10)
    assert hub.calls == []


def test_close_refuses_busy_gpu_or_changed_job():
    busy = Hub('Running', apps=[{'pid': 123}])
    with pytest.raises(gpu.GPUError, match='123'):
        gpu.close(busy, busy.machine(), timeout=10)
    assert busy.calls == []
    idle = Hub('Running')
    old_job = {**idle.machine(), 'jobId': 'old-job'}
    with pytest.raises(gpu.GPUError, match='job or state changed'):
        gpu.close(idle, old_job, timeout=10)
    assert idle.calls == []


def test_close_checks_provider_terminal_state_and_is_idempotent():
    hub = Hub('Running', 'Running', 'Stopped')
    assert gpu.close(hub, hub.machine(), timeout=10)['state'] == 'Stopped'
    assert gpu.close(hub, hub.machine(), timeout=10)['state'] == 'Stopped'
    assert hub.calls == [('devmachine', 'close', hub.id)]


def test_gpu_probe_cannot_treat_empty_or_invalid_inventory_as_idle():
    assert gpu.parse_gpu('0, RTX 4090, 1, 10, 24564, 39\n\n--compute--\n')['compute_processes'] == []
    busy = gpu.parse_gpu('0, RTX 4090, 50, 20000, 24564, 55\n\n--compute--\n432, python, 19900\n')
    assert busy['compute_processes'][0]['pid'] == 432
    for output in ['', '\n--compute--\n', '0, RTX 4090, 50, 20000, 24564, 55\n\n--compute--\nN/A']:
        with pytest.raises(gpu.GPUError, match='usage is unknown'):
            gpu.parse_gpu(output)
