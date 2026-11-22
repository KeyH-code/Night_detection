"""Local controller for this project's existing BitaHub development machine."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


CONFIG = Path(__file__).resolve().parents[1] / 'configs/gpu.local.json'
GATEWAY = 'https://www.bitahub.com/gateway'
BOOTING = {'Creating', 'Queued', 'Queuing', 'Pending', 'Deploying', 'Starting'}
STOPPABLE = BOOTING | {'Running'}
GPU_QUERY = (
    'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu '
    '--format=csv,noheader,nounits && printf "\\n--compute--\\n" && '
    'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits'
)


class GPUError(RuntimeError):
    pass


def report(data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def invoke(argv, timeout):
    try:
        return subprocess.run(argv, capture_output=True, encoding='utf-8', errors='replace',
                              stdin=subprocess.DEVNULL, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GPUError(f'{Path(argv[0]).name}: {type(exc).__name__}; check status before retrying') from exc


class BitaHub:
    def __init__(self, config):
        self.config = config
        self.id = config['id']
        cli = Path(config['cli_js']).expanduser()
        node = shutil.which('node')
        if not node or not cli.is_file():
            raise GPUError('Install Node.js and BitaCLI, then check cli_js in configs/gpu.local.json')
        self.cli = [node, str(cli), '--profile', config['profile'], '--output', 'json']
        profile = self.call('profile', 'get', config['profile'])
        if (profile.get('name'), profile.get('baseUrl'), profile.get('productId')) != (
            config['profile'], GATEWAY, 'heros'
        ):
            raise GPUError('BitaCLI profile does not point to the configured production gateway')
        if not profile.get('authenticated'):
            raise GPUError('Run BitaCLI device authorization for the configured profile')
        account = self.call('auth', 'whoami')
        if account.get('username') != config['username'] or account.get('tenantId') != config['tenant_id']:
            raise GPUError('BitaCLI login differs from the account bound to this project')

    def call(self, *args):
        result = invoke([*self.cli, *args], 40)
        try:
            payload = json.loads(result.stderr if result.returncode else result.stdout)
            if not result.returncode and payload.get('ok') is True:
                data = payload['data']
                if isinstance(data, dict):
                    return data
            code = payload.get('error', {}).get('code')
        except (ValueError, KeyError, AttributeError):
            code = None
        # Some raw platform responses include connection credentials; never print them.
        raise GPUError(f'BitaCLI {" ".join(args[:2])} failed (code {code or result.returncode})')

    def machine(self):
        machine = self.call('devmachine', 'get', self.id)
        if machine.get('id') != self.id or not isinstance(machine.get('state'), str):
            raise GPUError('BitaCLI did not return the bound machine and its state')
        return machine

    def connection(self, machine):
        if machine['state'] != 'Running':
            raise GPUError(f"SSH requires Running; platform state is {machine['state']}")
        if machine.get('sshId') != self.config['ssh_id']:
            raise GPUError('Platform SSH key differs from the project binding')
        endpoint = ((machine.get('latestJob') or {}).get('sshDto') or {})
        host = endpoint.get('url')
        port = str(endpoint.get('port', ''))
        if host != self.config['ssh_host'] or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]+', host or ''):
            raise GPUError('SSH endpoint does not match the trusted project gateway')
        if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise GPUError('No valid SSH port in latest platform job')
        identity, known_hosts = self.ssh_files()
        options = ['-F', 'none', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                   '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'UpdateHostKeys=no', '-o', 'GlobalKnownHostsFile=none',
                   '-o', f'UserKnownHostsFile="{known_hosts.as_posix()}"',
                   '-o', f"HostKeyAlias={self.config['host_key_alias']}", '-i', str(identity)]
        return {'host': host, 'port': int(port), 'user': 'root', 'options': options}

    def ssh_files(self):
        identity = Path(self.config['identity_file']).expanduser()
        hosts = Path(self.config['known_hosts']).expanduser()
        alias = self.config['host_key_alias']
        if not identity.is_file() or not hosts.is_file() or not any(
            alias in line.split(maxsplit=1)[0].split(',')
            for line in hosts.read_text(encoding='utf-8').splitlines() if line.strip()
        ):
            raise GPUError('This machine requires the local SSH identity and a pinned host key')
        if not shutil.which('ssh'):
            raise GPUError('OpenSSH client is required')
        return identity, hosts

    def gpu(self, machine):
        info = self.connection(machine)
        argv = ['ssh', *info['options'], '-p', str(info['port']),
                f"{info['user']}@{info['host']}", GPU_QUERY]
        result = invoke(argv, 25)
        if result.returncode:
            raise GPUError(f'SSH/GPU probe failed (exit {result.returncode}); GPU usage is unknown')
        return parse_gpu(result.stdout)


def parse_gpu(output):
    try:
        inventory, processes = output.split('\n--compute--\n')
        gpu_rows = csv.reader((line for line in inventory.splitlines() if line.strip()), skipinitialspace=True)
        gpus = [{'index': int(idx), 'name': name, 'utilization_pct': int(util),
                 'memory_used_mib': int(used), 'memory_total_mib': int(total), 'temperature_c': int(temp)}
                for idx, name, util, used, total, temp in gpu_rows]
        rows = [row for row in csv.reader(processes.splitlines(), skipinitialspace=True)
                if row and any(field.strip() for field in row)]
        apps = [{'pid': int(pid), 'name': name, 'memory_mib': memory} for pid, name, memory in rows]
        if not gpus or any(app['pid'] <= 0 for app in apps):
            raise ValueError('missing GPU or invalid PID')
    except (ValueError, TypeError) as exc:
        raise GPUError('Unexpected nvidia-smi output; GPU usage is unknown') from exc
    return {'gpus': gpus, 'compute_processes': apps}


def summary(machine):
    fields = ('id', 'name', 'state', 'clusterName', 'powerConfId', 'powerConfDisplayName',
              'autoCloseTime', 'autoCloseMillisecond', 'jobId')
    return {field: machine.get(field) for field in fields}


def wait_for(machine, accept, allowed, deadline):
    while True:
        current = machine()
        state = current['state']
        if state in accept:
            return current
        if state not in allowed:
            raise GPUError(f'unexpected platform state: {state}; inspect status')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GPUError(f'waiting for {"/".join(sorted(accept))} timed out; last state: {state}')
        time.sleep(min(10, remaining))


def deadline_visible(machine, target):
    """Read back an absolute deadline or a remaining-milliseconds countdown."""
    now = datetime.now(timezone.utc).timestamp()
    expected = target.timestamp()
    for field in ['autoCloseTime', 'autoCloseMillisecond']:
        value = machine.get(field)
        if isinstance(value, (str, int)) and str(value).isascii() and str(value).isdigit():
            number = int(value)
            if abs(number / 1000 - expected) <= 2 or abs(number - expected) <= 2:
                return True
            if field == 'autoCloseMillisecond' and number > 0 and abs(number / 1000 - (expected - now)) <= 10:
                return True
        elif isinstance(value, str):
            try:
                observed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                continue
            if observed.tzinfo is not None and abs(observed.timestamp() - expected) <= 2:
                return True
    return False


def close(hub, machine, timeout, *, startup_rollback=False):
    state = machine['state']
    if state == 'Stopped':
        return machine
    if state in STOPPABLE:
        if state == 'Running' and not startup_rollback:
            if not machine.get('jobId'):
                raise GPUError('Platform did not identify the running job; refusing shutdown')
            observed = hub.gpu(machine)
            if observed['compute_processes']:
                pids = ', '.join(str(app['pid']) for app in observed['compute_processes'])
                raise GPUError(f'GPU compute processes still running: {pids}')
            latest = hub.machine()
            if latest['state'] != 'Running' or latest.get('jobId') != machine.get('jobId'):
                raise GPUError('Machine job or state changed during shutdown check; recheck status')
        hub.call('devmachine', 'close', hub.id)
    elif state != 'Stopping':
        raise GPUError(f'refusing shutdown in unrecognized or failed state: {state}')
    return wait_for(hub.machine, {'Stopped'}, STOPPABLE | {'Stopping'}, time.monotonic() + timeout)


def start(hub, minutes, timeout):
    machine = hub.machine()
    if machine['state'] != 'Stopped':
        raise GPUError(f"start requires Stopped, got {machine['state']}; no power or timer change sent")
    config = hub.config
    for field, key in [('powerConfId', 'power_conf_id'), ('sshId', 'ssh_id'), ('startupType', 'startup_type')]:
        if machine.get(field) != config[key]:
            raise GPUError(f'Platform {field} differs from binding; review machine before starting')
    hub.ssh_files()
    end_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=minutes)
    startup_deadline = time.monotonic() + min(timeout, 60 * minutes - 60)
    # A transport failure here can be ambiguous. Do not retry or assume ownership.
    hub.call('devmachine', 'open', hub.id, '--startup-type', str(config['startup_type']))
    try:
        active = wait_for(hub.machine, STOPPABLE, {'Stopped'}, startup_deadline)
        timestamp = end_at.isoformat(timespec='seconds')
        try:
            timer = hub.call('devmachine', 'auto-shutdown', hub.id, '--at', timestamp)
        except GPUError:
            if active['state'] == 'Running':
                raise
            # The platform may reject scheduling until the machine is Running.
            active = wait_for(hub.machine, {'Running'}, BOOTING, startup_deadline)
            timer = hub.call('devmachine', 'auto-shutdown', hub.id, '--at', timestamp)
        if timer.get('id') != hub.id or timer.get('closeTime') != int(end_at.timestamp()) * 1000:
            raise GPUError('BitaCLI did not acknowledge the requested shutdown deadline')
        report({'event': 'platform_timer_acknowledged', 'id': hub.id, 'shutdown_at_utc': timestamp})
        active = wait_for(hub.machine, {'Running'}, BOOTING, startup_deadline)
        readback_deadline = min(time.monotonic() + 20, startup_deadline)
        while active['state'] == 'Running' and not deadline_visible(active, end_at):
            if time.monotonic() >= readback_deadline:
                raise GPUError('Platform details did not confirm the requested shutdown deadline')
            time.sleep(3)
            active = hub.machine()
        if active['state'] != 'Running' or (end_at - datetime.now(timezone.utc)).total_seconds() < 60:
            raise GPUError('Machine is no longer Running or there is less than one minute left')
        return {**summary(active), 'confirmed_shutdown_at_utc': timestamp}
    except (GPUError, KeyboardInterrupt) as exc:
        # No workload has been launched by this command. Undo an acknowledged start if setup fails.
        try:
            final = close(hub, hub.machine(), min(timeout, 180), startup_rollback=True)
        except (GPUError, KeyboardInterrupt) as rollback:
            raise GPUError(f'{exc or "start interrupted"}; rollback not confirmed: {rollback}; check the platform now') from exc
        raise GPUError(f'{exc or "start interrupted"}; post-failure platform state: {final["state"]}; inspect before retrying') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ['status', 'gpu', 'connection']:
        commands.add_parser(name)
    boot = commands.add_parser('start', help='start the bound machine and set a platform shutdown deadline')
    boot.add_argument('--minutes', type=int, required=True, choices=range(5, 1441), metavar='5..1440')
    stop = commands.add_parser('stop', help='check GPU processes, shut down, then confirm Stopped')
    stop.add_argument('--ready', action='store_true', required=True, help='authorized stop; jobs and result handoff checked')
    for command in [boot, stop]:
        command.add_argument('--wait-seconds', type=int, default=600, choices=range(1, 3601), metavar='1..3600')
    args = parser.parse_args()
    try:
        hub = BitaHub(json.loads(CONFIG.read_text(encoding='utf-8')))
        if args.action == 'start':
            report(start(hub, args.minutes, args.wait_seconds))
            return 0
        machine = hub.machine()
        if args.action == 'stop':
            report(summary(close(hub, machine, args.wait_seconds)))
        elif args.action == 'connection':
            report(hub.connection(machine))
        elif args.action == 'gpu':
            report({**summary(machine), 'telemetry': hub.gpu(machine) if machine['state'] == 'Running' else None})
        else:
            report(summary(machine))
        return 0
    except (GPUError, OSError, ValueError, KeyError) as exc:
        print(f'GPU: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
