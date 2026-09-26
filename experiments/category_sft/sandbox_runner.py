"""Fail-closed Linux execution isolation for Python benchmark samples.

Uses bubblewrap namespaces, a narrowly mounted read-only interpreter/stdlib,
no host data/home/proc, no network, an empty environment and bounded resources.
Reference: https://github.com/containers/bubblewrap
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

BWRAP = Path(os.environ.get('CG_BWRAP', '/usr/bin/bwrap'))
PYTHON = Path('/usr/bin/python3.10')
MAX_OUTPUT = 4 << 20

BOOT = r'''
import ctypes, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**2, 8*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
resource.setrlimit(resource.RLIMIT_CPU, (int(sys.argv[1]), int(sys.argv[1])+1))
# RLIMIT_NPROC alone does not constrain a root-mapped uid. Deny process/thread
# creation and namespace escape explicitly before running any sample code.
try:
    sc = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sc.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sc.seccomp_rule_add.restype = ctypes.c_int
    sc.seccomp_load.argtypes = [ctypes.c_void_p]
    sc.seccomp_load.restype = ctypes.c_int
    sc.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sc.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError('seccomp_init')
    for name in ('fork', 'vfork', 'clone', 'clone3', 'unshare', 'setns',
                 'execve', 'execveat', 'ptrace', 'process_vm_readv',
                 'process_vm_writev', 'mount', 'umount2', 'pivot_root',
                 'socket', 'socketpair', 'bpf', 'perf_event_open',
                 'keyctl', 'add_key', 'request_key'):
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        if nr >= 0 and sc.seccomp_rule_add(ctx, 0x00050001, nr, 0) != 0:
            raise RuntimeError('seccomp_rule_add: ' + name)
    if sc.seccomp_load(ctx) != 0:
        raise RuntimeError('seccomp_load')
    sc.seccomp_release(ctx)
except Exception as exc:
    print('CG_SANDBOX_BOOT_FAILURE: ' + str(exc), file=sys.stderr)
    raise SystemExit(121)
sys.path.insert(0, '/opt/packages')
source = sys.stdin.read(2*1024**2 + 1)
if len(source) > 2*1024**2:
    raise ValueError('sample exceeds execution byte limit')
exec(compile(source, '/tmp/solution.py', 'exec'), {'__name__': '__main__'})
'''

def command(timeout):
    if not BWRAP.is_file() or not PYTHON.is_file():
        raise RuntimeError('isolated executor is not provisioned')
    cmd = [str(BWRAP), '--unshare-all', '--cap-drop', 'ALL', '--new-session', '--die-with-parent']
    # No /data, /mnt, /root, /home or host /proc is exposed.
    for path in ['/usr/bin/python3.10', '/usr/lib/python3.10',
                 '/lib/x86_64-linux-gnu', '/lib64', '/usr/lib/x86_64-linux-gnu']:
        if Path(path).exists(): cmd += ['--ro-bind', path, path]
    # Benchmarks sometimes accept NumPy solutions. Mount only this public package,
    # never the project, the entire venv or cached access credentials.
    for folder in [Path(os.environ.get('CG_SANDBOX_PACKAGES', '/usr/local/lib/python3.10/dist-packages'))]:
        if (folder/'numpy').is_dir():
            cmd += ['--ro-bind', str(folder/'numpy'), '/opt/packages/numpy']
            if (folder/'numpy.libs').is_dir(): cmd += ['--ro-bind', str(folder/'numpy.libs'), '/opt/packages/numpy.libs']
            break
    cmd += ['--dev','/dev','--tmpfs','/tmp','--chdir','/tmp','--clearenv']
    for key, val in {'PATH':'/usr/bin','LANG':'C.UTF-8','OPENBLAS_NUM_THREADS':'1',
                     'OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():
        cmd += ['--setenv',key,val]
    if 'PYTHONHASHSEED' in os.environ: cmd += ['--setenv','PYTHONHASHSEED',os.environ['PYTHONHASHSEED']]
    return cmd + ['--',str(PYTHON),'-S','-c',BOOT,str(max(1,math.ceil(timeout)))]

def run_python(code, timeout=12.0):
    if not isinstance(code,str) or not 0 < timeout <= 120: raise ValueError('invalid execution request')
    cmd = command(timeout)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,
                                start_new_session=True,env={'PATH':'/usr/bin:/bin'},close_fds=True)
        try:
            proc.communicate(code.encode('utf-8'), timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.communicate(); rc = 124
        out.seek(0); err.seek(0)
        stdout = out.read(MAX_OUTPUT).decode('utf-8',errors='replace')
        stderr = err.read(MAX_OUTPUT).decode('utf-8',errors='replace')
    # Infrastructure failure must abort evaluation, not become a wrong answer.
    if 'bwrap:' in stderr or 'CG_SANDBOX_BOOT_FAILURE:' in stderr:
        raise RuntimeError('sandbox infrastructure failure: '+stderr[-1000:])
    return rc,stdout,stderr

def self_test():
    cases = {
        'normal_python': ('import math; assert math.sqrt(9)==3; print("ok")', 2, 0),
        'numpy': ('import numpy as np; assert np.arange(3).sum()==3', 3, 0),
        'no_host_files': ('import os; assert not os.path.exists("/data"); assert not os.path.exists("/root"); assert not os.path.exists("/proc/1/environ")', 2, 0),
        'read_only_runtime': ('try:\n open("/usr/lib/python3.10/os.py","a").write("bad")\nexcept OSError:\n pass\nelse:\n raise AssertionError("runtime writable")', 2, 0),
        'network_isolated': ('import socket\ntry:\n socket.create_connection(("1.1.1.1",443),timeout=.3)\nexcept OSError:\n pass\nelse:\n raise AssertionError("network available")', 2, 0),
        'fork_denied': ('import os\ntry:\n os.fork()\nexcept PermissionError:\n pass\nelse:\n raise AssertionError("fork allowed")', 2, 0),
        'exec_denied': ('import os\ntry:\n os.execv("/usr/bin/python3.10",["python3.10","-c","pass"])\nexcept PermissionError:\n pass\nelse:\n raise AssertionError("exec allowed")', 2, 0),
        'timeout': ('while True: pass', .3, 124),
        'assertion_is_failure': ('assert False', 2, 1),
    }
    results=[]
    for name,(code,timeout,expected) in cases.items():
        t=time.time(); rc,out,err=run_python(code,timeout)
        results.append({'case':name,'returncode':rc,'expected':expected,'seconds':time.time()-t,'passed':rc==expected,'stderr':err[-300:]})
    print(json.dumps({'results':results,'all_passed':all(x['passed'] for x in results)},indent=2))
    if not all(x['passed'] for x in results): raise SystemExit(1)

if __name__ == '__main__': self_test()
