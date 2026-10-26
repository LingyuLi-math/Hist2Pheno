########################################################
# 2026.10.26, feature/output integrity / 特征版本、全维有限值与原子缓存
########################################################
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _inventory(directory):
    # Include ctime/inode: replacing or overwriting an existing .pth invalidates cache.
    rows = []
    for path in sorted(Path(directory).glob('*.pth')):
        stat = path.stat()
        rows.append([path.name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino])
    return rows


def feature_manifest(directory, cache_dir):
    """Content SHA per token; unchanged files reuse hashes after complete stat inventory.

    No persistent in-process memo: later extraction/overwrites must be visible.
    First scan hashes all tokens; subsequent scans stat all tokens and hash changed ones.
    Active writers must finish before use; this is a snapshot, not proof of extraction completion.
    """
    directory = Path(directory).resolve()
    key = hashlib.sha256(str(directory).encode()).hexdigest()
    path = Path(cache_dir)/'feature_manifests'/f'{key}.json'
    with file_lock(str(path)+'.lock'):
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = {}
        inventory = _inventory(directory)
        if not inventory:
            raise RuntimeError(f'No .pth tokens in {directory}')
        previous = {row['stat'][0]: row for row in old.get('files', [])}
        files = []
        for stat in inventory:
            prior = previous.get(stat[0], {})
            value = prior.get('sha256') if prior.get('stat') == stat else sha256(directory/stat[0])
            files.append({'stat': stat, 'sha256': value})
        if inventory != _inventory(directory):
            raise RuntimeError(f'UNI2 extraction changed during manifest scan: {directory}; retry after extraction completes')
        content = [[row['stat'][0], row['sha256']] for row in files]
        fingerprint = hashlib.sha256(json.dumps(content, separators=(',', ':')).encode()).hexdigest()
        result = {'version': 2, 'directory': str(directory), 'count': len(files),
                  'fingerprint': fingerprint, 'files': files}
        if result != old:
            atomic_json(path, result)
        return {k: result[k] for k in ('version', 'directory', 'count', 'fingerprint')}


def finite_rows(matrix, rows=None, chunk_size=4096):
    """Bounded-memory all-dimension finite mask / 避免 N×D 布尔临时数组。"""
    n = len(matrix) if rows is None else len(rows)
    valid = np.empty(n, dtype=bool)
    for start in range(0, n, chunk_size):
        block = matrix[start:start+chunk_size] if rows is None else matrix[np.asarray(rows[start:start+chunk_size])]
        valid[start:start+len(block)] = np.isfinite(block).all(axis=1)
    return valid


def output_manifest(paths, root):
    root = Path(root)
    outputs = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f'Missing prediction output: {path}')
        outputs.append({'path': str(path.relative_to(root)), 'size': path.stat().st_size, 'sha256': sha256(path)})
    return outputs


def completed_run(marker, stamp, root, required=()):
    try:
        run = json.loads(Path(marker).read_text())
        if run.get('version') != 2 or run.get('stamp') != stamp or not run.get('outputs'):
            return None
        listed = {row['path'] for row in run['outputs']}
        if not set(required).issubset(listed):
            return None
        for row in run['outputs']:
            path = Path(root)/row['path']
            if not path.is_file() or path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
                return None
        return run
    except (OSError, ValueError, KeyError, TypeError):
        return None
