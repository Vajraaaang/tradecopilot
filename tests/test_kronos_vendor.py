"""The optional upstream model stays pinned, attributed, and packaged without credentials."""
import hashlib
import json
from pathlib import Path


def test_pinned_upstream_files_and_only_declared_import_patch():
    root = Path(__file__).resolve().parents[1]
    vendor = root / 'src/tradecopilot/_vendor/kronos'
    assert (vendor / 'UPSTREAM.json').is_file(), 'Kronos source must be pinned before use'
    manifest = json.loads((vendor / 'UPSTREAM.json').read_text())
    assert manifest['commit'] == '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
    for name in ('kronos.py', 'module.py', 'LICENSE'):
        data = (vendor / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['packaged_sha256']
    source = (vendor / 'kronos.py').read_text()
    assert 'sys.path.append' not in source
    assert 'from tradecopilot._vendor.kronos.module import *' in source
    assert 'MIT License' in (vendor / 'LICENSE').read_text()
