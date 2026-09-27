"""Preserve the exact dirty baseline before Phase 5.3 edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path('/root/autodl-tmp/eagle3-phase5.3-20260925')
    root.mkdir(exist_ok=False)
    files = [p for folder in ('nanovllm', 'tests', 'docs', 'benchmarks')
             for p in Path(folder).rglob('*') if p.is_file()
             and p.suffix in ('.py', '.md', '.json', '.csv')
             and '__pycache__' not in str(p)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'baseline.json').write_text(json.dumps(dict(hashes=hashes,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--short'], text=True)), indent=2))
    (root/'baseline.diff').write_bytes(subprocess.check_output(['git', 'diff', '--binary']))
    with tarfile.open(root/'baseline-source.tar.gz', 'w:gz') as archive:
        for p in files:
            archive.add(p)
    print(root, len(files))


if __name__ == '__main__':
    main()
