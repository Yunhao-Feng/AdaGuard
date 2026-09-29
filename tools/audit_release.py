"""Audit public source files. Findings print rule names and locations, never values."""
import argparse
import ast
import re
import subprocess
from pathlib import Path

ALLOWED = {'.py','.md','.toml','.json','.jsonl','.jinja','.txt','.bib','.cff'}
PUBLIC_URLS = {
    'https://github.com/Yunhao-Feng/AdaGuard',
    'https://arxiv.org/abs/2609.34241',
    'https://huggingface.co/papers/2609.34241',
    'https://huggingface.co/Yunhao-Feng/AdaGuard-0.6B',
    'https://huggingface.co/Yunhao-Feng/AdaGuard-4B',
    'https://huggingface.co/Yunhao-Feng/AdaGuard-8B',
}
PUBLIC_BADGE_URLS = {
    'https://img.shields.io/badge/arXiv-2609.34241-B31B1B?style=flat-square&logo=arxiv&logoColor=white',
    'https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E',
    'https://img.shields.io/badge/AdaGuard-0.6B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E',
    'https://img.shields.io/badge/AdaGuard-4B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E',
    'https://img.shields.io/badge/AdaGuard-8B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E',
}
PUBLIC_LINK_SUFFIXES = {'.md', '.toml', '.bib', '.cff'}
RULES = {
    'home-path': r'(?:/Users/|/home/|/root/|[A-Z]:\\Users\\)\S+',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'cloud-key': r'\b(?:AKIA|ASIA|LTAI)[A-Za-z0-9]{12,}\b',
    'service-token': r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b',
    'credential-assignment': r'''(?i)(?:api[_-]?key|access[_-]?key|secret[_-]?key|password|bearer)\s*[=:]\s*["'][^"'\s]{8,}["']''',
    'storage-uri': r'\b(?:oss|s3|gs)://\S+',
    'network-endpoint': r'''https?://[^\s)"'<>{}\[\]`\\]+''',
    'machine-address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}
NETWORK_IMPORTS = {'openai','requests','httpx','urllib','boto3','oss2','wandb','socket'}


def source_paths(root):
    """Include tracked and new source files, without traversing Git internals."""
    if (root / '.git').exists():
        try:
            result = subprocess.run(
                ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('could not enumerate Git source files') from exc
        names = set(result.stdout.decode('utf-8').split('\0')) - {''}
        for name in sorted(names):
            path = root / name
            # A deleted tracked file is not part of the current source tree.
            if path.exists() or path.is_symlink():
                yield path
    else:
        yield from sorted(root.rglob('*'))


def scan(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError('source directory does not exist')
    findings=[]
    for path in source_paths(root):
        relative=path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        if any(root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
            findings.append((relative,0,'symlink')); continue
        if not path.is_file():
            continue
        if any(x.startswith('.') and x != '.gitignore' for x in path.relative_to(root).parts):
            findings.append((relative,0,'hidden-file')); continue
        if path.name not in {'LICENSE','.gitignore'} and path.suffix not in ALLOWED:
            findings.append((relative,0,'unexpected-file-type')); continue
        try:
            text=path.read_text(encoding='utf-8')
        except (UnicodeError,OSError):
            findings.append((relative,0,'non-text')); continue
        # Scanner contains detection signatures, not release credentials.
        if relative == 'tools/audit_release.py':
            continue
        for line_number,line in enumerate(text.splitlines(),1):
            for name,pattern in RULES.items():
                matches = list(re.finditer(pattern, line))
                if name == 'network-endpoint' and path.suffix in PUBLIC_LINK_SUFFIXES:
                    allowed_urls = PUBLIC_URLS | (PUBLIC_BADGE_URLS if path.suffix == '.md' else set())
                    matches = [m for m in matches if m.group().rstrip('.,;:') not in allowed_urls]
                if matches:
                    findings.append((relative,line_number,name))
        if path.suffix=='.py':
            try:
                tree=ast.parse(text)
            except SyntaxError as exc:
                findings.append((relative, exc.lineno or 0, 'invalid-python')); continue
            for node in ast.walk(tree):
                modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any(m.split('.')[0] in NETWORK_IMPORTS for m in modules):
                    findings.append((relative,node.lineno,'network-client-import'))
    return findings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    args=p.parse_args()
    try:
        findings=scan(Path(args.directory).resolve())
    except (OSError, RuntimeError, UnicodeError):
        print('source-enumeration-error')
        raise SystemExit(1) from None
    for path,line,rule in findings:
        print(f'{path}:{line}: {rule}')
    print(f'{len(findings)} findings')
    raise SystemExit(bool(findings))


if __name__=='__main__':
    main()
