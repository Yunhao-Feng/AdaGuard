"""Fail-closed source audit. Findings print rule names and locations, never values."""
import argparse
import ast
import re
from pathlib import Path

ALLOWED = {'.py','.md','.toml','.json','.jsonl','.jinja','.txt'}
RULES = {
    'home-path': r'(?:/Users/|/home/|/root/|[A-Z]:\\Users\\)\S+',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'cloud-key': r'\b(?:AKIA|ASIA|LTAI)[A-Za-z0-9]{12,}\b',
    'service-token': r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b',
    'credential-assignment': r'''(?i)(?:api[_-]?key|access[_-]?key|secret[_-]?key|password|bearer)\s*[=:]\s*["'][^"'\s]{8,}["']''',
    'storage-uri': r'\b(?:oss|s3|gs)://\S+',
    'network-endpoint': r'https?://[^\s)"<>]+',
    'machine-address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}
NETWORK_IMPORTS = {'openai','requests','httpx','urllib','boto3','oss2','wandb','socket'}


def scan(root):
    findings=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root).as_posix()
        if path.is_symlink():
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
                if re.search(pattern,line):
                    findings.append((relative,line_number,name))
        if path.suffix=='.py':
            tree=ast.parse(text)
            for node in ast.walk(tree):
                modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any(m.split('.')[0] in NETWORK_IMPORTS for m in modules):
                    findings.append((relative,node.lineno,'network-client-import'))
    return findings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    args=p.parse_args()
    findings=scan(Path(args.directory).resolve())
    for path,line,rule in findings:
        print(f'{path}:{line}: {rule}')
    print(f'{len(findings)} findings')
    raise SystemExit(bool(findings))


if __name__=='__main__':
    main()
