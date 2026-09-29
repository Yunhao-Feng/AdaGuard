"""Public documentation and source-audit regression checks; no network calls."""
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit_release', ROOT / 'tools/audit_release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_resource_links_are_consistent():
    for name in ('README.md', 'README.zh-CN.md', 'pyproject.toml'):
        text = (ROOT / name).read_text(encoding='utf-8')
        for url in audit.PUBLIC_URLS:
            if 'github.com' not in url:
                assert url in text, (name, url)


def test_readme_badges_link_to_resources():
    for name in ('README.md', 'README.zh-CN.md'):
        text = (ROOT / name).read_text(encoding='utf-8')
        badges = re.findall(r'\[!\[[^\]]+\]\(([^)]+)\)\]\(([^)]+)\)', text)
        assert len(badges) == 5
        assert {image for image, _ in badges} == audit.PUBLIC_BADGE_URLS
        assert {target for _, target in badges} == {
            url for url in audit.PUBLIC_URLS if 'github.com' not in url
        }


def test_badge_allowlist_is_exact_and_markdown_only(tmp_path):
    badge = sorted(audit.PUBLIC_BADGE_URLS)[0]
    (tmp_path / 'README.md').write_text(badge, encoding='utf-8')
    assert audit.scan(tmp_path) == []
    (tmp_path / 'README.md').write_text(badge + '&unexpected=value', encoding='utf-8')
    (tmp_path / 'metadata.toml').write_text('badge = ' + repr(badge), encoding='utf-8')
    assert sorted(audit.scan(tmp_path)) == [
        ('README.md', 1, 'network-endpoint'),
        ('metadata.toml', 1, 'network-endpoint'),
    ]


def test_readme_citations_match_bib_file():
    bib = (ROOT / 'CITATION.bib').read_text(encoding='utf-8').strip()
    for name in ('README.md', 'README.zh-CN.md'):
        text = (ROOT / name).read_text(encoding='utf-8')
        blocks = re.findall(r'```bibtex\n(.*?)\n```', text, flags=re.S)
        assert blocks == [bib]
        assert text.rstrip().endswith('```')
    assert '2609.34241' in bib
    cff = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    assert 'preferred-citation:' in cff
    assert '10.48550/arXiv.2609.34241' in cff
    assert 'version: 0.1.0' in cff


def test_local_documentation_links_exist():
    pages = list(ROOT.glob('*.md')) + list((ROOT / 'docs').glob('*.md'))
    for page in pages:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', page.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            local = target.split('#', 1)[0]
            assert (page.parent / local).exists(), (page.name, target)


@pytest.mark.parametrize('suffix', ['.md', '.toml', '.bib', '.cff'])
def test_public_links_are_allowed_in_metadata(tmp_path, suffix):
    text = '\n'.join(audit.PUBLIC_URLS)
    (tmp_path / ('resources' + suffix)).write_text(text, encoding='utf-8')
    assert audit.scan(tmp_path) == []


def test_public_link_does_not_hide_an_unknown_endpoint(tmp_path):
    public = sorted(audit.PUBLIC_URLS)[0]
    unexpected = public + '?token=not-approved'
    (tmp_path / 'README.md').write_text(public + ' ' + unexpected, encoding='utf-8')
    assert audit.scan(tmp_path) == [('README.md', 1, 'network-endpoint')]


def test_public_links_do_not_exempt_executable_code(tmp_path):
    (tmp_path / 'client.py').write_text('url = ' + repr(sorted(audit.PUBLIC_URLS)[0]), encoding='utf-8')
    assert audit.scan(tmp_path) == [('client.py', 1, 'network-endpoint')]


def test_network_import_and_syntax_errors_are_reported(tmp_path):
    # Build the import as data so this test does not import a network client.
    (tmp_path / 'client.py').write_text('import ' + 'requests\n', encoding='utf-8')
    (tmp_path / 'broken.py').write_text('def incomplete(', encoding='utf-8')
    assert audit.scan(tmp_path) == [
        ('broken.py', 1, 'invalid-python'),
        ('client.py', 1, 'network-client-import'),
    ]


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match='does not exist'):
        audit.scan(tmp_path / 'missing')


def test_export_checks_unexpected_files_and_hidden_files(tmp_path):
    (tmp_path / 'weights.bin').write_bytes(b'\x00\x01')
    (tmp_path / '.private').write_text('private data', encoding='utf-8')
    assert audit.scan(tmp_path) == [
        ('.private', 0, 'hidden-file'),
        ('weights.bin', 0, 'unexpected-file-type'),
    ]


def test_git_scan_includes_tracked_ignored_and_new_files(tmp_path):
    if not shutil.which('git'):
        pytest.skip('Git is required for source enumeration')
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)

    git('init')
    (tmp_path / '.gitignore').write_text('outputs/\n', encoding='utf-8')
    (tmp_path / 'outputs').mkdir()
    (tmp_path / 'outputs' / 'tracked.bin').write_bytes(b'\x00')
    git('add', '-f', 'outputs/tracked.bin')
    (tmp_path / 'outputs' / 'ignored.bin').write_bytes(b'\x00')
    (tmp_path / 'new.bin').write_bytes(b'\x00')
    assert audit.scan(tmp_path) == [
        ('new.bin', 0, 'unexpected-file-type'),
        ('outputs/tracked.bin', 0, 'unexpected-file-type'),
    ]


def test_git_enumeration_fails_closed(tmp_path, monkeypatch):
    (tmp_path / '.git').mkdir()
    def unavailable(*args, **kwargs):
        raise OSError('Git unavailable')
    monkeypatch.setattr(audit.subprocess, 'run', unavailable)
    with pytest.raises(RuntimeError, match='enumerate Git'):
        audit.scan(tmp_path)


def test_current_source_tree_passes_audit():
    assert audit.scan(ROOT) == []
