from pathlib import Path

from scripts import git_health


def test_check_git_dir_writable_false_when_write_fails(monkeypatch, tmp_path):
    repo_root = tmp_path
    git_dir = repo_root / ".git"
    git_dir.mkdir()

    def _raise(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _raise, raising=True)
    writable, reason = git_health._check_git_dir_writable(repo_root)
    assert writable is False
    assert "read-only" in reason


def test_collect_git_health_parses_command_outputs(monkeypatch, tmp_path):
    outputs = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
        ("git", "rev-parse", "HEAD"): (0, "abc123"),
        ("git", "remote", "-v"): (0, "origin https://example/repo.git (fetch)"),
        ("git", "status", "--porcelain"): (0, " M README.md\n?? tests/new.py"),
        ("findmnt", "-T", str(tmp_path / ".git")): (0, "TARGET SOURCE FSTYPE OPTIONS\n/tmp/.git /dev/sda ext4 rw"),
    }

    def _fake_run(cmd, *, cwd):
        return outputs.get(tuple(cmd), (1, "missing"))

    monkeypatch.setattr(git_health, "_run", _fake_run)
    monkeypatch.setattr(git_health, "_check_git_dir_writable", lambda _repo: (True, "writable"))

    payload = git_health.collect_git_health(tmp_path)
    assert payload["branch"] == "main"
    assert payload["head"] == "abc123"
    assert payload["worktree_dirty"] is True
    assert len(payload["worktree_changes"]) == 2
    assert payload["git_dir_writable"] is True
    assert payload["remotes"] == ["origin https://example/repo.git (fetch)"]

