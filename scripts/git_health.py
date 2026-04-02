"""
输出仓库 Git 健康状态，重点检查 .git 是否可写。

用法：
    python scripts/git_health.py
    python scripts/git_health.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return 1, str(exc)
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def _check_git_dir_writable(repo_root: Path) -> tuple[bool, str]:
    test_file = repo_root / ".git" / ".codex_write_test"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True, "writable"
    except Exception as exc:
        return False, str(exc)


def collect_git_health(repo_root: Path) -> dict:
    status_code, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    head_code, head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    remotes_code, remotes = _run(["git", "remote", "-v"], cwd=repo_root)
    dirty_code, porcelain = _run(["git", "status", "--porcelain"], cwd=repo_root)
    mount_code, mount_info = _run(["findmnt", "-T", str(repo_root / ".git")], cwd=repo_root)
    writable, writable_reason = _check_git_dir_writable(repo_root)

    return {
        "repo_root": str(repo_root),
        "branch": branch if status_code == 0 else None,
        "head": head if head_code == 0 else None,
        "remotes": remotes.splitlines() if remotes_code == 0 and remotes else [],
        "worktree_dirty": bool(porcelain.strip()) if dirty_code == 0 else None,
        "worktree_changes": porcelain.splitlines() if dirty_code == 0 and porcelain else [],
        "git_dir_writable": writable,
        "git_dir_writable_reason": writable_reason,
        "git_mount": mount_info if mount_code == 0 else None,
    }


def _print_human_readable(payload: dict) -> None:
    print(f"repo_root: {payload['repo_root']}")
    print(f"branch: {payload.get('branch') or 'unknown'}")
    print(f"head: {payload.get('head') or 'unknown'}")
    print(f"git_dir_writable: {payload['git_dir_writable']} ({payload['git_dir_writable_reason']})")
    print(f"worktree_dirty: {payload.get('worktree_dirty')}")
    if payload.get("worktree_changes"):
        print("worktree_changes:")
        for line in payload["worktree_changes"][:20]:
            print(f"  {line}")
        if len(payload["worktree_changes"]) > 20:
            print(f"  ... ({len(payload['worktree_changes']) - 20} more)")
    print("remotes:")
    for line in payload.get("remotes", []):
        print(f"  {line}")
    if payload.get("git_mount"):
        print("git_mount:")
        print(payload["git_mount"])


def main() -> None:
    parser = argparse.ArgumentParser(description="检查仓库 Git 健康状态")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--strict", action="store_true", help="若 .git 不可写则返回非 0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    payload = collect_git_health(repo_root)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human_readable(payload)

    if args.strict and not payload.get("git_dir_writable", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
