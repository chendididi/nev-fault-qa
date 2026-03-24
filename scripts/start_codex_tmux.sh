#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${CODEX_TMUX_SESSION:-nev}"
WORKDIR="${CODEX_WORKDIR:-/root/ubuntuchen_file/nev-fault-qa}"
MODE="start"
ATTACH=1

usage() {
  cat <<'EOF'
Usage:
  scripts/start_codex_tmux.sh [start|resume|attach] [--no-attach] [--session NAME] [--workdir DIR]

Modes:
  start    Create the tmux session if needed and start a fresh Codex session.
  resume   Create the tmux session if needed and run `codex resume --last`.
  attach   Only attach to an existing tmux session.

Options:
  --no-attach      Create/start the session but do not attach immediately.
  --session NAME   Override the tmux session name. Default: nev
  --workdir DIR    Override the project directory.
  -h, --help       Show this help message.

Examples:
  scripts/start_codex_tmux.sh
  scripts/start_codex_tmux.sh resume
  scripts/start_codex_tmux.sh --session nev-prod
  scripts/start_codex_tmux.sh resume --no-attach
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

has_session() {
  tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

attach_session() {
  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "$SESSION_NAME"
  fi
  exec tmux attach -t "$SESSION_NAME"
}

start_tmux_session() {
  local codex_command="$1"

  tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"
  tmux send-keys -t "${SESSION_NAME}:0.0" "$codex_command" C-m
  echo "Started tmux session '$SESSION_NAME' in $WORKDIR"
}

build_codex_command() {
  local mode="$1"
  if [[ "$mode" == "resume" ]]; then
    printf "cd %q && codex resume --last --no-alt-screen -C %q" "$WORKDIR" "$WORKDIR"
  else
    printf "cd %q && codex --no-alt-screen -C %q" "$WORKDIR" "$WORKDIR"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|resume|attach)
      MODE="$1"
      shift
      ;;
    --no-attach)
      ATTACH=0
      shift
      ;;
    --session)
      SESSION_NAME="${2:?missing session name}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:?missing workdir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_cmd tmux
require_cmd codex

if [[ ! -d "$WORKDIR" ]]; then
  echo "Workdir does not exist: $WORKDIR" >&2
  exit 1
fi

case "$MODE" in
  attach)
    if ! has_session; then
      echo "tmux session '$SESSION_NAME' does not exist." >&2
      echo "Run: scripts/start_codex_tmux.sh start --session $SESSION_NAME" >&2
      exit 1
    fi
    ;;
  start|resume)
    if ! has_session; then
      start_tmux_session "$(build_codex_command "$MODE")"
    else
      echo "tmux session '$SESSION_NAME' already exists, attaching to it."
    fi
    ;;
esac

if [[ "$ATTACH" -eq 1 ]]; then
  attach_session
fi

echo "Session '$SESSION_NAME' is ready but not attached."
