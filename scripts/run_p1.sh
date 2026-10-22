#!/usr/bin/env bash
# P1 非交互式验证脚本：不等待输入，逐项执行并在末尾汇总失败。
#
# 用法：
#   scripts/run_p1.sh
#   PYTHON_BIN=/path/to/python scripts/run_p1.sh
#   RUN_FRONTEND=1 scripts/run_p1.sh
#
# 说明：脚本只负责验证与报告，不负责生成代码。单项失败不会中断其余检查，
# 最终只要有任何失败项就以非零码退出。
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/usr/local/anaconda3/.conda/envs/injection_guard/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
RUN_FRONTEND="${RUN_FRONTEND:-0}"
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/injection-guard-p1.XXXXXX")"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PIP_NO_INPUT=1

FAILURES=()

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  printf '\n==> [%s] %s\n' "$name" "$*"
  if "$@" >"$log" 2>&1; then
    printf '    PASS (%s)\n' "$log"
  else
    printf '    FAIL (%s)\n' "$log"
    tail -n 60 "$log" || true
    FAILURES+=("$name")
  fi
}

printf 'InjectionGuard P1 verification\n'
printf 'repo:   %s\n' "$ROOT_DIR"
printf 'python: %s\n' "$PYTHON_BIN"
printf 'logs:   %s\n' "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'ERROR: Python 不存在或不可执行: %s\n' "$PYTHON_BIN" >&2
  printf '可通过 PYTHON_BIN=/path/to/python 覆盖。\n' >&2
  exit 2
fi

run_step "git-diff-check" git diff --check
run_step "ruff" "$PYTHON_BIN" -m ruff check src tests
run_step "pytest" "$PYTHON_BIN" -m pytest -q
run_step "eval-dry-run" "$PYTHON_BIN" -m injection_guard.cli.eval_guardrails --dry-run --cases eval/golden_cases.jsonl

if [[ "$RUN_FRONTEND" == "1" ]]; then
  if [[ -d frontend/node_modules ]]; then
    run_step "frontend-type-check" npm --prefix frontend run type-check
    run_step "frontend-test" npm --prefix frontend test
  else
    printf '\n==> [frontend] SKIP (frontend/node_modules 不存在)\n'
  fi
else
  printf '\n==> [frontend] SKIP (RUN_FRONTEND=1 可启用)\n'
fi

printf '\n===== P1 summary =====\n'
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  printf 'ALL PASS\n'
  exit 0
fi

printf 'FAILED: %s\n' "${FAILURES[*]}"
printf 'See logs: %s\n' "$LOG_DIR"
exit 1
