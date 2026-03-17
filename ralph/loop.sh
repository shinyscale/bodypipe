#!/usr/bin/env bash
# Ralph Wiggum outer loop for bodypipe Qt port
# Usage: ./ralph/loop.sh [plan|build]
#
# plan  — one shot: re-generate IMPLEMENTATION_PLAN.md from specs
# build — loop: pick next [ ] task, implement, commit, repeat until done or stuck
set -euo pipefail

MODE="${1:-build}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

case "$MODE" in
  plan)  PROMPT_FILE="PROMPT_plan.md" ;;
  build) PROMPT_FILE="PROMPT_build.md" ;;
  *)     echo "Usage: $0 [plan|build]"; exit 1 ;;
esac

assemble_prompt() {
  local PROMPT=""

  # 1. Mode prompt
  PROMPT="$(cat "ralph/$PROMPT_FILE")"

  # 2. Implementation plan
  if [[ -f ralph/IMPLEMENTATION_PLAN.md ]]; then
    PROMPT="$PROMPT

---
# Current IMPLEMENTATION_PLAN.md
$(cat ralph/IMPLEMENTATION_PLAN.md)"
  fi

  # 3. All specs
  for spec in ralph/specs/*.md; do
    if [[ -f "$spec" ]]; then
      PROMPT="$PROMPT

---
# Spec: $(basename "$spec")
$(cat "$spec")"
    fi
  done

  # 4. AGENTS.md
  if [[ -f ralph/AGENTS.md ]]; then
    PROMPT="$PROMPT

---
# AGENTS.md
$(cat ralph/AGENTS.md)"
  fi

  echo "$PROMPT"
}

has_todo_tasks() {
  grep -q '\[ \]' ralph/IMPLEMENTATION_PLAN.md 2>/dev/null
}

has_blocked_tasks() {
  grep -q '\[!\]' ralph/IMPLEMENTATION_PLAN.md 2>/dev/null
}

iteration=0

if [[ "$MODE" == "plan" ]]; then
  # Plan mode: one shot
  PROMPT="$(assemble_prompt)"
  echo "=== Ralph Wiggum (plan mode) ==="
  echo "Prompt: $(echo "$PROMPT" | wc -c) bytes"
  echo ""
  claude --dangerously-skip-permissions -p "$PROMPT"
  exit 0
fi

# Build mode: loop until no [ ] tasks remain or we hit a blocker
echo "=== Ralph Wiggum (build loop) ==="
echo "Working directory: $PROJECT_DIR"
echo ""

while has_todo_tasks; do
  iteration=$((iteration + 1))

  # Safety: bail if any task got stuck
  if has_blocked_tasks; then
    echo ""
    echo "=== BLOCKED: A task is marked [!]. Fix it and re-run. ==="
    grep '\[!\]' ralph/IMPLEMENTATION_PLAN.md
    exit 1
  fi

  # Show what's next
  next_task="$(grep -m1 '\[ \]' ralph/IMPLEMENTATION_PLAN.md || true)"
  echo "--- Iteration $iteration ---"
  echo "Next: $next_task"
  echo ""

  PROMPT="$(assemble_prompt)"

  # Run one Claude Code session
  if ! claude --dangerously-skip-permissions -p "$PROMPT"; then
    echo ""
    echo "=== Claude exited non-zero on iteration $iteration. Stopping. ==="
    exit 1
  fi

  echo ""
  echo "=== Iteration $iteration complete ==="
  echo ""
done

echo ""
echo "=== All tasks done after $iteration iterations ==="
