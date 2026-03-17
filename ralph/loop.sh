#!/usr/bin/env bash
# Ralph Wiggum outer loop for bodypipe Qt port
# Usage: ralph/loop.sh [plan|build]
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

# Temp file for prompt (avoids "argument list too long" with large prompts)
PROMPT_TMP="$(mktemp)"
trap 'rm -f "$PROMPT_TMP"' EXIT

assemble_prompt() {
  # Write prompt to temp file instead of shell variable
  cat "ralph/$PROMPT_FILE" > "$PROMPT_TMP"

  # Append implementation plan
  if [[ -f ralph/IMPLEMENTATION_PLAN.md ]]; then
    printf '\n\n---\n# Current IMPLEMENTATION_PLAN.md\n' >> "$PROMPT_TMP"
    cat ralph/IMPLEMENTATION_PLAN.md >> "$PROMPT_TMP"
  fi

  # Append all specs
  for spec in ralph/specs/*.md; do
    if [[ -f "$spec" ]]; then
      printf '\n\n---\n# Spec: %s\n' "$(basename "$spec")" >> "$PROMPT_TMP"
      cat "$spec" >> "$PROMPT_TMP"
    fi
  done

  # Append AGENTS.md
  if [[ -f ralph/AGENTS.md ]]; then
    printf '\n\n---\n# AGENTS.md\n' >> "$PROMPT_TMP"
    cat ralph/AGENTS.md >> "$PROMPT_TMP"
  fi
}

run_claude() {
  cat "$PROMPT_TMP" | claude --dangerously-skip-permissions -p
}

has_todo_tasks() {
  grep -q '\[ \]' ralph/IMPLEMENTATION_PLAN.md 2>/dev/null
}

has_blocked_tasks() {
  grep -q '\[!\]' ralph/IMPLEMENTATION_PLAN.md 2>/dev/null
}

iteration=0

if [[ "$MODE" == "plan" ]]; then
  assemble_prompt
  echo "=== Ralph Wiggum (plan mode) ==="
  echo "Prompt: $(wc -c < "$PROMPT_TMP") bytes"
  echo ""
  run_claude
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

  assemble_prompt

  # Run one Claude Code session
  if ! run_claude; then
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
