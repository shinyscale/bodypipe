#!/usr/bin/env bash
# Ralph Wiggum outer loop for bodypipe Qt port
# Usage: ./loop.sh [plan|build]
set -euo pipefail

MODE="${1:-plan}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

case "$MODE" in
  plan)
    PROMPT_FILE="PROMPT_plan.md"
    ;;
  build)
    PROMPT_FILE="PROMPT_build.md"
    ;;
  *)
    echo "Usage: $0 [plan|build]"
    exit 1
    ;;
esac

PROMPT="$(cat "ralph/$PROMPT_FILE")"

# Append current implementation plan state
if [[ -f ralph/IMPLEMENTATION_PLAN.md ]]; then
  PROMPT="$PROMPT

---
# Current IMPLEMENTATION_PLAN.md
$(cat ralph/IMPLEMENTATION_PLAN.md)"
fi

# Append all specs
for spec in ralph/specs/*.md; do
  if [[ -f "$spec" ]]; then
    PROMPT="$PROMPT

---
# Spec: $(basename "$spec")
$(cat "$spec")"
  fi
done

# Append AGENTS.md
if [[ -f ralph/AGENTS.md ]]; then
  PROMPT="$PROMPT

---
# AGENTS.md
$(cat ralph/AGENTS.md)"
fi

echo "=== Ralph Wiggum ($MODE mode) ==="
echo "Working directory: $PROJECT_DIR"
echo "Prompt length: $(echo "$PROMPT" | wc -c) bytes"
echo ""

# Run Claude Code with the assembled prompt
claude --dangerously-skip-permissions -p "$PROMPT"
