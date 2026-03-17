#!/usr/bin/env bash
# Ralph Wiggum — MEOW loop (Molecular Expressions Of Work)
#
# Each iteration either PLANS one feature (traces it through GVHMR source)
# or BUILDS one feature (implements the Qt equivalent).
#
# Usage: ralph/loop.sh [--dry-run]
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GVHMR_ROOT="$(cd "$PROJECT_DIR/../GVHMR" 2>/dev/null && pwd || echo "")"

cd "$PROJECT_DIR"

if [[ -z "$GVHMR_ROOT" ]]; then
  echo "ERROR: Cannot find GVHMR root at $PROJECT_DIR/../GVHMR"
  exit 1
fi

FEATURES_FILE="ralph/FEATURES.md"
PROMPT_TMP="$(mktemp)"
trap 'rm -f "$PROMPT_TMP"' EXIT

MAX_RETRIES=3  # max attempts for a single feature before marking [!]

# --- Feature parsing ---

next_unplanned() {
  grep -m1 '^\[ \]' "$FEATURES_FILE" 2>/dev/null || true
}

next_planned() {
  grep -m1 '^\[P\]' "$FEATURES_FILE" 2>/dev/null || true
}

has_blocked() {
  grep -q '^\[!\]' "$FEATURES_FILE" 2>/dev/null
}

has_work() {
  grep -qE '^\[ \]|^\[P\]' "$FEATURES_FILE" 2>/dev/null
}

parse_feature_num() {
  echo "$1" | sed -E 's/^\[[^]]*\] ([0-9]+)\..*/\1/'
}

parse_feature_slug() {
  echo "$1" | sed -E 's/^\[[^]]*\] [0-9]+\. ([^ ]+) —.*/\1/'
}

mark_feature() {
  local num="$1" new_status="$2"
  if $DRY_RUN; then
    echo "[dry-run] Would mark feature $num as [$new_status]"
    return
  fi
  sed -i -E "s/^\[[^]]*\] ${num}\./[${new_status}] ${num}./" "$FEATURES_FILE"
}

feature_status() {
  local num="$1"
  grep " ${num}\." "$FEATURES_FILE" | head -1 | sed -E 's/^\[([^]]*)\].*/\1/'
}

# --- Prompt assembly ---

assemble_plan_prompt() {
  cat ralph/PROMPT_plan.md > "$PROMPT_TMP"
  printf '\n\n## GVHMR_ROOT\n%s\n' "$GVHMR_ROOT" >> "$PROMPT_TMP"
  printf '\n\n---\n# ralph/FEATURES.md\n' >> "$PROMPT_TMP"
  cat "$FEATURES_FILE" >> "$PROMPT_TMP"
  printf '\n\n---\n# ralph/AGENTS.md\n' >> "$PROMPT_TMP"
  cat ralph/AGENTS.md >> "$PROMPT_TMP"
}

assemble_build_prompt() {
  local feature_line="$1"
  local num slug trace_file
  num="$(parse_feature_num "$feature_line")"
  slug="$(parse_feature_slug "$feature_line")"
  trace_file="ralph/traces/${num}-${slug}.md"

  cat ralph/PROMPT_build.md > "$PROMPT_TMP"
  printf '\n\n## GVHMR_ROOT\n%s\n' "$GVHMR_ROOT" >> "$PROMPT_TMP"
  printf '\n\n---\n# ralph/FEATURES.md\n' >> "$PROMPT_TMP"
  cat "$FEATURES_FILE" >> "$PROMPT_TMP"

  if [[ -f "$trace_file" ]]; then
    printf '\n\n---\n# Trace: %s\n' "$(basename "$trace_file")" >> "$PROMPT_TMP"
    cat "$trace_file" >> "$PROMPT_TMP"
  else
    echo ""
    echo "ERROR: Trace file not found: $trace_file"
    echo "Cannot build without a plan. Marking feature $num as [!]."
    mark_feature "$num" "!"
    return 1
  fi

  printf '\n\n---\n# ralph/AGENTS.md\n' >> "$PROMPT_TMP"
  cat ralph/AGENTS.md >> "$PROMPT_TMP"
}

run_claude() {
  if $DRY_RUN; then
    echo "[dry-run] Would send $(wc -c < "$PROMPT_TMP") bytes to claude"
    return 0
  fi
  cat "$PROMPT_TMP" | claude --dangerously-skip-permissions -p
}

# --- Main loop ---

echo "=== Ralph Wiggum (MEOW loop) ==="
echo "Project:   $PROJECT_DIR"
echo "GVHMR:     $GVHMR_ROOT"
$DRY_RUN && echo "Mode:      DRY RUN"
echo ""

iteration=0
retries=0
last_feature=""

while has_work; do
  iteration=$((iteration + 1))

  if has_blocked; then
    echo ""
    echo "=== BLOCKED ==="
    grep '^\[!\]' "$FEATURES_FILE"
    exit 1
  fi

  # Decide: build a planned feature, or plan the next unplanned one
  planned="$(next_planned)"
  unplanned="$(next_unplanned)"

  if [[ -n "$planned" ]]; then
    action="build"
    feature_line="$planned"
  elif [[ -n "$unplanned" ]]; then
    action="plan"
    feature_line="$unplanned"
  else
    break
  fi

  num="$(parse_feature_num "$feature_line")"
  slug="$(parse_feature_slug "$feature_line")"

  # Track retries per feature
  if [[ "$num" == "$last_feature" ]]; then
    retries=$((retries + 1))
  else
    retries=0
    last_feature="$num"
  fi

  if [[ $retries -ge $MAX_RETRIES ]]; then
    echo "=== Feature $num failed after $MAX_RETRIES attempts. Marking [!]. ==="
    mark_feature "$num" "!"
    continue
  fi

  echo "--- Iteration $iteration: $action $num. $slug (attempt $((retries + 1))) ---"

  if [[ "$action" == "plan" ]]; then
    assemble_plan_prompt
  else
    if ! assemble_build_prompt "$feature_line"; then
      continue
    fi
  fi

  if ! run_claude; then
    echo "=== Claude exited non-zero. Will retry. ==="
    continue
  fi

  # Check if Claude updated the status
  status="$(feature_status "$num")"
  expected_status="P"
  [[ "$action" == "build" ]] && expected_status="x"

  if [[ "$status" != "$expected_status" ]]; then
    echo "WARNING: Feature $num status is [$status], expected [$expected_status]."
    if [[ "$action" == "plan" ]]; then
      # Check if trace file was created
      trace_file="ralph/traces/${num}-${slug}.md"
      if [[ -f "$trace_file" ]]; then
        echo "Trace file exists. Marking [P]."
        mark_feature "$num" "P"
      else
        echo "No trace file. Will retry."
      fi
    fi
    # For build: never auto-mark. Let it retry.
  fi

  echo "=== Iteration $iteration complete ==="
  echo ""

  # Dry-run: show what the first iteration would do, then stop
  if $DRY_RUN; then
    echo "[dry-run] Stopping after first iteration. In real mode, loop continues."
    break
  fi
done

echo ""
echo "=== Done after $iteration iterations ==="
