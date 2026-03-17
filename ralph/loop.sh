#!/bin/bash
# Usage: ./ralph/loop.sh [plan] [max_iterations]
# Examples:
#   ./ralph/loop.sh              # Build mode, unlimited
#   ./ralph/loop.sh 20           # Build mode, max 20 iterations
#   ./ralph/loop.sh plan         # Plan mode, unlimited
#   ./ralph/loop.sh plan 5       # Plan mode, max 5 iterations

if [ "$1" = "plan" ]; then
    MODE="plan"
    PROMPT_FILE="ralph/PROMPT_plan.md"
    MAX_ITERATIONS=${2:-0}
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    MODE="build"
    PROMPT_FILE="ralph/PROMPT_build.md"
    MAX_ITERATIONS=$1
else
    MODE="build"
    PROMPT_FILE="ralph/PROMPT_build.md"
    MAX_ITERATIONS=0
fi

ITERATION=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:   $MODE"
echo "Prompt: $PROMPT_FILE"
[ $MAX_ITERATIONS -gt 0 ] && echo "Max:    $MAX_ITERATIONS iterations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

while true; do
    if [ $MAX_ITERATIONS -gt 0 ] && [ $ITERATION -ge $MAX_ITERATIONS ]; then
        echo "Reached max iterations: $MAX_ITERATIONS"
        break
    fi

    cat "$PROMPT_FILE" | claude -p \
        --dangerously-skip-permissions \
        --model opus \
        --verbose

    ITERATION=$((ITERATION + 1))
    echo -e "\n\n======================== LOOP $ITERATION ========================\n"
done
