#!/usr/bin/env bash
# Guardrail hook for Claude Code PreToolUse events.
# Reads JSON from stdin, inspects Bash commands, blocks dangerous operations.
# Exit 2 = block the action. Exit 0 = allow.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# Block --no-verify (forces fixing pre-commit failures)
if echo "$COMMAND" | grep -q -- '--no-verify'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: --no-verify bypasses pre-commit hooks. Fix the underlying issue instead."}}'
  exit 2
fi

# Block force-push to protected branches
if echo "$COMMAND" | grep -q 'push' && echo "$COMMAND" | grep -q -- '--force' && echo "$COMMAND" | grep -qE '(main|dev)'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: force-push to protected branch. Use a PR workflow instead."}}'
  exit 2
fi

# Warn on rm -rf in data/ directories
if echo "$COMMAND" | grep -q 'rm' && echo "$COMMAND" | grep -q 'data/'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: destructive operation on data/ directory. Verify you have backups before proceeding."}}'
  exit 2
fi

exit 0
