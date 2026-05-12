---
name: pr
description: Create a pull request following the project PR template and the commit format defined in AGENTS.md.
---

Create a pull request for the current branch following project standards.

## Pre-flight checks

Before creating the PR, verify:

1. Not on a protected branch: `git branch --show-current` — must not be `main` or `dev`.
2. All tests pass: `make test`.
3. Type check passes: `make typecheck`.
4. Linters pass: `make lint`.
5. Working tree is clean: `git status`.

If any check fails, stop and fix the issues before proceeding.

## PR format

**Title:** concise, imperative, matches commit format from AGENTS.md (no `[AI]:` prefix).

Examples:
- `feat: add late delivery risk classifier`
- `fix: handle missing shipping mode in CSV adapter`
- `docs: update phase status in CLAUDE.md`

**Body:** populate `.github/PULL_REQUEST_TEMPLATE.md`:

```
Fixes #<issue_number>

### Proposed Changes
  - <change 1>
  - <change 2>
  - <change 3>
```

Ask the user for the issue number if not obvious from the branch name. If there is no linked issue, remove the `Fixes` line.

## Create the PR

```bash
git push -u origin <current-branch>
gh pr create --base dev --title "<type>: <title>" --body "$(cat <<'EOF'
Fixes #<issue_number>

### Proposed Changes
  - <change 1>
  - <change 2>
  - <change 3>
EOF
)"
```

Default base branch is `dev`. Confirm with the user before targeting `main`.

Return the PR URL when done.
