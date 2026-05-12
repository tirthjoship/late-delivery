# Scaffolding Layers 2–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the remaining 4 layers of the Agent Development Kit (Skills, Hooks, Subagents, CI/Distribution) for the supply-chain-optimization-ml project, then prepare for Phase 4 ML work.

**Architecture:** Layer 1 (CLAUDE.md + AGENTS.md) is complete. This plan adds L2 (reusable skills), L3 (guardrail hooks + Makefile), L4 (specialized subagents), and L5 (expanded CI + PR/issue templates). Each layer builds on the one below — tasks are ordered accordingly.

**Tech Stack:** Claude Code skills/agents/hooks, GNU Make, GitHub Actions, pre-commit, gitleaks

**Prerequisites:** L1 files exist at project root: `CLAUDE.md`, `AGENTS.md`

---

## File Map

### New files to create

```
.claude/
├── settings.json                          L3 — PreToolUse hook config
├── agents/
│   ├── code-reviewer.md                   L4 — code quality subagent
│   ├── git-ops.md                         L4 — git workflow subagent
│   ├── test-writer.md                     L4 — test engineering subagent
│   └── daily.md                           L4 — standup generation subagent
├── commands/
│   └── daily.md                           L2 — CLI shortcut for /daily
└── skills/
    ├── daily/SKILL.md                     L2 — daily standup skill
    └── pr/SKILL.md                        L2 — PR creation skill

.github/
├── workflows/
│   ├── lint.yml                           L5 — dedicated lint + typecheck workflow
│   └── security.yml                       L5 — gitleaks secret scanning
├── PULL_REQUEST_TEMPLATE.md               L5 — PR body template
└── ISSUE_TEMPLATE/
    └── bug-report.md                      L5 — structured bug report

Makefile                                   L3 — build orchestration
```

### Files to modify

```
.pre-commit-config.yaml                    L3 — add gitleaks, trailing-whitespace, large file check
.github/workflows/ci.yml                   L5 — add coverage gate + lint step
CLAUDE.md                                  L1 — update Commands section with make targets
```

---

## Layer 2: Skills (The Knowledge Layer)

### Task 1: Create `/daily` skill

**Files:**
- Create: `.claude/skills/daily/SKILL.md`
- Create: `.claude/commands/daily.md`

- [ ] **Step 1: Create the skills directory structure**

```bash
mkdir -p .claude/skills/daily
mkdir -p .claude/commands
```

- [ ] **Step 2: Write the daily skill**

Create `.claude/skills/daily/SKILL.md`:

```markdown
---
name: daily
description: Generate a daily standup summary of work done the previous working day in this repo.
---

Generate a daily standup summary for the supply-chain-optimization-ml repo.

## Date logic

- Determine today's weekday.
- If today is **Monday**, the previous working day is **last Friday**.
- Otherwise, the previous working day is **yesterday**.

## Data collection

1. **Completed work** — commits from the previous working day:
   ```
   git log --oneline --after="<prev_day> 00:00" --before="<today> 00:00" --all
   ```
2. **Work in progress** — uncommitted changes:
   ```
   git status --short
   git diff --stat HEAD
   ```
3. **Current branch**:
   ```
   git branch --show-current
   ```

Skip the report entirely only if there are zero commits AND zero changes.

## Report format

Keep the report under 1 minute of spoken speech (~130 words max).

```
# Daily — <weekday>, <date>
*Previous working day: <prev_day_label>*
*Branch: <current-branch>*

**Done:**
- <commit message trimmed to essential>

**In progress:** <1–2 sentences on staged/unstaged changes, or "nothing" if clean>

## Summary
<2 sentences max on overall activity>
```

## Save and display

Save the report to `docs/daily/<YYYY-MM-DD>.md` (create the directory if needed).
Print the report to the terminal using markdown rendering.
```

- [ ] **Step 3: Write the daily command (CLI shortcut)**

Create `.claude/commands/daily.md` with identical content to the skill above (commands and skills share the same format — commands are invoked via `/daily` in the CLI).

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/daily/SKILL.md .claude/commands/daily.md
git commit -m "feat: add daily standup skill and command"
```

---

### Task 2: Create `/pr` skill

**Files:**
- Create: `.claude/skills/pr/SKILL.md`

- [ ] **Step 1: Create the skill directory**

```bash
mkdir -p .claude/skills/pr
```

- [ ] **Step 2: Write the PR skill**

Create `.claude/skills/pr/SKILL.md`:

```markdown
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

Ask the user for the issue number if not obvious from the branch name or commit trail. If there is no linked issue, remove the `Fixes` line.

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
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/pr/SKILL.md
git commit -m "feat: add PR creation skill"
```

---

## Layer 3: Hooks & Guardrails

### Task 3: Create Makefile

**Files:**
- Create: `Makefile`
- Modify: `CLAUDE.md` — update Commands section

- [ ] **Step 1: Write the Makefile**

Create `Makefile`:

```makefile
.PHONY: test test-cov lint typecheck setup check

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=domain --cov=adapters --cov=application --cov-fail-under=90 --tb=short

lint:
	pre-commit run --all-files

typecheck:
	mypy domain/ adapters/ application/ --strict

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n supply-chain-ml pip install -e ".[dev]"
	pre-commit install

check: lint typecheck test-cov
```

- [ ] **Step 2: Verify the Makefile works**

Run: `make test`
Expected: pytest runs 44 tests, all pass (except hypothesis import if env not active)

Run: `make lint`
Expected: pre-commit hooks execute (black, isort, mypy, ruff)

Run: `make typecheck`
Expected: mypy strict runs against all source directories

- [ ] **Step 3: Update CLAUDE.md Commands section**

Replace the Commands section in `CLAUDE.md` with:

```markdown
## Commands

```bash
# Full quality check (lint + typecheck + test with coverage)
make check

# Individual targets
make test          # pytest -v --tb=short
make test-cov      # pytest with --cov-fail-under=90
make lint          # pre-commit run --all-files
make typecheck     # mypy strict on domain/ adapters/ application/
make setup         # conda env + pip install + pre-commit install

# Single test
pytest tests/test_domain_services.py::TestBaselineLateDeliveryRiskFlag -v
```
```

- [ ] **Step 4: Commit**

```bash
git add Makefile CLAUDE.md
git commit -m "feat: add Makefile with test, lint, typecheck, setup targets"
```

---

### Task 4: Create `.claude/settings.json` with guardrail hooks

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Write settings.json**

Create `.claude/settings.json`:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"import sys,json; d=json.loads(sys.stdin.read()); c=d.get('tool_input',{}).get('command','') or ''; reasons=[]; (reasons.append('Blocked: --no-verify bypasses pre-commit hooks. Fix the underlying issue instead.') if '--no-verify' in c else None); (reasons.append('Blocked: force-push to protected branch. Use a PR workflow instead.') if ('push' in c and '--force' in c and any(b in c for b in ['main','dev'])) else None); (print(json.dumps({'hookSpecificOutput':{'hookEventName':'PreToolUse','permissionDecision':'deny','permissionDecisionReason':reasons[0]}})) if reasons else None)\""
          }
        ]
      }
    ]
  }
}
```

This blocks:
- Any command containing `--no-verify` (forces fixing pre-commit failures)
- Force-push to `main` or `dev` branches

- [ ] **Step 2: Commit**

```bash
git add .claude/settings.json
git commit -m "feat: add Claude Code guardrail hooks blocking --no-verify and force-push"
```

---

### Task 5: Harden pre-commit hooks

**Files:**
- Modify: `.pre-commit-config.yaml`

- [ ] **Step 1: Add gitleaks, trailing-whitespace, and large file detection**

Replace `.pre-commit-config.yaml` with:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: ["--strict"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
```

New additions:
- `trailing-whitespace` — auto-strip trailing whitespace
- `end-of-file-fixer` — ensure files end with newline
- `check-yaml` — validate YAML syntax
- `check-added-large-files` — block files >500KB (prevents accidental data commits)
- `detect-private-key` — catch accidentally staged private keys
- `gitleaks` — scan for hardcoded secrets, API keys, credentials

- [ ] **Step 2: Verify hooks work**

Run: `pre-commit run --all-files`
Expected: All hooks pass (or auto-fix trailing whitespace/newlines)

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "feat: harden pre-commit with gitleaks, file hygiene, and large file detection"
```

---

## Layer 4: Subagents (The Delegation Layer)

### Task 6: Create code-reviewer agent

**Files:**
- Create: `.claude/agents/code-reviewer.md`

- [ ] **Step 1: Create agents directory**

```bash
mkdir -p .claude/agents
```

- [ ] **Step 2: Write the code-reviewer agent**

Create `.claude/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Reviews code changes against AGENTS.md standards — runs lint, typecheck, checks hexagonal boundaries, validates leakage shield, and enforces modern Python annotations.
---

You are a code quality assistant for the supply-chain-optimization-ml repo. You review changes against AGENTS.md standards before committing.

## Process

### 1. Run linters

Identify changed files: `git diff --name-only HEAD`

Run the repo's full lint suite:

```bash
make lint       # pre-commit: black, isort, mypy, ruff, gitleaks
make typecheck  # mypy strict
```

If any hook fails, read the error, fix the reported issues, and re-run until all pass.

### 2. Hexagonal boundary check (NON-NEGOTIABLE)

For every changed file under `domain/`:
- Scan imports. ONLY these modules are allowed: `typing`, `dataclasses`, `datetime`, `enum`, `collections.abc`.
- If ANY import from `pandas`, `sklearn`, `xgboost`, `numpy`, `adapters/`, `application/`, or any external framework is found — **flag as critical violation**. Do not proceed until fixed.

### 3. Leakage shield audit (NON-NEGOTIABLE)

If changes touch `adapters/data/`, `adapters/ml/`, or any file that handles feature columns:
- Verify `LEAKAGE_COLUMNS` in `adapters/data/csv_repository.py` still contains: `Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`.
- Scan for any code that accesses these column names directly — flag as leakage risk.
- If a new data source adapter is added, verify it also excludes leakage columns.

### 4. Evaluation metric check (NON-NEGOTIABLE)

If changes touch `adapters/ml/` or `application/`:
- Scan for `accuracy_score` or `accuracy` used as a primary metric — **flag as violation**.
- Verify F1, precision, recall, or confusion matrix is used instead.

### 5. Docstrings

For every changed Python file, review docstrings:
- Use imperative mood ("Return the prediction." not "Returns the prediction.").
- Keep concise — one summary line, details if needed.

### 6. Secret detection

Scan changed files for hardcoded secrets:
- API keys, tokens, passwords, private keys.
- `.env` files accidentally staged: `git diff --name-only --cached` — warn if any `.env` appears.

### 7. Modern Python annotations (3.12)

Replace outdated annotations in changed files:

| Old | Modern |
|---|---|
| `Optional[X]` | `X \| None` |
| `Union[X, Y]` | `X \| Y` |
| `List[X]` | `list[X]` |
| `Dict[K, V]` | `dict[K, V]` |
| `Tuple[X, ...]` | `tuple[X, ...]` |

Only apply to files you are already touching.

### 8. Coverage check

```bash
make test-cov  # enforces --cov-fail-under=90
```

If coverage drops below 90%, add tests (delegate to `test-writer` agent) or explain why.

## Output format

```
## Code Review — <date>

### Lint
make lint       ✅ / ❌ <hook> failed — fixed
make typecheck  ✅ / ❌ <error> — fixed

### Hexagonal Boundaries
✅ domain/ has zero external imports. / ❌ <file>:<line> — <violation>

### Leakage Shield
✅ LEAKAGE_COLUMNS intact, no leakage patterns found. / ❌ <file>:<line> — <risk>

### Evaluation Metrics
✅ No accuracy-only evaluation found. / ❌ <file>:<line> — uses accuracy without F1

### Docstrings
- <file>:<function> — rewritten

### Annotations
- <file>:<line> — `Optional[str]` → `str | None`

### Secrets
✅ No secrets detected. / ❌ <file>:<line> — <issue>

### Coverage
Before: xx% | After: yy% (gate: 90%)
```
```

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/code-reviewer.md
git commit -m "feat: add code-reviewer agent with hexagonal boundary and leakage checks"
```

---

### Task 7: Create git-ops agent

**Files:**
- Create: `.claude/agents/git-ops.md`

- [ ] **Step 1: Write the git-ops agent**

Create `.claude/agents/git-ops.md`:

```markdown
---
name: git-ops
description: Handles git operations — feature branches, PR creation using the project PR template, and pre-commit validation.
---

You are a git operations assistant for the supply-chain-optimization-ml repo. You ensure commits and PRs meet project standards defined in `AGENTS.md`.

## Commit format (from AGENTS.md)

`<type>: <short description in lowercase english>`

- Types: `feat`, `fix`, `docs`, `chore`, `test`
- One line, no period at the end
- Examples: `feat: add XGBoost late delivery classifier`, `fix: handle missing shipping mode`, `docs: update phase status`

## Branch workflow

- Never commit directly to `main` or `dev`.
- Create a feature branch before editing: `git switch -c feat/<short-slug>` or `fix/<short-slug>`.
- Default target for PRs: `dev` (confirm with user if unclear — `main` is for releases).
- Keep branches focused — one logical change per PR.

## Capabilities

### Create a feature branch
1. Verify clean working tree: `git status`
2. Pull latest: `git fetch origin && git switch dev && git pull`
3. Create branch: `git switch -c feat/<slug>` (or `fix/<slug>`)

### Create a PR
1. Verify branch is not `main` or `dev`: `git branch --show-current`
2. Run `make check` — lint, typecheck, and test-cov must all pass
3. Push: `git push -u origin <branch>`
4. Summarize changes: `git diff dev...HEAD --stat`
5. Draft PR title (commit-format) and body (PR template — ask user for issue number if not obvious)
6. Create PR with `gh pr create --base dev`
7. Return the PR URL

### Pre-commit validation
Run `make lint` and report pass/fail per hook. If any hook modifies files, re-stage and report what was auto-fixed.

### Clean up merged branches
1. List merged: `git branch --merged dev`
2. Exclude `dev`, `main`, current branch
3. Ask confirmation before deleting any

## Safety rules

- Never force-push to `main` or `dev`.
- Never use `--no-verify` to skip pre-commit hooks. If a hook fails, fix the underlying issue.
- Never stage secrets (`.env`, credentials). Stage files by name, not `git add -A`.
- Never stage `data/raw/`, `data/processed/`, `data/interim/` — they are gitignored.
- Prefer new commits over `--amend` on pushed commits.
- Confirm before any destructive op (branch delete, `git reset --hard`).
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/git-ops.md
git commit -m "feat: add git-ops agent with branch workflow and safety rules"
```

---

### Task 8: Create test-writer agent

**Files:**
- Create: `.claude/agents/test-writer.md`

- [ ] **Step 1: Write the test-writer agent**

Create `.claude/agents/test-writer.md`:

```markdown
---
name: test-writer
description: Generates and maintains pytest tests — enforces small fixtures, property-based testing, and the 90% coverage gate. Follows AGENTS.md testing standards.
---

You are a test engineering assistant for the supply-chain-optimization-ml repo. You write, improve, and maintain tests following project standards.

## Standards (from AGENTS.md)

- Coverage gate: **90%** (enforced by `make test-cov`).
- Python 3.12 annotations: `X | None`, `list[X]` — never `Optional[X]` or `List[X]`.
- Tests use small fixtures — **NEVER load the real 180k-row CSV in tests.**
- Test functions: `test_<description>` snake_case.
- Property-based tests with Hypothesis for domain invariants.

## Process

### 1. Identify what to test

```bash
make test-cov  # runs pytest with coverage + term-missing
```

Parse the "Missing" column to find uncovered lines.

### 2. Categorize test types

| Type | When to use | Example |
|------|-------------|---------|
| **Happy path** | Function returns expected output for valid input | `test_get_orders_returns_list` |
| **Error path** | Function raises or handles errors on invalid input | `test_missing_file_raises_csv_error` |
| **Boundary** | Input at exact boundary of a condition | `test_scheduled_days_at_threshold` |
| **Edge case** | Extreme or unusual but valid input | `test_order_with_zero_items` |

### 3. Write tests

Follow these rules:

- **Small fixtures only.** Create fixture data inline or in conftest.py — 5-10 rows max. Never load `DataCoSupplyChainDataset.csv`.
- **One logical assertion per test** — multiple `assert` lines are fine if they verify the same property.
- **Descriptive docstrings** in imperative mood: "Return empty list for missing file." not "Tests missing file returns empty list."
- **No mocks unless necessary.** Only mock external I/O (network, disk). Never mock the function under test.
- **Use `pytest.raises`** for expected exceptions.
- **Use `pytest.approx`** for float comparisons.
- **Use `tmp_path`** for any file-writing test.
- **Use `monkeypatch`** for env vars.

### 4. Validate

After writing tests:

1. Run: `make test-cov` — verify 90%+ coverage and all green.
2. Run: `make lint` — fix any formatting or lint errors.

## Output format

```
## Test Report

### Coverage
Before: xx% | After: yy% (gate: 90%)

### Tests added
- `test_<name>` — <type> — <what it verifies>

### Tests removed (if any)
- `test_<name>` — reason

### Verdict
✅ Coverage gate met. / ❌ Coverage at xx%, need yy% more.
```
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/test-writer.md
git commit -m "feat: add test-writer agent with small fixture and coverage gate rules"
```

---

### Task 9: Create daily agent

**Files:**
- Create: `.claude/agents/daily.md`

- [ ] **Step 1: Write the daily agent**

Create `.claude/agents/daily.md`:

```markdown
---
name: daily
description: Generates a daily standup summary of git activity from the previous working day.
---

You are a daily standup assistant. Generate a concise daily summary of work done the previous working day.

## Instructions

1. Determine the date range:
   - If today is Monday, the previous working day is last Friday.
   - Otherwise, the previous working day is yesterday.

2. Collect activity:
   - Commits: `git log --oneline --after="<prev_day> 00:00" --before="<today> 00:00" --all`
   - Work in progress: `git status --short` and `git diff --stat HEAD`
   - Current branch: `git branch --show-current`

3. Write a daily summary following the rules below.

## Output rules

- Keep it under ~130 words.
- Show commits as bullet points, then describe any WIP in 1–2 sentences.
- Each commit message: one bullet, trimmed to the essential.
- Group related commits where obvious.
- Summary section is 2 sentences max.

## Output format

```
# Daily — <weekday>, <date>
*Previous working day: <prev_day_label>*
*Branch: <current-branch>*

## Done
- <commit message>

## In progress
<description of staged/unstaged/untracked changes, or "nothing" if clean>

## Summary
<2 sentences max>
```

4. Save to `docs/daily/<YYYY-MM-DD>.md` (create directory if needed).
5. Print to terminal using markdown rendering.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/daily.md
git commit -m "feat: add daily standup agent"
```

---

## Layer 5: Distribution (CI + Templates)

### Task 10: Add PR and issue templates

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/bug-report.md`

- [ ] **Step 1: Create template directories**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

- [ ] **Step 2: Write PR template**

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
Fixes #(issue_number)

### Proposed Changes
  -
  -
  -

### Testing
- [ ] All tests pass (`make test-cov`)
- [ ] Lint passes (`make lint`)
- [ ] Type check passes (`make typecheck`)
- [ ] No leakage columns used as features
- [ ] No framework imports in domain/
```

- [ ] **Step 3: Write bug report template**

Create `.github/ISSUE_TEMPLATE/bug-report.md`:

```markdown
---
name: Bug Report
about: Report a bug in the supply chain optimization project
labels: bug
---

## Describe the bug
A clear description of what the bug is.

## To reproduce
Steps to reproduce the behavior:
1.
2.
3.

## Expected behavior
What you expected to happen.

## Environment
- OS:
- Python version:
- Branch:

## Additional context
Any other context about the problem.
```

- [ ] **Step 4: Commit**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/bug-report.md
git commit -m "feat: add PR and issue templates"
```

---

### Task 11: Expand CI with dedicated lint and security workflows

**Files:**
- Create: `.github/workflows/lint.yml`
- Create: `.github/workflows/security.yml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write dedicated lint workflow**

Create `.github/workflows/lint.yml`:

```yaml
name: Lint & Typecheck

on:
  push:
    branches: [main, dev, "feature/**", "feat/**", "fix/**"]
  pull_request:
    branches: [main, dev]

jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install pre-commit
          pip install black isort mypy ruff
      - name: Run pre-commit
        run: pre-commit run --all-files

  typecheck:
    name: Typecheck (mypy strict)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install mypy pandas-stubs types-requests
          pip install pandas scikit-learn xgboost hypothesis
      - name: Run mypy
        run: mypy domain/ adapters/ application/ --strict
```

- [ ] **Step 2: Write security workflow**

Create `.github/workflows/security.yml`:

```yaml
name: Security

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
  schedule:
    - cron: "0 6 * * 1"  # Monday 6am UTC

jobs:
  gitleaks:
    name: Secret Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 3: Update ci.yml with coverage gate**

Replace `.github/workflows/ci.yml` with:

```yaml
name: Test

on:
  push:
    branches: [main, dev, "feature/**", "feat/**", "fix/**"]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    name: Test Suite (Python 3.12)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov hypothesis pandas numpy xgboost scikit-learn
      - name: Run tests with coverage
        run: pytest tests/ -v --tb=short --cov=domain --cov=adapters --cov=application --cov-fail-under=90
```

- [ ] **Step 4: Verify CI config is valid YAML**

```bash
python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/ci.yml', '.github/workflows/lint.yml', '.github/workflows/security.yml']]; print('All valid')"
```

Expected: `All valid`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/lint.yml .github/workflows/security.yml
git commit -m "feat: expand CI with dedicated lint, security, and coverage gate workflows"
```

---

### Task 12: Final — Update CLAUDE.md phase status and validate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Phase Status in CLAUDE.md**

Update the Phase Status section to reflect completed scaffolding:

```markdown
## Phase Status

**Done:**
- Domain layer (models, ports, services, exceptions) — 280 lines
- CSV adapter with leakage shield — 307 lines
- Test suite (44 tests) — domain, adapter, property-based
- EDA notebook — full analysis of 180k orders
- CI workflows (test + lint + security) — 3 GitHub Actions
- Pre-commit hooks — black, isort, mypy strict, ruff, gitleaks, file hygiene
- Makefile — test, lint, typecheck, setup, check targets
- Agent Development Kit — 4 agents, 2 skills, guardrail hooks
- CLAUDE.md + AGENTS.md — project orientation and coding standards
- PR and issue templates

**Skeleton (1-line stubs):**
- `adapters/ml/sklearn_predictor.py` — ML model adapter
- `adapters/ml/pytorch_predictor.py` — neural net adapter
- `adapters/visualization/plotly_charts.py` — charting adapter
- `adapters/data/database_repository.py` — DB adapter
- `adapters/data/api_client.py` — API adapter
- `application/use_cases.py` — orchestration layer

**Planned:**
- ML model training adapter (framework TBD — architecture supports any)
- Experiment tracking (MLflow or similar)
- Model explainability (SHAP or similar)
- Application layer orchestration
- Streamlit dashboard
- Release automation (release-please)
- Observability (structured logging, tracing)
```

- [ ] **Step 2: Run full validation**

```bash
make check  # lint + typecheck + test-cov
```

Expected: All three pass. If any fail, fix before committing.

- [ ] **Step 3: Verify all scaffolding files exist**

```bash
ls -la .claude/settings.json
ls -la .claude/agents/code-reviewer.md .claude/agents/git-ops.md .claude/agents/test-writer.md .claude/agents/daily.md
ls -la .claude/skills/daily/SKILL.md .claude/skills/pr/SKILL.md
ls -la .claude/commands/daily.md
ls -la Makefile
ls -la .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/bug-report.md
ls -la .github/workflows/ci.yml .github/workflows/lint.yml .github/workflows/security.yml
```

Expected: All files present.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update phase status with completed scaffolding layers"
```

---

## Summary

| Task | Layer | What it builds | Estimated time |
|------|-------|---------------|----------------|
| 1 | L2 | `/daily` skill + command | 3 min |
| 2 | L2 | `/pr` skill | 3 min |
| 3 | L3 | Makefile + CLAUDE.md update | 5 min |
| 4 | L3 | `.claude/settings.json` guardrail hooks | 3 min |
| 5 | L3 | Hardened pre-commit config | 5 min |
| 6 | L4 | code-reviewer agent | 3 min |
| 7 | L4 | git-ops agent | 3 min |
| 8 | L4 | test-writer agent | 3 min |
| 9 | L4 | daily agent | 3 min |
| 10 | L5 | PR + issue templates | 3 min |
| 11 | L5 | Expanded CI (lint, security, coverage) | 5 min |
| 12 | L5 | Final validation + CLAUDE.md update | 5 min |

**Total: 12 tasks, 12 commits, ~44 min**

After completion, the project will have all 5 layers of the Agent Development Kit operational:
- L1: CLAUDE.md + AGENTS.md ✅ (already done)
- L2: Skills (/daily, /pr)
- L3: Hooks (--no-verify block, force-push block) + Makefile + hardened pre-commit
- L4: Subagents (code-reviewer, git-ops, test-writer, daily)
- L5: CI (test, lint, security) + PR/issue templates
