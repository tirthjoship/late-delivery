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
