---
name: domain-check
description: Validate hexagonal architecture compliance — verify domain/ has zero external imports and all adapters implement port Protocols.
---

Validate hexagonal architecture compliance for the supply-chain-optimization-ml project.

## What is hexagonal compliance?

The core rule: **dependencies point inward only.**

```
adapters/     →  domain/  ←  application/
(external)       (pure)      (orchestration)
```

- `domain/` imports ONLY: `typing`, `dataclasses`, `datetime`, `enum`, `collections.abc`, `__future__`
- `adapters/` imports from `domain/` to implement port Protocols
- `application/` imports from both `domain/` and `adapters/` to orchestrate

## Audit steps

### 1. Domain purity check (NON-NEGOTIABLE)

```bash
grep -rn "^import\|^from" domain/*.py
```

Filter out allowed modules. Any remaining import = **critical violation**.

Allowed imports in domain/:
- `from __future__ import annotations`
- `from typing import ...`
- `from dataclasses import ...`
- `from datetime import ...`
- `from enum import ...`
- `from collections.abc import ...`
- `from domain. ...` (internal cross-references)

Everything else = violation.

### 2. Port interface coverage

Read `domain/ports.py`. For each Protocol defined:
- Search `adapters/` for classes implementing it
- Verify method signatures match (name, parameters, return type)
- Flag any Protocol with zero implementations

### 3. Adapter → domain dependency direction

```bash
grep -rn "from application" adapters/
```

Adapters must NOT import from application/. If found = dependency inversion violation.

### 4. Application composition check

```bash
grep -rn "^import\|^from" application/*.py
```

Application layer SHOULD import from both domain/ and adapters/. Verify it wires them together (composition root pattern).

### 5. Frozen dataclass check

```bash
grep -n "dataclass" domain/models.py
```

Immutable entities (Product) should use `@dataclass(frozen=True)`. Mutable aggregates (Order) can use `@dataclass` without frozen.

### 6. No business logic in adapters

Review each adapter file. Business rules (risk classification, threshold logic, scoring) belong in `domain/services.py`. Adapters should only:
- Parse/transform external data into domain entities
- Delegate to domain services for logic
- Format domain results for external output

## Output format

```
## Domain Check — <date>

### Domain Purity
✅ Zero external imports / ❌ <file>:<line> — imports <module>

### Port Coverage
- SalesDataRepository → implemented by: DataCoCSVRepository ✅
- <other ports> → <status>

### Dependency Direction
✅ No reverse dependencies / ❌ <file> imports from wrong layer

### Frozen Dataclasses
✅ Immutable entities frozen / ⚠️ <class> should be frozen

### Business Logic Placement
✅ All logic in domain/services.py / ❌ <adapter>:<line> contains business rule

### Verdict
✅ Hexagonal architecture compliant / ❌ <N> violations found
```
