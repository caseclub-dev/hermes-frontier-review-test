# M3 Workflow Reference — Real Coder Delegation with Mock Claude

This document describes the end-to-end correction loop workflow that connects
the Frontier Review engine to real Hermes Coder agent delegations via
`delegate_task`, using mock review (MockAnthropicTransport) for code review.

## No Anthropic API Key Is Used

The entire M3 pipeline uses `MockAnthropicTransport`. The Anthropic API key is
never read or transmitted at any point in this workflow.

---

## Phase 1: Initial Delegation

```
Orchestrator receives task → Creates disposable workspace → Builds delegation request
→ delegate_task(profile="coder", goal=..., context=request) → Coder implements code
→ Orchestrator verifies commit SHA, file list, test output
```

### Tool Sequence

1. **Create disposable workspace** — isolated temp directory with git init
2. **Build initial request** — `build_initial_delegation_request()`
3. **Delegate to Coder** — via `delegate_task(profile="coder", ...)`
4. **Parse response** — `parse_delegation_response()` → extract files, SHA, test output
5. **Verify workspace** — `GitWorkspaceVerifier.verify_workspace()` → confirm syntax valid, SHA matches

---

## Phase 2: Mock Review

```
Orchestrator runs review pipeline on actual code → MockAnthropicTransport generates findings
→ If PASS → Phase 4 (READY_FOR_OPERATOR_REVIEW)
→ If CHANGES_REQUIRED → Phase 3 (correction loop)
→ If BLOCKED → stop
```

### Tool Sequence

1. **Build mock review payload** — `MockReviewSimulator.create_changes_required(findings)` or `create_pass_review()`
2. **Parse review result** — json.loads() → check status field
3. **Check limits** — `is_duplicate_review(review_count, max_reviews)`, `is_max_correction_cycles(...)`

---

## Phase 3: Correction Loop (up to 3 cycles)

```
For each correction cycle:
  Orchestrator builds correction request → delegate_task(profile="coder") → Coder fixes code
  → Orchestrator verifies new SHA, evidence fingerprint changed → re-run tests
  → Build next review → PASS or CHANGES_REQUIRED or BLOCKED or ESCALATED_ARCHITECT
```

### Key Checks

- **No-op detection**: Compare pre/post SHA + evidence fingerprint. If unchanged → `handle_delegation_failure("no_changes")`
- **Evidence fingerprint changed**: `compute_evidence_fingerprint(sha, test_output)` — must differ from previous cycle
- **Tests pass after fix**: Re-run tests in workspace, parse output via `parse_delegation_response()`
- **Max cycles**: Track corrections_made. If >= 3 → `handle_delegation_failure("max_cycles_exhausted")`

### Tool Sequence (per cycle)

1. **Build correction request** — `build_correction_delegation_request(findings_text=..., ...)`
2. **Delegate to Coder** — via `delegate_task(profile="coder", ...)`
3. **Parse response** — `parse_delegation_response()` → new SHA, files modified
4. **Verify changes** — GitWorkspaceVerifier: new SHA different from previous?
5. **Run tests** — pytest/unittest in workspace
6. **Compute fingerprint** — verify changed from pre-fix fingerprint

---

## Phase 4: Final Status Determination

| Condition | Status | Terminal? | Action |
|-----------|--------|-----------|--------|
| Mock review returns `pass` | READY_FOR_OPERATOR_REVIEW | No | Present to operator |
| Max corrections (3) without PASS | ESCALATED_ARCHITECT | Yes | Requires operator intervention |
| Max reviews (4) without PASS | BLOCKED | Yes | Stop pipeline |
| Budget exhausted | BLOCKED | Yes | Stop pipeline |
| Malformed review response | BLOCKED | Yes | Fix transport, retry or escalate |

---

## Failure Scenarios & Handling

### A. Coder timeout / no response
- `handle_delegation_failure("timeout")` → blocked, terminal
- Orchestrator stops; may re-attempt with shorter timeout

### B. Coder returns no commit
- `parse_delegation_response()` → sha=None, has_changes=False
- `handle_delegation_failure("no_commit", response_text)` → changes_required, non-terminal
- Re-attempt delegation (up to max corrections)

### C. No-op correction (unchanged SHA)
- GitWorkspaceVerifier detects no diff after correction
- Fingerprint unchanged: `fp_before == fp_after`
- `handle_delegation_failure("no_changes")` → changes_required, non-terminal
- If at max cycles → escalate

### D. Tests fail after correction
- Workspace test runner returns failures
- `handle_delegation_failure("tests_failed", test_output)` → changes_required, non-terminal
- Re-attempt delegation (up to max corrections)

### E. Malformed mock review response
- JSON parsing fails for review payload
- `handle_delegation_failure("malformed_review", response_text or None)` → blocked, terminal
- Fix transport issue; do not retry automatically

### F. Max correction cycles exhausted
- corrections_made >= max_corrections (3)
- `handle_delegation_failure("max_cycles_exhausted")` → escalated_architect, terminal
- Requires manual operator intervention

### G. Repeated identical review attempted
- Same commit SHA + same evidence fingerprint reviewed again
- `is_duplicate_review(review_count, max_reviews)` returns True at limit (4)
- No new review dispatched; task marked for different approach

---

## M3 Demonstration Summary

The following concrete events were executed during the M3 demonstration:

### Initial Delegation
- **Task**: Implement calculator module with 6 functions and comprehensive tests
- **Workspace**: `/tmp/frontier_coder_test_workspace` (isolated disposable repo)
- **Coder response**: Created `calculator.py` + `test_calculator.py`, 25/25 tests pass
- **Git commit**: `4b820bf3729913172a078b1f21eb5ec50a4c7ff8`

### First Mock Review (changes_required)
- **Finding 1** [MEDIUM]: Unused `Union` import in calculator.py
- **Finding 2** [MEDIUM]: Inconsistent error message format in divide()
- **Finding 3** [LOW]: Missing overflow edge case tests

### Correction Delegation
- **Coder response**: Removed Union import, standardized divide error message, added 4 new edge case tests
- **Git commit**: `92a3daa06df2d75b2c07c8c89f9491f836ddd702`
- **Tests**: 29/29 pass (25 original + 4 new)

### Second Mock Review (pass)
- All findings addressed: no-unused-imports ✅, consistent error format ✅, edge case tests ✅
- Status: READY_FOR_OPERATOR_REVIEW
- Evidence fingerprints differ: `dfa193619a1df732` → `5bd44cc20983f0cc`

### Failure Scenarios Verified
All 7 failure modes handled correctly via `handle_delegation_failure()`:
- timeout → blocked (terminal) ✅
- no_commit → changes_required (re-attempt) ✅
- no_changes → changes_required (re-attempt) ✅
- tests_failed → changes_required (re-attempt) ✅
- malformed_review → blocked (terminal) ✅
- max_cycles_exhausted → escalated_architect (terminal) ✅
- unknown_error → blocked (terminal) ✅

---

## M3 Modified Files

1. `src/frontier_review/coder_integration.py` — Core M3 module (request builders, parsers, verifiers, mock simulator, correction loop runner, failure handler)
2. `tests/test_integration.py` — Integration test specification for all flows and failure scenarios
3. `src/frontier_review/__init__.py` — Updated exports to include new M3 symbols

## Git Commits

- Baseline: `16c1b986d72e1893b677f7680f022753005ab694` — M3 implementation committed
- Bug fix: `f4f6ced` — handle_delegation_failure None-safety fixes

## Backup

- `/tmp/frontier_review_m3_backup.tar.gz` — Complete project backup at 252K
