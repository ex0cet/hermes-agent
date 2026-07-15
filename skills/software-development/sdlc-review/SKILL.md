---
name: sdlc-review
description: "Kanban SDLC review: verify PRs, ACs, diffs, and test results."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, sdlc, code-review, pr-review, quality]
    related_skills: [engineering-review, requesting-code-review, github-code-review]
---

# SDLC Review (Kanban)

Structured PR/change review for Kanban review-lane tasks. This skill is
loaded by the Kanban dispatcher when a task moves to the `review` status.
It guides the review agent through verification of the implementation
before the PR can be merged or sent back for rework.

**Core principle:** The reviewer must independently verify every claim
made by the implementer. Self-reported success is never sufficient
evidence for approval.

## When to Use

This skill is loaded automatically by the Kanban dispatcher for tasks in
the `review` column. It is not intended for manual invocation.

## How the Review Lane Works

A task enters `review` when the implementer submits a PR. The Kanban
dispatcher spawns a review agent with this skill loaded. The review
agent examines the PR and either:

- **Approves (→ `done`):** the PR is merged and the task is completed.
- **Requests changes (→ `running`):** the task is returned to the
  implementer with review feedback.

## Procedure

### Step 1 — Gather context

Read the Kanban task body, acceptance criteria, and any prior comments:

```text
Task body:  <read from the task's description / notes>
Assignee:   <the worker who implemented this>
Target:     PR #<number> at <URL>
Branch:     <branch-name>
SHA:        <commit-sha>
```

Use `search_files` and `read_file` (scoped to the workspace) to inspect
the actual code changes. Do NOT rely on the implementer's summary alone.

### Step 2 — Pin the review SHA

Record the exact SHA of the commit being reviewed. This is the SHA the
implementer's workspace branch points to at the time of review. If the
SHA changes during review, the review is stale and must be restarted.

```bash
git rev-parse HEAD
```

**Never approve a stale SHA.** If the commit has moved since the review
started, abort and signal that a fresh review is needed.

### Step 3 — Verify acceptance criteria

For each acceptance criterion listed in the task:

1. Find evidence in the diff or running system that it is met.
2. Note the exact line or test that satisfies it.
3. If evidence cannot be found, flag as incomplete.

### Step 4 — Review the diff

Examine the actual changes. Pay attention to:

- **Scope:** does the change address ONLY the stated requirements?
  Flag any scope-creep or unrelated changes.
- **Protected files:** if the repo has protected paths
  (config, secrets, build files, CI config), verify no unauthorized
  modifications.
- **Test coverage:** are there tests for the new/changed behavior?
  Do existing tests still pass?
- **Edge cases:** error handling, null inputs, boundary conditions,
  concurrent access.
- **Regressions:** could this change break existing functionality?

Use `terminal` to run tests if test infrastructure is available in the
workspace.

### Step 5 — Record findings in Kanban

Post structured review results as a Kanban comment via the
`kanban_comment` tool (available through the `kanban` toolset):

```text
## Review Result: <PASS | CHANGES REQUESTED>

### Acceptance Criteria
- [x] AC1: <verified at path:line>
- [x] AC2: <verified at path:line>
- [ ] AC3: <NOT verified — details>

### Issues Found
1. <severity>: <description> (path:line)
2. ...

### Verdict
<explicit pass or changes_requested statement>
```

### Step 6 — Determine outcome

**PASS conditions (ALL must hold):**
- Every acceptance criterion is verified by independent evidence.
- The diff is scoped to the stated requirements.
- No protected files are modified without justification.
- All tests pass (or no regressions vs baseline).
- The SHA is current and matches what was pinned.

**CHANGES REQUESTED if ANY of:**
- An acceptance criterion is not met or cannot be verified.
- The change introduces a regression.
- The diff contains scope-creep or unauthorized changes.
- Tests fail or are missing for new functionality.
- The SHA has changed since review started.

### Step 7 — Take action

**If PASS:**

1. Call `kanban_comment` with the structured review result and a
   merge plan.
2. Merge the PR (via `gh pr merge --squash --delete-branch` or
   equivalent).
3. Complete the task via the `kanban_complete` tool with outcome
   `passed`.

**If CHANGES REQUESTED:**

1. Call `kanban_comment` with the structured review result, listing
   every issue clearly.
2. Do NOT call `kanban_complete` — the implementer needs the task to
   stay in `running` for rework. Instead, signal rejection by
   recording the review outcome as `changes_requested`.
3. The Kanban dispatcher handles the status transition. Do not attempt
   to manipulate task status directly.

## Pitfalls

- **Accepting self-reported claims.** Always verify — run tests, read
  the diff, check the evidence. Do not take "I tested it" as proof.
- **Approving stale code.** If the branch SHA changed mid-review,
  restart. Never approve an unverified SHA.
- **Scope creep.** A PR that refactors unrelated code or adds features
  beyond the stated task must be flagged even if the code is correct.
- **Not recording detailed feedback.** A "changes requested" verdict
  without specific, actionable issues is not useful. Every issue must
  include the file path and line number.
- **Marking a task as `done` from `changes_requested`.** Pass is the
  only path to completion. If changes are needed, the task goes back
  to the implementer.
- **Conflicting with Kanban lifecycle hooks.** Do not call
  `complete_task` directly — use `kanban_complete` which respects
  lifecycle hooks. The Kanban lifecycle (claim → work → complete or
  block) is managed by the dispatcher and the task worker; the review
  agent only reads task state and posts results.

## Verification

- [ ] All acceptance criteria are verified against the actual diff/code
- [ ] Review SHA is pinned and current
- [ ] Diff scope matches task requirements
- [ ] Protected files checked
- [ ] Test results reviewed (run if possible)
- [ ] Review result posted as Kanban comment
- [ ] PASS → PR merged + task completed
- [ ] CHANGES REQUESTED → detailed feedback posted, task NOT completed
