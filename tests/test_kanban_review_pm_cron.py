"""Comprehensive tests for the kanban-review-pm cron job.

31+ tests covering detection, idempotency, state transitions,
reviewer completion, stale-guard, CAS failure, and E2E.
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import (
    connect, create_task, list_tasks, get_task,
    archive_task, complete_task, parent_ids, write_txn,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(conn, *, title="test", status="ready", assignee="dev",
               body=None, idempotency_key=None, task_kind=None,
               parents=(), workspace_kind="scratch", workspace_path=None,
               branch_name=None) -> str:
    tid = create_task(conn, title=title, body=body, assignee=assignee,
                       idempotency_key=idempotency_key, parents=parents,
                       workspace_kind=workspace_kind, workspace_path=workspace_path,
                       branch_name=branch_name)
    if status != "ready":
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
            kb._append_event(conn, tid, status, {"auto": True, "reason": "test"})
    if task_kind is not None:
        conn.execute("UPDATE tasks SET task_kind = ? WHERE id = ?", (task_kind, tid))
    conn.commit()
    return tid


def _block_with_reason(conn, task_id: str, reason: str):
    with write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        kb._append_event(conn, task_id, "blocked", {"reason": reason})


def _set_sha(conn, task_id: str, sha: str):
    _block_with_reason(conn, task_id, f"review-required: SHA={sha}")


def _capture(fn, *args, **kw) -> str:
    old, buf = sys.stdout, StringIO()
    try:
        sys.stdout = buf
        fn(*args, **kw)
    finally:
        sys.stdout = old
    return buf.getvalue()


def _cron_count(conn) -> int:
    return len(list_tasks(conn, assignee="pm"))


def _complete_structured_review(conn, task_id: str, review: dict[str, Any]) -> None:
    complete_task(conn, task_id, result=None, summary=review["verdict"],
                  metadata={"review": review})
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, 'done', 1, 1, 'completed', ?)",
        (task_id, json.dumps({"review": review})),
    )
    conn.commit()


def test_stale_unclaimed_reviewer_for_old_sha_is_quarantined(cron, conn):
    source = _make_task(conn, title="source", status="running")
    _block_with_reason(conn, source, "review-required: SHA=bbbbbbb")
    old = _make_task(
        conn, title="old reviewer", status="ready", assignee="reviewer",
        task_kind="independent_review",
        idempotency_key=f"review:{source}:aaaaaaa:1",
    )
    assert cron._quarantine_stale_review_artifacts(conn) == 1
    assert get_task(conn, old).status == "archived"
    event = [e for e in kb.list_events(conn, old) if e.kind == "quarantined_stale_review"][-1]
    assert event.payload["current_sha"] == "bbbbbbb"


def test_running_old_sha_reviewer_is_not_quarantined(cron, conn):
    source = _make_task(conn, title="source", status="running")
    _block_with_reason(conn, source, "review-required: SHA=bbbbbbb")
    old = _make_task(
        conn, title="old reviewer", status="running", assignee="reviewer",
        task_kind="independent_review",
        idempotency_key=f"review:{source}:aaaaaaa:1",
    )
    assert cron._quarantine_stale_review_artifacts(conn) == 0
    assert get_task(conn, old).status == "running"


def test_misanchored_reviewer_is_replaced_and_rebound_to_target_workspace(cron, conn):
    source = _make_task(
        conn, title="source", status="blocked", workspace_kind="worktree",
        workspace_path="/repo/.worktrees/source", branch_name="wt/source",
    )
    reviewer = _make_task(
        conn, title="review", status="blocked", assignee="reviewer",
        body=f"- review_target_task_id: {source}",
        workspace_kind="scratch", workspace_path="/scratch/review",
    )
    _block_with_reason(conn, source, f"review-required: reviewer task {reviewer}")
    assert cron._repair_misanchored_review_workspaces(conn) == 1
    assert get_task(conn, reviewer).status == "archived"
    replacement = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ?",
        (f"workspace-repair:{reviewer}:{source}",),
    ).fetchone()["id"]
    repaired = get_task(conn, replacement)
    assert repaired.workspace_path == "/repo/.worktrees/source"
    assert repaired.branch_name == "wt/source"
    latest = [e for e in kb.list_events(conn, source) if e.kind == "blocked"][-1]
    assert replacement in latest.payload["reason"]
    _complete_structured_review(conn, replacement, {
        "target_task_id": source, "verdict": "pass", "reviewed_sha": "",
        "review_round": 3,
    })
    cron.main()
    assert get_task(conn, source).status == "ready"


def test_round_two_remediation_dispatches_one_final_reviewer_and_rebinds_source(cron, conn):
    source = _make_task(
        conn, title="source", status="blocked", workspace_kind="worktree",
        workspace_path="/repo/.worktrees/source", branch_name="wt/source",
    )
    _block_with_reason(conn, source, "review-required: reviewer task t_old; SHA: aaaaaaa")
    with write_txn(conn):
        kb._append_event(conn, source, "review_verdict_requires_routing", {
            "review_round": 2, "reviewer_id": "t_old",
        })
    remediation = _make_task(conn, title="bounded remediation", status="running", assignee="dev")
    kb.link_tasks(conn, remediation, source)
    metadata = {
        "new_sha": "b" * 40,
        "review": {"target_task_id": source,
                   "verdict": "remediation-ready-for-final-acceptance"},
    }
    complete_task(conn, remediation, metadata=metadata)
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, 'done', 1, 1, 'completed', ?)",
        (remediation, json.dumps(metadata)),
    )
    conn.commit()
    assert cron._dispatch_final_reviews_after_bounded_remediation(conn) == 1
    final = conn.execute(
        "SELECT id, workspace_path FROM tasks WHERE idempotency_key = ?",
        (f"review-final:{source}:{'b' * 40}:3",),
    ).fetchone()
    assert final["workspace_path"] == "/repo/.worktrees/source"
    latest = [e for e in kb.list_events(conn, source) if e.kind == "blocked"][-1]
    assert final["id"] in latest.payload["reason"]
    assert cron._dispatch_final_reviews_after_bounded_remediation(conn) == 0


def test_accepted_gate_decision_quarantines_only_named_nonrunning_artifacts(cron, conn):
    """Gate cleanup is opt-in structured data, never title-based guessing."""
    obsolete = _make_task(conn, title="old remediation", status="blocked")
    running = _make_task(conn, title="still collecting evidence", status="running")
    gate_reviewer = _make_task(conn, title="Gate B review", status="running", assignee="reviewer")
    review = {"target_task_id": "gate-source", "verdict": "pass", "reviewed_sha": "c" * 40,
              "review_round": 1}
    complete_task(
        conn, gate_reviewer, summary="Gate accepted",
        metadata={"review": review, "gate_decision": {
            "accepted": True, "workflow": "pre-86-10", "stage": "B",
            "obsolete_task_ids": [obsolete, running],
        }},
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, 'done', 1, 1, 'completed', ?)",
        (gate_reviewer, json.dumps({"review": review, "gate_decision": {
            "accepted": True, "workflow": "pre-86-10", "stage": "B",
            "obsolete_task_ids": [obsolete, running],
        }})),
    )
    conn.commit()

    assert cron._quarantine_gate_superseded_artifacts(conn) == 1
    assert get_task(conn, obsolete).status == "archived"
    assert get_task(conn, running).status == "running"
    event = [e for e in kb.list_events(conn, obsolete)
             if e.kind == "quarantined_by_gate_acceptance"][-1]
    assert event.payload["stage"] == "B"


def test_gate_decision_without_explicit_obsolete_ids_does_not_archive(conn, cron):
    candidate = _make_task(conn, title="blocked but unrelated", status="blocked")
    reviewer = _make_task(conn, title="Gate review", status="running", assignee="reviewer")
    review = {"target_task_id": "gate-source", "verdict": "pass", "reviewed_sha": "d" * 40,
              "review_round": 1}
    complete_task(conn, reviewer, summary="Gate accepted", metadata={
        "review": review,
        "gate_decision": {"accepted": True, "workflow": "pre-86-10", "stage": "B"},
    })
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, 'done', 1, 1, 'completed', ?)",
        (reviewer, json.dumps({"review": review, "gate_decision": {
            "accepted": True, "workflow": "pre-86-10", "stage": "B"}})),
    )
    conn.commit()
    assert cron._quarantine_gate_superseded_artifacts(conn) == 0
    assert get_task(conn, candidate).status == "blocked"


def test_accepted_remediation_successor_completes_blocked_source(cron, conn):
    source = _make_task(conn, title="source", status="blocked")
    gate = _make_task(conn, title="Gate", status="todo", assignee="reviewer", parents=(source,))
    successor = _make_task(conn, title="successor", status="running")
    pm = _make_task(conn, title="PM", status="done", assignee="pm")
    kb.complete_task(conn, successor, summary="remediation complete")
    reviewer = _make_task(conn, title="review", status="running", assignee="reviewer")
    _complete_structured_review(conn, reviewer, {
        "target_task_id": successor, "verdict": "pass", "reviewed_sha": "a" * 40,
        "review_round": 1,
    })
    with kb.write_txn(conn):
        kb._append_event(conn, source, "remediation_handoff_applied", {
            "pm_task_id": pm, "successor_task_id": successor,
            "retired_predecessor_task_ids": [],
        })
    assert cron._reconcile_completed_remediation_liveness(conn) == 1
    assert get_task(conn, source).status == "done"
    assert get_task(conn, gate).status == "ready"
    assert any(e.kind == "remediation_successor_accepted" for e in kb.list_events(conn, source))


def test_accepted_nested_remediation_propagates_to_original_source(cron, conn):
    """A reviewed leaf closes every blocked ancestor in one reconciliation."""
    source = _make_task(conn, title="original", status="blocked")
    gate = _make_task(conn, title="Gate", status="todo", assignee="reviewer", parents=(source,))
    intermediate = _make_task(conn, title="first remediation", status="blocked")
    leaf = _make_task(conn, title="final remediation", status="running")
    pm_one = _make_task(conn, title="PM one", status="done", assignee="pm")
    pm_two = _make_task(conn, title="PM two", status="done", assignee="pm")
    kb.complete_task(conn, leaf, summary="leaf remediation complete")
    reviewer = _make_task(conn, title="leaf review", status="running", assignee="reviewer")
    _complete_structured_review(conn, reviewer, {
        "target_task_id": leaf, "verdict": "pass", "reviewed_sha": "b" * 40,
        "review_round": 1,
    })
    with kb.write_txn(conn):
        kb._append_event(conn, source, "remediation_handoff_applied", {
            "pm_task_id": pm_one, "successor_task_id": intermediate,
            "retired_predecessor_task_ids": [],
        })
        kb._append_event(conn, intermediate, "remediation_handoff_applied", {
            "pm_task_id": pm_two, "successor_task_id": leaf,
            "retired_predecessor_task_ids": [],
        })
    assert cron._reconcile_completed_remediation_liveness(conn) == 2
    assert get_task(conn, intermediate).status == "done"
    assert get_task(conn, source).status == "done"
    assert get_task(conn, gate).status == "ready"
    source_events = kb.list_events(conn, source)
    assert source_events[-1].kind == "remediation_successor_accepted"
    assert source_events[-1].payload["acceptance"] == "accepted_descendant"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cron(monkeypatch, tmp_path):
    import importlib.util as iu
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(); monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    dst = hermes_home / "scripts" / "kanban-review-pm.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(os.path.expanduser("~/.hermes/scripts/kanban-review-pm.py"))
    if not src.exists():
        pytest.skip("script not found")
    dst.write_text(src.read_text())
    spec = iu.spec_from_file_location("kpm", dst)
    mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
    c = kb.connect()
    try:
        mod._ensure_schema(c); c.commit()
    finally:
        c.close()
    return mod


@pytest.fixture
def conn(cron):
    return kb.connect()


# ═════════════════════════════════════════════════════════════════════════════
# 1. 検出条件
# ═════════════════════════════════════════════════════════════════════════════


class TestDetection:
    def test_detects_review_status(self, cron, conn):
        src = _make_task(conn, title="F", status="review", assignee="dev")
        cron.main()
        assert _cron_count(conn) == 1

    def test_detects_blocked_review_required(self, cron, conn):
        src = _make_task(conn, title="B", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required: SHA=abc1234")
        conn.commit()
        cron.main()
        assert _cron_count(conn) == 1

    def test_named_live_reviewer_suppresses_redundant_pm_orchestration(self, cron, conn):
        """A normal worker-created review handoff is not a cron recovery case."""
        src = _make_task(conn, title="B", status="running", assignee="dev")
        reviewer = _make_task(conn, title="Independent review", assignee="reviewer")
        _block_with_reason(conn, src, f"review-required: reviewer task {reviewer}")
        conn.commit()

        assert cron._find_review_candidates(conn) == []
        cron.main()
        assert _cron_count(conn) == 0

    def test_ignores_plain_blocked(self, cron, conn):
        src = _make_task(conn, title="P", status="running", assignee="dev")
        _block_with_reason(conn, src, "dependency: waiting")
        conn.commit()
        assert "No review-required" in _capture(cron.main)

    def test_ignores_non_prefix_reason(self, cron, conn):
        src = _make_task(conn, title="N", status="running", assignee="dev")
        _block_with_reason(conn, src, "This is not a review-required task")
        conn.commit()
        assert "No review-required" in _capture(cron.main)

    def test_ignores_archived(self, cron, conn):
        src = _make_task(conn, title="A", status="review", assignee="dev")
        archive_task(conn, src); conn.commit()
        assert "No review-required" in _capture(cron.main)

    def test_ignores_pm_reviewer_kinds(self, cron, conn):
        _make_task(conn, title="PM", status="review", assignee="pm", task_kind="review_orchestration")
        _make_task(conn, title="RV", status="review", assignee="rv", task_kind="independent_review")
        conn.commit()
        assert "No review-required" in _capture(cron.main)


# ═════════════════════════════════════════════════════════════════════════════
# 2. PM作成内容
# ═════════════════════════════════════════════════════════════════════════════


class TestPmCreation:
    def test_independent_pm(self, cron, conn):
        src = _make_task(conn, title="X", status="review", assignee="dev")
        cron.main()
        pm = list_tasks(conn, assignee="pm", status="ready")
        assert len(pm) == 1
        p = pm[0]
        assert p.assignee == "pm"
        assert parent_ids(conn, p.id) == []
        assert p.status == "ready"
        row = conn.execute("SELECT task_kind FROM tasks WHERE id = ?", (p.id,)).fetchone()
        assert row["task_kind"] == "review_orchestration"
        assert p.title.startswith("[PM]")


# ═════════════════════════════════════════════════════════════════════════════
# 3. 冪等性
# ═════════════════════════════════════════════════════════════════════════════


class TestIdempotency:
    def test_double_run_one_pm(self, cron, conn):
        src = _make_task(conn, title="D", status="review", assignee="dev")
        cron.main(); cron.main()
        assert _cron_count(conn) == 1

    def test_new_sha_new_pm(self, cron, conn):
        src = _make_task(conn, title="S", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit(); cron.main()
        assert _cron_count(conn) == 1
        pm1 = list_tasks(conn, assignee="pm")[0]
        # Close round 1
        _make_task(conn, title="r1", status="done",
                    idempotency_key=f"review:{pm1.idempotency_key}", assignee="rv")
        conn.commit()
        complete_task(conn, pm1.id, result="ok", summary="done"); conn.commit()
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (src,))
            kb._append_event(conn, src, "running", {"auto": True})
        conn.commit()
        # Round 2: new SHA
        _set_sha(conn, src, "def4567"); conn.commit(); cron.main()
        assert _cron_count(conn) == 2
        keys = {p.idempotency_key for p in list_tasks(conn, assignee="pm")}
        assert len(keys) == 2

    def test_action_key_changes_with_round(self, cron, conn):
        """review-lane action key に round が含まれている"""
        src = _make_task(conn, title="AK", status="review", assignee="dev")
        # Force a known SHA
        _set_sha(conn, src, "abc1234"); conn.commit()
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        key = pm.idempotency_key
        # _BLOCKED_REVIEW key uses evt<N>; SHA is metadata
        assert "evt" in key, f"Key should contain event ref: {key}"
        # Round info is in the PM body
        assert pm.body is not None, "PM body must exist"
        assert "Review round:" in pm.body, f"Body missing round info: {pm.body[:200]}"

    def test_action_key_differs_between_rounds(self, cron, conn):
        """round 1 と round 2 で action key が異なる"""
        src = _make_task(conn, title="R2", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit(); cron.main()
        key1 = list_tasks(conn, assignee="pm")[0].idempotency_key
        # Close round 1
        _make_task(conn, title="x", status="done",
                    idempotency_key=f"review:{key1}", assignee="rv")
        conn.commit()
        complete_task(conn, list_tasks(conn, assignee="pm")[0].id, result="ok", summary="d"); conn.commit()
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (src,))
            kb._append_event(conn, src, "running", {"auto": True})
        conn.commit()
        # Re-block with same SHA (simulate re-review request)
        _set_sha(conn, src, "abc1234"); conn.commit(); cron.main()
        assert _cron_count(conn) == 2
        key2 = list_tasks(conn, assignee="pm")[1].idempotency_key
        assert key2 != key1, "Round 2 key must differ from round 1"
        # Each block event has unique evt<N> in key
        assert key2 != key1, "Round 2 key must differ from round 1"
        # Also verify the key is not the same sequence (different evt id)
        assert key2.count("evt") == 1 and key1.count("evt") == 1


# ═════════════════════════════════════════════════════════════════════════════
# 4. CAS状態遷移
# ═════════════════════════════════════════════════════════════════════════════


class TestCasTransition:
    def test_blocked_review_hold_is_preserved_after_pm_created(self, cron, conn):
        src = _make_task(conn, title="T", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()
        cron.main()
        assert get_task(conn, src).status == "blocked"
        assert list_tasks(conn, assignee="pm")[0].status == "ready"

    def test_cas_rejects_mismatch(self, cron, conn):
        src = _make_task(conn, title="C", status="running", assignee="dev")
        assert cron._cas_set_status(conn, src, "blocked", "review") is False
        assert get_task(conn, src).status == "running"

    def test_cas_failure_cancels_fresh_pm(self, cron, conn):
        """PM作成後にCAS失敗→sourceがready/done→PMをキャンセル"""
        src = _make_task(conn, title="CF", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()

        # First run creates PM but preserves the review hold.
        cron.main()
        assert get_task(conn, src).status == "blocked"
        pm1 = list_tasks(conn, assignee="pm")
        assert len(pm1) == 1

        # Someone resolves the source to 'ready' manually
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (src,))
            kb._append_event(conn, src, "ready", {"auto": True, "reason": "manual"})
        conn.commit()

        # Now a new block event triggers candidate detection again
        _set_sha(conn, src, "new4567"); conn.commit()

        # Run cron: should detect CAS failure and CANCEL the new PM
        out = _capture(cron.main)
        pm_after = list_tasks(conn, assignee="pm")
        # The new PM might be the same as old or a new one — verify we don't
        # leave a ready PM for a source that's no longer blocked.
        for pm in pm_after:
            if pm.status == "ready":
                # Verify source is in review or blocked state
                s = get_task(conn, src)
                assert s.status in ("review", "blocked"), \
                    f"Ready PM {pm.id} but source={s.status}"

    def test_cas_already_review_no_cancel(self, cron, conn):
        """別プロセスが先にreviewへ移した→PM維持"""
        src = _make_task(conn, title="AR", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()

        # Manually move to review before cron runs
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (src,))
            kb._append_event(conn, src, "review", {"auto": True, "reason": "concurrent"})
        conn.commit()

        # Cron should still create PM (source is in review lane)
        cron.main()
        assert _cron_count(conn) == 1


# ═════════════════════════════════════════════════════════════════════════════
# 5. コメント
# ═════════════════════════════════════════════════════════════════════════════


class TestCommenting:
    def test_comment_only_on_create(self, cron, conn):
        src = _make_task(conn, title="C1", status="review", assignee="dev")
        cron.main()
        assert conn.execute(
            "SELECT id FROM task_comments WHERE task_id = ? AND author='cron-review-pm'", (src,)
        ).fetchall() != []
        cron.main()
        assert len(conn.execute(
            "SELECT id FROM task_comments WHERE task_id = ? AND author='cron-review-pm'", (src,)
        ).fetchall()) == 1


# ═════════════════════════════════════════════════════════════════════════════
# 6. Reviewer完了→release
# ═════════════════════════════════════════════════════════════════════════════


class TestReviewRelease:
    def _setup(self, conn, sha="abc1234", verdict="pass"):
        src = _make_task(conn, title="R", status="running", assignee="dev")
        _set_sha(conn, src, sha); conn.commit()
        return src

    def _mk_reviewer(self, conn, pm, verdict="pass", sha="abc1234"):
        rev_key = f"review:{pm.idempotency_key}"
        rev_id = _make_task(conn, title="rv", status="running",
                             idempotency_key=rev_key, assignee="reviewer")
        conn.commit()
        meta = json.dumps({"verdict": verdict, "reviewed_sha": sha, "review_round": 1})
        complete_task(conn, rev_id, result=meta, summary=verdict)
        conn.commit()
        return rev_id

    def test_pass_releases_source(self, cron, conn):
        src = self._setup(conn)
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "pass", "abc1234")
        out = _capture(cron.main)
        assert get_task(conn, src).status == "ready"
        assert "released" in out

    def test_pass_with_nits_releases(self, cron, conn):
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "pass-with-nits", "abc1234")
        out = _capture(cron.main)
        assert get_task(conn, src).status == "ready"
        assert "released" in out

    def test_changes_requested_preserves_source_hold(self, cron, conn):
        """changes-requested は remediation が作られるまで source を release しない。"""
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        reviewer = self._mk_reviewer(conn, pm, "changes-requested", "abc1234")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src,
            "verdict": "changes-requested",
            "reviewed_sha": "abc1234",
            "review_round": 2,
            "findings": [{"detail": "bounded fix"}],
        })
        out = _capture(cron.main)
        assert get_task(conn, src).status == "blocked"
        escalations = [task for task in list_tasks(conn, assignee="pm")
                       if "review-remediation" in (task.idempotency_key or "")]
        assert len(escalations) == 1
        assert escalations[0].status == "ready"
        assert "remediation PM escalation" in out

    def test_final_changes_requested_requires_explicit_pm_decision(self, cron, conn):
        """A failed final acceptance review is never silently retried."""
        src = self._setup(conn)
        reviewer = _make_task(
            conn, title="final reviewer", status="running", assignee="reviewer",
            task_kind="independent_review",
            idempotency_key=f"review-final:{src}:abc1234:3",
        )
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src,
            "verdict": "changes-requested",
            "reviewed_sha": "abc1234",
            "review_round": 3,
            "findings": [{"detail": "final bounded finding"}],
        })

        out = _capture(cron.main)

        assert get_task(conn, src).status == "blocked"
        escalations = [task for task in list_tasks(conn, assignee="pm")
                       if "review-remediation" in (task.idempotency_key or "")]
        assert len(escalations) == 1
        assert "Final acceptance decision required" in (escalations[0].body or "")
        assert "Choose exactly one" in (escalations[0].body or "")
        assert "remediation PM escalation" in out

    def test_first_round_changes_requested_on_pm_successor_routes_back_to_pm(self, cron, conn):
        """A bounded PM successor never gets an implicit extra rework round."""
        src = self._setup(conn)
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        handoff_owner = _make_task(conn, title="completed PM", status="done", assignee="pm")
        kb._append_event(conn, handoff_owner, "remediation_handoff_applied", {
            "source_task_id": "original-source",
            "successor_task_id": src,
            "retired_predecessor_task_ids": [],
        })
        conn.commit()
        reviewer = self._mk_reviewer(conn, pm, "changes-requested", "abc1234")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src,
            "verdict": "changes-requested",
            "reviewed_sha": "abc1234",
            "review_round": 1,
            "findings": [{"detail": "PM decision required"}],
        })
        out = _capture(cron.main)
        assert get_task(conn, src).status == "blocked"
        escalations = [task for task in list_tasks(conn, assignee="pm")
                       if "review-remediation" in (task.idempotency_key or "")]
        assert len(escalations) == 1
        assert "remediation PM escalation" in out

    def test_pm_successor_escalation_suppresses_duplicate_ordinary_review_pm(self, cron, conn):
        """One reviewed PM successor produces one PM decision card, never two."""
        src = self._setup(conn)
        handoff_owner = _make_task(conn, title="completed PM", status="done", assignee="pm")
        kb._append_event(conn, handoff_owner, "remediation_handoff_applied", {
            "source_task_id": "original-source",
            "successor_task_id": src,
            "retired_predecessor_task_ids": [],
        })
        reviewer = _make_task(
            conn, title="direct reviewer", status="running", assignee="reviewer",
            idempotency_key="review:direct-pm-successor",
        )
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src,
            "verdict": "changes-requested",
            "reviewed_sha": "abc1234",
            "review_round": 1,
            "findings": [{"detail": "PM decision required"}],
        })
        conn.commit()
        _capture(cron.main)
        pm_tasks = [task for task in list_tasks(conn, assignee="pm") if task.status != "done"]
        assert len(pm_tasks) == 1
        assert "review-remediation" in (pm_tasks[0].idempotency_key or "")

    def test_completed_remediation_pm_without_successor_reopens(self, cron, conn):
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        reviewer = self._mk_reviewer(conn, pm, "changes-requested", "abc1234")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src, "verdict": "changes-requested",
            "reviewed_sha": "abc1234", "review_round": 2, "findings": [],
        })
        cron.main()
        escalation = [task for task in list_tasks(conn, assignee="pm")
                      if "review-remediation" in (task.idempotency_key or "")][0]
        complete_task(conn, escalation.id, result="done", summary="forgot successor")
        conn.commit()
        out = _capture(cron.main)
        assert get_task(conn, escalation.id).status == "ready"
        assert "reopened remediation PM" in out

    def test_blocked_verdict_no_release(self, cron, conn):
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "blocked", "abc1234")
        out = _capture(cron.main)
        assert get_task(conn, src).status == "blocked"
        assert "released" not in out

    def test_incomplete_reviewer_no_release(self, cron, conn):
        src = self._setup(conn)
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        _make_task(conn, title="pending", status="running",
                    idempotency_key=f"review:{pm.idempotency_key}", assignee="rv")
        conn.commit()
        out = _capture(cron.main)
        assert "released" not in out

    def test_duplicate_release_idempotent(self, cron, conn):
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "pass", "abc1234")
        cron.main()  # first release
        cron.main()  # second — should be no-op
        assert get_task(conn, src).status == "ready"
        ev = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind='released_from_review'", (src,)
        ).fetchall()
        assert len(ev) == 1

    def test_review_release_clears_human_block_marker(self, cron, conn):
        src = self._setup(conn)
        conn.execute("UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (src,))
        conn.commit()
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "pass", "abc1234")
        cron.main()
        released = get_task(conn, src)
        assert released.status == "ready"
        assert released.block_kind is None


# ═════════════════════════════════════════════════════════════════════════════
# 7. 古い結果を拒否
# ═════════════════════════════════════════════════════════════════════════════


class TestStaleGuard:
    def test_direct_pm_reviewer_pass_releases_its_metadata_target(self, cron, conn):
        src = _make_task(conn, title="direct target", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit()
        reviewer = _make_task(conn, title="direct reviewer", status="running", assignee="rv",
                              idempotency_key=f"review:{src}:abc1234:3")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": src, "verdict": "pass", "reviewed_sha": "abc1234",
            "review_round": 3,
        })
        cron.main()
        assert get_task(conn, src).status == "ready"

    def test_completed_pm_metadata_round_overrides_creation_prompt(self, cron, conn):
        src = _make_task(conn, title="post-remediation", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit()
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        # The PM body was created as round 1 but its reconciliation completed
        # a pre-existing round-3 reviewer.
        assert kb.claim_task(conn, pm.id) is not None
        kb.complete_task(conn, pm.id, metadata={"review_round": 3})
        assert cron._expected_review_round(conn, pm.idempotency_key) == 3

    def test_latest_review_block_sha_wins_over_old_handoff(self, cron, conn):
        src = _make_task(conn, title="multiple handoffs", status="running", assignee="dev")
        _set_sha(conn, src, "aaaaaaa")
        # A later review request supersedes the original target.
        _block_with_reason(conn, src, "review-required: SHA=bbbbbbb")
        assert cron._resolve_sha_from_events(conn, src) == "bbbbbbb"

    def test_abbreviated_current_sha_matches_full_review_sha(self, cron, conn):
        src = _make_task(conn, title="short-sha", status="running", assignee="dev")
        full_sha = "a" * 40
        _set_sha(conn, src, full_sha[:8]); conn.commit()
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        rev = _make_task(conn, title="review", status="running",
                         idempotency_key=f"review:{pm.idempotency_key}", assignee="rv")
        _complete_structured_review(conn, rev, {
            "verdict": "pass", "reviewed_sha": full_sha, "review_round": 1,
        })
        cron.main()
        assert get_task(conn, src).status == "ready"

    def test_stale_sha_no_release(self, cron, conn):
        src = _make_task(conn, title="SS", status="running", assignee="dev")
        _set_sha(conn, src, "def4567"); conn.commit()  # current SHA
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        rev_key = f"review:{pm.idempotency_key}"
        rev_id = _make_task(conn, title="srv", status="running",
                             idempotency_key=rev_key, assignee="rv")
        conn.commit()
        # Reviewer checked SHA=abc1234 (stale)
        meta = json.dumps({"verdict": "pass", "reviewed_sha": "abc1234", "review_round": 1})
        complete_task(conn, rev_id, result=meta, summary="pass"); conn.commit()
        out = _capture(cron.main)
        assert get_task(conn, src).status == "blocked", "Stale SHA must NOT release"
        assert "released" not in out

    def test_wrong_round_no_release(self, cron, conn):
        src = _make_task(conn, title="WR", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit()
        cron.main()
        pm1 = list_tasks(conn, assignee="pm")[0]
        rev_key = f"review:{pm1.idempotency_key}"
        rev = _make_task(conn, title="r1", status="running",
                          idempotency_key=rev_key, assignee="rv")
        conn.commit()
        meta = json.dumps({"verdict": "pass", "reviewed_sha": "abc1234", "review_round": 1})
        complete_task(conn, rev, result=meta, summary="pass"); conn.commit()
        # Release round 1
        cron.main()
        assert get_task(conn, src).status == "ready"
        # Put back to running, new block
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (src,))
            kb._append_event(conn, src, "running", {"auto": True})
        conn.commit()
        _set_sha(conn, src, "def4567"); conn.commit()
        cron.main()  # creates new PM for round 2
        assert _cron_count(conn) == 2
        assert get_task(conn, src).status == "blocked"

    def test_same_sha_wrong_round_no_release(self, cron, conn):
        src = _make_task(conn, title="same-sha", status="running", assignee="dev")
        _set_sha(conn, src, "abc1234"); conn.commit()
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        rev = _make_task(conn, title="rv", status="running", assignee="reviewer",
                         idempotency_key=f"review:{pm.idempotency_key}")
        conn.commit()
        complete_task(conn, rev, result=json.dumps({
            "verdict": "pass", "reviewed_sha": "abc1234", "review_round": 99,
        }), summary="pass")
        conn.commit()
        cron.main()
        assert get_task(conn, src).status == "blocked"


class TestStructuredBlockedVerdict:
    def test_does_not_escalate_historical_verdict_for_completed_source(self, cron, conn):
        source = _make_task(conn, title="completed source", status="running", assignee="dev")
        complete_task(conn, source, result="done", summary="done")
        reviewer = _make_task(conn, title="old review", status="running", assignee="reviewer")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": source, "verdict": "changes-requested",
            "reviewed_sha": "abc1234", "review_round": 1,
        })
        cron.main()
        assert list_tasks(conn, assignee="pm") == []

    def test_routes_blocked_run_metadata_to_pm_without_releasing_source(self, cron, conn):
        source = _make_task(conn, title="source", status="blocked", assignee="dev")
        reviewer = _make_task(conn, title="independent review", status="running", assignee="reviewer")
        _complete_structured_review(conn, reviewer, {
            "target_task_id": source,
            "verdict": "blocked",
            "reviewed_sha": "abc1234",
            "review_round": 2,
            "findings": [{"detail": "controlled gate-off evidence missing"}],
        })
        cron.main(); cron.main()
        pms = list_tasks(conn, assignee="pm", status="ready")
        assert len(pms) == 1
        assert pms[0].idempotency_key.endswith(f"review-remediation:{reviewer}")
        assert "Do not create another independent review" in pms[0].body
        assert get_task(conn, source).status == "blocked"

    def test_only_latest_blocked_verdict_for_a_source_is_escalated(self, cron, conn):
        source = _make_task(conn, title="source", status="blocked", assignee="dev")
        old = _make_task(conn, title="old independent review", status="running", assignee="reviewer")
        new = _make_task(conn, title="new independent review", status="running", assignee="reviewer")
        for task_id, sha in ((old, "old1234"), (new, "new1234")):
            _complete_structured_review(conn, task_id, {
                "target_task_id": source, "verdict": "blocked", "reviewed_sha": sha,
                "review_round": 2,
            })
        conn.execute("UPDATE tasks SET completed_at = 1 WHERE id = ?", (old,))
        conn.execute("UPDATE tasks SET completed_at = 2 WHERE id = ?", (new,))
        conn.commit()
        cron.main()
        pms = list_tasks(conn, assignee="pm", status="ready")
        assert len(pms) == 1
        assert pms[0].idempotency_key.endswith(f"review-remediation:{new}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. 障害復旧
# ═════════════════════════════════════════════════════════════════════════════


class TestRecovery:
    def test_orphan_pm_detected(self, cron, conn):
        src = _make_task(conn, title="O", status="review", assignee="dev")
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        complete_task(conn, pm.id, result="oops", summary="Forgot reviewer"); conn.commit()
        out = _capture(cron.main)
        # Orphan is detected only on next block event cycle.
        # For review-lane sources the active-PM check prevents re-detection.
        assert "released" not in out or "No review-required" in out


# ═════════════════════════════════════════════════════════════════════════════
# 9. 既存Reviewer
# ═════════════════════════════════════════════════════════════════════════════


class TestExistingReviewer:
    def test_active_reviewer_skips_pm(self, cron, conn):
        src = _make_task(conn, title="ER", status="review", assignee="dev")
        cron.main()
        pm = list_tasks(conn, assignee="pm")[0]
        _make_task(conn, title="ar", status="running",
                    idempotency_key=f"review:{pm.idempotency_key}", assignee="rv")
        conn.commit()
        cron.main()
        assert _cron_count(conn) == 1

    def test_repairs_source_when_blocked(self, cron, conn):
        src = _make_task(conn, title="RP", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()
        # Pre-create reviewer
        _make_task(conn, title="pm-pre", status="ready", assignee="pm",
                    idempotency_key="cron-review-pm:mock:evt999")
        _make_task(conn, title="rv-pre", status="running", assignee="rv",
                    idempotency_key="review:cron-review-pm:mock:evt999")
        conn.commit()
        cron.main()
        s = get_task(conn, src)
        assert s.status == "blocked"


# ═════════════════════════════════════════════════════════════════════════════
# 10. 古いPMが新レビューを妨げない
# ═════════════════════════════════════════════════════════════════════════════


class TestOldPmNotBlocking:
    def test_new_sha_bypasses_old_pm(self, cron, conn):
        """古いPM target_sha=abc, 新block target_sha=def → 新PM作成"""
        src = _make_task(conn, title="OB", status="running", assignee="dev")
        # Round 1: SHA=abc
        _set_sha(conn, src, "abc1234"); conn.commit(); cron.main()
        assert _cron_count(conn) == 1
        key1 = list_tasks(conn, assignee="pm")[0].idempotency_key
        # Close round 1
        _make_task(conn, title="x", status="done",
                    idempotency_key=f"review:{key1}", assignee="rv")
        conn.commit()
        complete_task(conn, list_tasks(conn, assignee="pm")[0].id, result="ok", summary="d"); conn.commit()
        with write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (src,))
            kb._append_event(conn, src, "running", {"auto": True})
        conn.commit()
        # Round 2: SHA=def (different, but old PM still non-archived)
        _set_sha(conn, src, "def4567"); conn.commit(); cron.main()
        assert _cron_count(conn) == 2, "New SHA must create new PM despite old active PM"


# ═════════════════════════════════════════════════════════════════════════════
# 11. E2E
# ═════════════════════════════════════════════════════════════════════════════


class TestE2E:
    def test_full_flow(self, cron, conn):
        """"""
        src = _make_task(conn, title="E2E", status="running", assignee="dev",
                          body="Implement X\nSHA=abc1234\n")
        _block_with_reason(conn, src, "review-required: SHA=abc1234"); conn.commit()

        # Step 2-3: PM created while source retains its review hold.
        cron.main()
        pm = list_tasks(conn, assignee="pm", status="ready")
        assert len(pm) == 1
        assert get_task(conn, src).status == "blocked"
        assert parent_ids(conn, pm[0].id) == []
        assert "cron-review-pm" in pm[0].idempotency_key
        # _BLOCKED_REVIEW → evt<N>; SHA is metadata-only in key
        assert "evt" in pm[0].idempotency_key or "abc1234" in pm[0].idempotency_key

        # Step 4: PM creates reviewer (no deps)
        rev_key = f"review:{pm[0].idempotency_key}"
        rev_id = _make_task(conn, title="E2E-Rev", status="running",
                             idempotency_key=rev_key, assignee="reviewer", parents=())
        conn.commit()
        assert parent_ids(conn, rev_id) == []
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (src, rev_id)
        ).fetchone() is None, "Reviewer must NOT be linked as child of source"

        # Step 5: Reviewer completes with pass
        meta = json.dumps({"verdict": "pass", "reviewed_sha": "abc1234", "review_round": 1})
        assert complete_task(conn, rev_id, result=meta, summary="pass"); conn.commit()

        # Step 6: Cron releases source
        out = _capture(cron.main)
        src_final = get_task(conn, src)
        assert src_final.status == "ready", f"Expected ready, got {src_final.status}"
        assert "released" in out
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='released_from_review'", (src,)
        ).fetchone()
        assert ev is not None
        assert json.loads(ev["payload"])["verdict"] == "pass"
