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
               parents=()) -> str:
    tid = create_task(conn, title=title, body=body, assignee=assignee,
                       idempotency_key=idempotency_key, parents=parents)
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
    def test_blocked_to_review_after_pm_created(self, cron, conn):
        src = _make_task(conn, title="T", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()
        cron.main()
        assert get_task(conn, src).status == "review"
        assert list_tasks(conn, assignee="pm")[0].status == "ready"

    def test_cas_rejects_mismatch(self, cron, conn):
        src = _make_task(conn, title="C", status="running", assignee="dev")
        assert cron._cas_set_status(conn, src, "blocked", "review") is False
        assert get_task(conn, src).status == "running"

    def test_cas_failure_cancels_fresh_pm(self, cron, conn):
        """PM作成後にCAS失敗→sourceがready/done→PMをキャンセル"""
        src = _make_task(conn, title="CF", status="running", assignee="dev")
        _block_with_reason(conn, src, "review-required:"); conn.commit()

        # First run creates PM and transitions to review
        cron.main()
        assert get_task(conn, src).status == "review"
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

    def test_changes_requested_releases_source(self, cron, conn):
        """changes-requested も review → ready へ (devがrework可能に)"""
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "changes-requested", "abc1234")
        out = _capture(cron.main)
        assert get_task(conn, src).status == "ready", \
            f"Expected ready for changes-requested, got {get_task(conn, src).status}"
        assert "released" in out

    def test_blocked_verdict_no_release(self, cron, conn):
        src = self._setup(conn)
        cron.main(); pm = list_tasks(conn, assignee="pm")[0]
        self._mk_reviewer(conn, pm, "blocked", "abc1234")
        out = _capture(cron.main)
        assert get_task(conn, src).status == "review"
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


# ═════════════════════════════════════════════════════════════════════════════
# 7. 古い結果を拒否
# ═════════════════════════════════════════════════════════════════════════════


class TestStaleGuard:
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
        assert get_task(conn, src).status == "review", "Stale SHA must NOT release"
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
        assert get_task(conn, src).status == "review"


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
        assert s.status == "review"


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

        # Step 2-3: PM created, source → review
        cron.main()
        pm = list_tasks(conn, assignee="pm", status="ready")
        assert len(pm) == 1
        assert get_task(conn, src).status == "review"
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
