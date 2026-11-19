#!/usr/bin/env python3
"""Cron-driven Frontier Review controller.

Processes pending review tasks from the WorkflowLedger, claims them atomically,
runs the autonomous correction loop with mock transport, and records results.

CLI args:
  --once          Process one pending task, run review pipeline, exit
  --recover       Find and release expired claims back to PENDING_REVIEW
  --worker-id     Worker identity (default from HERMES_WORKER_ID or "cron-1")
  --ttl           Claim TTL in seconds (default 300)
  --state-dir     Ledger state directory (default ~/.hermes/frontier_review_state)
  --dry-run       List eligible tasks without claiming

Output: JSON lines with action, task_id, worker_id, status transitions, etc.
Exit codes: 0 = success, 1 = unrecoverable failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project source is importable
sys.path.insert(0, "/home/atlas/projects/frontier_review/src")

from frontier_review.autonomous_loop import (
    AutonomousCorrectionLoop,
    Disposition,
)
from frontier_review.completion_adapter import CoderCompletionAdapter
from frontier_review.pr_adapter import (
    PRRegistry,
    GitHubClient,
    PRReviewTask,
    discover_reviewable_prs,
    build_review_context,
    push_correction_to_github,
    verify_approval_prerequisites,
)
from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
from frontier_review.workflow_state import (
    WorkflowLedger,
    default_ledger_path,
    TaskRecord,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _json_line(data: dict) -> None:
    """Print a single JSON line to stdout."""
    print(json.dumps(data, default=str))


def _create_transport_from_env(task_id: str = "", ledger_path: str = "") -> MockAnthropicTransport:
    """Build a transport from FR_REVIEW_SEQUENCE env var or default PASS.

    Expected format: JSON array like [\"changes_required\",\"pass\"] or just [\"pass\"].

    When task_id and ledger_path are provided, persists consumed scenario index
    in the ledger so scenarios advance across controller invocations.
    """
    seq_str = os.environ.get("FR_REVIEW_SEQUENCE", '[\"pass\"]')
    try:
        full_sequence = json.loads(seq_str)
    except json.JSONDecodeError:
        logger.warning(f"Invalid FR_REVIEW_SEQUENCE: {seq_str!r}, defaulting to [\\\"pass\\\"]")
        full_sequence = ["pass"]

    if not full_sequence:
        full_sequence = ["pass"]

    # Determine starting index from ledger (persists across invocations)
    start_idx = 0
    remaining_seq = list(full_sequence)
    if task_id and ledger_path:
        try:
            with open(ledger_path) as f:
                ledger = json.load(f)
            idx_key = "_mock_scenario_index"
            idx = ledger.get(task_id, {}).get(idx_key, 0)
            start_idx = int(idx)
            if start_idx >= len(full_sequence):
                # Sequence exhausted — reset to full sequence for re-reviews
                start_idx = 0
                remaining_seq = list(full_sequence)
            else:
                remaining_seq = full_sequence[start_idx:]
        except Exception as e:
            logger.debug(f"Could not read scenario index from ledger: {e}")

    scenarios = []
    for status in remaining_seq:
        status_lower = status.lower()
        if status_lower == "pass":
            scenarios.append(MockAnthropicTransport.ok_response(status="pass"))
        elif status_lower == "changes_required":
            scenarios.append(
                MockAnthropicTransport.ok_response(
                    status="changes_required",
                    findings=[{"title": "needs_fixes", "severity": "MEDIUM", "category": "quality", "description": "Issues found"}],
                )
            )
        else:
            scenarios.append(MockAnthropicTransport.ok_response(status=status))

    transport = MockAnthropicTransport()
    if len(scenarios) == 1:
        transport.configure(scenarios[0])
    else:
        transport.configure_sequence(scenarios)

    # Persist consumed scenario count to ledger for cross-invocation tracking
    if task_id and ledger_path and scenarios:
        try:
            with open(ledger_path) as f:
                ledger = json.load(f)
            ledger.setdefault(task_id, {})["_mock_scenario_index"] = start_idx + len(scenarios)
            with open(ledger_path, "w") as f:
                json.dump(ledger, f, indent=2)
        except Exception:
            pass  # non-critical — don't break controller on ledger write failure

    return transport


def _archive_correction_job(job_path: str, task_id: str, ws_path: str) -> str:
    """Archive a correction job file for audit evidence.

    Moves the job file to an archive directory rather than deleting it,
    preserving complete history of all dispatch attempts regardless of outcome.

    Returns the archived path or empty string on failure.
    """
    try:
        import shutil
        archive_dir = os.path.join(ws_path, "correction_jobs_archive", task_id)
        os.makedirs(archive_dir, exist_ok=True)
        basename = os.path.basename(job_path)
        import time as _time_mod
        archived_name = f"{_time_mod.time():.0f}_{basename}"
        dest = os.path.join(archive_dir, archived_name)
        shutil.move(job_path, dest)
        return dest
    except Exception as e:
        logger.warning(f"Failed to archive correction job {job_path}: {e}")
        return ""


def _archive_orphaned_jobs(task_id: str, ws_path: str) -> None:
    """Archive any orphaned correction job files for audit evidence.

    Called when CORRECTION_PENDING has no matching job file — all remaining
    jobs in the task's directory are stale and should be preserved as history.
    """
    try:
        import shutil
        old_dir = os.path.join(ws_path, "correction_jobs", task_id)
        if os.path.isdir(old_dir):
            _archive_correction_job(
                os.path.join(old_dir, ""),
                task_id, ws_path,
            )
            # Move entire directory to archive
            archive_dir = os.path.join(ws_path, "correction_jobs_archive", task_id)
            os.makedirs(archive_dir, exist_ok=True)
            shutil.move(old_dir, archive_dir)
    except Exception as e:
        logger.warning(f"Failed to archive orphaned jobs for {task_id}: {e}")



def _process_once(
    ledger: WorkflowLedger,
    worker_id: str,
    ttl: int,
    state_dir: str,
    dry_run: bool,
) -> int:
    """Process one pending task. Returns exit code."""
    pending = ledger.get_pending_tasks()
    if not pending:
        _json_line({
            "action": "no_pending_tasks",
            "worker_id": worker_id,
            "message": "No pending tasks",
        })
        return 0

    # Pick first by created_at (already sorted)
    task = pending[0]
    task_id = task.task_id

    if dry_run:
        _json_line({
            "action": "dry_run_eligible",
            "task_id": task_id,
            "worker_id": worker_id,
            "workspace_path": task.workspace_path,
            "status": task.status,
            "message": "Would claim and process this task",
        })
        return 0

    # Try to claim
    claimed = ledger.claim_task(task_id, worker_id=worker_id, ttl_seconds=ttl)
    if claimed is None:
        _json_line({
            "action": "claim_failed",
            "task_id": task_id,
            "worker_id": worker_id,
            "reason": "Task not claimable (already claimed or removed)",
        })
        return 0

    # Verify workspace is a git repo with at least one commit
    ws_path = claimed.workspace_path
    if not os.path.isdir(ws_path):
        _json_line({
            "action": "error",
            "task_id": task_id,
            "previous_status": "PENDING_REVIEW",
            "new_status": "IN_PROGRESS",
            "error": f"Workspace path does not exist: {ws_path}",
        })
        ledger.release_task(task_id, reason="workspace_not_found")
        return 1

    # Check for at least one commit
    git_dir = os.path.join(ws_path, ".git")
    has_commits = False
    if os.path.isdir(git_dir):
        try:
            result = os.popen(f"cd {ws_path} && git rev-parse HEAD 2>/dev/null").read().strip()
            has_commits = bool(result)
        except Exception:
            pass

    if not has_commits:
        _json_line({
            "action": "error",
            "task_id": task_id,
            "previous_status": "PENDING_REVIEW",
            "new_status": "IN_PROGRESS",
            "error": "Workspace is not a git repo or has no commits",
        })
        ledger.release_task(task_id, reason="no_git_commits")
        return 1

    # ── Correction job file validation (Phase 3B) ────────────────
    # A correction job file is only valid if the ledger confirms
    # CORRECTION_PENDING.  Stale files from prior runs must NOT
    # trigger Coder — ledger state is authoritative.
    loop_for_check = AutonomousCorrectionLoop(
        workspace_path=ws_path, state_dir=state_dir,
    )
    job_file = loop_for_check._correction_job_path(task_id)

    should_dispatch_correction = False
    if claimed.status == "CORRECTION_PENDING":
        # Validate the job file exists and matches ledger data.
        # If it does not exist or is corrupted, archive it as audit evidence.
        if os.path.isfile(job_file):
            try:
                with open(job_file) as jf:
                    import json as _json_module
                    job_data = _json_module.load(jf)
                if (job_data.get("task_id") == task_id and
                        job_data.get("current_sha") == claimed.current_sha):
                    should_dispatch_correction = True
                else:
                    logger.warning(
                        f"[{task_id}] Correction job metadata mismatch — "
                        f"archiving (ledger sha={claimed.current_sha}, "
                        f"file task_id={job_data.get('task_id')}, "
                        f"file sha={job_data.get('current_sha')})"
                    )
                    # Archive for audit evidence
                    _archive_correction_job(job_file, task_id, ws_path)
            except Exception:
                logger.warning(
                    f"[{task_id}] Cannot read correction job file — "
                    f"archiving as audit evidence"
                )
                _archive_correction_job(job_file, task_id, ws_path)
        else:
            # No job file but ledger says CORRECTION_PENDING — stale state.
            logger.warning(
                f"[{task_id}] CORRECTION_PENDING without job file "
                f"— checking for prior verdict before deciding recovery path"
            )
            _archive_orphaned_jobs(task_id, ws_path)

            # Check ledger for existing review verdicts to avoid re-reviewing a SHA that was already rejected.
            task_data = loop_for_check.ledger._data.get(task_id, {})
            review_seq = task_data.get("review_sequence", [])
            current_sha_val = task_data.get("current_sha", "") or claimed.current_sha or ""

            has_changes_required_verdict = any(
                e.get("type") == "review" and e.get("status") == "changes_required" and (
                    not e.get("sha") or e.get("sha") == current_sha_val
                )
                for e in review_seq
            )

            if has_changes_required_verdict:
                # Same SHA was already rejected — cannot recover without a valid correction job.
                # Keep task blocked; do NOT re-enter PENDING_REVIEW where it could trigger
                # an infinite re-review loop or a second Coder dispatch for the same bad code.
                logger.error(
                    f"[{task_id}] CORRECTION_PENDING without job file AND "
                    f"prior CHANGES_REQUIRED verdict for current SHA — transitioning to BLOCKED"
                )
                loop_for_check.ledger.set_task_status(task_id, "BLOCKED")
            else:
                # No prior rejection recorded — safe to re-enter PENDING_REVIEW for fresh review.
                logger.info(
                    f"[{task_id}] CORRECTION_PENDING without job file and no prior CHANGES_REQUIRED "
                    f"— transitioning to PENDING_REVIEW"
                )
                loop_for_check.ledger.set_task_status(task_id, "PENDING_REVIEW")

    if should_dispatch_correction:
        # ── Direct correction dispatch (Phase 3B) ────────────────
        logger.info(f"[{task_id}] Correction job validated — dispatching real Coder (Phase 3B)")
        _json_line({
            "action": "correction_dispatch",
            "task_id": task_id,
            "worker_id": worker_id,
            "previous_status": claimed.status,
        })

        result = loop_for_check.dispatch_correction(
            task_id=task_id,
            cli_command="hermes",
            timeout_s=300,
            extra_cli_args=["--cli", "--accept-hooks"],
        )

        if result["success"]:
            # Archive the completed job file for audit evidence
            _archive_correction_job(job_file, task_id, ws_path)
            # Transition back to PENDING_REVIEW for re-evaluation
            loop_for_check.ledger.set_task_status(task_id, "PENDING_REVIEW")
            _json_line({
                "action": "correction_complete",
                "task_id": task_id,
                "previous_sha": result["previous_sha"],
                "new_sha": result["new_sha"],
                "output_path": result["output_path"],
                "new_status": "PENDING_REVIEW",
            })
            ledger.release_task(task_id, reason="correction_dispatched")
            return 0
        else:
            _json_line({
                "action": "correction_failed",
                "task_id": task_id,
                "previous_sha": result["previous_sha"],
                "error": result["error"],
            })
            ledger.release_task(task_id, reason=f"correction_failed: {result['error']}")
            return 1

    # ── Normal pipeline: PENDING_REVIEW → review loop ──────────
    # Build transport from env or default
    transport = _create_transport_from_env()

    # Run the autonomous correction loop
    try:
        loop = AutonomousCorrectionLoop(
            workspace_path=ws_path,
            state_dir=state_dir,
        )
        ws = loop.run(
            task_id=task_id,
            requirements=claimed.requirements,
            acceptance_criteria=[],  # Handled by transport sequence
            code_context="",
            mock_transport=transport,
        )

        final_disposition = ws.final_disposition or ""
        review_count = loop.ledger.get_task_review_count(task_id)
        correction_count = loop.ledger.get_task(task_id).correction_count
        # Grab the actual git HEAD so approval validation has the correct SHA.
        final_sha = ws.task_record.current_sha if hasattr(ws.task_record, 'current_sha') else ""
        if not final_sha:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                cwd=ws_path, timeout=5,
            )
            if r.returncode == 0:
                final_sha = r.stdout.strip()

        # C2: Heartbeat — extend lease after correction loop iteration
        extended = loop.ledger._extend_lease(task_id, ttl_seconds=300)
        logger.info(f"[{task_id}] Lease extended: {extended}")
        _json_line({
            "action": "lease_extended",
            "task_id": task_id,
            "result": extended,
        })

        # C3: Approval validation before PASS disposition
        # Reload from disk — the autonomous loop writes reviews via its own
        # ledger object sharing the same file, so the controller's in-memory
        # state is stale after the loop returns.
        if final_disposition == Disposition.READY_FOR_OPERATOR_REVIEW:
            ledger._load_under_lock(task_id)
            approval = ledger.validate_approval_record(
                task_id=task_id,
                workspace_path=ws_path,
                commit_sha=final_sha,
            )
            if not approval["approved"]:
                logger.warning(f"[{task_id}] Approval gate failed: {approval['reason']}")
                final_disposition = Disposition.BLOCKED
                loop.ledger.finalize_task(task_id, final_disposition)

        _json_line({
            "action": "review_complete",
            "task_id": task_id,
            "worker_id": worker_id,
            "previous_status": "PENDING_REVIEW",
            "new_status": final_disposition,
            "final_disposition": final_disposition,
            "review_count": review_count,
            "correction_count": correction_count,
            "sha_at_review": final_sha,
        })

        if final_disposition in (Disposition.BLOCKED, Disposition.MAX_CORRECTION_EXHAUSTED):
            return 1

        # Phase 3B: If still CORRECTION_NEEDED after loop, dispatch real Coder.
        # This path is reached when the loop exhausted its internal correction
        # iterations and exited with a pending job file.
        if final_disposition == Disposition.CORRECTION_NEEDED:
            logger.info(f"[{task_id}] Correction needed — dispatching real Coder (Phase 3B)")
            _json_line({
                "action": "correction_dispatch_phase3b",
                "task_id": task_id,
                "new_status": final_disposition,
            })

            result = loop.dispatch_correction(
                task_id=task_id,
                cli_command="hermes",
                timeout_s=300,
                extra_cli_args=["--cli", "--accept-hooks"],
            )

            if result["success"]:
                loop.ledger.set_task_status(task_id, "PENDING_REVIEW")
                _json_line({
                    "action": "correction_complete",
                    "task_id": task_id,
                    "previous_sha": result["previous_sha"],
                    "new_sha": result["new_sha"],
                    "output_path": result["output_path"],
                    "new_status": "PENDING_REVIEW",
                })
            else:
                _json_line({
                    "action": "correction_failed",
                    "task_id": task_id,
                    "previous_sha": result["previous_sha"],
                    "error": result["error"],
                })

            ledger.release_task(task_id, reason="correction_dispatched")
            return 1 if not result["success"] else 0

        return 0

    except Exception as exc:
        logger.exception(f"Review failed for task {task_id}")
        _json_line({
            "action": "error",
            "task_id": task_id,
            "previous_status": "PENDING_REVIEW",
            "new_status": "IN_PROGRESS",
            "error": str(exc),
        })
        ledger.release_task(task_id, reason=f"review_error: {exc}")
        return 1


def _recover_expired(ledger: WorkflowLedger) -> int:
    """Find and release expired claims. Returns count of recovered tasks."""
    expired = ledger.find_expired_claims(max_ttl_seconds=600)
    recovered = 0

    for task in expired:
        released = ledger.release_task(task.task_id, reason="expired_by_controller")
        if released:
            _json_line({
                "action": "recovered_expired",
                "task_id": task.task_id,
                "claim_worker": task.claim_worker,
                "claim_expires_at": task.claim_expires_at,
            })
            recovered += 1

    if not expired:
        _json_line({
            "action": "no_expired_claims",
            "message": "No expired claims to recover",
        })

    return recovered


def _process_pr_once(
    state_dir: str,
    worker_id: str,
    ttl: int,
    allowlist: list[str] | None = None,
) -> int:
    """Process pending PR review tasks in --pr-mode.

    Loads the PRRegistry, discovers reviewable PRs via SHA comparison,
    runs Frontier Review with mock transport, dispatches real Coder
    on CHANGES_REQUIRED, or marks PASS and updates GitHub status checks.

    Returns exit code: 0 = done (no work / all processed), 1 = errors.
    """
    import fcntl

    ledger_path = default_ledger_path(state_dir)
    registry = PRRegistry(ledger_path)
    if allowlist:
        registry.set_allowlist(allowlist)

    # Acquire exclusive lock to prevent concurrent --once invocations
    lock_path = os.path.join(state_dir, ".pr_lock")
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        _json_line({
            "action": "already_claimed",
            "worker_id": worker_id,
            "message": "Another controller instance is already running PR review",
        })
        return 0

    try:
        # Discover reviewable PRs — use real GitHub API if token is set
        gh_token = os.environ.get("FR_GITHUB_TOKEN", "")
        gh_client = GitHubClient(token=gh_token)
        reviewable = discover_reviewable_prs(registry, gh_client)

        if not reviewable:
            _json_line({
                "action": "no_reviewable_prs",
                "worker_id": worker_id,
                "message": "No PRs need review at this time",
            })
            return 0

        error_count = 0
        for task_id, info in reviewable:
            sha = info["sha"]
            diff = info["diff"]
            pr_info = info["pr_info"]

            # Claim the task atomically
            claimed = registry._ledger.claim_task(
                task_id, worker_id=worker_id, ttl_seconds=ttl
            )
            if claimed is None:
                _json_line({
                    "action": "claim_failed",
                    "task_id": task_id,
                    "reason": "Task not claimable (already claimed or removed)",
                })
                continue

            try:
                # ── Push-state reconciliation (Blocker 1 recovery) ─────────────
                # Must run BEFORE the review pipeline to avoid consuming mock
                # scenarios on stale/lost-push state.
                ws_path = claimed.workspace_path or "/tmp"
                raw_claim = registry._ledger._data.get(task_id, {})
                push_error = raw_claim.get("push_error") or ""
                is_correction_pending = raw_claim.get("status") == "CORRECTION_PENDING"

                sha_stale = False
                if claimed.workspace_path:
                    try:
                        r_check = subprocess.run(
                            ["git", "rev-parse", "--short=12", "HEAD"],
                            capture_output=True, text=True, timeout=5,
                            cwd=claimed.workspace_path,
                        )
                        local_head = r_check.stdout.strip()
                        gh_cl_quick = GitHubClient(
                            repo_owner=pr_info.get("repo_owner", ""),
                            repo_name=pr_info.get("repo_name", ""),
                            token=gh_token,
                        )
                        remote_head = gh_cl_quick.get_remote_head_sha(
                            pr_info.get("head_branch", "")
                        )
                        intended = (raw_claim.get("current_sha") or "").strip()
                        if intended and local_head and remote_head:
                            if int(intended[:12], 16) != int(remote_head[:12], 16):
                                sha_stale = True
                                logger.info(
                                    f"[{task_id}] Stale SHA detected: "
                                    f"ledger={intended[:8]} remote={remote_head[:8]} local={local_head[:8]}"
                                )
                    except Exception as e:
                        logger.debug(f"[{task_id}] Could not check SHA staleness: {e}")

                should_reconcile = (push_error or is_correction_pending or sha_stale) and claimed.workspace_path
                logger.debug(f"[{task_id}] should_reconcile={should_reconcile} push_error={bool(push_error)} correction_pending={is_correction_pending} sha_stale={sha_stale}")

                if should_reconcile:
                    _json_line({
                        "action": "push_state_reconciliation",
                        "task_id": task_id,
                        "reason": f"push_error={bool(push_error)}, correction_pending={is_correction_pending}, sha_stale={sha_stale}",
                    })

                    # Archive stale correction job files from prior dispatches
                    ws_path = claimed.workspace_path or ""
                    if ws_path:
                        _archive_orphaned_jobs(task_id, ws_path)

                    # Query GitHub for the actual remote PR head SHA
                    gh_cl = GitHubClient(
                        repo_owner=pr_info.get("repo_owner", ""),
                        repo_name=pr_info.get("repo_name", ""),
                        token=gh_token,
                    )
                    remote_sha_on_branch = gh_cl.get_remote_head_sha(
                        pr_info.get("head_branch", "")
                    )

                    # Get local SHA from workspace
                    r_local = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True, text=True, timeout=5,
                        cwd=ws_path,
                    )
                    local_sha = r_local.stdout.strip()

                    # Compare remote vs expected corrected SHA
                    intended_sha = raw_claim.get("current_sha", "") or local_sha

                    if remote_sha_on_branch == intended_sha:
                        _json_line({
                            "action": "push_reconciled",
                            "task_id": task_id,
                            "remote_sha": remote_sha_on_branch[:8],
                            "intended_sha": intended_sha[:8],
                        })
                        registry._ledger.set_task_status(task_id, "PENDING_REVIEW")
                        raw = registry._ledger._data.get(task_id)
                        if raw:
                            raw["push_error"] = ""
                            raw["remote_head_sha"] = remote_sha_on_branch
                            # NEVER reset correction_count, _mock_scenario_index, or review_status here.
                            # Those are task accounting data — reconciliation only updates push state.
                            registry._ledger._save()
                    elif remote_sha_on_branch == local_sha:
                        _json_line({
                            "action": "push_reconciled",
                            "task_id": task_id,
                            "remote_sha": remote_sha_on_branch[:8],
                            "local_sha": local_sha[:8],
                        })
                        registry._ledger.set_task_status(task_id, "PENDING_REVIEW")
                        raw = registry._ledger._data.get(task_id)
                        if raw:
                            raw["push_error"] = ""
                            raw["remote_head_sha"] = remote_sha_on_branch
                            raw["current_sha"] = local_sha
                            registry._ledger._save()
                    else:
                        _json_line({
                            "action": "sha_mismatch_detected",
                            "task_id": task_id,
                            "remote_sha": remote_sha_on_branch[:8],
                            "intended_sha": intended_sha[:8] if intended_sha else "(none)",
                            "local_sha": local_sha[:8] if local_sha else "(none)",
                        })

                        # If local workspace already has the corrected SHA, no push needed
                        if local_sha == remote_sha_on_branch:
                            _json_line({
                                "action": "push_reconciled",
                                "task_id": task_id,
                                "remote_sha": remote_sha_on_branch[:8],
                                "local_sha": local_sha[:8],
                                "note": "workspace already matches remote — no push needed",
                            })
                            registry._ledger.set_task_status(task_id, "PENDING_REVIEW")
                            raw = registry._ledger._data.get(task_id)
                            if raw:
                                raw["push_error"] = ""
                                raw["remote_head_sha"] = remote_sha_on_branch
                                raw["current_sha"] = local_sha
                                registry._ledger._save()
                            # Skip full pipeline — fall through to review_status handler
                            # which will re-review the corrected SHA directly
                        else:
                            # Bounded push retries
                            push_retries = raw_claim.get("_push_retry_count", 0)
                            max_push_retries = 3
                            if push_retries >= max_push_retries:
                                _json_line({
                                    "action": "push_max_retries_exhausted",
                                    "task_id": task_id,
                                    "retries": push_retries,
                                })
                                registry._ledger.release_task(
                                    task_id, reason="push_max_retries_exhausted"
                                )
                                continue

                            pushed, push_result = push_correction_to_github(
                                repo_owner=pr_info.get("repo_owner", ""),
                                repo_name=pr_info.get("repo_name", ""),
                                head_branch=pr_info.get("head_branch", ""),
                                token=gh_token,
                                workspace_path=ws_path,
                            )

                            if pushed:
                                _json_line({
                                    "action": "push_retry_success",
                                    "task_id": task_id,
                                    "remote_sha": push_result[:8],
                                    "retry_count": push_retries + 1,
                                })
                                registry._ledger.set_task_status(task_id, "PENDING_REVIEW")
                                raw = registry._ledger._data.get(task_id)
                                if raw:
                                    raw["push_error"] = ""
                                    raw["remote_head_sha"] = push_result
                                    raw["_push_retry_count"] = push_retries + 1
                                    registry._ledger._save()
                            else:
                                _json_line({
                                    "action": "push_retry_failed",
                                    "task_id": task_id,
                                    "error": str(push_result)[:200],
                                    "retry_count": push_retries + 1,
                                })
                                raw = registry._ledger._data.get(task_id)
                                if raw:
                                    raw["_push_retry_count"] = push_retries + 1
                                    raw["status"] = "PUSH_PENDING"
                                    registry._ledger._save()

                # Always refresh claim data from ledger for post-reconciliation state
                _now_data = registry._ledger._data.get(task_id, {})
                _claimed_now = PRReviewTask.from_dict(_now_data)

                # Build review context — NO credentials included (only after reconciliation)
                review_ctx = build_review_context(
                    _claimed_now,
                    diff,
                    registry,
                )

                _json_line({
                    "action": "review_started",
                    "task_id": task_id,
                    "sha": sha[:8],
                    "pr_number": pr_info["pr_number"],
                    "repo": f"{pr_info['repo_owner']}/{pr_info['repo_name']}",
                })

                # Run Frontier Review via mock transport
                ws_path = claimed.workspace_path or "/tmp"
                loop = AutonomousCorrectionLoop(
                    workspace_path=ws_path, state_dir=state_dir,
                )

                # Set up mock transport from env sequence (with per-task persistence)
                transport = _create_transport_from_env(
                    task_id=task_id,
                    ledger_path=os.path.join(state_dir, "workflow_ledger.json"),
                )

                ws = loop.run(
                    task_id=task_id,
                    requirements=review_ctx.get("requirements", _claimed_now.requirements),
                    acceptance_criteria=[],
                    code_context="",
                    mock_transport=transport,
                    initial_commit_sha=sha,
                    pr_mode=True,
                )

                final_disposition = ws.final_disposition or ""
                new_sha = sha  # default — may change after correction

                if final_disposition in (
                    Disposition.BLOCKED, Disposition.MAX_CORRECTION_EXHAUSTED
                ):
                    _json_line({
                        "action": "review_complete",
                        "task_id": task_id,
                        "status": final_disposition,
                        "sha": sha[:8],
                    })
                    error_count += 1
                    registry._ledger.release_task(task_id, reason="blocked")
                    continue

                # Fall through to review_status handler (state refreshed post-reconciliation)

                rec = PRReviewTask.from_dict(_now_data)
                if rec.review_status == "unreviewed":
                    # First review cycle — always expect changes first or pass
                    if final_disposition == Disposition.READY_FOR_OPERATOR_REVIEW:
                        _json_line({
                            "action": "review_pass",
                            "task_id": task_id,
                            "sha": sha[:8],
                            "pr_number": pr_info["pr_number"],
                        })
                        # ── Blocker 4: Verify approval prerequisites before PASS ──
                        ok, reason = verify_approval_prerequisites(
                            repo_owner=pr_info.get("repo_owner", ""),
                            repo_name=pr_info.get("repo_name", ""),
                            pr_number=pr_info.get("pr_number", 0),
                            head_branch=pr_info.get("head_branch", ""),
                            reviewed_sha=sha,
                            token=gh_token,
                        )
                        if not ok:
                            _json_line({
                                "action": "approval_blocked",
                                "task_id": task_id,
                                "reason": reason,
                            })
                            # Do NOT mark pass — leave as changes_required for re-review
                            registry._ledger.release_task(task_id, reason=f"approval_blocked: {reason}")
                            continue

                        # Mark pass in registry
                        evidence_fp = loop.ledger.compute_fingerprint(sha, "")
                        registry.mark_pass(task_id, sha, evidence_fp)

                        gh_pr = GitHubClient(
                            repo_owner=pr_info["repo_owner"],
                            repo_name=pr_info["repo_name"],
                        )
                        if gh_client._mock_state:
                            gh_pr._mock_state = dict(gh_client._mock_state)
                        gh_pr.update_status(
                            ref_sha=sha, state="success",
                            description="Frontier Review passed",
                            context="frontier-review",
                        )
                        gh_pr.update_pr_ready(pr_info["pr_number"])

                        registry._ledger.release_task(task_id, reason="pass")
                    else:
                        # CHANGES_REQUIRED
                        review_count = loop.ledger.get_task_review_count(task_id)
                        max_c = claimed.max_correction_cycles
                        if review_count >= max_c:
                            _json_line({
                                "action": "correction_exhausted",
                                "task_id": task_id,
                                "max_corrections": max_c,
                            })
                            registry.mark_correction_exhausted(task_id)
                            error_count += 1
                        else:
                            # Persist findings, dispatch correction
                            review_status = loop.ledger._data[task_id].get("review_status", "changes_required")
                            if "review_status" not in loop.ledger._data[task_id]:
                                raw = loop.ledger._data[task_id]
                                raw["review_status"] = "changes_required"
                                registry._ledger._save()

                            _json_line({
                                "action": "correction_needed",
                                "task_id": task_id,
                                "sha": sha[:8],
                            })
                            # Dispatch real Coder correction
                            result = loop.dispatch_correction(
                                task_id=task_id,
                                cli_command="hermes",
                                timeout_s=300,
                                extra_cli_args=["--cli", "--accept-hooks"],
                            )

                            if result["success"]:
                                new_sha = result["new_sha"] or sha
                                registry.update_task_sha(task_id, new_sha)
                                _json_line({
                                    "action": "correction_applied",
                                    "task_id": task_id,
                                    "old_sha": sha[:8],
                                    "new_sha": (new_sha or "")[:8],
                                })

                                # ── Blocker 1: Push correction to GitHub and verify remote SHA ──
                                pushed, push_result = push_correction_to_github(
                                    repo_owner=pr_info.get("repo_owner", ""),
                                    repo_name=pr_info.get("repo_name", ""),
                                    head_branch=pr_info.get("head_branch", ""),
                                    token=gh_token,
                                    workspace_path=claimed.workspace_path or ws_path,
                                )

                                if not pushed:
                                    # BLOCKED: correction committed but push failed or remote mismatch
                                    _json_line({
                                        "action": "push_blocked",
                                        "task_id": task_id,
                                        "error": str(push_result),
                                        "local_sha": ((new_sha or "")[:8] if new_sha else sha[:8]),
                                    })
                                    # Keep in PENDING_REVIEW but mark as needing push-verify first
                                    # so next controller run retries push before doing a fresh review
                                    registry._ledger.set_task_status(
                                        task_id, "PENDING_REVIEW", push_error=push_result,
                                    )
                                    raw = registry._ledger._data.get(task_id)
                                    if raw:
                                        raw["review_status"] = "unreviewed"  # re-discoverable
                                        raw["remote_head_sha"] = ""  # force SHA mismatch → discovered
                                    continue

                                # Update ledger with verified remote SHA, then clear remote_head_sha
                                # so the NEXT controller run will discover and re-review this PR
                                # (the recorded SHA must differ from current GitHub SHA to trigger review)
                                new_remote_sha = push_result  # this is the verified remote HEAD
                                registry._ledger.set_task_status(task_id, "PENDING_REVIEW")
                                raw = registry._ledger._data.get(task_id)
                                if raw:
                                    raw["remote_head_sha"] = ""  # clear to force re-discovery next run
                                    raw["review_status"] = "unreviewed"
                                    registry._ledger._save()
                                _json_line({
                                    "action": "push_verified",
                                    "task_id": task_id,
                                    "remote_sha": new_remote_sha[:8],
                                })
                            else:
                                _json_line({
                                    "action": "correction_failed",
                                    "task_id": task_id,
                                    "error": result["error"],
                                })
                                error_count += 1

                elif rec.review_status == "changes_required":
                    # Re-review after correction
                    if final_disposition == Disposition.READY_FOR_OPERATOR_REVIEW:
                        _json_line({
                            "action": "review_pass_after_correction",
                            "task_id": task_id,
                            "sha": sha[:8],
                            "pr_number": pr_info["pr_number"],
                        })
                        # ── Blocker 4: Verify approval prerequisites before PASS (re-review) ──
                        ok, reason = verify_approval_prerequisites(
                            repo_owner=pr_info.get("repo_owner", ""),
                            repo_name=pr_info.get("repo_name", ""),
                            pr_number=pr_info.get("pr_number", 0),
                            head_branch=pr_info.get("head_branch", ""),
                            reviewed_sha=sha,
                            token=gh_token,
                        )
                        if not ok:
                            _json_line({
                                "action": "approval_blocked",
                                "task_id": task_id,
                                "reason": reason,
                            })
                            registry._ledger.release_task(task_id, reason=f"approval_blocked: {reason}")
                            continue

                        evidence_fp = loop.ledger.compute_fingerprint(sha, "")
                        registry.mark_pass(task_id, sha, evidence_fp)
                        gh_pr = GitHubClient(
                            repo_owner=pr_info["repo_owner"],
                            repo_name=pr_info["repo_name"],
                        )
                        if gh_client._mock_state:
                            gh_pr._mock_state = dict(gh_client._mock_state)
                        gh_pr.update_status(
                            ref_sha=sha, state="success",
                            description="Frontier Review passed after correction",
                            context="frontier-review",
                        )
                        gh_pr.update_pr_ready(pr_info["pr_number"])
                        registry._ledger.release_task(task_id, reason="pass_after_correction")
                    else:
                        review_count = loop.ledger.get_task_review_count(task_id)
                        if review_count >= claimed.max_correction_cycles:
                            registry.mark_correction_exhausted(task_id)
                            _json_line({
                                "action": "correction_exhausted",
                                "task_id": task_id,
                                "max_corrections": claimed.max_correction_cycles,
                            })
                            error_count += 1
                        else:
                            _json_line({
                                "action": "review_continued_changes_required",
                                "task_id": task_id,
                            })
                            registry._ledger.release_task(task_id, reason="more_corrections_needed")

            except Exception as exc:
                logger.exception(f"PR review error for {task_id}: {exc}")
                _json_line({
                    "action": "error",
                    "task_id": task_id,
                    "error": str(exc),
                })
                error_count += 1
            finally:
                # Always release claim if still IN_PROGRESS
                try:
                    current_rec = registry._ledger.get_task(task_id)
                    if current_rec.status == "IN_PROGRESS":
                        registry._ledger.release_task(task_id, reason="review_complete")
                except ValueError:
                    pass

        return 0 if error_count == 0 else 1

    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Frontier Review Controller")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Process one pending task")
    group.add_argument("--recover", action="store_true", help="Recover expired claims")
    parser.add_argument("--pr-mode", action="store_true", help="PR-based review mode (requires --once)")
    parser.add_argument("--worker-id", default=None, help="Worker identity")
    parser.add_argument("--ttl", type=int, default=300, help="Claim TTL in seconds")
    parser.add_argument("--state-dir", default=None, help="Ledger state directory")
    parser.add_argument("--dry-run", action="store_true", help="List eligible tasks without claiming")
    parser.add_argument("--allowlist", nargs="*", default=None, help="Allowed repos (owner/repo pairs)")

    args = parser.parse_args()

    if args.pr_mode and not args.once:
        print("ERROR: --pr-mode requires --once", file=sys.stderr)
        return 1

    # Worker identity
    worker_id = args.worker_id or os.environ.get("HERMES_WORKER_ID", "cron-1")

    # State directory
    if args.state_dir:
        state_dir = args.state_dir
    else:
        home = os.environ.get("HOME", "/home/atlas")
        state_dir = os.path.join(home, ".hermes", "frontier_review_state")
    os.makedirs(state_dir, exist_ok=True)

    ledger_path = default_ledger_path(state_dir)
    ledger = WorkflowLedger(ledger_path)

    if args.once:
        if args.pr_mode:
            allowlist = args.allowlist or []
            return _process_pr_once(state_dir, worker_id, args.ttl, allowlist if allowlist else None)
        return _process_once(ledger, worker_id, args.ttl, state_dir, args.dry_run)
    elif args.recover:
        recovered = _recover_expired(ledger)
        _json_line({
            "action": "recover_complete",
            "recovered_count": recovered,
        })
        return 0


if __name__ == "__main__":
    sys.exit(main())
