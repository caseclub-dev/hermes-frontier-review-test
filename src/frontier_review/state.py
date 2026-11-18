"""Persistent task-state machine for Frontier Review.

States: PENDING → REREVIEWING → COMPLETE | BLOCKED

The state machine persists to JSON on disk so that process restarts
do not lose in-progress review cycles. It also supports test isolation
via an optional in-memory storage backend.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TaskState:
    """A single review task's persistent state."""

    task_id: str
    cycle_status: str = "PENDING"  # PENDING | REREVIEWING | COMPLETE | BLOCKED
    commit_sha: str = ""
    diff: str = ""
    requirements: str = ""
    last_review_result: str = ""  # JSON string of ReviewResult or empty
    findings_count: int = 0
    corrections_made: int = 0  # number of correction cycles attempted
    max_reviews: int = 5
    review_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StateMachineError(Exception):
    """Raised when a state transition is invalid."""


class TaskStateMachine:
    """Manages review task states with persistence and transitions.

    Supports both file-backed (production) and dict-backed (test) storage.
    """

    VALID_TRANSITIONS = {
        "PENDING": {"REREVIEWING"},
        "REREVIEWING": {"COMPLETE", "BLOCKED", "REREVIEWING"},
        "COMPLETE": set(),  # terminal
        "BLOCKED": set(),  # terminal
    }

    def __init__(self, store_path: Optional[str] = None) -> None:
        self._store_path = store_path
        # _store is either a file path or an in-memory dict (for tests)
        if store_path is not None:
            self._store_dict: dict[str, Any] = self._load_store(store_path)
        else:
            self._store_dict = {}

    def _load_store(self, path: str) -> dict[str, Any]:
        """Load state store from disk."""
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_store(self) -> None:
        """Persist state store to disk."""
        if self._store_path is not None:
            tmp_path = self._store_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self._store_dict, f, indent=2)
            os.replace(tmp_path, self._store_path)

    def create_task(
        self,
        task_id: str,
        commit_sha: str = "",
        diff: str = "",
        requirements: str = "",
    ) -> TaskState:
        """Create a new task in PENDING state."""
        if task_id in self._store_dict:
            raise StateMachineError(f"Task '{task_id}' already exists.")

        ts = datetime.now(timezone.utc).isoformat()
        task = TaskState(
            task_id=task_id,
            cycle_status="PENDING",
            commit_sha=commit_sha,
            diff=diff,
            requirements=requirements,
            created_at=ts,
            updated_at=ts,
        )
        self._store_dict[task_id] = task.to_dict()
        self._save_store()
        return task

    def get_task(self, task_id: str) -> TaskState:
        """Get a task by ID."""
        if task_id not in self._store_dict:
            raise StateMachineError(f"Task '{task_id}' does not exist.")
        return TaskState.from_dict(self._store_dict[task_id])

    def transition(
        self,
        task_id: str,
        new_status: str,
        **kwargs: Any,
    ) -> TaskState:
        """Transition a task to a new state.

        Additional kwargs (commit_sha, diff, findings_count, etc.) update
        the task's fields along with the status change.
        """
        if task_id not in self._store_dict:
            raise StateMachineError(f"Task '{task_id}' does not exist.")

        task = TaskState.from_dict(self._store_dict[task_id])

        # Validate transition
        current = task.cycle_status
        allowed = self.VALID_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise StateMachineError(
                f"Invalid transition: {current} → {new_status}. "
                f"Allowed: {allowed or {'(terminal)'}}"
            )

        # Check review count cap (only for REREVIEWING → REREVIEWING)
        if new_status == "REREVIEWING" and current == "REREVIEWING":
            if task.review_count >= task.max_reviews:
                raise StateMachineError(
                    f"Task '{task_id}' exceeded max reviews ({task.max_reviews})."
                )

        # Apply transition
        old_status = task.cycle_status
        task.cycle_status = new_status
        task.updated_at = datetime.now(timezone.utc).isoformat()
        task.review_count = kwargs.get("review_count", task.review_count)
        task.corrections_made = kwargs.get("corrections_made", task.corrections_made)

        # Update any additional fields
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self._store_dict[task_id] = task.to_dict()
        self._save_store()
        return task

    def list_tasks(self) -> list[TaskState]:
        """List all tasks."""
        return [TaskState.from_dict(d) for d in self._store_dict.values()]

    def get_task_count(self) -> int:
        return len(self._store_dict)

    @staticmethod
    def reset_store(path: str) -> None:
        """Delete and recreate the store file (for tests)."""
        if os.path.exists(path):
            os.remove(path)
