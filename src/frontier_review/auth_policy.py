"""Operator authorization policy engine for Frontier Review corrections.

Manages per-PR authorization records (create, get, revoke, consume) with
strict input validation, atomic file writes, and path-injection prevention.
All operations are local-file based — the AuthPolicy class does NOT check
caller identity; that is enforced by deployment config (file permissions,
Unix identity).

Request schema validation:
  - repository_owner, repository_name: must match AUTHORIZED_REPOS allowlist
  - pr_number: positive integer 1-999999
  - head_branch: must match BRANCH_PATTERN (hermes/<task-id>)
  - initial_remote_sha: 40 lowercase hex characters
  - max_corrections: integer in [1, 10]
  - ttl_hours: float/real number in [1.0, 720.0]

Security constraints:
  - No shell command execution from arguments
  - Path injection prevention (../, newlines, null bytes rejected)
  - File permissions 0o640 on all record files
  - Atomic writes via temp file + rename
  - Never log SHAs/tokens beyond first 8 chars
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants — never configurable by caller
# ---------------------------------------------------------------------------

AUTHORIZED_REPOS: set[tuple[str, str]] = {("caseclub-dev", "hermes-frontier-review-test")}

BRANCH_PATTERN = re.compile(r"^hermes/[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

DEFAULT_AUTH_DIR: str = "/var/lib/frontier-review/publisher/auth/"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("frontier_review.auth_policy")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _mask_sha(value: str) -> str:
    """Return the first 8 characters of a SHA or value for safe logging."""
    if len(value) >= 8:
        return value[:8] + "..."
    return value


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class AuthorizationError(Exception):
    """Base exception for authorization policy failures."""


class ValidationError(AuthorizationError):
    """Raised when inputs fail validation checks."""


class RecordNotFoundError(AuthorizationError):
    """Raised when a requested authorization record does not exist."""


class PolicyViolationError(AuthorizationError):
    """Raised when an operation violates policy (expired, revoked, budget exceeded)."""


# ---------------------------------------------------------------------------
# Input validation helpers (called BEFORE any file I/O)
# ---------------------------------------------------------------------------

def _validate_repo(owner: Any, name: Any) -> None:
    """Validate repository owner and name against allowlist.

    Raises ValidationError on invalid input or disallowed repo.
    """
    if not isinstance(owner, str) or not isinstance(name, str):
        raise ValidationError("repository_owner and repository_name must be strings")
    # Path injection prevention
    _validate_safe_string(owner, "repository_owner")
    _validate_safe_string(name, "repository_name")
    if (owner, name) not in AUTHORIZED_REPOS:
        raise ValidationError(
            f"Repository {owner}/{name} not on allowlist"
        )


def _validate_branch(branch: Any) -> None:
    """Validate that branch matches the hermes/* pattern.

    Raises ValidationError on invalid input.
    """
    if not isinstance(branch, str):
        raise ValidationError("head_branch must be a string")
    _validate_safe_string(branch, "head_branch")
    if not BRANCH_PATTERN.match(branch):
        raise ValidationError(
            "head_branch must match pattern hermes/<task-id> (alphanumeric start)"
        )


def _validate_sha(sha: Any) -> None:
    """Validate SHA is exactly 40 lowercase hex characters.

    Raises ValidationError on invalid input.
    """
    if not isinstance(sha, str):
        raise ValidationError("initial_remote_sha must be a string")
    if not SHA_PATTERN.match(sha):
        raise ValidationError(
            "initial_remote_sha must be exactly 40 lowercase hex characters"
        )


def _validate_pr_number(n: Any) -> None:
    """Validate PR number is a positive integer in [1, 999999].

    Raises ValidationError on invalid input.
    """
    if isinstance(n, bool):
        raise ValidationError("pr_number must be an integer, not boolean")
    if not isinstance(n, int):
        raise ValidationError("pr_number must be a positive integer")
    if n < 1 or n > 999999:
        raise ValidationError("pr_number must be between 1 and 999999")


def _validate_max_corrections(n: Any) -> None:
    """Validate max_corrections is an integer in [1, 10].

    Raises ValidationError on invalid input.
    """
    if isinstance(n, bool):
        raise ValidationError("max_corrections must be an integer")
    if not isinstance(n, int):
        try:
            n = int(n)
        except (ValueError, TypeError):
            raise ValidationError("max_corrections must be an integer in [1, 10]")
    if n < 1 or n > 10:
        raise ValidationError("max_corrections must be between 1 and 10")


def _validate_ttl(ttl_hours: Any) -> None:
    """Validate TTL is a positive number in [1.0, 720.0].

    Raises ValidationError on invalid input.
    """
    if isinstance(ttl_hours, bool):
        raise ValidationError("ttl_hours must be a number")
    try:
        ttl = float(ttl_hours)
    except (ValueError, TypeError):
        raise ValidationError("ttl_hours must be a positive number")
    if ttl < 1.0 or ttl > 720.0:
        raise ValidationError("ttl_hours must be between 1 and 720")


def _validate_safe_string(value: str, field_name: str) -> None:
    """Reject path injection vectors in string values.

    Rejects if the value contains '../', newlines, null bytes, or '/'
    (other than within the expected structure).

    Raises ValidationError on dangerous input.
    """
    if "../" in value or value.startswith("/") or value.endswith("/"):
        raise ValidationError(
            f"{field_name} contains path injection characters"
        )
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValidationError(
            f"{field_name} contains forbidden characters (newline, carriage return, null)"
        )


# ---------------------------------------------------------------------------
# AuthPolicy — core authorization record manager
# ---------------------------------------------------------------------------

class AuthPolicy:
    """Manage per-PR operator authorization records.

    Records are stored as JSON files in a local directory with atomic writes.
    All input validation happens before any file I/O.

    Usage::

        policy = AuthPolicy(auth_dir="/tmp/auth-test")
        record = policy.create("caseclub-dev", "hermes-frontier-review-test", 42,
                               "hermis/fix-build", "abc123" * 7)
    """

    def __init__(self, auth_dir: str | None = None) -> None:
        """Initialize AuthPolicy with the authorization directory.

        Resolves auth_dir in this order of precedence:
          1. Explicit parameter (auth_dir)
          2. Environment variable FR_AUTH_DIR
          3. Default path DEFAULT_AUTH_DIR
        """
        if auth_dir is not None:
            self._auth_dir = auth_dir
        else:
            env_dir = os.environ.get("FR_AUTH_DIR")
            if env_dir is not None:
                self._auth_dir = env_dir
            else:
                self._auth_dir = DEFAULT_AUTH_DIR

        # Ensure the directory exists
        Path(self._auth_dir).mkdir(parents=True, exist_ok=True)

    @property
    def auth_dir(self) -> str:
        """The resolved authorization directory path."""
        return self._auth_dir

    def _record_path(self, authorization_id: str) -> str:
        """Return the filesystem path for an authorization record.

        The authorization_id is a UUID string; it is safe to use as a filename.
        """
        # Double-check: auth_id should be a clean UUID — no injection possible
        if "/" in authorization_id or "\\" in authorization_id:
            raise ValidationError(
                f"Invalid authorization_id format: {authorization_id[:12]}..."
            )
        return os.path.join(self._auth_dir, f"{authorization_id}.json")

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        repository_owner: str,
        repository_name: str,
        pr_number: int,
        head_branch: str,
        initial_remote_sha: str,
        max_corrections: int = 3,
        ttl_hours: float = 72.0,
    ) -> dict[str, Any]:
        """Create a new authorization record.

        All inputs are validated BEFORE any file I/O occurs.

        Args:
            repository_owner: Owner of the repository (must be on allowlist).
            repository_name: Name of the repository (must be on allowlist).
            pr_number: Pull request number (positive integer 1-999999).
            head_branch: Feature branch name (hermes/* pattern).
            initial_remote_sha: 40-char lowercase hex SHA of remote HEAD.
            max_corrections: Max corrections allowed [1, 10]. Default 3.
            ttl_hours: Time-to-live in hours [1.0, 720.0]. Default 72.

        Returns:
            The saved record dict.

        Raises:
            ValidationError: On invalid inputs or duplicate active record.
        """
        # --- Validate ALL inputs BEFORE any file I/O -------------------
        _validate_repo(repository_owner, repository_name)
        _validate_pr_number(pr_number)
        _validate_branch(head_branch)
        _validate_sha(initial_remote_sha)
        _validate_max_corrections(max_corrections)
        _validate_ttl(ttl_hours)

        # Generate a unique authorization ID
        auth_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)

        record: dict[str, Any] = {
            "authorization_id": auth_id,
            "repository_owner": repository_owner,
            "repository_name": repository_name,
            "pr_number": pr_number,
            "head_branch": head_branch,
            "initial_remote_sha": initial_remote_sha,
            "max_corrections": max_corrections,
            "corrections_used": 0,
            "ttl_hours": ttl_hours,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "status": "active",
        }

        # Check for duplicate active record
        filepath = self._record_path(auth_id)
        if self._is_duplicate_active(repository_owner, repository_name, pr_number):
            raise ValidationError(
                f"Active authorization already exists for "
                f"{repository_owner}/{repository_name} PR {pr_number}"
            )

        # Atomic write: temp file + rename
        self._save_record(record, filepath)

        logger.info(
            "Created auth record %s for %s/%s#%d (%s)",
            auth_id[:8], repository_owner, repository_name,
            int(pr_number),
            _mask_sha(initial_remote_sha),
        )
        return record

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self, authorization_id: str) -> dict[str, Any]:
        """Return an authorization record by ID.

        Args:
            authorization_id: UUID string of the record.

        Returns:
            The record dict.

        Raises:
            RecordNotFoundError: If no active or inactive record exists.
        """
        filepath = self._record_path(authorization_id)
        if not os.path.isfile(filepath):
            raise RecordNotFoundError(
                f"Authorization record not found: {authorization_id[:12]}..."
            )
        return self._load_record(filepath)

    # ------------------------------------------------------------------
    # List active
    # ------------------------------------------------------------------

    def list_active(self) -> list[dict[str, Any]]:
        """List all authorization records with status='active'.

        Records are sorted by created_at in descending order (newest first).
        Expired and revoked records are excluded.

        Returns:
            List of active record dicts.
        """
        auth_path = Path(self._auth_dir)
        if not auth_path.is_dir():
            return []

        now = datetime.now(timezone.utc)

        results: list[dict[str, Any]] = []
        for filepath in auth_path.glob("*.json"):
            try:
                record = self._load_record(str(filepath))
                if record.get("status") != "active":
                    continue

                # Also exclude expired records from the active listing
                try:
                    expires = datetime.fromisoformat(record["expires_at"])
                    if expires < now:
                        continue
                except (KeyError, ValueError):
                    pass

                results.append(record)
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted records
                continue

        results.sort(
            key=lambda r: r.get("created_at", ""), reverse=True
        )
        return results

    # ------------------------------------------------------------------
    # Revoke
    # ------------------------------------------------------------------

    def revoke(self, authorization_id: str) -> dict[str, Any]:
        """Revoke an active authorization record.

        Sets status='revoked' and writes atomically.

        Args:
            authorization_id: UUID string of the record to revoke.

        Returns:
            The updated record dict.

        Raises:
            RecordNotFoundError: If no record exists for this ID.
        """
        filepath = self._record_path(authorization_id)
        if not os.path.isfile(filepath):
            raise RecordNotFoundError(
                f"Authorization record not found: {authorization_id[:12]}..."
            )

        record = self._load_record(filepath)
        record["status"] = "revoked"
        self._save_record(record, filepath)

        logger.info("Revoked auth record %s", authorization_id[:8])
        return record

    # ------------------------------------------------------------------
    # Check authorized
    # ------------------------------------------------------------------

    def check_authorized(
        self, repository_owner: str, repository_name: str, pr_number: int
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Check if a valid authorization exists for a repo+PR pair.

        The authorization must be active, not expired, and within the
        correction budget (corrections_used < max_corrections).

        Args:
            repository_owner: Owner of the repository.
            repository_name: Name of the repository.
            pr_number: Pull request number to check.

        Returns:
            Tuple of (is_authorized, record) — record is None if not authorized.
        """
        # Validate inputs
        _validate_repo(repository_owner, repository_name)
        _validate_pr_number(pr_number)

        auth_path = Path(self._auth_dir)
        if not auth_path.is_dir():
            return False, None

        now = datetime.now(timezone.utc)

        for filepath in auth_path.glob("*.json"):
            try:
                record = self._load_record(str(filepath))
            except (json.JSONDecodeError, KeyError):
                continue

            if (
                record.get("repository_owner") != repository_owner or
                record.get("repository_name") != repository_name or
                record.get("pr_number") != pr_number
            ):
                continue

            if record.get("status") != "active":
                continue

            # Check expiration
            try:
                expires = datetime.fromisoformat(record["expires_at"])
            except (KeyError, ValueError):
                continue
            if expires < now:
                continue

            # Check correction budget
            corrections_used = record.get("corrections_used", 0)
            max_corr = record.get("max_corrections", 3)
            if corrections_used >= max_corr:
                continue

            return True, record

        return False, None

    # ------------------------------------------------------------------
    # Consume correction
    # ------------------------------------------------------------------

    def consume_correction(self, authorization_id: str) -> bool:
        """Increment corrections_used and write atomically.

        Args:
            authorization_id: UUID string of the authorization record.

        Returns:
            True if the operation succeeded within budget.

        Raises:
            RecordNotFoundError: If the record does not exist.
            PolicyViolationError: If corrections_used exceeds max_corrections.
        """
        filepath = self._record_path(authorization_id)
        if not os.path.isfile(filepath):
            raise RecordNotFoundError(
                f"Authorization record not found: {authorization_id[:12]}..."
            )

        record = self._load_record(filepath)
        record["corrections_used"] = record.get("corrections_used", 0) + 1

        corrections_used = record["corrections_used"]
        max_corr = record.get("max_corrections", 3)

        # Write the incremented counter atomically
        self._save_record(record, filepath)

        if corrections_used > max_corr:
            logger.warning(
                "Correction budget exceeded for %s: %d/%d",
                authorization_id[:8], corrections_used, max_corr,
            )
            raise PolicyViolationError(
                f"Correction budget exceeded: {corrections_used}/{max_corr}"
            )

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_record(self, filepath: str) -> dict[str, Any]:
        """Load and parse a JSON record file.

        Args:
            filepath: Absolute path to the .json record file.

        Returns:
            Parsed record dict.

        Raises:
            RecordNotFoundError: If the file does not exist or is corrupt.
        """
        if not os.path.isfile(filepath):
            raise RecordNotFoundError(f"Record file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            raise RecordNotFoundError(
                f"Failed to load record from {filepath}: {exc}"
            ) from exc

    def _save_record(self, record: dict[str, Any], filepath: str) -> None:
        """Atomically write a JSON record file (temp + rename).

        File permissions are set to 0o640.

        Args:
            record: The authorization record dict to save.
            filepath: Absolute path for the output file.
        """
        dirpath = os.path.dirname(filepath)
        # Atomic write via temp file in the same directory + rename
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix="auth_", dir=dirpath
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, indent=2)
            # Set permissions to 0o640 before rename
            os.chmod(tmp_path, 0o640)
            os.rename(tmp_path, filepath)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _is_duplicate_active(
        self, owner: str, name: str, pr_number: int
    ) -> bool:
        """Check whether an active authorization already exists for this repo+PR.

        Args:
            owner: Repository owner.
            name: Repository name.
            pr_number: PR number.

        Returns:
            True if a matching active record exists.
        """
        auth_path = Path(self._auth_dir)
        if not auth_path.is_dir():
            return False

        for filepath in auth_path.glob("*.json"):
            try:
                record = self._load_record(str(filepath))
            except (RecordNotFoundError, json.JSONDecodeError):
                continue

            if (
                record.get("repository_owner") == owner
                and record.get("repository_name") == name
                and record.get("pr_number") == pr_number
                and record.get("status") == "active"
            ):
                return True

        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the authorization policy tool.

    Usage::

        fr-auth authorize --repo OWNER/NAME --pr N --branch BRANCH --sha SHA [options]
        fr-auth list
        fr-auth revoke --id AUTHORIZATION_ID

    Returns:
        0 on success, 1 on error (errors written to stderr).
    """
    parser = argparse.ArgumentParser(
        prog="fr-auth",
        description="Frontier Review operator authorization CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # authorize subcommand
    auth_parser = subparsers.add_parser("authorize", help="Create a new authorization")
    auth_parser.add_argument(
        "--repo", required=True, dest="repo",
        help="Repository in OWNER/NAME format (e.g. caseclub-dev/hermes-frontier-review-test)"
    )
    auth_parser.add_argument(
        "--pr", required=True, type=int, dest="pr_number",
        help="Pull request number"
    )
    auth_parser.add_argument(
        "--branch", required=True, dest="branch",
        help="Feature branch name (hermes/* pattern)"
    )
    auth_parser.add_argument(
        "--sha", required=True, dest="sha",
        help="40-character lowercase hex SHA"
    )
    auth_parser.add_argument(
        "--max", type=int, default=3, dest="max_corrections",
        help="Maximum corrections allowed [1-10] (default: 3)"
    )
    auth_parser.add_argument(
        "--ttl", type=float, default=72.0, dest="ttl_hours",
        help="Time-to-live in hours [1-720] (default: 72)"
    )

    # list subcommand
    subparsers.add_parser("list", help="List active authorizations")

    # revoke subcommand
    rev_parser = subparsers.add_parser("revoke", help="Revoke an authorization")
    rev_parser.add_argument(
        "--id", required=True, dest="authorization_id",
        help="Authorization ID (UUID)"
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 1

    try:
        policy = AuthPolicy()

        if args.command == "authorize":
            # Parse OWNER/NAME from --repo
            repo_parts = args.repo.split("/")
            if len(repo_parts) != 2 or not repo_parts[0] or not repo_parts[1]:
                print(json.dumps({"error": "repo must be in OWNER/NAME format"}), file=sys.stderr)
                return 1
            owner, name = repo_parts

            # Validate PR number range
            if args.pr_number < 1 or args.pr_number > 999999:
                print(json.dumps({"error": "pr_number must be between 1 and 999999"}), file=sys.stderr)
                return 1

            record = policy.create(
                repository_owner=owner,
                repository_name=name,
                pr_number=args.pr_number,
                head_branch=args.branch,
                initial_remote_sha=args.sha,
                max_corrections=args.max_corrections,
                ttl_hours=args.ttl_hours,
            )
            print(json.dumps(record, indent=2))
            return 0

        elif args.command == "list":
            records = policy.list_active()
            print(json.dumps(records, indent=2))
            return 0

        elif args.command == "revoke":
            record = policy.revoke(args.authorization_id)
            print(json.dumps(record, indent=2))
            return 0

    except ValidationError as exc:
        print(json.dumps({"error": f"validation error: {exc}"}), file=sys.stderr)
        return 1
    except RecordNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except PolicyViolationError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except AuthorizationError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    # Should be unreachable — all command paths return above
    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
