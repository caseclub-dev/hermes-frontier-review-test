"""Coder delegation adapter for Frontier Review.

When a review returns CHANGES_REQUIRED or NEEDS_EVIDENCE, Orchestrator
delegates the findings back to Coder via this adapter. The adapter
builds a structured correction request that Coder can process.

M1 note: actual delegate_task calls are NOT made in M1. This module
provides the interface and request builder only, for testing the
serialization format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class DelegationRequest:
    """A structured request to Coder for code corrections.

    This is the serializable envelope sent from Orchestrator → Coder
    when a review cycle requires changes.
    """

    task_id: str
    verdict: str  # CHANGES_REQUIRED | NEEDS_EVIDENCE
    summary: str
    findings: list[dict[str, str]] = field(default_factory=list)
    requirements_met: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    deadline_note: str = ""  # Optional constraint for Coder
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> "DelegationRequest":
        if isinstance(data, str):
            data = json.loads(data)
        # Filter only known fields
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CorrectionTicket:
    """A specific correction item extracted from a review finding."""

    task_id: str
    finding_title: str
    severity: str  # HIGH | MEDIUM | LOW
    category: str  # correctness | security | performance
    location: str  # file:line reference or "global"
    description: str
    suggested_correction: str = ""


def build_delegation_request(
    task_id: str,
    verdict: str,
    summary: str,
    findings: list[dict[str, str]],
    requirements_met: list[str] | None = None,
    suggestions: list[str] | None = None,
) -> DelegationRequest:
    """Build a DelegationRequest from review results.

    This function is the interface between Orchestrator and Coder.
    In M1 it only builds the request object; no actual delegation occurs.
    """
    return DelegationRequest(
        task_id=task_id,
        verdict=verdict,
        summary=summary,
        findings=findings or [],
        requirements_met=requirements_met or [],
        suggestions=suggestions or [],
    )


def extract_correction_tickets(
    task_id: str,
    findings: list[dict[str, str]],
) -> list[CorrectionTicket]:
    """Extract individual CorrectionTickets from review findings.

    Each HIGH-severity finding becomes a mandatory correction ticket.
    MEDIUM/LOW are advisory but still tracked.
    """
    tickets = []
    for f in (findings or []):
        if not isinstance(f, dict):
            continue
        # Find the description and title — they should exist based on Finding model
        title = f.get("title", "")
        severity = f.get("severity", "LOW")
        category = f.get("category", "general")
        location = f.get("location", "unknown")
        description = f.get("description", "")

        tickets.append(CorrectionTicket(
            task_id=task_id,
            finding_title=title,
            severity=severity,
            category=category,
            location=location,
            description=description,
            suggested_correction=f.get("suggested_correction", ""),
        ))
    return tickets


def serialize_for_coder(request: DelegationRequest) -> str:
    """Serialize a delegation request in the format Coder expects.

    In M1 this returns a JSON string. In later phases it may add
    context about which files to edit, test commands to run, etc.
    """
    return request.to_json()
