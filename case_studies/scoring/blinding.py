"""
Cryptographic blinding engine for anonymised multi-system evaluation.

Protocol
--------
1.  A fresh session UUID is generated (or supplied) for each scoring run.
2.  For every task, the three system answers are shuffled using a
    deterministic HMAC-based seed:  seed = HMAC-SHA256(session_id, task_id).
    This ensures the permutation is reproducible given the session key but
    unpredictable without it.
3.  Shuffled answers are labelled "Response A", "Response B", "Response C".
4.  The full mapping (system -> label per task) is stored in a sealed
    manifest JSON that is written to disk *before* any judge is called,
    then used for de-anonymisation after all scores are collected.
5.  System-identifying strings are scrubbed from answer text using a
    configurable deny-list regex.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


RESPONSE_LABELS: list[str] = ["Response A", "Response B", "Response C"]

_SYSTEM_NAME_PATTERNS: list[str] = [
    r"\bBioClaw\b",
    r"\bBiomni\b",
    r"\bChatGPT\b",
    r"\bOpenClaude\b",
    r"\bGPT[-\s]?5\.5\b",
    r"\bgpt[-_]5\.5[-_]thinking[-_]extended\b",
    r"\bPLEAS\b",
    r"\bBioPLEAS\b",
    r"\bSystem\s*[123]\b",
]

_SCRUB_RE = re.compile("|".join(_SYSTEM_NAME_PATTERNS), re.IGNORECASE)


def _derive_seed(session_id: str, task_id: str) -> int:
    digest = hmac.new(
        session_id.encode(), task_id.encode(), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:8], "big")


def scrub_system_names(text: str) -> str:
    return _SCRUB_RE.sub("[SYSTEM]", text)


@dataclass
class BlindedResponse:
    label: str
    text: str


@dataclass
class TaskBlindingRecord:
    task_id: str
    mapping: dict[str, str]   # original system_id -> blinded label
    reverse: dict[str, str]   # blinded label -> original system_id


@dataclass
class BlindingManifest:
    session_id: str
    created_utc: str
    task_records: list[TaskBlindingRecord] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> BlindingManifest:
        with open(path) as f:
            data = json.load(f)
        records = [TaskBlindingRecord(**r) for r in data["task_records"]]
        return cls(
            session_id=data["session_id"],
            created_utc=data["created_utc"],
            task_records=records,
        )


def blind_answers(
    task_id: str,
    system_answers: dict[str, str],
    session_id: str,
) -> tuple[list[BlindedResponse], TaskBlindingRecord]:
    """Shuffle and anonymise answers for one task.

    Parameters
    ----------
    task_id : str
        E.g. "T01".
    system_answers : dict[str, str]
        Maps system identifier -> raw answer text.  Exactly 3 entries.
    session_id : str
        UUID for this scoring session.

    Returns
    -------
    blinded : list[BlindedResponse]
        Three responses labelled "Response A/B/C" in shuffled order.
    record : TaskBlindingRecord
        The mapping needed to de-anonymise later.
    """
    system_ids = sorted(system_answers.keys())
    if len(system_ids) != 3:
        raise ValueError(f"Expected 3 systems, got {len(system_ids)}: {system_ids}")

    seed = _derive_seed(session_id, task_id)
    import random
    rng = random.Random(seed)
    shuffled = list(system_ids)
    rng.shuffle(shuffled)

    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    blinded: list[BlindedResponse] = []

    for label, sys_id in zip(RESPONSE_LABELS, shuffled):
        clean_text = scrub_system_names(system_answers[sys_id])
        blinded.append(BlindedResponse(label=label, text=clean_text))
        mapping[sys_id] = label
        reverse[label] = sys_id

    record = TaskBlindingRecord(task_id=task_id, mapping=mapping, reverse=reverse)
    return blinded, record


def create_session(session_id: str | None = None) -> str:
    return session_id or uuid.uuid4().hex
