"""Operations registry — everything the Activity tab shows.

An Operation is a download, a node install, a restart, or an update. Each has
progress, a state, and (when failed) an error + a recovery hint. Terminal
operations are kept (bounded) so the user can see what happened, and the
registry is persisted to disk so a restart doesn't lose the picture.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
STATE_PAUSED = "paused"

TERMINAL = {STATE_DONE, STATE_FAILED, STATE_CANCELLED}


@dataclass
class Operation:
    id: str
    kind: str                       # download | install_node | restart | update | verify
    title: str                      # user-facing
    state: str = STATE_QUEUED
    progress: Optional[float] = None  # 0..1
    detail: Optional[str] = None      # one-line status ("9.8 of 24 GB · 184 MB/s")
    error: Optional[str] = None       # user-facing failure text
    error_kind: Optional[str] = None  # gated | auth | not_found | network | disk | verify | other
    fix: Optional[dict] = None        # {"action": "hf_license"|"hf_token"|"retry"|"restart"|"add_url", ...}
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # kind-specific (filename, size, bytes, url, targets, workflow slugs)
    group: Optional[str] = None       # e.g. "setup:<slug>" — ops queued together

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class OperationRegistry:
    def __init__(self, persist_path: Path, max_terminal: int = 60):
        self._ops: Dict[str, Operation] = {}
        self._persist_path = persist_path
        self._max_terminal = max_terminal
        self._listeners: List[Callable[[Operation], Any]] = []
        self._load()

    # -- persistence --
    def _load(self):
        try:
            if self._persist_path.exists():
                data = json.loads(self._persist_path.read_text())
                for d in data.get("ops", []):
                    op = Operation(**{k: v for k, v in d.items() if k in Operation.__dataclass_fields__})
                    # A process restart interrupts everything non-terminal.
                    if op.state in (STATE_RUNNING, STATE_QUEUED, STATE_PAUSED):
                        if op.kind == "download":
                            op.state = STATE_QUEUED  # resumable — re-queue
                            op.detail = "Waiting to resume"
                        elif op.kind == "restart":
                            op.state = STATE_DONE
                            op.finished_at = time.time()
                            op.detail = "Restarted"
                        else:
                            op.state = STATE_FAILED
                            op.error = "Interrupted by a restart"
                            op.error_kind = "other"
                            op.fix = {"action": "retry"}
                    self._ops[op.id] = op
        except Exception:
            logger.warning("could not load operations state", exc_info=True)

    def save(self):
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ops": [o.to_dict() for o in self._ops.values()]}))
            os.replace(tmp, self._persist_path)
        except Exception:
            logger.debug("could not persist operations", exc_info=True)

    # -- listeners --
    def on_change(self, cb: Callable[[Operation], Any]):
        self._listeners.append(cb)

    def _emit(self, op: Operation):
        for cb in list(self._listeners):
            try:
                r = cb(op)
                if asyncio.iscoroutine(r):
                    asyncio.ensure_future(r)
            except Exception:
                logger.debug("operation listener failed", exc_info=True)

    # -- CRUD --
    def create(self, kind: str, title: str, *, meta: Optional[dict] = None, group: Optional[str] = None,
               op_id: Optional[str] = None) -> Operation:
        op = Operation(id=op_id or uuid.uuid4().hex[:12], kind=kind, title=title, meta=meta or {}, group=group)
        self._ops[op.id] = op
        self._trim()
        self.save()
        self._emit(op)
        return op

    def get(self, op_id: str) -> Optional[Operation]:
        return self._ops.get(op_id)

    def all(self) -> List[Operation]:
        return sorted(self._ops.values(), key=lambda o: o.created_at, reverse=True)

    def active(self) -> List[Operation]:
        return [o for o in self._ops.values() if o.state not in TERMINAL]

    def find(self, kind: str, **meta_match) -> Optional[Operation]:
        for o in self._ops.values():
            if o.kind == kind and o.state not in TERMINAL and all(o.meta.get(k) == v for k, v in meta_match.items()):
                return o
        return None

    def update(self, op: Operation, **changes) -> Operation:
        for k, v in changes.items():
            setattr(op, k, v)
        if op.state == STATE_RUNNING and op.started_at is None:
            op.started_at = time.time()
        if op.state in TERMINAL and op.finished_at is None:
            op.finished_at = time.time()
        self._emit(op)
        return op

    def remove(self, op_id: str) -> bool:
        op = self._ops.pop(op_id, None)
        if op:
            self.save()
            return True
        return False

    def clear_done(self):
        for oid in [o.id for o in self._ops.values() if o.state in TERMINAL]:
            self._ops.pop(oid, None)
        self.save()

    def _trim(self):
        terminal = sorted([o for o in self._ops.values() if o.state in TERMINAL], key=lambda o: o.finished_at or 0)
        while len(terminal) > self._max_terminal:
            victim = terminal.pop(0)
            self._ops.pop(victim.id, None)
