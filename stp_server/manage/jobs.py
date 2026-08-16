"""Registry of ComfyUI prompts we queued, so the manager can label them."""

import time
from typing import Dict, Optional

_jobs: Dict[str, dict] = {}


def register(prompt_id: str, title: str, request_id: str, addr: Optional[str] = None) -> None:
    _jobs[prompt_id] = {"title": title, "request_id": request_id, "addr": addr,
                        "started_at": time.time(), "progress": 0.0}


def progress(prompt_id: str, value: float) -> None:
    j = _jobs.get(prompt_id)
    if j is not None:
        j["progress"] = value


def unregister(prompt_id: Optional[str]) -> None:
    if prompt_id:
        _jobs.pop(prompt_id, None)


def snapshot() -> Dict[str, dict]:
    return dict(_jobs)


def request_id_for(prompt_id: str) -> Optional[str]:
    j = _jobs.get(prompt_id)
    return j["request_id"] if j else None
