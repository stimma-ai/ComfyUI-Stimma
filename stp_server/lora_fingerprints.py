"""Cheap, cached content fingerprints for LoRA files (STP `file_fingerprints`).

A fingerprint identifies a file's weights, so anything a host attaches to it
survives moving and renaming the file. It samples a few chunks instead of
reading the whole file: a library can be hundreds of GB on network storage.

Hashing runs on one background thread. Lookups only read memory, so they are
safe to answer from the event loop and never wait for hashing.
"""

import hashlib
import json
import logging
import os
import struct
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256 * 1024
CHUNK_COUNT = 4
_MAX_HEX = 32
_CACHE_VERSION = 1
_SAVE_EVERY_SECONDS = 10.0
_RESCAN_AFTER_SECONDS = 60.0
# A name the last scan didn't find stays null until this long after that scan.
_MISSING_RESCAN_SECONDS = 5.0
_SLEEP_BETWEEN_FILES = 0.002


def fingerprint_file(path: str) -> Optional[str]:
    """Return a file's `stfp1` fingerprint, or None if it can't have one.

    Hashes the safetensors data section only, so editing a file's metadata
    keeps its identity.
    """
    if not path.lower().endswith(".safetensors"):
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            start = 8 + struct.unpack("<Q", raw)[0]
            data_size = size - start
            if data_size <= 0:
                return None
            digest = hashlib.sha256(struct.pack("<Q", data_size))
            if data_size <= CHUNK_COUNT * CHUNK_SIZE:
                f.seek(start)
                digest.update(f.read())
            else:
                for i in range(CHUNK_COUNT):
                    f.seek(start + ((data_size - CHUNK_SIZE) * i) // (CHUNK_COUNT - 1))
                    digest.update(f.read(CHUNK_SIZE))
    except OSError:
        return None
    return f"stfp1:{digest.hexdigest()[:_MAX_HEX]}"


def pair_fingerprint(high: Optional[str], low: Optional[str]) -> Optional[str]:
    """Fingerprint a paired LoRA from its two files. Null if either is null."""
    if not high or not low:
        return None
    joined = "\n".join(sorted((high, low))).encode()
    return f"stfp1p:{hashlib.sha256(joined).hexdigest()[:_MAX_HEX]}"


def _default_cache_path() -> str:
    import folder_paths

    return os.path.join(folder_paths.get_user_directory(), "default", "stimma", "lora-fingerprints.json")


def _list_lora_files() -> List[Tuple[str, Optional[str]]]:
    import folder_paths

    return [(name, folder_paths.get_full_path("loras", name)) for name in folder_paths.get_filename_list("loras")]


class LoraFingerprints:
    """Background fingerprinting of ComfyUI's LoRA list, cached on disk."""

    def __init__(self, cache_path: Optional[str] = None, list_files=_list_lora_files):
        self._cache_path = cache_path
        self._list_files = list_files
        self._lock = threading.Lock()
        self._entries: Dict[str, dict] = {}
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._scanning = False
        self._last_complete: Optional[float] = None
        # Names the running scan has re-checked. Until then a cached value may
        # be stale (an overwritten checkpoint), so it counts as pending.
        self._validated: set = set()

    def kick(self) -> None:
        """Ask the worker to scan. Never blocks."""
        with self._lock:
            self._validated = set()
            self._scanning = True
            self._start_worker()
        self._wake.set()

    def _start_worker(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="stimma-lora-fingerprints", daemon=True)
            self._thread.start()

    def lookup(self, names: List[str]) -> Tuple[Dict[str, Optional[str]], int]:
        """Return ({name: fingerprint or None}, pending count) from memory."""
        now = time.monotonic()
        with self._lock:
            age = None if self._last_complete is None else now - self._last_complete
            missing = any(name not in self._entries for name in names)
            if not self._scanning and (
                age is None or age > _RESCAN_AFTER_SECONDS or (missing and age >= _MISSING_RESCAN_SECONDS)
            ):
                self._validated = set()
                self._scanning = True
                self._start_worker()
                self._wake.set()
            scanning = self._scanning
            result: Dict[str, Optional[str]] = {}
            pending = 0
            for name in names:
                entry = self._entries.get(name)
                result[name] = entry.get("fingerprint") if entry else None
                if scanning and name not in self._validated:
                    pending += 1
        return result, pending

    def _run(self) -> None:
        self._load()
        while True:
            self._wake.wait()
            self._wake.clear()
            try:
                self._scan()
            except Exception:
                logger.exception("LoRA fingerprint scan failed")
            with self._lock:
                self._scanning = self._wake.is_set()
                self._last_complete = time.monotonic()

    def _scan(self) -> None:
        with self._lock:
            self._validated = set()
        files = self._list_files()
        seen = set()
        dirty = False
        last_save = time.monotonic()
        for name, path in files:
            seen.add(name)
            try:
                stat = os.stat(path) if path else None
            except OSError:
                stat = None
            if stat is None:
                with self._lock:
                    dirty = self._entries.pop(name, None) is not None or dirty
                    self._validated.add(name)
                continue
            with self._lock:
                entry = self._entries.get(name)
            if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
                with self._lock:
                    self._validated.add(name)
                continue
            fingerprint = fingerprint_file(path)
            with self._lock:
                self._entries[name] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "fingerprint": fingerprint,
                }
                self._validated.add(name)
            dirty = True
            if time.monotonic() - last_save > _SAVE_EVERY_SECONDS:
                self._save()
                dirty = False
                last_save = time.monotonic()
            time.sleep(_SLEEP_BETWEEN_FILES)
        with self._lock:
            for name in [n for n in self._entries if n not in seen]:
                del self._entries[name]
                dirty = True
        if dirty:
            self._save()

    def _resolved_cache_path(self) -> Optional[str]:
        if self._cache_path is None:
            try:
                self._cache_path = _default_cache_path()
            except Exception:
                logger.debug("No ComfyUI user directory for the LoRA fingerprint cache", exc_info=True)
        return self._cache_path

    def _load(self) -> None:
        path = self._resolved_cache_path()
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("Ignoring an unreadable LoRA fingerprint cache")
            return
        entries = data.get("entries") if isinstance(data, dict) and data.get("version") == _CACHE_VERSION else None
        if isinstance(entries, dict):
            with self._lock:
                self._entries = {k: v for k, v in entries.items() if isinstance(v, dict)}

    def _save(self) -> None:
        path = self._resolved_cache_path()
        if not path:
            return
        with self._lock:
            payload = {"version": _CACHE_VERSION, "entries": dict(self._entries)}
        tmp = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".lora-fingerprints-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except OSError:
            logger.warning("Could not write the LoRA fingerprint cache", exc_info=True)
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
