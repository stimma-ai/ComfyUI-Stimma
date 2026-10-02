"""Prepare the isolated runtime; called by the connector CLI and manager."""

from contextlib import contextmanager
import os
import shutil
import subprocess
import sys

from audio_runtime import MOSS_SOURCE, PLUGIN_ROOT, runtime_python, runtime_ready

UV_VERSION = "0.12.7"


@contextmanager
def installation_lock():
    # All local ComfyUI workers share this runtime. OS locks release even when
    # an installer crashes or ComfyUI is restarted during setup.
    path = PLUGIN_ROOT / ".runtimes/moss-install.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            import time
            lock.seek(0)
            lock.write(b"\0")
            lock.flush()
            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.5)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def install():
    if os.environ.get("STIMMA_MOSS_PYTHON"):
        if not runtime_ready():
            raise RuntimeError("The configured external MOSS runtime is missing.")
        return
    print("Preparing model runtime", flush=True)
    with installation_lock():
        if runtime_ready():
            return
        python = runtime_python()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
            env.pop(key, None)

        def run(command):
            subprocess.run(command, check=True, env=env)

        uv = shutil.which("uv")
        if uv:
            installer = [uv]
        else:
            # Bootstrap into a private directory, never ComfyUI's environment.
            bootstrap = PLUGIN_ROOT / ".runtimes/installer"
            print("Preparing runtime installer", flush=True)
            run([sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "--upgrade", f"uv=={UV_VERSION}"])
            installer = [sys.executable, "-c",
                         "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('uv',run_name='__main__')",
                         str(bootstrap)]
        marker = python.parent.parent / ".stimma-moss-source"
        marker.unlink(missing_ok=True)
        if not python.is_file():
            print("Preparing Python 3.12", flush=True)
            run(installer + ["venv", "--python", "3.12", "--python-preference", "only-managed", str(python.parent.parent)])
        source = f"moss-soundeffect-v2[torch-cu128] @ git+https://github.com/OpenMOSS/MOSS-TTS.git@{MOSS_SOURCE}#subdirectory=moss_soundeffect_v2"
        print("Installing model dependencies", flush=True)
        run(installer + ["pip", "install", "--python", str(python),
                         "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                         "--index-strategy", "unsafe-best-match", source])
        print("Checking model runtime", flush=True)
        run([str(python), "-c", "from moss_soundeffect_v2 import MossSoundEffectPipeline"])
        marker.write_text(MOSS_SOURCE + "\n")
        print("Model runtime ready", flush=True)
