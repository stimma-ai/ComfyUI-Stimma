"""Isolated model runtimes prepared through the connector's first-party CLI."""

import asyncio
import os
import signal
import sys

from audio_runtime import MOSS_RUNTIME, PLUGIN_ROOT, moss_dependencies, runtime_status


def status(runtime_id):
    result = runtime_status(runtime_id)
    import folder_paths
    result["models"] = moss_dependencies(folder_paths.models_dir)
    return result


async def install(runtime_id, log):
    if runtime_id != MOSS_RUNTIME:
        raise KeyError(runtime_id)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(PLUGIN_ROOT / "tools/stimma-comfy"), "setup-runtime",
        env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            log(line.decode(errors="replace").rstrip())
        code = await process.wait()
        if code:
            raise RuntimeError(f"Runtime installer exited with code {code}")
        if not runtime_status(runtime_id)["installed"]:
            raise RuntimeError("Runtime installation did not pass validation")
    finally:
        if process.returncode is None:
            # Stop the installer and its uv/pip children together. Otherwise a
            # restart can release the lock while a child still mutates the venv.
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
                                                               stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await killer.wait()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
