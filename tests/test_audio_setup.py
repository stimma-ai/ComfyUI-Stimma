"""Audio dependencies follow validation, Get ready, Activity, and retry paths."""

import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audio_runtime as audio
from stp_server.discovery import _validate_workflow
from stp_server.manage import manager as manager_module, runtimes, resolve
from stp_server.manage.manager import Manager
from stp_server.manage.ops import OperationRegistry
from stp_server.manage.routes import make_routes
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from stp_server.provider import StimmaPluginProvider

spec = importlib.util.spec_from_file_location('install_moss_runtime', ROOT / 'tools/install_moss_runtime.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

PROMPT = {'1': {'class_type': 'StimmaMossSoundEffect', 'inputs': {}}}
INFO = {'StimmaMossSoundEffect': {'input': {'required': {}}}}


def folder_module(base):
    models = Path(base) / 'models'
    models.mkdir(exist_ok=True)
    return types.SimpleNamespace(models_dir=str(models), base_path=base,
                                 get_folder_paths=lambda name: [str(models / name)])


def write_models(base):
    root = audio.model_directory(Path(base) / 'models')
    for name in audio.MOSS_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'model')


class AudioDependencyTests(unittest.TestCase):
    def test_files_and_runtime_are_distinct_repairable_issues(self):
        with tempfile.TemporaryDirectory() as base, patch.dict(sys.modules, {'folder_paths': folder_module(base)}), \
             patch.object(audio, 'runtime_ready', return_value=False):
            issues = []
            warnings = _validate_workflow(PROMPT, INFO, issues)
            self.assertEqual(len(warnings), len(audio.MOSS_FILES) + 1)
            self.assertEqual(sum(i['kind'] == 'missing_runtime' for i in issues), 1)
            self.assertEqual(sum(i['kind'] == 'missing_model' for i in issues), len(audio.MOSS_FILES))
            self.assertFalse(any('setup-audio' in w for w in warnings))
            for issue in issues:
                if issue['kind'] == 'missing_model':
                    source = resolve.resolve_source(issue['name'])
                    self.assertIn(audio.MOSS_REVISION, source['url'])
                    self.assertEqual(len(source['sha256']), 64)
                    self.assertGreater(source['size'], 0)
                    self.assertEqual(Path(resolve.dest_path_for(issue['name'], issue['folder'])),
                                     Path(base) / 'models' / issue['folder'] / issue['name'])
            write_models(base)
            issues = []
            _validate_workflow(PROMPT, INFO, issues)
            self.assertEqual([i['kind'] for i in issues], ['missing_runtime'])
            with patch.object(audio, 'runtime_ready', return_value=True):
                self.assertEqual(_validate_workflow(PROMPT, INFO), [])

    def test_readiness_fingerprint_tracks_pipeline_and_runtime(self):
        with tempfile.TemporaryDirectory() as base, patch.dict(sys.modules, {'folder_paths': folder_module(base)}), \
             patch.object(audio, 'runtime_ready', return_value=False):
            provider = object.__new__(StimmaPluginProvider)
            before = provider._snapshot_comfyui_deps()
            write_models(base)
            files_added = provider._snapshot_comfyui_deps()
            with patch.object(audio, 'runtime_ready', return_value=True):
                runtime_added = provider._snapshot_comfyui_deps()
            self.assertNotEqual(before, files_added)
            self.assertNotEqual(files_added, runtime_added)
            (audio.model_directory(Path(base) / 'models') / 'model_index.json').unlink()
            self.assertNotEqual(files_added, provider._snapshot_comfyui_deps())

    def test_stable_audio_uses_normal_model_catalog(self):
        for path in audio.SA3_FILES:
            source = resolve.resolve_source(Path(path).name)
            self.assertIsNotNone(source)
            self.assertEqual(source['folder'], path.split('/')[0])


class AudioManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folders = folder_module(self.temp.name)
        patcher = patch.dict(sys.modules, {'folder_paths': self.folders})
        patcher.start()
        self.addCleanup(patcher.stop)
        ready = patch.object(audio, 'runtime_ready', return_value=False)
        ready.start()
        self.addCleanup(ready.stop)
        self.manager = object.__new__(Manager)
        self.manager.ops = OperationRegistry(Path(self.temp.name) / 'ops.json')
        self.manager.provider = MagicMock(object_info=INFO)
        self.manager.provider.discover_and_register_tools = AsyncMock()
        self.manager.provider.notify_tools_changed = AsyncMock()
        self.manager.provider.push_state = AsyncMock()
        self.manager.instances = types.SimpleNamespace(statuses=[])
        self.manager.downloads = MagicMock()
        self.manager.downloads.enqueue.side_effect = lambda **kw: self.manager.ops.create('download', kw['filename'], meta=kw, group=kw['group'])
        self.manager._session = None
        issues = []
        warnings = _validate_workflow(PROMPT, INFO, issues)
        self.workflow = types.SimpleNamespace(tool_info={'slug': 'moss', 'display_name': 'MOSS'}, api_prompt=PROMPT,
                                              issues=issues, warnings=warnings, model_hints={}, file_path='moss.json')
        self.manager._workflows = [self.workflow]
        self.probe = patch.object(manager_module, 'probe_hf', new=AsyncMock(return_value={}))
        self.probe.start()
        self.addCleanup(self.probe.stop)

    async def test_get_ready_queues_models_and_one_deduplicated_runtime(self):
        with patch.object(self.manager, '_run_runtime_install', new=AsyncMock()) as install:
            result = await self.manager.start_setup('moss')
            second = await self.manager.start_setup('moss')
            await asyncio.sleep(0)
        self.assertEqual(len(result['plan']['downloads']), len(audio.MOSS_FILES))
        self.assertFalse(result['plan']['blockers'])
        self.assertEqual(len(result['plan']['runtimes']), 1)
        runtime_ops = [op for op in self.manager.ops.all() if op.kind == 'install_runtime']
        self.assertEqual(len(runtime_ops), 1)
        self.assertEqual(install.await_count, 1)
        self.assertIn(runtime_ops[0].id, second['queued'])
        self.assertTrue(all(op.group == 'setup:moss' for op in self.manager.ops.all()))
        detail = await self.manager.workflow_detail('moss')
        self.assertEqual(len(detail['models']), len(audio.MOSS_FILES))
        self.assertFalse(detail['runtimes'][0]['installed'])

    async def test_runtime_only_setup_does_not_download_models(self):
        write_models(self.temp.name)
        self.workflow.issues = []
        _validate_workflow(PROMPT, INFO, self.workflow.issues)
        with patch.object(self.manager, '_run_runtime_install', new=AsyncMock()):
            result = await self.manager.start_setup('moss')
            await asyncio.sleep(0)
        self.assertEqual(result['plan']['downloads'], [])
        self.assertEqual(len(result['queued']), 1)
        self.manager.downloads.enqueue.assert_not_called()

    async def test_completion_refreshes_tools_without_restart(self):
        op = self.manager.ops.create('install_runtime', 'Runtime', meta={'runtime_id': audio.MOSS_RUNTIME, 'target': 'local'})
        with patch.object(runtimes, 'install', new=AsyncMock()):
            await self.manager._run_runtime_install(op)
        self.assertEqual(op.state, 'done')
        self.manager.provider.discover_and_register_tools.assert_awaited_once_with(force=True)
        self.manager.provider.notify_tools_changed.assert_awaited_once()

    async def test_failed_install_can_be_retried(self):
        op = self.manager.ops.create('install_runtime', 'Runtime', meta={'runtime_id': audio.MOSS_RUNTIME, 'target': 'local'})
        with patch.object(runtimes, 'install', new=AsyncMock(side_effect=RuntimeError('network unavailable'))):
            await self.manager._run_runtime_install(op)
        self.assertEqual(op.state, 'failed')
        self.assertEqual(op.fix, {'action': 'retry'})
        self.assertIn('network unavailable', op.error)
        self.manager.provider.discover_and_register_tools.assert_not_awaited()
        with patch.object(runtimes, 'install', new=AsyncMock()):
            await self.manager._run_runtime_install(op)
        self.assertEqual(op.state, 'done')
        self.assertIsNone(op.error)

    async def test_local_ready_still_plans_missing_peer_dependencies(self):
        write_models(self.temp.name)
        self.workflow.issues = []
        self.workflow.warnings = []
        dependency = audio.moss_dependencies(self.folders.models_dir)[0]
        peer_status = {'id': audio.MOSS_RUNTIME, 'title': 'Runtime', 'installed': False, 'installable': True,
                       'models': [{**dependency, 'installed': False}]}
        with patch.object(audio, 'runtime_ready', return_value=True), \
             patch.object(self.manager, '_peer_runtime_statuses', new=AsyncMock(return_value=[('worker:8188', peer_status)])):
            plan = await self.manager.plan_setup('moss')
        self.assertEqual(plan['runtimes'][0]['target'], 'worker:8188')
        self.assertEqual(len(plan['downloads']), 1)
        self.assertTrue(plan['downloads'][0]['already_present'])
        self.assertEqual(plan['downloads'][0]['peers'], ['worker:8188'])

    async def test_fanout_runs_once_per_host_and_waits_for_activity(self):
        self.manager.instances.statuses = [types.SimpleNamespace(addr='worker:8188', local=False, healthy=True),
                                           types.SimpleNamespace(addr='worker:8189', local=False, healthy=True)]
        with patch.object(self.manager, '_run_peer_download', new=AsyncMock()) as run:
            ids = await self.manager._fanout_download({'filename': 'file', 'url': 'https://example.test/file'}, 'moss')
            repeated = await self.manager._fanout_download({'filename': 'file', 'url': 'https://example.test/file'}, 'moss')
            await asyncio.sleep(0)
        self.assertEqual(len(ids), 1)
        self.assertEqual(repeated, ids)
        run.assert_awaited_once()
        self.assertEqual(self.manager.ops.get(ids[0]).state, 'queued')

    async def test_external_missing_runtime_is_a_visible_blocker(self):
        with patch.dict(os.environ, {'STIMMA_MOSS_PYTHON': '/missing/external/python'}):
            plan = await self.manager.plan_setup('moss')
        self.assertFalse(plan['runtimes'][0]['installable'])
        self.assertEqual(plan['blockers'][0]['kind'], 'runtime')

    async def test_stable_audio_does_not_queue_a_runtime(self):
        self.workflow.api_prompt = {'1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {}}}
        self.workflow.issues = [{'kind': 'missing_model', 'name': 'stable_audio_3_medium.safetensors', 'folder': 'checkpoints'}]
        result = await self.manager.start_setup('moss')
        self.assertEqual(result['plan']['runtimes'], [])
        self.assertEqual(len(result['queued']), 1)
        self.assertEqual(self.manager.ops.all()[0].kind, 'download')


class AudioSetupApiTests(unittest.IsolatedAsyncioTestCase):
    setUp = AudioManagerTests.setUp
    # Exercise the actual routes as well as the manager, with tiny fixtures in
    # place of multi-GB weights and the CUDA installer.
    async def asyncSetUp(self):
        app = web.Application()
        app.add_routes(make_routes(self.manager))
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_runtime_only_get_ready_and_activity_retry(self):
        write_models(self.temp.name)
        self.workflow.issues = []
        _validate_workflow(PROMPT, INFO, self.workflow.issues)
        plan = await self.client.get('/stp-v1/manage/api/workflows/moss/plan')
        data = await plan.json()
        self.assertEqual(data['downloads'], [])
        self.assertEqual(len(data['runtimes']), 1)
        with patch.object(runtimes, 'install', new=AsyncMock(side_effect=RuntimeError('offline'))):
            response = await self.client.post('/stp-v1/manage/api/workflows/moss/setup', json={})
            self.assertEqual(response.status, 200)
            data = await response.json()
            await asyncio.sleep(0)
        op_id = data['queued'][0]
        self.assertEqual(self.manager.ops.get(op_id).state, 'failed')
        with patch.object(runtimes, 'install', new=AsyncMock()):
            response = await self.client.post(f'/stp-v1/manage/api/activity/{op_id}/retry', json={})
            self.assertEqual(response.status, 200)
            await asyncio.sleep(0)
        self.assertEqual(self.manager.ops.get(op_id).state, 'done')
        self.assertIsNone(self.manager.ops.get(op_id).error)

    async def test_peer_status_unknown_id_and_csrf_guard(self):
        response = await self.client.get(f'/stp-v1/manage/api/runtimes/{audio.MOSS_RUNTIME}')
        self.assertEqual(response.status, 200)
        self.assertEqual(len((await response.json())['models']), len(audio.MOSS_FILES))
        response = await self.client.get('/stp-v1/manage/api/runtimes/arbitrary')
        self.assertEqual(response.status, 404)
        response = await self.client.post('/stp-v1/manage/api/runtimes/arbitrary/install', json={})
        self.assertEqual(response.status, 404)
        response = await self.client.post(f'/stp-v1/manage/api/runtimes/{audio.MOSS_RUNTIME}/install', json={},
                                          headers={'Origin': 'https://unrelated.test'})
        self.assertEqual(response.status, 403)

    async def test_unknown_or_duplicate_install_does_not_start_extra_tasks(self):
        with patch.object(self.manager, '_run_runtime_install', new=AsyncMock()) as install:
            first = await self.client.post(f'/stp-v1/manage/api/runtimes/{audio.MOSS_RUNTIME}/install', json={'slug': 'moss'})
            data = await first.json()
            second = await self.client.post(f'/stp-v1/manage/api/runtimes/{audio.MOSS_RUNTIME}/install', json={'slug': 'moss'})
            self.assertEqual((await second.json())['operation']['id'], data['operation']['id'])
            op_id = data['operation']['id']
            response = await self.client.post(f'/stp-v1/manage/api/activity/{op_id}/retry', json={})
            self.assertEqual(response.status, 409)
        install.assert_awaited_once()


class PeerActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_peer_progress_is_tracked_until_completion_and_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as base:
            manager = object.__new__(Manager)
            manager.ops = OperationRegistry(Path(base) / 'operations.json')
            op = manager.ops.create('peer_download', 'File', meta={'target': 'worker:8188'})
            remote = {'id': 'peer-op', 'state': 'running', 'progress': .25, 'detail': 'Downloading'}
            response = MagicMock()
            response.__aenter__ = AsyncMock(return_value=response)
            response.__aexit__ = AsyncMock(return_value=False)
            response.json = AsyncMock(return_value={'operations': [{**remote, 'state': 'done'}]})
            session = MagicMock()
            session.get.return_value = response
            manager._get_session = AsyncMock(return_value=session)
            with patch.object(manager_module.asyncio, 'sleep', new=AsyncMock()):
                await manager._wait_peer_operation(op, remote)
            self.assertEqual(op.progress, .25)
            self.assertEqual(op.meta['peer_operation_id'], 'peer-op')
            session.get.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                await manager._wait_peer_operation(op, {**remote, 'state': 'failed', 'error': 'disk full'})


class RuntimeProcessTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX process-group cleanup')
    async def test_installer_cancellation_stops_its_process_group(self):
        started = asyncio.Event()
        block = asyncio.Event()
        async def readline():
            started.set()
            await block.wait()
        process = MagicMock(pid=12345, returncode=None)
        process.stdout.readline = readline
        process.wait = AsyncMock(return_value=-15)
        with patch.object(runtimes.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=process)) as spawn, \
             patch.object(runtimes.os, 'killpg') as terminate:
            task = asyncio.create_task(runtimes.install(audio.MOSS_RUNTIME, lambda line: None))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(spawn.call_args.args[1:], (str(ROOT / 'tools/stimma-comfy'), 'setup-runtime'))
        if os.name != 'nt':
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            terminate.assert_called_once_with(12345, runtimes.signal.SIGTERM)
        process.wait.assert_awaited_once()


class RuntimeInstallerTests(unittest.TestCase):
    def test_no_uv_bootstraps_privately_and_marks_ready_only_after_import(self):
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            python = root / '.runtimes/moss-soundeffect-v2/bin/python'
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                self.assertNotIn('PYTHONPATH', kwargs['env'])
                self.assertNotIn('PYTHONHOME', kwargs['env'])
                if 'venv' in command:
                    python.parent.mkdir(parents=True)
                    python.write_text('')
                if command[-1] == 'from moss_soundeffect_v2 import MossSoundEffectPipeline':
                    self.assertFalse((python.parent.parent / '.stimma-moss-source').exists())
            with patch.object(audio, 'PLUGIN_ROOT', root), patch.object(installer, 'PLUGIN_ROOT', root), \
                 patch.object(audio, 'runtime_python', return_value=python), \
                 patch.object(installer, 'runtime_python', return_value=python), \
                 patch.object(installer.shutil, 'which', return_value=None), \
                 patch.object(installer.subprocess, 'run', side_effect=run), \
                 patch.dict(os.environ, {'PYTHONPATH': 'host-packages', 'PYTHONHOME': 'host-python'}, clear=True):
                installer.install()
                self.assertTrue(audio.runtime_ready())
                installer.install()
            self.assertEqual(len(calls), 4)
            bootstrap = calls[0]
            self.assertIn('--target', bootstrap)
            self.assertIn(str(root / '.runtimes/installer'), bootstrap)
            self.assertIn('only-managed', calls[1])
            self.assertIn(str(python), calls[2])
            self.assertIn(audio.MOSS_SOURCE, calls[2][-1])

    def test_failed_import_never_marks_runtime_ready(self):
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            python = root / '.runtimes/moss-soundeffect-v2/bin/python'
            python.parent.mkdir(parents=True)
            python.write_text('')
            def run(command, **kwargs):
                if command[0] == str(python):
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(audio, 'PLUGIN_ROOT', root), patch.object(installer, 'PLUGIN_ROOT', root), \
                 patch.object(audio, 'runtime_python', return_value=python), \
                 patch.object(installer, 'runtime_python', return_value=python), \
                 patch.object(installer.shutil, 'which', return_value='uv'), \
                 patch.object(installer.subprocess, 'run', side_effect=run), \
                 patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(subprocess.CalledProcessError):
                    installer.install()
                self.assertFalse(audio.runtime_ready())


if __name__ == '__main__':
    unittest.main()
