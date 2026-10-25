"""Provider routing and image transport tests; no paid generation calls."""
import base64
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from railways import claude_generation as claude
from railways import orchestrator as loop


class ClaudeGenerationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'frame.png'
        self.pixels = np.random.default_rng(4).integers(0, 256, (120, 240, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.path), self.pixels))
        self.images = [{'path': str(self.path), 'label': 'Original positive target'}]

    def test_request_embeds_images_in_order_and_disables_tools_without_bare(self):
        command, payload = claude.request('Task', self.images, 'sonnet', 'medium')
        self.assertEqual(command[0], 'claude')
        self.assertEqual(command[command.index('--tools') + 1], '')
        self.assertEqual(command[command.index('--setting-sources') + 1], '')
        self.assertIn('--strict-mcp-config', command)
        self.assertIn('--disable-slash-commands', command)
        self.assertIn('--no-session-persistence', command)
        self.assertNotIn('--bare', command)
        self.assertNotIn('--dangerously-skip-permissions', command)
        self.assertEqual(command[command.index('--output-format') + 1], 'stream-json')
        self.assertIn('--verbose', command)
        message = json.loads(payload)
        self.assertEqual(message['type'], 'user')
        content = message['message']['content']
        self.assertEqual(content[0]['text'], 'Task')
        self.assertIn('Image 1: Original positive target', content[1]['text'])
        self.assertEqual(base64.b64decode(content[2]['source']['data']), self.path.read_bytes())

    def test_large_images_are_compressed_without_resizing(self):
        with patch.object(claude, 'MAX_IMAGE_BYTES', 50000), patch.object(
            cv2, 'resize', side_effect=AssertionError('No downsampling')):
            block = claude.image_block(self.path)
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        data = base64.b64decode(block['source']['data'])
        self.assertLessEqual(len(data), 50000)
        self.assertEqual(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape, self.pixels.shape)

    def test_oversized_image_fails_instead_of_silently_dropping_or_resizing(self):
        with patch.object(claude, 'MAX_IMAGE_BYTES', 1):
            with self.assertRaisesRegex(RuntimeError, 'too large'):
                claude.image_block(self.path)

    def test_request_budget_is_checked(self):
        with patch.object(claude, 'MAX_REQUEST_BYTES', 1):
            with self.assertRaisesRegex(RuntimeError, 'transport budget'):
                claude.request('task', [], 'sonnet', 'medium')

    def test_response_requires_success_not_just_exit_zero(self):
        code = '```python\ndef process_image(path): return []\n```'
        self.assertEqual(claude.response(json.dumps(dict(type='result', subtype='success', result=code))), code)
        stream = '\n'.join(json.dumps(event) for event in [
            dict(type='system', subtype='init'), dict(type='assistant', message={'content': []}),
            dict(type='result', subtype='success', result=code)]) + '\n'
        self.assertEqual(claude.response(stream), code)
        for result in ['oops', '{}', '[]', json.dumps(dict(type='result', subtype='success', result='')),
                       json.dumps(dict(type='result', subtype='error_max_turns', errors=['turn limit'])),
                       json.dumps(dict(type='result', subtype='success', is_error=True, result='auth failed'))]:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                claude.response(result)

    def invoke(self, process):
        with (patch.object(loop, 'LLM_PROVIDER', 'claude'),
              patch.object(loop, 'load_model_context_images', return_value=self.images),
              patch.object(loop, 'Spinner', return_value=nullcontext()),
              patch.object(loop.subprocess, 'Popen', return_value=process) as launch,
              redirect_stdout(io.StringIO())):
            result = loop.call_codex_cli('task', 'rules', model='sonnet', reasoning_effort='high',
                                         timeout=42, diagnostic_montage='supplied')
        return result, launch

    def test_dispatch_uses_claude_images_model_effort_and_timeout(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (json.dumps(dict(type='result', subtype='success', result='CODE')), '')
        result, launch = self.invoke(process)
        self.assertEqual(result, 'CODE')
        cmd = launch.call_args.args[0]
        self.assertEqual(cmd[0], 'claude')
        self.assertEqual(cmd[cmd.index('--model') + 1], 'sonnet')
        self.assertEqual(cmd[cmd.index('--effort') + 1], 'high')
        self.assertNotIn('--image', cmd)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual(process.communicate.call_args.kwargs['timeout'], 42)
        content = json.loads(process.communicate.call_args.kwargs['input'])['message']['content']
        self.assertIn('rules', content[0]['text'])
        self.assertEqual(sum(block['type'] == 'image' for block in content), 1)

    def test_claude_timeout_terminates_group(self):
        process = Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired('claude', 42)
        with patch.object(loop, 'terminate_process_group') as stop:
            with self.assertRaisesRegex(RuntimeError, 'RAILWAYS_CLAUDE_TIMEOUT'):
                self.invoke(process)
        stop.assert_called_once_with(process)

    def test_claude_exit_failure_is_not_parsed_as_code(self):
        process = Mock(returncode=1)
        process.communicate.return_value = ('', 'authentication failed')
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.invoke(process)
        self.assertEqual(error.exception.stderr, 'authentication failed')

    def test_provider_configuration_does_not_reuse_codex_models(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith('RAILWAYS_')}
        env.update(RAILWAYS_LLM_PROVIDER='claude', RAILWAYS_CODEX_MODEL='codex-only',
                   RAILWAYS_CLAUDE_MODEL='sonnet', RAILWAYS_CLAUDE_TIMEOUT='321')
        code = ('import json; from railways import orchestrator as o; '
                'print(json.dumps([o.GENERATION_MODEL,o.ARCHITECT_MODEL,o.GENERATION_TIMEOUT,o.GENERATION_EFFORT]))')
        output = subprocess.check_output([sys.executable, '-B', '-c', code], env=env, text=True)
        self.assertEqual(json.loads(output), ['sonnet', 'opus', 321, 'medium'])
        env['RAILWAYS_LLM_PROVIDER'] = 'typo'
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be codex or claude', result.stderr)

    def test_auth_error_in_stdout_is_surfaced_when_stderr_empty(self):
        process = Mock(returncode=1)
        process.communicate.return_value = (json.dumps(dict(
            type='result', subtype='success', is_error=True, result='Not logged in')), '')
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.invoke(process)
        self.assertIn('Not logged in', error.exception.stderr)


if __name__ == '__main__':
    unittest.main()
