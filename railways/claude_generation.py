"""Claude Code print-mode transport; no tools or repository access needed."""
import base64
import json
from pathlib import Path


# Bound image payloads without resizing. Eight attachments fit comfortably below
# the standard request limit, leaving room for source code and diagnostics.
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 28 * 1024 * 1024


def image_block(path):
    import cv2

    path = Path(path)
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pixels is None:
        raise RuntimeError(f"Cannot decode Claude attachment: {path}")
    if max(pixels.shape[:2]) > 8000:
        raise RuntimeError(f"Claude attachment exceeds 8000 pixels: {path}")
    media_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(path.suffix.lower())
    data = path.read_bytes()
    if media_type is None or len(data) > MAX_IMAGE_BYTES:
        # Only the model's transport copy is compressed. Dataset files and
        # detector/evaluator pixels are untouched; never downsample here.
        for quality in (95, 90, 85, 80):
            ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and encoded.nbytes <= MAX_IMAGE_BYTES:
                data, media_type = encoded.tobytes(), 'image/jpeg'
                break
        else:
            raise RuntimeError(f"Claude attachment too large at native dimensions: {path}")
    return {'type': 'image', 'source': {
        'type': 'base64', 'media_type': media_type,
        'data': base64.b64encode(data).decode('ascii'),
    }}


def request(prompt, images, model, effort):
    content = [{'type': 'text', 'text': prompt}]
    for index, entry in enumerate(images, 1):
        content.append({'type': 'text', 'text': f"Image {index}: {entry['label']}"})
        content.append(image_block(entry['path']))
    payload = json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}) + '\n'
    if len(payload.encode('utf-8')) > MAX_REQUEST_BYTES:
        raise RuntimeError('Claude request exceeds the local 28 MiB transport budget.')
    command = [
        'claude', '--print', '--model', model, '--effort', effort,
        '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
        '--tools', '', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
        '--setting-sources', '', '--settings', '{"disableAllHooks":true}',
        '--disable-slash-commands', '--permission-mode', 'dontAsk',
        '--no-session-persistence', '--no-chrome',
        '--system-prompt', 'Respond only from the supplied task and images. Do not use tools.',
    ]
    # Do not use --bare: the installed CLI disables subscription/OAuth login in
    # that mode. Normal CLI authentication remains available with tools disabled.
    return command, payload


def response(stdout):
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except (ValueError, TypeError) as error:
        raise RuntimeError('Claude CLI returned invalid result JSON.') from error
    results = [event for event in events if isinstance(event, dict) and event.get('type') == 'result']
    if len(results) != 1:
        raise RuntimeError('Claude CLI did not return a result envelope.')
    result = results[0]
    if result.get('is_error') or result.get('subtype') != 'success':
        detail = result.get('errors') or result.get('result') or result.get('subtype')
        raise RuntimeError(f'Claude CLI generation failed: {str(detail)[:2000]}')
    text = result.get('result')
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError('Claude CLI returned no generated text.')
    return text.strip()
