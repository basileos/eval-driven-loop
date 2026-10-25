/* Local-file player: no requests, uploads, detector calls, or saved-state writes. */
(() => {
  'use strict';
  const R = window.RailwayReview;
  const $ = id => document.getElementById(id);
  const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d');
  const state = {model: null, range: null, ready: false, videoReady: false, videoName: '',
    url: null, jsonVersion: 0, index: 0, pendingIndex: null, presentedTime: 0,
    records: [], filtered: [], page: 0, warnings: [], error: '', lastUIIndex: null};
  const pageSize = 100;
  const controls = ['play', 'prev-frame', 'next-frame', 'prev-result', 'next-result', 'scrubber', 'frame-input'];
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const seconds = value => number(value) ? `${value.toFixed(3)} s` : '—';

  function messages(error = state.error) {
    state.error = error;
    $('messages').textContent = error || state.warnings.join('\n');
    $('messages').className = error ? 'error' : '';
    $('messages').hidden = !$('messages').textContent;
  }

  function clearOverlay() { ctx.clearRect(0, 0, overlay.width, overlay.height); }

  function disableReview() {
    video.pause(); state.ready = false; state.range = null; state.pendingIndex = null;
    state.records = []; state.filtered = []; state.page = 0; state.lastUIIndex = null;
    state.warnings = [];
    for (const id of controls) $(id).disabled = true;
    $('frame-badge').hidden = true; $('empty').hidden = false;
    $('range-info').textContent = 'Load both files to select the recorded range';
    $('current-time').textContent = '— / —'; $('range-ends').textContent = 'Start — · End — (exclusive)';
    $('result-status').textContent = 'Waiting for files'; $('result-status').className = 'badge';
    for (const id of ['mode', 'processing', 'available', 'age', 'confidence', 'size']) $(`detail-${id}`).textContent = '—';
    $('frame-note').textContent = 'Unprocessed frames are not counted as negative detections.';
    $('record-error').hidden = true;
    clearOverlay(); renderTable(); drawMarks();
  }

  function configureReview() {
    disableReview(); messages('');
    if (!state.model || !state.videoReady) return;
    try {
      state.range = R.selectedRange(state.model, video.duration);
      state.records = state.model.frames.filter(r => r.frame_index >= state.range.first && r.frame_index <= state.range.last);
      state.warnings = [...state.model.warnings];
      if (state.range.clipped) state.warnings.push('The JSON end time exceeds this video; playback stops at the video end.');
      const expectedName = String(state.model.run.source ?? '').split(/[\\/]/).pop();
      if (expectedName && expectedName !== state.videoName) state.warnings.push(`Source filename differs: JSON expects “${expectedName}”. Verify you selected the matching, uncropped source video.`);
      if (state.records.some(row => !R.aspectMatches(row, video.videoWidth, video.videoHeight))) {
        state.warnings.push('Some result dimensions have a different aspect ratio from the video. Their overlays are disabled to avoid misleading alignment.');
      } else if (state.records.some(row => row.width !== video.videoWidth || row.height !== video.videoHeight)) {
        state.warnings.push('Video and result resolutions differ. Overlays are scaled proportionally; use the original, uncropped video for reliable review.');
      }
      const excluded = state.model.frames.length - state.records.length;
      if (excluded) state.warnings.push(`${excluded} recorded result(s) outside the selected video range are not shown.`);
      state.ready = true;
      $('empty').hidden = true; $('frame-badge').hidden = false;
      $('stage').style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      for (const id of controls) $(id).disabled = false;
      for (const id of ['scrubber', 'frame-input']) {
        $(id).min = state.range.first; $(id).max = state.range.last;
      }
      $('range-info').textContent = `${R.formatTime(state.range.start)} – ${R.formatTime(state.range.end)} · ${state.model.fps} FPS · ${video.videoWidth} × ${video.videoHeight}`;
      $('range-ends').textContent = `Start ${R.formatTime(state.range.start)} · End ${R.formatTime(state.range.end)} (exclusive)`;
      state.index = state.range.first;
      filterRecords(); messages(''); resizeCanvases(); seekFrame(state.range.first);
    } catch (error) { messages(error.message); }
  }

  $('video-file').addEventListener('change', () => {
    disableReview(); state.videoReady = false; messages('');
    const file = $('video-file').files[0];
    if (state.url) URL.revokeObjectURL(state.url);
    state.url = null; video.removeAttribute('src'); video.load();
    if (!file) return;
    state.videoName = file.name;
    state.url = URL.createObjectURL(file); video.src = state.url; video.load();
  });

  $('json-file').addEventListener('change', async () => {
    const version = ++state.jsonVersion;
    disableReview(); state.model = null; messages('');
    const file = $('json-file').files[0];
    if (!file) return;
    try {
      const text = await file.text();
      if (version !== state.jsonVersion) return;
      state.model = R.normalizeResults(JSON.parse(text));
      configureReview();
    } catch (error) {
      if (version === state.jsonVersion) messages(`Could not load results: ${error.message}`);
    }
  });

  video.addEventListener('loadedmetadata', () => {
    if (!video.videoWidth || !video.videoHeight || !Number.isFinite(video.duration)) {
      messages('The browser could not read this video’s dimensions or duration.'); return;
    }
    state.videoReady = true; configureReview();
  });
  video.addEventListener('error', () => {
    if (!state.url) return;
    disableReview(); state.videoReady = false;
    messages('Cannot play this video in this browser. Try a browser that supports its codec, or an H.264 MP4 copy with unchanged timing and no crop.');
  });

  function seekFrame(index, pause = true) {
    if (!state.ready || !Number.isFinite(index)) return;
    if (pause) video.pause();
    state.index = Math.max(state.range.first, Math.min(state.range.last, Math.round(index)));
    state.pendingIndex = state.index; clearOverlay();
    $('scrubber').value = state.index; $('frame-input').value = state.index;
    video.currentTime = R.seekTime(state.index, state.model.fps, state.range);
    if (!video.seeking) { state.pendingIndex = null; state.presentedTime = video.currentTime; }
    render(state.presentedTime, true);
  }

  function stepFrame(direction) {
    seekFrame((state.pendingIndex ?? state.index) + direction);
  }

  function stepResult(direction) {
    if (!state.ready) return;
    const next = R.adjacentRecord(state.filtered, state.pendingIndex ?? state.index, direction);
    if (next) {
      state.page = Math.floor(state.filtered.indexOf(next) / pageSize);
      renderTable(); seekFrame(next.frame_index);
    }
  }

  async function playRange() {
    if (!state.ready) return;
    if (!video.paused) { video.pause(); return; }
    if (state.index >= state.range.last || video.currentTime < state.range.start || video.currentTime >= state.range.end) {
      seekFrame(state.range.first, false);
    }
    try { await video.play(); }
    catch (error) { if (error.name !== 'AbortError') messages(`Playback failed: ${error.message}`); }
  }

  $('play').addEventListener('click', playRange);
  $('prev-frame').addEventListener('click', () => stepFrame(-1));
  $('next-frame').addEventListener('click', () => stepFrame(1));
  $('prev-result').addEventListener('click', () => stepResult(-1));
  $('next-result').addEventListener('click', () => stepResult(1));
  $('scrubber').addEventListener('input', event => seekFrame(Number(event.target.value)));
  $('frame-input').addEventListener('change', event => {
    if (event.target.value.trim()) seekFrame(Number(event.target.value));
    else event.target.value = state.index;
  });
  $('show-overlay').addEventListener('change', () => render(state.presentedTime, true));
  $('speed').addEventListener('change', event => { video.playbackRate = Number(event.target.value); });
  video.addEventListener('play', () => { $('play').textContent = '❚❚ Pause'; });
  video.addEventListener('pause', () => { $('play').textContent = '▶ Play range'; });
  video.addEventListener('seeking', clearOverlay);
  video.addEventListener('seeked', () => {
    state.pendingIndex = null; state.presentedTime = video.currentTime;
    render(state.presentedTime, true);
  });

  document.addEventListener('keydown', event => {
    if (!state.ready || event.ctrlKey || event.metaKey || event.altKey ||
        event.target.closest('input,select,textarea,button,a,[contenteditable]')) return;
    if (event.code === 'Space') { event.preventDefault(); playRange(); }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      const direction = event.key === 'ArrowLeft' ? -1 : 1;
      if (event.shiftKey) stepResult(direction); else stepFrame(direction);
    }
  });

  function resizeCanvases() {
    const ratio = window.devicePixelRatio || 1;
    for (const canvas of [overlay, $('marks')]) {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    }
    drawMarks(); render(state.presentedTime, true);
  }
  new ResizeObserver(resizeCanvases).observe($('stage'));

  function drawOverlay(row) {
    clearOverlay();
    if (!$('show-overlay').checked || !row || R.recordStatus(row).kind !== 'positive' ||
        !R.aspectMatches(row, video.videoWidth, video.videoHeight)) return;
    const ratio = window.devicePixelRatio || 1;
    const path = points => {
      ctx.beginPath(); points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    };
    R.scaledTracks(row, overlay.width, overlay.height).forEach((track, i) => {
      path(R.corridor(track)); ctx.closePath(); ctx.fillStyle = 'rgba(64,230,126,0.18)'; ctx.fill();
      for (const [points, color] of [[track.left_rail, '#00e5f1'], [track.right_rail, '#ff5add']]) {
        path(points); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.strokeStyle = '#00171bb0'; ctx.lineWidth = 4 * ratio; ctx.stroke();
        ctx.strokeStyle = color; ctx.lineWidth = 2 * ratio; ctx.stroke();
      }
      const label = `T${i + 1}` + (number(track.confidence) ? ` · ${(track.confidence * 100).toFixed(0)}%` : '');
      ctx.font = `600 ${12 * ratio}px system-ui`;
      const width = ctx.measureText(label).width + 12 * ratio;
      const anchor = track.left_rail[Math.floor(track.left_rail.length / 2)];
      const x = Math.max(0, Math.min(overlay.width - width, anchor[0]));
      const y = Math.max(22 * ratio, Math.min(overlay.height - 3 * ratio, anchor[1]));
      ctx.fillStyle = '#071923e8'; ctx.fillRect(x, y - 20 * ratio, width, 23 * ratio);
      ctx.fillStyle = '#fff'; ctx.fillText(label, x + 6 * ratio, y - 4 * ratio);
    });
  }

  function render(time, force = false) {
    if (!state.ready) return;
    const rawIndex = R.frameAtTime(time, state.model.fps);
    const seeking = video.seeking || state.pendingIndex !== null || rawIndex < state.range.first || rawIndex > state.range.last;
    if (!seeking) state.index = rawIndex;
    const index = state.pendingIndex ?? state.index;
    const row = seeking ? null : state.model.byFrame.get(index);
    drawOverlay(row);
    const key = `${index}:${seeking}`;
    if (!force && state.lastUIIndex === key) return;
    state.lastUIIndex = key;
    $('scrubber').value = index;
    if (document.activeElement !== $('frame-input')) $('frame-input').value = index;
    $('frame-badge').textContent = `${seeking ? 'Seeking · ' : ''}Frame ${index} · ${R.formatTime(index / state.model.fps)}`;
    $('current-time').textContent = `${R.formatTime(index / state.model.fps)} · ${index - state.range.first + 1} / ${state.range.last - state.range.first + 1}`;
    const status = seeking ? {kind: 'waiting', label: 'Seeking…'} : R.recordStatus(row);
    $('result-status').textContent = status.label; $('result-status').className = `badge ${status.kind}`;
    $('detail-mode').textContent = row ? row.streaming?.mode ?? 'full_search' : '—';
    $('detail-processing').textContent = seconds(row?.processing_seconds ?? row?.elapsed_seconds);
    $('detail-available').textContent = R.formatTime(row?.result_available_at_video_seconds);
    $('detail-age').textContent = seconds(row?.result_age_seconds);
    $('detail-size').textContent = row ? `${row.width} × ${row.height}` : '—';
    $('detail-confidence').textContent = row?.predictions.length ? row.predictions.map((track, i) => `T${i + 1}: ${number(track.confidence) ? (track.confidence * 100).toFixed(1) + '%' : 'n/a'}`).join(' · ') : '—';
    $('record-error').hidden = !row?.error; $('record-error').textContent = row?.error ? String(row.error) : '';
    $('frame-note').textContent = !row ? 'No result on this source frame. This does not mean no railway was present.'
      : !R.aspectMatches(row, video.videoWidth, video.videoHeight) ? 'Overlay hidden: video and result aspect ratios do not match.'
      : status.kind === 'waiting' ? 'No fresh railway verification on this frame. No previous result is carried forward.'
      : 'Overlay uses this source frame’s coordinates. Live inference arrival time is shown separately; confidence is not a probability.';
    $('prev-frame').disabled = index <= state.range.first; $('next-frame').disabled = index >= state.range.last;
    $('prev-result').disabled = !R.adjacentRecord(state.filtered, index, -1);
    $('next-result').disabled = !R.adjacentRecord(state.filtered, index, 1);
    for (const tr of $('records').children) tr.classList.toggle('active', Number(tr.dataset.frame) === index);
  }

  function drawMarks() {
    const canvas = $('marks'), context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.range) return;
    const span = state.range.last - state.range.first + 1;
    for (const row of state.records) {
      const kind = R.recordStatus(row).kind;
      context.fillStyle = {positive: '#20ad76', negative: '#9cabb4', waiting: '#d8b36c', error: '#d65865'}[kind] || '#aabac2';
      context.fillRect((row.frame_index - state.range.first) / span * canvas.width, 0,
                       Math.max(1, canvas.width / span), canvas.height);
    }
  }

  function filterRecords() {
    const filter = $('filter').value;
    state.filtered = state.records.filter(row => filter === 'tracks' ? R.recordStatus(row).kind === 'positive' : filter === 'errors' ? Boolean(row.error) : true);
    state.page = 0; renderTable(); render(state.presentedTime, true);
  }

  function renderTable() {
    const tbody = $('records'); tbody.replaceChildren();
    const pages = Math.max(1, Math.ceil(state.filtered.length / pageSize));
    state.page = Math.min(state.page, pages - 1);
    const rows = state.filtered.slice(state.page * pageSize, (state.page + 1) * pageSize);
    for (const row of rows) {
      const tr = document.createElement('tr'); tr.dataset.frame = row.frame_index;
      const time = document.createElement('td'), link = document.createElement('button');
      link.textContent = `${row.frame_index} / ${R.formatTime(row.frame_index / state.model.fps)}`;
      link.addEventListener('click', () => seekFrame(row.frame_index)); time.append(link);
      const mode = document.createElement('td'); mode.textContent = row.error ? 'error' : row.streaming?.mode ?? 'full_search';
      const tracks = document.createElement('td'); tracks.textContent = row.predictions.length;
      tr.append(time, mode, tracks); tr.classList.toggle('active', row.frame_index === state.index); tbody.append(tr);
    }
    if (!rows.length) { const tr = document.createElement('tr'), td = document.createElement('td'); td.colSpan = 3; td.textContent = 'No recorded results in this selection.'; tr.append(td); tbody.append(tr); }
    $('record-count').textContent = state.ready ? `(${state.records.length})` : '';
    $('page-info').textContent = `${state.filtered.length} results · page ${state.page + 1} / ${pages}`;
    $('prev-page').disabled = state.page === 0; $('next-page').disabled = state.page + 1 >= pages;
  }
  $('filter').addEventListener('change', filterRecords);
  $('prev-page').addEventListener('click', () => { state.page--; renderTable(); });
  $('next-page').addEventListener('click', () => { state.page++; renderTable(); });

  // Use presented video timestamps where available, rather than inference
  // arrival timestamps or an assumed detector processing frequency.
  if ('requestVideoFrameCallback' in video) {
    const onFrame = (_, metadata) => {
      state.presentedTime = metadata.mediaTime; render(metadata.mediaTime);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  }
  function tick() {
    if (state.ready && !video.paused) {
      if (video.currentTime >= state.range.end - 1e-7 || video.ended) {
        if ($('loop').checked) { seekFrame(state.range.first, false); video.play().catch(() => {}); }
        else seekFrame(state.range.last);
      } else if (!('requestVideoFrameCallback' in video)) {
        state.presentedTime = video.currentTime; render(state.presentedTime);
      }
    }
    requestAnimationFrame(tick);
  }
  video.addEventListener('ended', () => {
    if (!state.ready) return;
    seekFrame($('loop').checked ? state.range.first : state.range.last);
    if ($('loop').checked) video.play().catch(() => {});
  });
  window.addEventListener('pagehide', () => { if (state.url) URL.revokeObjectURL(state.url); });
  requestAnimationFrame(tick);
})();
