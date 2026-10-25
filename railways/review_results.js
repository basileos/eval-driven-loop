/* Shared, dependency-free result validation and frame navigation. */
(function (root) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const positive = value => finite(value) && value > 0;
  const frameAtTime = (seconds, fps) => Math.floor(seconds * fps + 1e-5);

  function normalizeResults(data) {
    if (!data || !data.run || !Array.isArray(data.frames)) {
      throw new Error('Expected a consolidated JSON with "run" and "frames". Use the live player’s --results file.');
    }
    const {run} = data;
    if (!positive(run.source_fps)) throw new Error('run.source_fps must be a positive number.');
    const start = run.start_time_seconds ?? 0;
    const end = run.end_time_seconds ?? null;
    if (!finite(start) || start < 0 || (end !== null && (!finite(end) || end <= start))) {
      throw new Error('Invalid start_time_seconds / end_time_seconds range.');
    }
    const warnings = [];
    if (data.finalized === false) warnings.push('This is an unfinished run; only saved results are available.');
    let inconsistentTimes = 0;
    const frames = data.frames.map((row, i) => {
      const fail = message => { throw new Error(`Result ${i + 1}: ${message}`); };
      if (!row || !Number.isSafeInteger(row.frame_index) || row.frame_index < 0) fail('invalid frame_index.');
      if (!positive(row.width) || !positive(row.height)) fail('missing or invalid original width / height.');
      if (!Array.isArray(row.predictions)) fail('predictions must be an array.');
      const timestamp = row.timestamp_seconds ?? row.frame_index / run.source_fps;
      if (!finite(timestamp) || timestamp < 0) fail('invalid timestamp_seconds.');
      if (Math.abs(timestamp - row.frame_index / run.source_fps) > 0.5 / run.source_fps) inconsistentTimes++;
      const predictions = row.predictions.map((track, j) => {
        if (!track || typeof track !== 'object') fail(`track ${j + 1} is invalid.`);
        for (const key of ['left_rail', 'right_rail']) {
          const points = track[key];
          if (!Array.isArray(points) || points.length < 2 || points.some(p =>
            !Array.isArray(p) || p.length !== 2 || !p.every(finite) ||
            p[0] < 0 || p[0] > row.width || p[1] < 0 || p[1] > row.height)) {
            fail(`track ${j + 1} has invalid ${key} coordinates.`);
          }
        }
        return track;
      });
      return {...row, timestamp_seconds: timestamp, predictions};
    }).sort((a, b) => a.frame_index - b.frame_index);
    for (let i = 1; i < frames.length; i++) {
      if (frames[i].frame_index === frames[i - 1].frame_index) throw new Error(`Duplicate result for frame ${frames[i].frame_index}.`);
    }
    if (inconsistentTimes) warnings.push(`${inconsistentTimes} timestamps disagree with frame index / FPS. Review aligns by frame_index; check the source video and FPS.`);
    return {run, frames, fps: run.source_fps, start, end, warnings,
            byFrame: new Map(frames.map(row => [row.frame_index, row]))};
  }

  function selectedRange(model, duration) {
    if (!positive(duration)) throw new Error('The video has no readable finite duration.');
    const start = model.start;
    const end = Math.min(model.end ?? duration, duration);
    if (start >= end) throw new Error('The selected range is outside this video. Load the original full-length source, not an exported clip.');
    const first = Math.ceil(start * model.fps - 1e-5);
    const last = Math.ceil(end * model.fps - 1e-5) - 1;
    if (last < first) throw new Error('The selected range contains no source frames.');
    return {start, end, first, last, clipped: model.end !== null && model.end > duration};
  }

  function seekTime(index, fps, range) {
    // Seek inside a frame, not on a floating-point boundary between two frames.
    const bounded = Math.max(range.first, Math.min(range.last, Math.round(index)));
    return Math.min(range.end - 1e-7, Math.max(range.start, (bounded + 0.5) / fps));
  }

  function adjacentRecord(frames, current, direction) {
    let lo = 0, hi = frames.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (frames[mid].frame_index < current) lo = mid + 1; else hi = mid;
    }
    if (direction < 0) return frames[lo - 1] ?? null;
    if (frames[lo]?.frame_index === current) lo++;
    return frames[lo] ?? null;
  }

  function recordStatus(row) {
    if (!row) return {label: 'No recorded result', kind: 'unprocessed'};
    if (row.error) return {label: 'Detector error', kind: 'error'};
    const mode = row.streaming?.mode;
    if (mode === 'search_wait') return {label: 'Search cooldown · not analyzed', kind: 'waiting'};
    if (mode === 'lost') return {label: 'Tracking lost', kind: 'waiting'};
    if (row.predictions.length) return {label: `${row.predictions.length} track${row.predictions.length === 1 ? '' : 's'} found`, kind: 'positive'};
    return {label: 'Checked · no tracks', kind: 'negative'};
  }

  function aspectMatches(row, videoWidth, videoHeight) {
    return Math.abs((row.width / row.height) / (videoWidth / videoHeight) - 1) < 0.01;
  }

  function scaledTracks(row, width, height) {
    return row.predictions.map(track => ({
      ...track,
      left_rail: track.left_rail.map(([x, y]) => [x * width / row.width, y * height / row.height]),
      right_rail: track.right_rail.map(([x, y]) => [x * width / row.width, y * height / row.height]),
    }));
  }

  function corridor(track) {
    const left = track.left_rail, right = track.right_rail;
    const distance = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    const same = distance(left[0], right[0]) + distance(left.at(-1), right.at(-1));
    const reversed = distance(left[0], right.at(-1)) + distance(left.at(-1), right[0]);
    return [...left, ...(same <= reversed ? [...right].reverse() : right)];
  }

  function formatTime(seconds) {
    if (!finite(seconds)) return '—';
    const ms = Math.max(0, Math.round(seconds * 1000));
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor(ms / 60000) % 60;
    const secs = Math.floor(ms / 1000) % 60;
    return `${hours ? String(hours).padStart(2, '0') + ':' : ''}${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`;
  }

  const api = {normalizeResults, selectedRange, frameAtTime, seekTime, adjacentRecord,
               recordStatus, aspectMatches, scaledTracks, corridor, formatTime};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RailwayReview = api;
})(globalThis);
