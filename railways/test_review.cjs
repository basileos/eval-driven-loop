/* Run with: node --test railways/test_review.cjs */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const R = require('./review_results.js');

const track = {left_rail: [[100, 200], [100, 1800]], right_rail: [[200, 200], [200, 1800]], confidence: 0.8};
const row = (index, overrides = {}) => ({frame_index: index, timestamp_seconds: index / 25,
  width: 3840, height: 2160, predictions: [], processing_seconds: 0.2, error: null, ...overrides});
const fixture = () => ({finalized: true, run: {source: '/video/source.mp4', source_fps: 25,
  start_time_seconds: 65, end_time_seconds: 65.2}, frames: [
    row(1629), row(1625, {predictions: [track], result_available_at_video_seconds: 68,
                         result_age_seconds: 3, streaming: {mode: 'search'}}),
    row(1627, {streaming: {mode: 'search_wait'}}),
  ]});

test('sorts sparse records and matches by input frame, not late arrival time', () => {
  const model = R.normalizeResults(fixture());
  assert.deepEqual(model.frames.map(r => r.frame_index), [1625, 1627, 1629]);
  assert.equal(model.byFrame.get(R.frameAtTime(65.02, 25)).predictions.length, 1);
  assert.equal(model.byFrame.get(R.frameAtTime(65.06, 25)), undefined);
  assert.equal(model.byFrame.get(R.frameAtTime(68, 25)), undefined);
});

test('range is start-inclusive/end-exclusive and steps inside each source frame', () => {
  const model = R.normalizeResults(fixture()), range = R.selectedRange(model, 100);
  assert.deepEqual(range, {start: 65, end: 65.2, first: 1625, last: 1629, clipped: false});
  for (let frame = range.first; frame <= range.last; frame++) {
    const timestamp = R.seekTime(frame, model.fps, range);
    assert.ok(timestamp >= range.start && timestamp < range.end);
    assert.equal(R.frameAtTime(timestamp, model.fps), frame);
  }
  assert.equal(R.frameAtTime(R.seekTime(-1, 25, range), 25), 1625);
  assert.equal(R.frameAtTime(R.seekTime(2000, 25, range), 25), 1629);
});

test('fractional start, partial final frame, open end, and shorter video', () => {
  const model = R.normalizeResults(fixture());
  const range = R.selectedRange({...model, start: 65.01, end: 65.09}, 100);
  assert.equal(range.first, 1626); assert.equal(range.last, 1627);
  assert.equal(R.frameAtTime(R.seekTime(1627, 25, range), 25), 1627);
  assert.equal(R.selectedRange({...model, end: null}, 65.4).end, 65.4);
  assert.equal(R.selectedRange(model, 65.1).clipped, true);
  assert.throws(() => R.selectedRange(model, 25), /original full-length/);
  assert.throws(() => R.selectedRange({...model, start: 65.01, end: 65.02}, 100), /no source frames/);
});

test('fractional FPS stepping and floating point boundaries', () => {
  const fps = 30000 / 1001;
  const range = {start: 10, end: 20, first: Math.ceil(10 * fps), last: Math.ceil(20 * fps) - 1};
  for (let i = range.first; i <= range.last; i++) assert.equal(R.frameAtTime(R.seekTime(i, fps, range), fps), i);
  assert.equal(R.frameAtTime(65.07999999999999, 25), 1627);
});

test('previous/next result navigation respects sparse gaps and endpoints', () => {
  const rows = R.normalizeResults(fixture()).frames;
  assert.equal(R.adjacentRecord(rows, 1625, -1), null);
  assert.equal(R.adjacentRecord(rows, 1625, 1).frame_index, 1627);
  assert.equal(R.adjacentRecord(rows, 1626, -1).frame_index, 1625);
  assert.equal(R.adjacentRecord(rows, 1626, 1).frame_index, 1627);
  assert.equal(R.adjacentRecord(rows, 1629, 1), null);
  assert.equal(R.adjacentRecord([], 0, 1), null);
});

test('unprocessed, checked-negative, cooldown, lost, positive, and error are distinct', () => {
  assert.equal(R.recordStatus(null).kind, 'unprocessed');
  assert.equal(R.recordStatus(row(0)).kind, 'negative');
  assert.equal(R.recordStatus(row(0, {streaming: {mode: 'search_wait'}})).kind, 'waiting');
  assert.equal(R.recordStatus(row(0, {streaming: {mode: 'lost'}})).kind, 'waiting');
  assert.equal(R.recordStatus(row(0, {predictions: [track]})).kind, 'positive');
  assert.equal(R.recordStatus(row(0, {predictions: [track], error: 'failure'})).kind, 'error');
});

test('geometry scales independently by both axes and never mutates saved predictions', () => {
  const input = row(0, {predictions: [track]}), before = JSON.stringify(input);
  const scaled = R.scaledTracks(input, 960, 540)[0];
  assert.deepEqual(scaled.left_rail, [[25, 50], [25, 450]]);
  assert.equal(JSON.stringify(input), before);
  assert.equal(R.aspectMatches(input, 1920, 1080), true);
  assert.equal(R.aspectMatches(input, 1080, 1920), false);
});

test('corridor fill handles either direction of the second rail', () => {
  const same = R.corridor(track);
  const opposite = R.corridor({...track, right_rail: [...track.right_rail].reverse()});
  assert.deepEqual(same, opposite);
  assert.deepEqual(same, [[100, 200], [100, 1800], [200, 1800], [200, 200]]);
});

test('validates schema, timing, duplicate frames, and coordinate arrays', () => {
  for (const bad of [null, {}, {run: {}, frames: []}]) assert.throws(() => R.normalizeResults(bad));
  const check = (mutate, pattern) => { const data = fixture(); mutate(data); assert.throws(() => R.normalizeResults(data), pattern); };
  check(d => { d.run.source_fps = 0; }, /source_fps/);
  check(d => { d.run.end_time_seconds = 64; }, /range/);
  check(d => { d.frames.push(d.frames[0]); }, /Duplicate/);
  check(d => { d.frames[0].frame_index = -1; }, /frame_index/);
  check(d => { d.frames[0].width = 0; }, /width/);
  check(d => { d.frames[0].predictions = null; }, /array/);
  check(d => { d.frames[0].predictions = [{left_rail: [[NaN, 2], [3, 4]], right_rail: track.right_rail}]; }, /coordinates/);
  check(d => { d.frames[0].predictions = [{...track, left_rail: [[1, 2], [4000, 4]]}]; }, /coordinates/);
});

test('empty/unfinished runs and inconsistent timestamps have honest diagnostics', () => {
  const data = fixture(); data.frames = []; data.finalized = false;
  assert.equal(R.normalizeResults(data).frames.length, 0);
  assert.match(R.normalizeResults(data).warnings[0], /unfinished/);
  data.frames = [row(1625, {timestamp_seconds: 68})];
  assert.ok(R.normalizeResults(data).warnings.some(w => w.includes('disagree')));
});

test('timestamp formatting handles millisecond carry and long videos', () => {
  assert.equal(R.formatTime(65.08), '01:05.080');
  assert.equal(R.formatTime(59.9999), '01:00.000');
  assert.equal(R.formatTime(3661.001), '01:01:01.001');
  assert.equal(R.formatTime(undefined), '—');
});

// A deterministic DOM/media double exercises controller wiring, not browser
// decoding. Native codec support and actual presented frames still need a browser.
function harness({videoFrames = false} = {}) {
  class Element {
    constructor(id = '') {
      this.id = id; this.listeners = {}; this.children = []; this.dataset = {};
      this.style = {}; this.value = ''; this.checked = false; this.width = 960; this.height = 540;
      this.classList = {toggle() {}}; this.disabled = false;
      this.calls = [];
      this.context = new Proxy({measureText: text => ({width: text.length * 7})}, {
        get: (target, key) => key in target ? target[key] : (...args) => this.calls.push([key, ...args]),
        set: (target, key, value) => { target[key] = value; return true; },
      });
    }
    addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); }
    async emit(type, props = {}) { for (const handler of this.listeners[type] ?? []) await handler({target: this, ...props}); }
    getContext() { return this.context; }
    getBoundingClientRect() { return {width: 960, height: this.id === 'marks' ? 10 : 540}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    closest() { return null; }
    removeAttribute(key) { delete this[key]; }
  }
  const html = fs.readFileSync(path.join(__dirname, 'review.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element(m[1])]));
  const doc = new Element(); doc.getElementById = id => { assert.ok(elements[id], `Missing HTML ID: ${id}`); return elements[id]; };
  doc.createElement = () => new Element();
  const video = elements.video;
  Object.assign(video, {videoWidth: 1920, videoHeight: 1080, duration: 70, paused: true, currentTime: 0, seeking: false,
    load() {}, pause() { this.paused = true; this.emit('pause'); }, async play() { this.paused = false; await this.emit('play'); }});
  if (videoFrames) video.requestVideoFrameCallback = callback => { video.onFrame = callback; };
  elements['show-overlay'].checked = true; elements.filter.value = 'all';
  const ticks = [], revoked = [];
  const window = new Element(); window.RailwayReview = R; window.devicePixelRatio = 1;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'review.js'), 'utf8'), {
    window, document: doc, URL: {createObjectURL: () => 'blob:local-video', revokeObjectURL: value => revoked.push(value)},
    ResizeObserver: class {constructor(cb) { this.cb = cb; } observe() { this.cb(); }},
    requestAnimationFrame: cb => ticks.push(cb), console,
  }, {filename: 'review.js'});
  const loadJSON = async data => {
    elements['json-file'].files = [{text: async () => typeof data === 'string' ? data : JSON.stringify(data)}];
    await elements['json-file'].emit('change');
  };
  const loadVideo = async () => {
    elements['video-file'].files = [{name: 'source.mp4'}];
    await elements['video-file'].emit('change'); await video.emit('loadedmetadata');
  };
  return {elements, video, doc, loadJSON, loadVideo, tick: () => ticks.shift()(), revoked};
}

test('controller loads either order, seeks range start, and draws late results retrospectively', async () => {
  for (const jsonFirst of [false, true]) {
    const h = harness();
    if (jsonFirst) { await h.loadJSON(fixture()); await h.loadVideo(); }
    else { await h.loadVideo(); await h.loadJSON(fixture()); }
    assert.equal(h.elements.play.disabled, false);
    assert.equal(h.video.currentTime, 65.02);
    assert.equal(h.elements['result-status'].textContent, '1 track found');
    assert.equal(h.elements['detail-available'].textContent, '01:08.000');
    assert.ok(h.elements.overlay.calls.some(c => c[0] === 'fill'));
    assert.equal(h.elements.scrubber.min, 1625); assert.equal(h.elements.scrubber.max, 1629);
  }
});

test('controller frame stepping clears overlays in gaps and result stepping jumps sparse rows', async () => {
  const h = harness(); await h.loadVideo(); await h.loadJSON(fixture());
  h.elements.overlay.calls = [];
  await h.elements['next-frame'].emit('click');
  assert.equal(h.video.currentTime, 65.06);
  assert.equal(h.elements['result-status'].textContent, 'No recorded result');
  assert.ok(!h.elements.overlay.calls.some(c => c[0] === 'fill'));
  await h.elements['next-result'].emit('click');
  assert.equal(h.video.currentTime, 65.1);
  assert.match(h.elements['result-status'].textContent, /not analyzed/);
  await h.elements['next-result'].emit('click');
  assert.equal(h.video.currentTime, 65.18);
  assert.equal(h.elements['result-status'].textContent, 'Checked · no tracks');
  assert.equal(h.elements['next-frame'].disabled, true);
  assert.equal(h.elements['next-result'].disabled, true);
});

test('playback pauses at the exclusive end or loops, without leaving range', async () => {
  const h = harness(); await h.loadVideo(); await h.loadJSON(fixture());
  await h.elements.play.emit('click'); assert.equal(h.video.paused, false);
  h.video.currentTime = 65.21; h.tick();
  assert.equal(h.video.paused, true); assert.equal(h.video.currentTime, 65.18);
  h.elements.loop.checked = true; await h.elements.play.emit('click');
  assert.equal(h.video.currentTime, 65.02);
  h.video.currentTime = 65.21; h.tick();
  assert.equal(h.video.currentTime, 65.02); assert.equal(h.video.paused, false);
});

test('presented-frame callbacks select exact records, not nearby or arriving results', async () => {
  const h = harness({videoFrames: true}); await h.loadVideo(); await h.loadJSON(fixture());
  h.video.onFrame(0, {mediaTime: 65.04});
  assert.equal(h.elements['result-status'].textContent, 'No recorded result');
  h.video.onFrame(0, {mediaTime: 65.08});
  assert.match(h.elements['result-status'].textContent, /not analyzed/);
});

test('malformed replacement JSON disables playback and clears old geometry', async () => {
  const h = harness(); await h.loadVideo(); await h.loadJSON(fixture());
  h.elements.overlay.calls = []; await h.loadJSON('{bad');
  assert.equal(h.elements.play.disabled, true);
  assert.match(h.elements.messages.textContent, /Could not load results/);
  assert.ok(h.elements.overlay.calls.some(c => c[0] === 'clearRect'));
  assert.ok(!h.elements.overlay.calls.some(c => c[0] === 'fill'));
});

test('filtering changes result navigation and overlay toggle does not change records', async () => {
  const h = harness(); await h.loadVideo(); await h.loadJSON(fixture());
  h.elements.filter.value = 'tracks'; await h.elements.filter.emit('change');
  assert.equal(h.elements.records.children.length, 1);
  assert.equal(h.elements['next-result'].disabled, true);
  h.elements.overlay.calls = []; h.elements['show-overlay'].checked = false;
  await h.elements['show-overlay'].emit('change');
  assert.equal(h.elements['result-status'].textContent, '1 track found');
  assert.ok(!h.elements.overlay.calls.some(c => c[0] === 'fill'));
});

test('unsupported video and out-of-range source show actionable errors', async () => {
  const h = harness(); await h.loadJSON(fixture()); h.video.duration = 25; await h.loadVideo();
  assert.equal(h.elements.play.disabled, true); assert.match(h.elements.messages.textContent, /original full-length/);
  await h.video.emit('error'); assert.match(h.elements.messages.textContent, /codec/);
});
