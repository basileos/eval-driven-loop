"""One-slot application buffer: slow inference never queues old live frames."""
from __future__ import annotations

import threading
import time


class LatestFrameReader:
    def __init__(self, frames, *, stop_capture=None):
        self.frames = iter(frames)
        self.stop_capture = stop_capture
        self.condition = threading.Condition()
        self.pending = None
        self.done = False
        self.stopped = False
        self.error = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._read, name='latest-video-frame', daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for item in self.frames:
                with self.condition:
                    if self.stopped:
                        break
                    if self.pending is not None:
                        self.dropped += 1
                    self.pending = (item, time.perf_counter())
                    self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            try:
                if self.stop_capture is not None:
                    self.stop_capture()
            except Exception as error:
                if self.error is None:
                    self.error = error
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        with self.condition:
            while self.pending is None and not self.done and not self.stopped:
                self.condition.wait(timeout=0.5)
            if self.stopped:
                raise StopIteration
            if self.pending is None:
                if self.error is not None:
                    raise self.error
                raise StopIteration
            (frame, metadata), captured_at = self.pending
            self.pending = None
        metadata = dict(metadata)
        metadata['application_queue_seconds'] = time.perf_counter() - captured_at
        return frame, metadata

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify_all()
        # The producer owns/releases the capture. Never release a backend while
        # another thread is inside read(); blocked sources finish cleanup when
        # their read returns. The thread is a daemon and this wait is bounded.
        self.thread.join(timeout=1.0)
