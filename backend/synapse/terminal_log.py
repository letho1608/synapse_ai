"""
Capture stdout/stderr to a ring buffer for dashboard terminal log.
Ghi lại mọi dòng in ra terminal để hiển thị trên dashboard.
"""
import sys
import threading

MAX_LINES = 2000
_lock = threading.Lock()
_lines = []  # list of str, oldest first


class _TeeWriter:
    """Ghi ra stream gốc và append từng dòng vào _lines (thread-safe)."""
    def __init__(self, stream, name="stdout"):
        self._stream = stream
        self._name = name
        self._buf = ""

    def write(self, s):
        if s is None:
            return
        # Handle Unicode encoding errors on Windows (cp1252 doesn't support Vietnamese)
        try:
            self._stream.write(s)
        except UnicodeEncodeError:
            # Replace unsupported characters with '?'
            self._stream.write(s.encode('cp1252', errors='replace').decode('cp1252'))
        self._stream.flush()
        self._buf += s
        while "\n" in self._buf:
            idx = self._buf.index("\n")
            line = self._buf[:idx].rstrip("\r\n")
            self._buf = self._buf[idx + 1:]
            with _lock:
                _lines.append(line)
            while len(_lines) > MAX_LINES:
                _lines.pop(0)

    def flush(self):
        self._stream.flush()
        if self._buf.strip():
            with _lock:
                _lines.append(self._buf.rstrip("\r\n"))
                while len(_lines) > MAX_LINES:
                    _lines.pop(0)
            self._buf = ""

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install():
    """Thay sys.stdout và sys.stderr bằng Tee để capture mọi print()."""
    global _lines
    with _lock:
        _lines = []
    sys.stdout = _TeeWriter(sys.__stdout__, "stdout")
    sys.stderr = _TeeWriter(sys.__stderr__, "stderr")


def get_lines():
    """Trả về bản sao danh sách dòng (cũ trước, mới sau)."""
    with _lock:
        return list(_lines)
