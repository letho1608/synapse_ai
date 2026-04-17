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
        self._stream.write(s)
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


def _wrap_utf8(stream):
    """Bọc stream gốc với UTF-8 encoding để tránh lỗi UnicodeEncodeError trên Windows."""
    import io
    try:
        if hasattr(stream, 'buffer'):
            return io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass
    return stream


def install():
    """Thay sys.stdout và sys.stderr bằng Tee để capture mọi print()."""
    global _lines
    with _lock:
        _lines = []
    utf8_stdout = _wrap_utf8(sys.__stdout__)
    utf8_stderr = _wrap_utf8(sys.__stderr__)
    sys.stdout = _TeeWriter(utf8_stdout, "stdout")
    sys.stderr = _TeeWriter(utf8_stderr, "stderr")



def get_lines():
    """Trả về bản sao danh sách dòng (cũ trước, mới sau)."""
    with _lock:
        return list(_lines)
