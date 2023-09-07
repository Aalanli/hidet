from typing import ContextManager
import contextlib
import os
import tempfile


class CapturedStdout:
    def __init__(self):
        self.content: str = ""

    def __str__(self):
        return self.content

    def set_output(self, content: str):
        self.content = content


@contextlib.contextmanager
def capture_stdout() -> ContextManager[CapturedStdout]:
    captured_stdout = CapturedStdout()

    with tempfile.TemporaryFile(mode='w+') as f:
        new_fd = f.fileno()
        original_fd = os.dup(1)
        os.dup2(new_fd, 1)
        yield captured_stdout
        os.dup2(original_fd, 1)
        os.close(original_fd)
        f.seek(0)
        captured_stdout.set_output(f.read())
