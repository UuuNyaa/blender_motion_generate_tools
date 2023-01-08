# -*- coding: utf-8 -*-
# Copyright 2022 UuuNyaa <UuuNyaa@gmail.com>
# This file is part of Motion Generate Tools.

import contextlib
import io
import logging
import subprocess
import sys
import threading
import traceback
from typing import Any, Callable, Iterable, Optional

FinallyCallback = Callable[['FunctionExecutor'], None]
LineCallback = Callable[[str], None]


def _invoke_callback(callback: Optional[Callable], *args: Any):
    if callback is None:
        return

    try:
        callback(*args)
    except:  # pylint: disable=bare-except
        logging.exception("Callback failed:")


class FunctionExecutor:
    def __init__(self):
        self._is_running = False
        self._return_value = None
        self._exception = None

    def exec_function(self, function: Callable, *args: Any, line_callback: Optional[LineCallback] = None, finally_callback: Optional[FinallyCallback] = None):
        class OutBuffer(io.StringIO):
            def write(self, text: str) -> int:
                _invoke_callback(line_callback, text)
                return super().write(text)

            def writelines(self, lines: Iterable[str]) -> None:
                lines_buffer = list(l for l in lines)
                for line in lines_buffer:
                    _invoke_callback(line_callback, line)
                return super().writelines(lines_buffer)

        def _run_background():
            buffer = OutBuffer()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    self._return_value = function(*args)
            except Exception as exception:  # pylint: disable=broad-except
                self._exception = exception
                self.write_exception(exception, line_callback=line_callback)
            finally:
                self._is_running = False
                _invoke_callback(finally_callback, self)

        self._is_running = True
        self._return_value = None
        self._exception = None

        thread = threading.Thread(target=_run_background)
        thread.daemon = True
        thread.start()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def return_value(self) -> Optional[Any]:
        return self._return_value

    @property
    def exception(self) -> Exception:
        return self._exception

    @staticmethod
    def write_exception(exception: Exception, line_callback: Optional[LineCallback]):
        if exception is None:
            return

        for line in (l for f in traceback.format_exception(exception) for l in f.splitlines()):
            _invoke_callback(line_callback, line)


class CommandExecutor(FunctionExecutor):
    _process: Optional[subprocess.Popen]
    _exit_code: int
    _command_line: str

    def __init__(self):
        super().__init__()
        self._process = None
        self._exit_code = -1
        self._command_line = ''

    def exec_command(self, *args: Any, line_callback: Optional[LineCallback] = None, finally_callback: Optional[FinallyCallback] = None):
        if self.is_running:
            raise ValueError(f"Process is running: pid={self._process.pid}")

        self._exit_code = -1
        self._command_line = ' '.join(args)
        self._process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def _enqueue_output():
            encoding = sys.getdefaultencoding()
            input_text_io = self._process.stdout

            buffer: bytearray
            while self._process.poll() is None:
                for buffer in iter(input_text_io.readline, b''):
                    text = buffer.decode(encoding).rstrip()
                    _invoke_callback(line_callback, text)

            input_text_io.close()
            self._exit_code = self._process.poll()
            self._process = None

        super().exec_function(_enqueue_output, finally_callback=finally_callback)

    @property
    def command_line(self) -> int:
        return self._command_line

    @property
    def exit_code(self) -> int:
        return self._exit_code
