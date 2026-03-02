import itertools
import shutil
import sys
import threading
import time


class Spinner:
    """A simple spinner class"""

    def __init__(
        self,
        message: str = "Loading...",
        delay: float = 0.1,
        plain_output: bool = False,
    ) -> None:
        """Initialize the spinner class"""
        self._message = message
        self.delay = delay
        self.plain_output = plain_output
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.running = False
        self.spinner_thread = None
        self.lock = threading.Lock()

    def spin(self) -> None:
        """Spin the spinner"""
        if self.plain_output:
            self.print_message()
            return
        while self.running:
            self.print_message()
            time.sleep(self.delay)

    def _clear_line(self):
        sys.stdout.write('\r')
        sys.stdout.write(' ' * (shutil.get_terminal_size().columns))
        sys.stdout.write('\r')
        
    def print_message(self):
        with self.lock:
            msg = self._message
        self._clear_line()
        sys.stdout.write(f"{next(self.spinner)} {msg}\r")
        sys.stdout.flush()

    def set_message(self, message: str):
        """Update the spinner message"""
        with self.lock:
            self._message = message

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread is not None:
            self.spinner_thread.join()
        with self.lock:
            msg = self._message
        self._clear_line()
        sys.stdout.flush()

    def __enter__(self):
        """Start the spinner"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop the spinner"""
        self.stop()

class _Spinner:
    """Simple spinner that invokes a callback on each frame."""

    def __init__(self, tick_fn, interval=0.1):
        self.tick_fn = tick_fn
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _spin(self):
        for frame in itertools.cycle("|/-\\"):
            if self._stop_event.is_set():
                break
            if self.tick_fn is not None:
                self.tick_fn(frame)
            time.sleep(self.interval)
