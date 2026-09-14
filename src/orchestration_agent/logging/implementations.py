"""Default Logger implementations: no-op and stdout."""

from typing import Optional

from .base import Logger


class NoOpLogger(Logger):
    """Default no-op logger (logs nothing)."""

    def debug(self, message: str, **kwargs) -> None:
        pass

    def info(self, message: str, **kwargs) -> None:
        pass

    def warning(self, message: str, **kwargs) -> None:
        pass

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        pass


class StdoutLogger(Logger):
    """Simple stdout logger, useful for local development and examples."""

    def debug(self, message: str, **kwargs) -> None:
        print(f"[DEBUG] {message}")

    def info(self, message: str, **kwargs) -> None:
        print(f"[INFO] {message}")

    def warning(self, message: str, **kwargs) -> None:
        print(f"[WARNING] {message}")

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        if exception:
            print(f"[ERROR] {message}: {exception}")
        else:
            print(f"[ERROR] {message}")
