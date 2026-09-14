"""Pluggable logging interface — the agent makes no data-store assumptions."""

from abc import ABC, abstractmethod
from typing import Optional


class Logger(ABC):
    """Pluggable logging interface."""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None: ...

    @abstractmethod
    def info(self, message: str, **kwargs) -> None: ...

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None: ...

    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None: ...
