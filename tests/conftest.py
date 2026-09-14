import os
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv():
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def redis_available() -> bool:
    return _port_open("localhost", 6379)


@pytest.fixture(scope="session")
def mongo_available() -> bool:
    return _port_open("localhost", 27017)


@pytest.fixture(scope="session")
def openai_api_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
