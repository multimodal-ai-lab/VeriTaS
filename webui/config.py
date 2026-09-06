"""Settings for the web UI.

Environment variables win over `config.yaml`, so the same image can be pointed
at a different database or media registry without rebuilding. `config.yaml` is
only ever read - never written - and is expected to be mounted read-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml() -> dict:
    """Loads `config.yaml` if it is available. Missing or broken files are not
    fatal: everything it provides can also be given via the environment."""
    path = Path(os.environ.get("VERITAS_CONFIG") or (REPO_ROOT / "config.yaml"))
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - a broken config must not crash startup
        return {}


def _pick(env: str, section: dict, key: str, default):
    """Environment variable, else the `config.yaml` section, else the default."""
    value = os.environ.get(env)
    if value not in (None, ""):
        return value
    value = section.get(key)
    return default if value in (None, "") else value


@dataclass(frozen=True)
class Settings:
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int

    #: Root of the ezMM item registry: holds `item_registry.db` and the media files.
    ezmm_path: Path

    #: Default and maximum page size of the claim browser.
    page_size: int
    max_page_size: int

    #: Largest media file streamed in one piece; larger files are served in ranges.
    chunk_size: int

    @property
    def dsn(self) -> str:
        return (f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}")

    @property
    def registry_db(self) -> Path:
        return self.ezmm_path / "item_registry.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config = _load_yaml()
    database = config.get("database") or {}
    return Settings(
        db_name=str(_pick("VERITAS_DB_NAME", database, "database", "veritas_db")),
        db_user=str(_pick("VERITAS_DB_USER", database, "user", "postgres")),
        db_password=str(_pick("VERITAS_DB_PASSWORD", database, "password", "")),
        db_host=str(_pick("VERITAS_DB_HOST", database, "host", "localhost")),
        db_port=int(_pick("VERITAS_DB_PORT", database, "port", 5432)),
        ezmm_path=Path(str(_pick("EZMM_PATH", config, "ezmm_path", "temp"))),
        page_size=int(os.environ.get("WEBUI_PAGE_SIZE") or 25),
        max_page_size=int(os.environ.get("WEBUI_MAX_PAGE_SIZE") or 200),
        chunk_size=int(os.environ.get("WEBUI_CHUNK_SIZE") or 1024 * 1024),
    )
