import logging
import yaml

from ezmm import set_ezmm_path

# Load global configuration variables
globals = yaml.safe_load(open("config.yaml"))

api_secrets = globals.get("api_secrets")

database = globals.get("database")

selfhosted = globals.get("selfhosted")

proxy = globals.get("proxy")

ezmm_path = globals.get("ezmm_path")

nextcloud = globals.get("nextcloud")

# Any other initialization
set_ezmm_path(ezmm_path)

logging.getLogger("ezMM").setLevel(logging.WARNING)
logger = logging.getLogger("VeriTaS")


def log_to_console(level: str | int = "INFO") -> None:
    """Prints VeriTaS log records to the console at the given level. Without a
    handler, Python falls back to printing WARNING and above only, so a script's
    `logger.setLevel("DEBUG")` alone shows nothing."""
    logger.setLevel(level)
    if not any(getattr(h, "_veritas_console", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        handler._veritas_console = True
        logger.addHandler(handler)
