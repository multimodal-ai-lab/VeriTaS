import logging
import os
from typing import Any, Callable, Mapping

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

#: Model versions behind the singletons in `veritas.models` and the members of the
#: global ensemble. Every key has a code default, so a config without it still works
#: (see `veritas.models.base.configured_model_name`).
models_config: dict = globals.get("models") or {}

#: Where the scrapeMM client finds its server. Applied to this process at import
#: time (see `configure_scrapemm` below).
scrapemm_config: dict = globals.get("scrapemm") or {}

# Any other initialization
set_ezmm_path(ezmm_path)

logging.getLogger("ezMM").setLevel(logging.WARNING)
logger = logging.getLogger("VeriTaS")

#: The scrapeMM client settings configurable here, with the environment variable
#: that scrapeMM itself reads for each of them.
SCRAPEMM_ENV_VARS = {
    "api_url": "SCRAPEMM_API_URL",
    "api_key": "SCRAPEMM_API_KEY",
}


def configure_scrapemm(section: Mapping[str, Any] | None,
                       configure: Callable[..., Any],
                       *,
                       persist: bool = False,
                       respect_environment: bool = True,
                       environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Passes the `scrapemm:` config section on to scrapeMM's `configure`.

    scrapeMM's own precedence is: saved client file < SCRAPEMM_* environment
    variables < `configure()` calls in the running process. A `configure()` call
    would therefore override the environment, so with `respect_environment` a value
    is only passed when its environment variable is not set - the environment
    keeps the last word, as it would without this config section.

    Missing or empty values are skipped, i.e. keep scrapeMM's current value.
    Returns the keyword arguments actually passed (empty if `configure` was not
    called at all). Never logs the API key."""
    if not section:
        return {}
    environ = os.environ if environ is None else environ

    values: dict[str, Any] = {}
    for name, env_var in SCRAPEMM_ENV_VARS.items():
        value = section.get(name)
        if not value:
            continue
        if respect_environment and environ.get(env_var):
            logger.debug(f"scrapeMM: {env_var} is set in the environment and takes "
                         f"precedence over `scrapemm.{name}` in config.yaml.")
            continue
        values[name] = value

    if not values:
        return {}
    configure(**values, persist=persist)
    logger.debug(f"Configured scrapeMM from config.yaml (persist={persist}): "
                 f"api_url={values.get('api_url', '<unchanged>')}, "
                 f"api_key={'set' if 'api_key' in values else '<unchanged>'}.")
    return values


if scrapemm_config:
    try:
        import scrapemm
    except ImportError:
        logger.debug("scrapeMM is not installed; ignoring the `scrapemm` config section.")
    else:
        try:
            configure_scrapemm(scrapemm_config, scrapemm.configure, persist=False)
        except Exception as e:
            # Importing VeriTaS must not fail over the scrapeMM client settings -
            # e.g. an older scrapeMM whose `configure` has no `persist` parameter.
            # The saved client configuration (or the environment) then applies.
            logger.warning(f"Could not configure scrapeMM from config.yaml: "
                           f"{type(e).__name__}: {e}")


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
