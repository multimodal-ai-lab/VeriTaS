import logging
import yaml

import scrapemm
from ezmm import set_ezmm_path

# Load global configuration variables
globals = yaml.safe_load(open("config.yaml"))

api_secrets = globals.get("api_secrets")

database = globals.get("database")

selfhosted = globals.get("selfhosted")

scrapemm_config = globals.get("scrapemm") or {}

proxy = globals.get("proxy")

ezmm_path = globals.get("ezmm_path")

nextcloud = globals.get("nextcloud")

# Any other initialization
set_ezmm_path(ezmm_path)

# Point the scrapeMM client at its server. persist=False keeps the API key out of
# scrapeMM's per-user config file.
if scrapemm_config.get("url"):
    scrapemm.configure(api_url=scrapemm_config["url"], api_key=scrapemm_config.get("api_key"), persist=False)

logging.getLogger("ezMM").setLevel(logging.WARNING)
logger = logging.getLogger("VeriTaS")
