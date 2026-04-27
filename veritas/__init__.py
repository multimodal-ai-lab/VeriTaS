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
