import asyncio
from veritas.pipeline.util.signatories import update_signatories
from veritas import log_to_console, logger

if __name__ == "__main__":
    log_to_console("INFO")
    asyncio.run(update_signatories())
