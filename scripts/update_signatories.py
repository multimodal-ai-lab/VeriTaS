import asyncio
from veritas.pipeline.util.signatories import update_signatories
from veritas import logger

if __name__ == "__main__":
    logger.setLevel("INFO")
    asyncio.run(update_signatories())
