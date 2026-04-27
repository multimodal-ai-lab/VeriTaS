import asyncio

from veritas import logger, globals
from veritas.pipeline.main import run_pipeline


def _build_stages_config() -> dict[int, dict]:
    """Builds the stage-specific configurations from `config.yaml`.
    Reads the `pipeline` section, drops disabled stages, strips the
    `enabled` flag, and converts `start`/`end` lists to tuples."""
    pipeline_cfg = globals.get("pipeline") or {}
    stages_config: dict[int, dict] = {}
    for key, params in pipeline_cfg.items():
        if not key.startswith("stage_"):
            continue
        stage_no = int(key.split("_", 1)[1])
        params = dict(params or {})
        if not params.pop("enabled", False):
            continue
        for k in ("start", "end"):
            if isinstance(params.get(k), list):
                params[k] = tuple(params[k])
        stages_config[stage_no] = params
    return stages_config


if __name__ == "__main__":
    logger.setLevel("INFO")
    asyncio.run(run_pipeline(_build_stages_config()))
