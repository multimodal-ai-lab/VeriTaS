import asyncio
import importlib
import logging
import os
from typing import Any
from multiprocessing import Process, set_start_method, freeze_support, current_process

from veritas.db import db
from veritas.pipeline.util.stage import Stage

logger = logging.getLogger("VeriTaS")


# Logging settings for the main process (subprocesses configure themselves
# via `configure_third_party_logging()` in `_run_stage`).
def configure_third_party_logging() -> None:
    """Silence noisy third-party loggers in the current (sub)process.

    Called once per stage subprocess from ``main._run_stage``. Previously each
    ``stage_*.py`` did this at import time; consolidating it here removes that
    duplication.
    """
    quiet_critical = (
        "httpx", "httpcore", "urllib3", "hlsnative", "websockets",
        "yt_dlp", "youtube_dl", "asyncio", "filelock", "seleniumbase",
    )
    quiet_error = (
        "scrapeMM", "ezMM", "firecrawl", "tweepy", "telethon", "aiohttp",
    )
    quiet_warning = ("PIL",)

    for name in quiet_critical:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    for name in quiet_error:
        logging.getLogger(name).setLevel(logging.ERROR)
    for name in quiet_warning:
        logging.getLogger(name).setLevel(logging.WARNING)


configure_third_party_logging()

# Suppress FFmpeg/OpenCV video codec warnings (h264 decoder messages)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Quiet mode

# All available stages (1..N). Each stage module must expose a `Stage{N}`
# subclass of `Stage` (see `veritas.pipeline.util.stage`).
STAGES = (1, 2, 3, 4, 5, 6, 7)


async def run_pipeline(
        stages_config: dict[int, dict[str, Any]],
):
    """Runs the whole pipeline. Centerpiece of the project."""

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Configure multiprocessing
    freeze_support()
    set_start_method("spawn", force=True)

    # Create processes for configured stages
    processes: dict[int, Process | None] = {}
    for stage_no in STAGES:
        if stage_no in stages_config:
            kwargs = stages_config[stage_no]
            p = Process(target=_run_stage, args=(stage_no,), kwargs=kwargs, name=f"stage-{stage_no}")
            processes[stage_no] = p
            p.start()
        else:
            processes[stage_no] = None

    # Monitor loop: show high-level live status and queue sizes
    await db.connect_maybe_initialize()
    try:
        while any(p and p.is_alive() for p in processes.values()):
            try:
                # Review counts per stage (non-dismissed)
                summary = await db.get_stage_summary()

                # Build status line
                parts = [str(summary.get(0, 0))]
                for i in STAGES:
                    p = processes.get(i)
                    if p is None:
                        status = "💤"
                    elif p.is_alive():
                        status = "✅"
                    elif p.exitcode == 0:
                        status = "🏁"
                    else:
                        status = f"❌(e:{p.exitcode}) "
                    parts.append(f"{status}S{i}")
                    parts.append(str(summary.get(i, 0)))

                # Get number of dismissed reviews
                dismissed_count = await db.count_dismissed_reviews()
                parts.append(f"🗑️{dismissed_count}")

                print(" ".join(parts), flush=True)
            except Exception:
                # Never crash the monitor; just continue
                pass
            await asyncio.sleep(5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupt received, shutting down pipeline...")
    finally:
        # Terminate live subprocesses, then wait for all to exit.
        for p in processes.values():
            if p and p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in processes.values():
            if p:
                try:
                    p.join()
                except KeyboardInterrupt:
                    p.join()


def _run_stage(stage_no: int, **kwargs):
    """Subprocess entry point: configures logging and runs the stage's `run_loop`."""
    # Get root logger to capture logs from all loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Add file handler to save all logs to file
    log_file = f"logs/{current_process().name}.txt"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(fh)

    # Add handler to propagate WARNING and above to console of main process
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s - %(processName)s - %(name)s: %(message)s"))
    root_logger.addHandler(sh)

    # Silence noisy third-party loggers in this subprocess.
    configure_third_party_logging()

    module = importlib.import_module(f"veritas.pipeline.stage_{stage_no}")
    stage_cls: type[Stage] = getattr(module, f"Stage{stage_no}")
    try:
        asyncio.run(stage_cls().run(**kwargs))
    except KeyboardInterrupt:
        # Quiet shutdown on Ctrl+C from the parent process.
        pass
