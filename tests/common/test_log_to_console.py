"""`logger.setLevel()` alone prints nothing below WARNING: the VeriTaS logger has
no handler of its own, so Python falls back to its last-resort handler. Scripts
therefore call `log_to_console`, which must attach exactly one handler."""

import logging

from veritas import log_to_console, logger


def _console_handlers():
    return [h for h in logger.handlers if getattr(h, "_veritas_console", False)]


def test_attaches_one_handler_and_sets_the_level(monkeypatch):
    monkeypatch.setattr(logger, "handlers", [h for h in logger.handlers
                                             if not getattr(h, "_veritas_console", False)])
    monkeypatch.setattr(logger, "level", logger.level)

    log_to_console("DEBUG")
    log_to_console("DEBUG")

    assert len(_console_handlers()) == 1
    assert logger.level == logging.DEBUG
