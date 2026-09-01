import logging
from synth_platform.logging import get_logger


def test_logger_redacts_secrets(caplog):
    log = get_logger("t")
    with caplog.at_level(logging.INFO):
        log.info("connecting with password=hunter2 and dsn=postgres://u:p@h/db")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "hunter2" not in joined and "[REDACTED]" in joined
