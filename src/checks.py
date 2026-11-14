"""Run a list of data checks: log PASS or FAIL for each one, and stop if any failed."""

import logging

logger = logging.getLogger(__name__)


def run_checks(checks, group):
    """Log PASS or FAIL for each (name, passed) pair under the label `group`, and raise if any check failed."""
    for name, passed in checks:
        if passed:
            logger.info(f"PASS  [{group}] {name}")
        else:
            logger.error(f"FAIL  [{group}] {name}")
    failed = [name for name, passed in checks if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} {group} check(s) failed: {failed}")
