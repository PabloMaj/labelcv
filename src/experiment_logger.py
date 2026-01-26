import logging
from pathlib import Path


class ExperimentLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(str(log_dir))
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_dir / "experiment.log")
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warn(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


if __name__ == "__main__":
    logger = ExperimentLogger(Path("runs/logger_test"))
    logger.info("Logger initialized correctly")
    logger.warn("This is a warning")
