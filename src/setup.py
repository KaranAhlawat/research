import logging


def setup() -> None:
    """
    Primary method for initial setup.

    The method will be called before running the project to setup logging for
    the entire project.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    tracker = logging.getLogger("tracker")
    tracker.info("Setting up project.")
