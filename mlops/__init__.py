from loguru import logger
from rich.logging import RichHandler


def setup_logging():
    """Configures loguru to use RichHandler for beautiful logging.
    """
    logger.remove()  # Remove default handler
    logger.add(
        RichHandler(rich_tracebacks=True, markup=True),
        format="{message}",
        level="INFO",
    )
