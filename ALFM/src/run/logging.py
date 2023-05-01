"""Custom logging config."""

import logging

from rich.logging import RichHandler


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[RichHandler()],
)


logger = logging.getLogger("rich")