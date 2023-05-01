"""Registry of all supported inital query methods."""

from enum import Enum

from ALFM.src.init_strategies.random_init import RandomInit


class InitType(Enum):
    """Enum of supported inital query methods."""

    random_init = RandomInit
