"""Registry of all supported Active Learning query methods."""

from enum import Enum

from ALFM.src.query_strategies.random import Random


class QueryType(Enum):
    """Enum of supported Active Learning query methods."""

    random = Random
