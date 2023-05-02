"""Registry of all supported Active Learning query methods."""

from enum import Enum

from ALFM.src.query_strategies.random import Random
from ALFM.src.query_strategies.uncertainty import Uncertainty
from ALFM.src.query_strategies.entropy import Entropy
from ALFM.src.query_strategies.margins import Margins
from ALFM.src.query_strategies.coreset import Coreset
from ALFM.src.query_strategies.probcover import ProbCover


class QueryType(Enum):
    """Enum of supported Active Learning query methods."""

    random = Random
    uncertainty = Uncertainty
    entropy = Entropy
    margins = Margins
    coreset = Coreset
    probcover = ProbCover
