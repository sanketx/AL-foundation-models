"""Registry of all supported Active Learning query methods."""

from enum import Enum

from ALFM.src.query_strategies.bald import BALD
from ALFM.src.query_strategies.balentacq import BalEntAcq
from ALFM.src.query_strategies.badge import BADGE
from ALFM.src.query_strategies.coreset import Coreset
from ALFM.src.query_strategies.entropy import Entropy
from ALFM.src.query_strategies.margins import Margins
from ALFM.src.query_strategies.powerbald import PowerBALD
from ALFM.src.query_strategies.probcover import ProbCover
from ALFM.src.query_strategies.random import Random
from ALFM.src.query_strategies.uncertainty import Uncertainty


class QueryType(Enum):
    """Enum of supported Active Learning query methods."""

    random = Random
    uncertainty = Uncertainty
    entropy = Entropy
    margins = Margins
    coreset = Coreset
    bald = BALD
    powerbald = PowerBALD
    balentacq = BalEntAcq
    badge = BADGE
    probcover = ProbCover
