"""Registry of all supported inital query methods."""

from enum import Enum

from ALFM.src.init_strategies.centroid_init import CentroidInit
from ALFM.src.init_strategies.probcover_init import ProbcoverInit
from ALFM.src.init_strategies.random_init import RandomInit
from ALFM.src.init_strategies.typiclust_init import TypiclustInit


class InitType(Enum):
    """Enum of supported inital query methods."""

    random_init = RandomInit
    typiclust_init = TypiclustInit
    centroid_init = CentroidInit
    probcover_init = ProbcoverInit
