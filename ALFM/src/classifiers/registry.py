"""Registry of all supported Classifiers."""

from enum import Enum

from ALFM.src.classifiers.linear_classifier import LinearClassifier
from ALFM.src.classifiers.residual_adapter import ResidualAdapter


class ClassifierType(Enum):
    """Enum of supported Classifiers."""

    linear_classifier = LinearClassifier
    residual_adapter = ResidualAdapter
