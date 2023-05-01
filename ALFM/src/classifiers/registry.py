"""Registry of all supported Classifiers."""

from enum import Enum

from ALFM.src.classifiers.linear_classifier import LinearClassifier


class ClassifierType(Enum):
    """Enum of supported Classifiers."""

    linear_classifier = LinearClassifier
