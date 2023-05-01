"""Active Learning training and experimentation code."""
from typing import Tuple

import h5py
import numpy as np
import torch
from ALFM.src.classifiers.classifier_wrapper import ClassifierWrapper
from ALFM.src.init_strategies.registry import InitType
from ALFM.src.query_strategies.registry import QueryType
from ALFM.src.run.utils import print_composition
from ALFM.src.run.utils import print_scores
from numpy.typing import NDArray
from omegaconf import DictConfig


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    """Fix the NumPy and PyTorch seeds.

    Args:
        seed: the value of the random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vectors(
    vector_file: str,
) -> Tuple[
    NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]
]:
    with h5py.File(vector_file) as fh:
        train_x = fh["train/features"][()].astype(np.float32)
        train_y = fh["train/labels"][()].astype(np.int64)
        test_x = fh["test/features"][()].astype(np.float32)
        test_y = fh["test/labels"][()].astype(np.int64)

    return train_x, train_y, test_x, test_y


def al_train(vector_file: str, cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    train_x, train_y, test_x, test_y = load_vectors(vector_file)

    num_classes = cfg.dataset.num_classes
    budget = cfg.budget.step * num_classes  # budget for each iteration
    budget_init = cfg.budget.init * num_classes

    iterations = np.arange(1, cfg.iterations.n + 1)

    if cfg.iterations.exp:  # exponential number of samples
        iterations = 2 ** (iterations - 1)

    # create a sampler for the intial pool and query it
    init_strategy = InitType[cfg.init_strategy.name]
    sampler = init_strategy.value(train_x, train_y, **cfg.init_strategy.params)
    labeled_pool = sampler.query(budget_init)

    # create the active learning query strategy
    query_strategy = QueryType[cfg.query_strategy.name]
    sampler = query_strategy.value(**cfg.query_strategy.params)

    # create a model from the classifier
    model = ClassifierWrapper(cfg)

    for i, iteration in enumerate(iterations, 1):
        print_composition(train_x, train_y, labeled_pool)

        model.fit(train_x, train_y, labeled_pool)
        scores = model.eval(test_x, test_y)

        print_scores(scores, i, len(iterations), budget)

        if i == len(iterations):
            return  # no need to run the query for the last iteration

        sampler.update_state(iteration, train_x, train_y, labeled_pool, None)

        budget = (iterations[i] - iterations[i - 1]) * num_classes * cfg.budget.step
        labeled_pool |= sampler.query(budget)  # update labeled pool

    #   log everything
