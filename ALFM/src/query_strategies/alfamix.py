"""ALFA-Mix query strategy."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.query_strategies.base_query import BaseQuery


class AlfaMix(BaseQuery):
    """ALFA-Mix Active Learning - https://arxiv.org/abs/2203.07034 ."""

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        D = params["features"].shape[1]
        self.eps = 0.2 / np.sqrt(D)

    def _get_anchors(
        self, features: torch.Tensor, labels: NDArray[np.int64]
    ) -> torch.Tensor:
        seen_classes = len(np.unique(labels))
        anchors = []

        for i in range(seen_classes):
            anchor = features[labels == i].mean(dim=0)
            anchors.append(anchor)

        return torch.stack(anchors)

    def _get_candidates(
        self,
        z_u: torch.Tensor,
        z_star: torch.Tensor,
        z_grad: torch.Tensor,
        y_star: torch.Tensor,
    ) -> torch.Tensor:
        self.model.classifier.linear.cuda()
        candidates = torch.zeros_like(y_star, dtype=torch.bool, device="cuda")
        grad_norm = torch.norm(z_grad, dim=1, keepdim=True)

        for z_s in track(z_star, description="[green]ALFA-Mix candidate selection"):
            # find unlabeled samples not in the candidate pool
            current = torch.nonzero(~candidates).flatten()
            z = z_u[current]
            ys = y_star[current]
            zg = z_grad[current]
            zg_norm = grad_norm[current]

            z_diff = z_s - z
            diff_norm = torch.norm(z_diff, dim=1, keepdim=True)
            alpha = self.eps * diff_norm * zg / (zg_norm * z_diff)
            z_lerp = alpha * z_s + (1 - alpha) * z

            probs = self.model.classifier.linear(z_lerp).softmax(dim=1)
            y_pred = torch.argmax(probs, dim=1)
            mismatch_idx = torch.nonzero(y_pred != ys).flatten()
            candidates[current[mismatch_idx]] = True

        self.model.classifier.linear.cpu()
        return torch.nonzero(candidates).cpu().flatten()

    def _random_samples(
        self, candidates: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        num_unlabeled = np.count_nonzero(~self.labeled_pool)
        unlabeled_pool = torch.ones(num_unlabeled, dtype=torch.bool)
        unlabeled_pool[candidates] = False  # all candidates will be labeled

        remaining = torch.nonzero(unlabeled_pool).flatten()
        idx = np.random.choice(len(remaining), num_samples, replace=False)
        return remaining[idx]

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        z_l = self.model.get_embedding(self.features[self.labeled_pool]).cuda()
        z_star = self._get_anchors(z_l, self.labels[self.labeled_pool].flatten())

        probs, z_u, grads = self.model.get_alpha_grad(self.features[~self.labeled_pool])
        y_star = probs.argmax(dim=1).cuda()

        candidates = self._get_candidates(z_u.cuda(), z_star, grads.cuda(), y_star)

        if len(candidates) < num_samples:
            delta = num_samples - len(candidates)
            random_samples = self._random_samples(candidates, delta)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)

        else:
            candidate_vectors = F.normalize(z_u[candidates]).numpy()
            selected, _ = cluster_features(candidate_vectors, num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
