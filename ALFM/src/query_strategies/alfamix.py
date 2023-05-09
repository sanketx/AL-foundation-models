"""ALFA-Mix query strategy."""

from typing import Any

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.query_strategies.base_query import BaseQuery


class AlfaMix(BaseQuery):
    """ALFA-Mix Active Learning - https://arxiv.org/abs/2203.07034 ."""

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        D = params["features"].shape[1]
        self.eps = 0.2 / np.sqrt(D)

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

    def _cluster_candidates(
        self, features: NDArray[np.float32], num_samples: int
    ) -> torch.Tensor:
        kmeans = faiss.Kmeans(features.shape[1], num_samples, niter=300, gpu=1)
        init_idx = kmeans_plus_plus_init(features, num_samples)
        kmeans.train(features, init_centroids=features[init_idx])

        sq_dist, cluster_idx = kmeans.index.search(features, 1)
        sq_dist = torch.from_numpy(sq_dist).ravel()
        cluster_idx = torch.from_numpy(cluster_idx).ravel()
        selected = torch.zeros(num_samples, dtype=torch.int64)

        for i in range(num_samples):
            idx = torch.nonzero(cluster_idx == i).ravel()
            min_idx = sq_dist[idx].argmin()  # point closest to the centroid
            selected[i] = idx[min_idx]  # add that id to the selected set

        return selected

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

        z_star = self.model.get_embedding(self.features[self.labeled_pool]).cuda()
        probs, z_u, grads = self.model.get_alpha_grad(self.features[~self.labeled_pool])
        y_star = probs.argmax(dim=1).cuda()

        candidates = self._get_candidates(z_u.cuda(), z_star, grads.cuda(), y_star)
        selected = self._cluster_candidates(z_u[candidates].numpy(), int(num_samples))

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
