"""ALFA-Mix query strategy."""

import sys
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import verbose
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
        z_u: NDArray[np.float32],
        z_star: NDArray[np.float32],
        y_star: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        z_u = torch.tensor(z_u, requires_grad=True)  # type: ignore[assignment]
        z_star = torch.from_numpy(z_star)  # type: ignore[assignment]
        y_star = torch.from_numpy(y_star)  # type: ignore[assignment]
        candidates = torch.zeros_like(y_star, dtype=torch.bool)

        logits = self.model.classifier.linear(z_u)
        loss = F.cross_entropy(logits, y_star, reduction="sum")
        z_grad = torch.autograd.grad(loss, z_u)[0]

        z_u.requires_grad = False
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

        return torch.nonzero(candidates).flatten().numpy()

    def _cluster_candidates(
        self, features: NDArray[np.float32], num_samples: int
    ) -> NDArray[np.int64]:
        kmeans = faiss.Kmeans(features.shape[1], int(num_samples), niter=300)
        init_idx = kmeans_plus_plus_init(features, num_samples)
        val = kmeans.train(features, init_centroids=features[init_idx])
        print("KMEANS: ", val)

        sq_dist, cluster_idx = kmeans.index.search(features, 1)
        sq_dist, cluster_idx = sq_dist.ravel(), cluster_idx.ravel()
        selected = np.zeros(num_samples, dtype=np.int64)

        for i in range(num_samples):
            idx = np.flatnonzero(cluster_idx == i)  # id of all points in the cluster
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

        probs, embedding = self.model.get_probs_and_embedding(self.features)
        z_star, z_u = embedding[self.labeled_pool], embedding[~self.labeled_pool]
        y_star = probs[~self.labeled_pool].argmax(axis=1)

        candidates = self._get_candidates(z_u, z_star, y_star)
        selected = self._cluster_candidates(z_u[candidates], num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
