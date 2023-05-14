"""Semi-supervised learning with label propagation."""

import logging
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track
from scipy.sparse import coo_matrix
from scipy.sparse import diags


class LabelPropagation:
    def __init__(
        self,
        enable: bool,
        alpha: float,
        knn: int,
        gamma: float,
        n_iter: int,
    ) -> None:
        self.enable = enable
        self.alpha = alpha
        self.knn = knn
        self.gamma = gamma
        self.n_iter = n_iter

    def fit(self, features: NDArray[np.float32]) -> None:
        if not self.enable:
            return

        num_samples, num_features = features.shape
        vectors = torch.from_numpy(features)
        vectors = F.normalize(vectors)

        index = faiss.IndexFlatL2(num_features)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

        gpu_index.add(vectors.numpy())
        sq_dist, idx = gpu_index.search(vectors.numpy(), 1 + self.knn)

        data = np.zeros(self.knn * num_samples, dtype=np.float32)
        rows = np.zeros(self.knn * num_samples, dtype=np.int64)
        cols = np.zeros(self.knn * num_samples, dtype=np.int64)

        for i in track(range(num_samples), description="[green]Cosine similarity"):
            start, end = i * self.knn, (i + 1) * self.knn
            rows[start:end] = i
            cols[start:end] = idx[i, 1:]
            # convert sq_dist to a similarity measure
            data[start:end] = (0.25 * (4 - sq_dist[i, 1:])) ** self.gamma

        data = np.clip(data, 0, 1)
        A = coo_matrix((data, (rows, cols)), shape=(num_samples, num_samples)).tocsr()

        W = A + A.T
        degree = np.bincount(W.nonzero()[0], minlength=num_samples)
        D = diags(degree ** (-0.5))
        diffusion = D @ W @ D

        crow_indices = torch.tensor(diffusion.indptr, dtype=torch.int64)
        col_indices = torch.tensor(diffusion.indices, dtype=torch.int64)
        values = torch.tensor(diffusion.data, dtype=torch.float32)
        size = torch.Size(diffusion.shape)
        self.diffusion = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size
        )

        logging.info("Diffusion matrix computed")

    def predict(
        self, labels: NDArray[np.int64], mask: NDArray[np.bool_]
    ) -> Optional[NDArray[np.float32]]:
        if not self.enable:
            return None

        num_classes = len(np.unique(labels))
        one_hot = F.one_hot(torch.tensor(labels.flatten()), num_classes).cuda()
        one_hot[~mask] = 0
        result = one_hot.clone().float()
        diffusion = self.diffusion.cuda()

        for i in track(range(self.n_iter), description="[green]Propagating labels"):
            result = (1 - self.alpha) * diffusion @ result + self.alpha * one_hot

        result[mask] = one_hot[mask].float()
        return result.cpu().numpy()
