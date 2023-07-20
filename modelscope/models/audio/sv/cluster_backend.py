# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import numpy as np
import scipy
import sklearn
from sklearn.cluster._kmeans import k_means

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.utils.constant import Tasks


class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=0, max_num_spks=30):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks

    def __call__(self, X, pval, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval):
        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=4):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks - 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


@MODELS.register_module(
    Tasks.speaker_diarization, module_name=Models.cluster_backend)
class ClusterBackend(TorchModel):
    r"""Perfom clustering for input embeddings and output the labels.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.other_config = kwargs

        if self.model_config['cluster_type'] == 'spectral':
            self.cluster = SpectralCluster(self.model_config['min_num_spks'],
                                           self.model_config['max_num_spks'])
        else:
            raise ValueError(
                'modelscope error: Only spectral clustering is currently supported.'
            )

    def forward(self, X, **params):
        # clustering and return the labels
        k = params['oracle_num'] if 'oracle_num' in params else None
        pval = params['pval'] if 'pval' in params else self.model_config['pval']
        assert len(
            X.shape
        ) == 2, 'modelscope error: the shape of input should be [N, C]'
        if X.shape[0] < 20:
            return np.zeros(X.shape[0], dtype='int')
        if self.model_config['cluster_type'] == 'spectral':
            if X.shape[0] * pval < 6:
                pval = 6. / X.shape[0]
            labels = self.cluster(X, pval, k)
        else:
            raise ValueError(
                'modelscope error: Only spectral clustering is currently supported.'
            )

        if k is None and 'merge_thr' in self.model_config:
            labels = self.merge_by_cos(labels, X,
                                       self.model_config['merge_thr'])

        return labels

    def merge_by_cos(self, labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(
                spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)
            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels
