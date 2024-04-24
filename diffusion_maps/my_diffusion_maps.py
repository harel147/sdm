import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kstest
from scipy.linalg import eigh

class MyDiffusionMaps:
    def __init__(self, eps=10, t=1, K_norm2=None):
        self.eps = eps
        self.t = t

        self.K_norm2 = K_norm2
        self.sorted_eigenvalues = None
        self.sorted_eigenvectors = None
        self.weighted_vectors = None

    def compute_kernel(self, data, metric='euclidean', prior_index=None, prior_strategy='zeros'):
        # affinity matrix W
        distances = cdist(data, data, metric=metric)
        W = np.exp(-((distances ** 2) / self.eps))
        if prior_index is not None:
            if prior_strategy == 'zeros':
                W[prior_index, :] = 0
                W[:, prior_index] = 0
                W[prior_index, prior_index] = 1
            elif prior_strategy == 'const':
                const = 0.1
                W[prior_index, :] = const
                W[:, prior_index] = const

        # diagonal normalization matrix Q
        row_sums = np.sum(W, axis=1)
        row_sums = row_sums ** -1
        Q = np.diag(row_sums)

        # normalized kernel
        K_norm1 = Q @ W @ Q

        # second diagonal normalization matrix Q2
        row_sums = np.sum(K_norm1, axis=1)
        row_sums = row_sums ** -0.5
        Q2 = np.diag(row_sums)

        # second normalized kernel
        self.K_norm2 = Q2 @ K_norm1 @ Q2

        U, S, Vt = np.linalg.svd(self.K_norm2)
        sorted_eigenvalues = np.sort(S ** 2)[::-1]
        if sorted_eigenvalues[2] >= 1:
            raise RuntimeError(f"third largest eigenvalue shouldn't be 1. you probably need to choose a bigger eps.")

        # this is a possible solution to enforce not too small eigenvalues
        U, S, Vt = np.linalg.svd(self.K_norm2)
        S[S < 1e-4] = 1e-4
        self.K_norm2 = U @ np.diag(S) @ U.T

    def force_first_eigenvectors_index_positive(self):
        negative_first_element = self.sorted_eigenvectors[:, 0] < 0
        self.sorted_eigenvectors[negative_first_element] *= -1

    def calculate_kernel_eigen(self):
        if self.K_norm2 is None:
            raise RuntimeError(f"can't compute eigen if K_norm2 hasn't been computed before")
        # calculate the second normalized kernel eigenvalues and eigenvectors
        # eigenvalues, eigenvectors = eigh(self.K_norm2)
        # indices = np.argsort(eigenvalues)[::-1]
        # self.sorted_eigenvalues = eigenvalues[indices]
        # self.sorted_eigenvectors = eigenvectors[:, indices].T

        U, S, Vt = np.linalg.svd(self.K_norm2)
        eigenvalues = S ** 2
        eigenvectors = U
        indices = np.argsort(eigenvalues)[::-1]
        self.sorted_eigenvalues = eigenvalues[indices]
        self.sorted_eigenvectors = eigenvectors[:, indices].T

        self.force_first_eigenvectors_index_positive()

    def calculate_weighted_vectors(self, ):
        if self.sorted_eigenvalues is None or self.sorted_eigenvectors is None:
            raise RuntimeError(f"can't compute weighted vectors if eigen vals/vecs haven't been computed before")
        # create weighted vectors
        self.weighted_vectors = []
        first_eigenvec = np.copy(self.sorted_eigenvectors[0])
        first_eigenvec[first_eigenvec == 0] = 1e-89
        for i in range(1, len(self.sorted_eigenvalues)):
            vec = self.sorted_eigenvectors[i] / first_eigenvec
            self.weighted_vectors.append(vec)

    def project_sample(self, sample_num, out_dims):
        sample_embeddings = []
        for d in range(out_dims):
            value = (self.sorted_eigenvalues[d + 1] ** self.t) * self.weighted_vectors[d][sample_num]
            sample_embeddings.append(value)
        sample_embeddings = np.array(sample_embeddings).T
        return sample_embeddings

    def project_to_low_dimensions(self, data, out_dims=2):
        if self.weighted_vectors is None:
            raise RuntimeError(f"can't project if weighted vectors haven't been computed before")
        # project to low dimensions
        embeddings = []
        for i in range(len(data)):
            sample_embeddings = self.project_sample(sample_num=i, out_dims=out_dims)
            embeddings.append(sample_embeddings)

        return np.array(embeddings)

    def fit_transform(self, data, out_dims=2, metric='euclidean'):

        self.compute_kernel(data, metric=metric)
        self.calculate_kernel_eigen()
        self.calculate_weighted_vectors()
        embeddings = self.project_to_low_dimensions(data, out_dims)

        return embeddings

    @staticmethod
    def auto_select_aps(data, metric='euclidean'):
        distances = cdist(data, data, metric=metric)
        possible_eps = [10**i for i in range(-5, 11)]
        eps_candidates_count = {}
        eps_candidates_eigenvalues = {}
        lower_bound = 1e-4
        upper_bound = 0.9999
        for eps in possible_eps:
            W = np.exp(-((distances ** 2) / eps))

            # diagonal normalization matrix Q
            row_sums = np.sum(W, axis=1)
            row_sums = row_sums ** -1
            Q = np.diag(row_sums)

            # normalized kernel
            K_norm1 = Q @ W @ Q

            # second diagonal normalization matrix Q2
            row_sums = np.sum(K_norm1, axis=1)
            row_sums = row_sums ** -0.5
            Q2 = np.diag(row_sums)

            # second normalized kernel
            K_norm2 = Q2 @ K_norm1 @ Q2

            U, S, Vt = np.linalg.svd(K_norm2)
            sorted_eigenvalues = np.sort(S ** 2)[::-1]

            if sorted_eigenvalues[1] > upper_bound or sorted_eigenvalues[1] < lower_bound:
                continue

            count = np.sum((lower_bound <= sorted_eigenvalues) & (sorted_eigenvalues <= upper_bound))
            if count > 0:
                eps_candidates_count[eps] = count
                eps_candidates_eigenvalues[eps] = sorted_eigenvalues

        max_count = max(list(eps_candidates_count.values()))
        best_count_eps = [eps for eps, count in eps_candidates_count.items() if count >= max_count]
        best_p_value = 0
        best_eps = None
        for eps, eigenvalues in eps_candidates_eigenvalues.items():
            if eps not in best_count_eps:
                continue
            valid_eigenvalues = eigenvalues[(lower_bound <= eigenvalues) & (eigenvalues <= upper_bound)]
            uniform_dist = np.random.uniform(0, 1, size=len(valid_eigenvalues))
            # Perform the Kolmogorov-Smirnov test
            ks_statistic, p_value = kstest(valid_eigenvalues, uniform_dist)
            if p_value > best_p_value:
                best_p_value = p_value
                best_eps = eps

        if best_eps is None:
            raise RuntimeError(f"Failed to select eps automatically.")

        print(f"selected eps: {best_eps}")
        return best_eps









