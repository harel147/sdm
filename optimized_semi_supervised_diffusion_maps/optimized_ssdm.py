import numpy as np
import torch
from scipy.spatial.distance import cdist


class optimized_SSDM:
    def __init__(self, n_components=30, labels_type='classification', data_eps=10, data_dist_metric='euclidean',
                 sample_ratio=1, device=None):
        if labels_type == 'regression':
            raise RuntimeError(f"optimized SSDM currently only supports classification labels")

        self.n_components = n_components
        self.data_eps = data_eps
        self.data_dist_metric = data_dist_metric
        self.sample_ratio = sample_ratio
        self.device = device
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None

    def preprocess(self, train_data, train_labels, test_data):
        sample_num = int(self.sample_ratio * (len(train_data) + len(test_data)))
        self.sampled_train_data = train_data[:sample_num]
        self.train_and_test_data = np.vstack((train_data, test_data))

        self.sampled_train_labels = train_labels[:sample_num]
        self.train_labels = train_labels


    def fit_transform(self, train_data, train_labels, test_data, t=-1):
        self.preprocess(train_data, train_labels, test_data)

        distances = cdist(self.sampled_train_data, self.train_and_test_data, metric=self.data_dist_metric)
        W = np.exp(-((distances ** 2) / self.data_eps))
        W = torch.from_numpy(W.astype(np.float32)).to(self.device)
        D = self.normalize_tensor(W)

        sampled_train_labels_kernel = (self.sampled_train_labels.reshape(1, -1) == self.sampled_train_labels.reshape(-1, 1)).astype(float)
        sampled_train_labels_kernel = torch.from_numpy(sampled_train_labels_kernel.astype(np.float32)).to(self.device)
        sampled_train_labels_kernel = self.normalize_tensor(sampled_train_labels_kernel)

        P = sampled_train_labels_kernel

        if t != -1:
            U1, s1, Vt1 = torch.linalg.svd(P)
            U2, s2, Vt2 = torch.linalg.svd(D, full_matrices=False)

            sigma = np.linspace(-5, 5, P.shape[0])  # Spread the range to make the sigmoid gradual
            sigma = 1 / (1 + np.exp(sigma))  # Sigmoid function
            sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
            s2 = torch.from_numpy(sigma.astype(np.float32)).to(self.device)

            along_path_kernel = self.compute_kernel_crazy_interpolation_for_t(U1, s1, Vt1, U2, s2, Vt2, t=t)

        else:
            along_path_kernel = P @ D  # dont use t

        U, s, V = torch.linalg.svd(along_path_kernel, full_matrices=False)
        vectors = V.T
        weighted_vectors = vectors @ torch.diag(s)

        train_embeddings = weighted_vectors[:len(self.train_labels)]
        test_embeddings = weighted_vectors[len(self.train_labels):]

        train_embeddings = train_embeddings.cpu().numpy()
        test_embeddings = test_embeddings.cpu().numpy()

        train_embeddings = train_embeddings[:, :self.n_components]
        test_embeddings = test_embeddings[:, :self.n_components]

        return train_embeddings, test_embeddings

    @staticmethod
    def compute_kernel_crazy_interpolation_for_t(U1, s1, Vt1, U2, s2, Vt2, t):
        sigma_raise = torch.diag(s1 ** (1 - t))
        kernel1_reconstruct_and_raised = U1 @ sigma_raise @ Vt1

        sigma_raise = torch.diag(s2 ** (t))
        kernel2_reconstruct_and_raised = U2 @ sigma_raise @ Vt2

        to_raise = (kernel1_reconstruct_and_raised @ kernel2_reconstruct_and_raised)  # asymmetric but same components

        return to_raise

    @staticmethod
    def normalize_tensor(mx):
        rowsum = torch.sum(mx, 1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

