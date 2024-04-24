import numpy as np
from scipy.spatial.distance import cdist
from diffusion_maps.my_diffusion_maps import MyDiffusionMaps


class SDM:
    def __init__(self, n_components=30, labels_type='regression', direction='labels_to_data', data_eps='auto', labels_eps='auto',
                 data_dist_metric='euclidean', labels_dist_metric='euclidean'):
        self.n_components = n_components
        self.labels_type = labels_type
        self.direction = direction
        self.data_eps = data_eps
        self.labels_eps = labels_eps
        self.data_dist_metric = data_dist_metric
        self.labels_dist_metric = labels_dist_metric
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.embedding_ = None

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

        if self.data_eps == 'auto':
            self.data_eps = MyDiffusionMaps.auto_select_aps(train_data)
        if self.labels_eps == 'auto':
            self.labels_eps = MyDiffusionMaps.auto_select_aps(train_labels)

        if self.labels_type == 'classification':
            num_classes = len(np.unique(train_labels))
            # You need to copy classes_mean_distances manually to create discrete_labels_distance metric
            classes_mean_distances = SDM.calculate_data_distances_for_metric(train_data, train_labels, num_classes, data_metric=self.data_dist_metric)
            # generate the distance metric function dynamically based on computed classes_mean_distances for this fold
            discrete_labels_distance = SDM.generate_discrete_labels_distance(classes_mean_distances)
            self.labels_dist_metric = discrete_labels_distance

    def fit_transform(self, train_data, train_labels, t=0.5, prior_strategy='zeros'):
        self.fit(train_data, train_labels)

        dm_train_data = MyDiffusionMaps(eps=self.data_eps)
        dm_train_data.compute_kernel(self.train_data)
        along_path_embeddings = []
        for i in range(len(self.train_labels)):
            train_labels_with_prior = np.copy(self.train_labels).astype(np.float64)
            if self.labels_type == 'classification':
                train_labels_with_prior[i] = np.array([-1]).reshape(-1, 1)
            else:
                train_labels_with_prior[i] = self.train_labels.mean().reshape(-1, 1)
            prior_index = i
            dm_train_labels_with_prior = MyDiffusionMaps(eps=self.labels_eps)
            dm_train_labels_with_prior.compute_kernel(train_labels_with_prior, metric=self.labels_dist_metric,
                                                      prior_index=prior_index, prior_strategy=prior_strategy)

            if self.direction == 'labels_to_data':
                start_kernel = dm_train_labels_with_prior.K_norm2
                end_kernel = dm_train_data.K_norm2
            else:
                start_kernel = dm_train_data.K_norm2
                end_kernel = dm_train_labels_with_prior.K_norm2

            along_path_kernel = SDM.compute_kernel_along_path_for_t(start_kernel, end_kernel, t=t)

            dm_along_path = MyDiffusionMaps(K_norm2=along_path_kernel)
            dm_along_path.calculate_kernel_eigen()
            dm_along_path.calculate_weighted_vectors()

            along_path_embeddings_i = dm_along_path.project_sample(sample_num=i, out_dims=self.n_components)
            along_path_embeddings.append(along_path_embeddings_i)

            if i % 10 == 0:
                print(f"done {round(i / len(self.train_labels), 2) * 100}% generating embeddings to trainset for t={t}")

        along_path_embeddings = np.array(along_path_embeddings)
        self.embedding_ = along_path_embeddings
        return along_path_embeddings

    def transform(self, test_data, t=0.5, prior_strategy='zeros'):
        if self.train_data is None:
            raise RuntimeError(f"You must use fit or fit_transform before using transform")

        if self.labels_type == 'classification':
            train_labels_with_prior = np.append(self.train_labels, np.array([-1]).reshape(-1, 1), axis=0)
        else:
            train_labels_with_prior = np.append(self.train_labels, self.train_labels.mean().reshape(-1, 1),
                                                axis=0)
        prior_index = len(train_labels_with_prior) - 1
        dm_train_labels_with_prior = MyDiffusionMaps(eps=self.labels_eps)
        dm_train_labels_with_prior.compute_kernel(train_labels_with_prior, metric=self.labels_dist_metric,
                                                  prior_index=prior_index, prior_strategy=prior_strategy)
        along_path_embeddings = []
        for i in range(test_data.shape[0]):
            train_data_with_unseen_sample = np.append(self.train_data, test_data[i].reshape(1, -1), axis=0)
            dm_train_data = MyDiffusionMaps(eps=self.data_eps)
            dm_train_data.compute_kernel(train_data_with_unseen_sample)

            if self.direction == 'labels_to_data':
                start_kernel = dm_train_labels_with_prior.K_norm2
                end_kernel = dm_train_data.K_norm2
            else:
                start_kernel = dm_train_data.K_norm2
                end_kernel = dm_train_labels_with_prior.K_norm2

            along_path_kernel = self.compute_kernel_along_path_for_t(start_kernel, end_kernel, t=t)
            dm_along_path = MyDiffusionMaps(K_norm2=along_path_kernel)
            dm_along_path.calculate_kernel_eigen()
            dm_along_path.calculate_weighted_vectors()

            along_path_embeddings_i = dm_along_path.project_sample(sample_num=len(train_data_with_unseen_sample) - 1,
                                                                   out_dims=self.n_components)
            along_path_embeddings.append(along_path_embeddings_i)

            if i%10 == 0:
                print(f"done {round(i/test_data.shape[0], 2)*100}% generating embeddings to testset  for t={t}")

        along_path_embeddings = np.array(along_path_embeddings)
        return along_path_embeddings


    @staticmethod
    def calculate_data_distances_for_metric(X_train, y_train, num_classes, force_zero_for_same_class=True,
                                            data_metric='euclidean'):
        y_train = y_train.flatten()
        classes_data = [X_train[y_train == i] for i in range(num_classes)]
        classes_mean_distances = np.zeros((num_classes, num_classes))
        for i, data1 in enumerate(classes_data):
            for j, data2 in enumerate(classes_data):
                distances = cdist(data1, data2, metric=data_metric)
                if force_zero_for_same_class and i == j:
                    classes_mean_distances[i, j] = 0
                else:
                    classes_mean_distances[i, j] = np.mean(distances)

        return classes_mean_distances

    @staticmethod
    def generate_discrete_labels_distance(classes_mean_distances):
        def discrete_labels_distance(label1, label2):
            if label1 == -1 and label2 == -1:
                return np.mean(classes_mean_distances)
            elif label1 == -1:
                return np.mean(classes_mean_distances[:, int(label2[0])])
            elif label2 == -1:
                return np.mean(classes_mean_distances[int(label1[0]), :])
            else:
                return classes_mean_distances[int(label1[0]), int(label2[0])]

        return discrete_labels_distance

    @staticmethod
    def compute_kernel_along_path_for_t(kernel1, kernel2, t):
        # kernel1_sqrt = sqrtm(kernel1)
        U, sigma, Vt = np.linalg.svd(kernel1)
        sqrt_sigma = np.sqrt(sigma)
        Sigma_sqrt = np.diag(sqrt_sigma)
        kernel1_sqrt = U @ Sigma_sqrt @ Vt

        if np.iscomplex(kernel1_sqrt).any():
            raise RuntimeError(f"complex numbers in kernel1_sqrt, smaller eps might solve it")
        kernel1_sqrt_inv = np.linalg.inv(kernel1_sqrt)

        # fractional matrix power by t using svd
        to_raise = kernel1_sqrt_inv @ kernel2 @ kernel1_sqrt_inv
        U, Sigma, Vt = np.linalg.svd(to_raise)
        Sigma_t = np.diag(Sigma ** t)
        raised_by_t = U @ Sigma_t @ Vt
        chosen_kernel = kernel1_sqrt @ raised_by_t @ kernel1_sqrt

        # chosen_kernel = kernel1_sqrt @ (
        #     fractional_matrix_power(kernel1_sqrt_inv @ kernel2 @ kernel1_sqrt_inv, t)) @ kernel1_sqrt

        if np.iscomplex(chosen_kernel).any():
            raise RuntimeError(f"complex numbers in chosen_kernel. i saw that it happen when the combination of the "
                               f"two kernels eps was not good. you probably want to make one of the eps bigger.")

        return chosen_kernel

