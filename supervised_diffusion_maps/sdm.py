import numpy as np
from scipy.spatial.distance import cdist
from diffusion_maps.my_diffusion_maps import MyDiffusionMaps


class SDM:
    def __init__(self, n_components=30, labels_type='regression', data_eps='auto', labels_eps='auto',
                 data_dist_metric='euclidean', labels_dist_metric='euclidean', setting=None):
        self.n_components = n_components
        self.labels_type = labels_type
        self.data_eps = data_eps
        self.labels_eps = labels_eps
        self.data_dist_metric = data_dist_metric
        self.labels_dist_metric = labels_dist_metric
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.setting = setting

        if self.setting != 'supervised' and self.setting != 'semi-supervised':
            raise RuntimeError(f"setting must be set to 'supervised' or 'semi-supervised'")

    def preprocess(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

        if self.labels_type == 'classification':
            num_classes = len(np.unique(train_labels))
            # You need to copy classes_mean_distances manually to create discrete_labels_distance metric
            classes_mean_distances = SDM.calculate_data_distances_for_metric(train_data, train_labels, num_classes, data_metric=self.data_dist_metric)
            # generate the distance metric function dynamically based on computed classes_mean_distances for this fold
            discrete_labels_distance = SDM.generate_discrete_labels_distance(classes_mean_distances)
            self.labels_dist_metric = discrete_labels_distance

        if self.data_eps == 'auto':
            self.data_eps = MyDiffusionMaps.auto_select_aps(train_data, self.data_dist_metric)
        if self.labels_eps == 'auto':
            self.labels_eps = MyDiffusionMaps.auto_select_aps(train_labels, self.labels_dist_metric)

    def fit_transform_supervised(self, train_data, train_labels, t=0.5):
        self.preprocess(train_data, train_labels)

        dm_train_data = MyDiffusionMaps(eps=self.data_eps)
        dm_train_data.compute_kernel(self.train_data)
        along_path_embeddings = []

        for i in range(len(self.train_labels)):
            train_labels_with_prior = np.copy(self.train_labels).astype(np.float64)
            if self.labels_type == 'classification':
                # we use -1 to mark the unlabeled sample, this is used
                train_labels_with_prior[i] = np.array([-1]).reshape(-1, 1)
            else:
                # this doesn't matter as distances for prior label are set to infinity later inside compute_kernel.
                train_labels_with_prior[i] = 0
            prior_index = i
            dm_train_labels_with_prior = MyDiffusionMaps(eps=self.labels_eps)
            dm_train_labels_with_prior.compute_kernel(train_labels_with_prior, metric=self.labels_dist_metric,
                                                      prior_index=prior_index, prior_type='single')

            start_kernel = dm_train_labels_with_prior.K_norm2
            end_kernel = dm_train_data.K_norm2

            U1, s1, Vt1 = np.linalg.svd(start_kernel)
            U2, s2, Vt2 = np.linalg.svd(end_kernel)

            along_path_kernel = SDM.compute_kernel_along_path_for_t(U1, s1, Vt1, U2, s2, Vt2, t=t)

            dm_along_path = MyDiffusionMaps(K_norm2=along_path_kernel)
            dm_along_path.calculate_kernel_eigen()
            dm_along_path.calculate_weighted_vectors()

            along_path_embeddings_i = dm_along_path.project_sample(sample_num=i, out_dims=self.n_components)
            along_path_embeddings.append(along_path_embeddings_i)

            if i % 10 == 0:
                print(f"done {round(i / len(self.train_labels), 2) * 100}% generating embeddings to trainset for t={t}")

        along_path_embeddings = np.array(along_path_embeddings)
        return along_path_embeddings

    def transform_supervised(self, test_data, t=0.5):
        if self.train_data is None:
            raise RuntimeError(f"You must use fit or fit_transform before using transform")

        train_labels_with_prior = np.append(self.train_labels, np.array([-1]).reshape(-1, 1), axis=0)
        prior_index = len(train_labels_with_prior) - 1
        dm_train_labels_with_prior = MyDiffusionMaps(eps=self.labels_eps)
        dm_train_labels_with_prior.compute_kernel(train_labels_with_prior, metric=self.labels_dist_metric,
                                                  prior_index=prior_index, prior_type='single')
        along_path_embeddings = []
        for i in range(test_data.shape[0]):
            train_data_with_unseen_sample = np.append(self.train_data, test_data[i].reshape(1, -1), axis=0)
            dm_train_data = MyDiffusionMaps(eps=self.data_eps)
            dm_train_data.compute_kernel(train_data_with_unseen_sample)

            start_kernel = dm_train_labels_with_prior.K_norm2
            end_kernel = dm_train_data.K_norm2

            U1, s1, Vt1 = np.linalg.svd(start_kernel)
            U2, s2, Vt2 = np.linalg.svd(end_kernel)

            along_path_kernel = self.compute_kernel_along_path_for_t(U1, s1, Vt1, U2, s2, Vt2, t=t)
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

    def fit_transform_semi_supervised(self, train_data, train_labels, test_data, t=0.5):
        self.preprocess(train_data, train_labels)

        array_of_minus_ones = np.ones((len(test_data), 1)) * -1
        train_labels_with_prior = np.vstack((self.train_labels, array_of_minus_ones))
        prior_index = len(self.train_labels)
        dm_train_labels_with_prior = MyDiffusionMaps(eps=self.labels_eps)
        dm_train_labels_with_prior.compute_kernel(train_labels_with_prior, metric=self.labels_dist_metric,
                                                  prior_index=prior_index, prior_type='batch')

        train_data_with_unseen_sample = np.vstack((self.train_data, test_data))
        dm_train_data = MyDiffusionMaps(eps=self.data_eps)
        dm_train_data.compute_kernel(train_data_with_unseen_sample)

        start_kernel = dm_train_labels_with_prior.K_norm2
        end_kernel = dm_train_data.K_norm2
        U1, s1, Vt1 = np.linalg.svd(start_kernel)
        U2, _, Vt2 = np.linalg.svd(end_kernel)

        # SVD shrinkage-based denoising tactic to the data kernel
        sigma = np.linspace(-5, 5, start_kernel.shape[0])  # Spread the range to make the sigmoid gradual
        sigma = 1 / (1 + np.exp(sigma))  # Sigmoid function
        sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
        s2 = sigma

        along_path_kernel = SDM.compute_kernel_along_path_for_t(U1, s1, Vt1, U2, s2, Vt2, t=t)

        dm_along_path = MyDiffusionMaps(K_norm2=along_path_kernel)
        dm_along_path.calculate_kernel_eigen()
        dm_along_path.calculate_weighted_vectors()

        along_path_embeddings = dm_along_path.project_to_low_dimensions(train_data_with_unseen_sample,
                                                                        out_dims=self.n_components)
        along_path_embeddings_train = along_path_embeddings[:len(self.train_labels), :]
        along_path_embeddings_test = along_path_embeddings[len(self.train_labels):, :]

        return along_path_embeddings_train, along_path_embeddings_test

    def fit(self, train_data, train_labels):
        if self.setting != 'supervised':
            print("fit() only support 'supervised' setting")

        self.preprocess(train_data, train_labels)

    def transform(self, test_data, t=0.5):
        if self.setting != 'supervised':
            print("transform() only support 'supervised' setting")

        embeddings_test = self.transform_supervised(test_data, t)
        return embeddings_test

    def fit_transform(self, train_data, train_labels, test_data=None, t=0.5):
        if self.setting == 'supervised':
            if test_data is not None:
                raise RuntimeError(f"fit_transform() don't accept test_data in the 'supervised' setting")
            embeddings_train = self.fit_transform_supervised(train_data, train_labels, t)
            return embeddings_train
        elif self.setting == 'semi-supervised':
            if test_data is None:
                raise RuntimeError(f"fit_transform() have to get test_data in the 'semi-supervised' setting")
            embeddings_train, embeddings_test = self.fit_transform_semi_supervised(train_data, train_labels, test_data, t)
            return embeddings_train, embeddings_test

    @staticmethod
    def calculate_data_distances_for_metric(X_train, y_train, num_classes, force_zero_for_same_class=True,
                                            data_metric='euclidean'):

        X_train, y_train = X_train[:2000], y_train[:2000]  # if train is too big this takes too much time, 2000 samples is enough.

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
            if label1 == -1 or label2 == -1:
                # this doesn't matter as distances for prior label are set to infinity later inside compute_kernel.
                return 0
            else:
                return classes_mean_distances[int(label1[0]), int(label2[0])]

        return discrete_labels_distance

    @staticmethod
    def compute_kernel_along_path_for_t(U1, s1, Vt1, U2, s2, Vt2, t):
        sigma_raise = np.diag(s1 ** (1 - t))
        kernel1_reconstruct_and_raised = U1 @ sigma_raise @ Vt1

        sigma_raise = np.diag(s2 ** (t))
        kernel2_reconstruct_and_raised = U2 @ sigma_raise @ Vt2

        to_raise = (kernel1_reconstruct_and_raised @ kernel2_reconstruct_and_raised)  # asymmetric but same components

        return to_raise

