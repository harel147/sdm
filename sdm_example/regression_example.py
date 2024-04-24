import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from supervised_diffusion_maps.sdm import SDM
from sdm_example.datasets_loaders import load_yacht_dataset
from sdm_example.tools import leave_1_out_sdm_select_optimal_t, plot_2d_visualization


def main():
    # load dataset
    train_data, test_data, train_labels, test_labels = load_yacht_dataset()

    # set parameters
    n_components = 2
    f = 5
    t_values = np.linspace(0, 1.0, f)

    # select best t along geodesic using leave-1-out on train set
    best_t = leave_1_out_sdm_select_optimal_t(train_data, train_labels, t_values, n_components, dataset_type='regression')

    # evaluate SDM with selected t on test set
    model = SDM(n_components=n_components, labels_type='regression')
    sdm_train_embeddings = model.fit_transform(train_data, train_labels, t=best_t)
    sdm_test_embeddings = model.transform(test_data, t=best_t)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(sdm_train_embeddings, train_labels.flatten())
    y_pred = model.predict(sdm_test_embeddings)
    mse = mean_squared_error(test_labels, y_pred)
    print(f"MSE for SDM using n_components={n_components} and selected t={best_t}: {mse}")
    plot_2d_visualization(sdm_train_embeddings, train_labels, sdm_test_embeddings, test_labels,
                          dataset_type='regression', algorithm_name='SDM')

    # evaluate unsupervised diffusion maps (t=1.0) on test set for comparison
    model = SDM(n_components=n_components, labels_type='regression')
    dm_train_embeddings = model.fit_transform(train_data, train_labels, t=1.0)
    dm_test_embeddings = model.transform(test_data, t=1.0)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(dm_train_embeddings, train_labels.flatten())
    y_pred = model.predict(dm_test_embeddings)
    mse = mean_squared_error(test_labels, y_pred)
    print(f"MSE for diffusion maps using n_components={n_components}: {mse}")
    plot_2d_visualization(dm_train_embeddings, train_labels, dm_test_embeddings, test_labels,
                          dataset_type='regression', algorithm_name='diffusion maps')


if __name__ == '__main__':
    main()



