from sklearn.neighbors import KNeighborsRegressor

from supervised_diffusion_maps.sdm import SDM
from sdm_example.datasets_loaders import load_yacht_dataset
from sdm_example.tools import plot_2d_visualization, normalized_mse


def main():
    # load dataset
    train_data, test_data, train_labels, test_labels = load_yacht_dataset()

    # set parameters
    n_components = 2
    selected_t = 0.02  # select best t along interpolation using leave-1-out on train set or any other method

    # evaluate SDM with selected t on test set
    model = SDM(n_components=n_components, labels_type='regression', setting='supervised')
    sdm_train_embeddings = model.fit_transform(train_data, train_labels, t=selected_t)
    sdm_test_embeddings = model.transform(test_data, t=selected_t)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(sdm_train_embeddings, train_labels.flatten())
    y_pred = model.predict(sdm_test_embeddings)
    nmse = normalized_mse(test_labels, y_pred)
    print(f"NMSE for SDM using n_components={n_components} and selected t={selected_t}: {nmse}")
    plot_2d_visualization(sdm_train_embeddings, train_labels, sdm_test_embeddings, test_labels,
                          dataset_type='regression', algorithm_name='SDM')

    # evaluate unsupervised diffusion maps (t=1.0) on test set for comparison
    model = SDM(n_components=n_components, labels_type='regression', setting='supervised')
    dm_train_embeddings = model.fit_transform(train_data, train_labels, t=1.0)
    dm_test_embeddings = model.transform(test_data, t=1.0)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(dm_train_embeddings, train_labels.flatten())
    y_pred = model.predict(dm_test_embeddings)
    nmse = normalized_mse(test_labels, y_pred)
    print(f"NMSE for diffusion maps using n_components={n_components}: {nmse}")
    plot_2d_visualization(dm_train_embeddings, train_labels, dm_test_embeddings, test_labels,
                          dataset_type='regression', algorithm_name='diffusion maps')


if __name__ == '__main__':
    main()



