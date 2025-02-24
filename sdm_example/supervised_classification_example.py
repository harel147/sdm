from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from supervised_diffusion_maps.sdm import SDM
from sdm_example.datasets_loaders import load_ionosphere
from sdm_example.tools import plot_2d_visualization, scale_data


def main():
    # load dataset
    train_data, test_data, train_labels, test_labels = load_ionosphere()

    # set parameters
    n_components = 2
    selected_t = 0.03  # select best t along interpolation using leave-1-out on train set or any other method

    # evaluate SDM with selected t on test set
    model = SDM(n_components=n_components, labels_type='classification', setting='supervised')
    sdm_train_embeddings = model.fit_transform(train_data, train_labels, t=selected_t)
    sdm_test_embeddings = model.transform(test_data, t=selected_t)

    X_train, X_test = scale_data(sdm_train_embeddings, sdm_test_embeddings)
    clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, train_labels.flatten().astype(int))
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Misclassification Rate for SDM using n_components={n_components} and selected t={selected_t}: {1-accuracy}")
    plot_2d_visualization(sdm_train_embeddings, train_labels, sdm_test_embeddings, test_labels,
                          dataset_type='classification', algorithm_name='SDM')

    # evaluate unsupervised diffusion maps (t=1.0) on test set for comparison
    model = SDM(n_components=n_components, labels_type='classification', setting='supervised')
    dm_train_embeddings = model.fit_transform(train_data, train_labels, t=1.0)
    dm_test_embeddings = model.transform(test_data, t=1.0)

    clf = KNeighborsClassifier(n_neighbors=1).fit(dm_train_embeddings, train_labels.flatten().astype(int))
    y_pred = clf.predict(dm_test_embeddings)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Misclassification Rate for diffusion maps using n_components={n_components}: {1 - accuracy}")
    plot_2d_visualization(dm_train_embeddings, train_labels, dm_test_embeddings, test_labels,
                          dataset_type='classification', algorithm_name='diffusion maps')

if __name__ == '__main__':
    main()



