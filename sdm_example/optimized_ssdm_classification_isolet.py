import time
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from optimized_semi_supervised_diffusion_maps.optimized_ssdm import optimized_SSDM
from sdm_example.datasets_loaders import load_isolet_dataset
from sdm_example.tools import plot_2d_visualization, scale_data


def main():
    # load dataset
    train_data, test_data, train_labels, test_labels = load_isolet_dataset()

    # set parameters
    n_components = 26
    selected_t = -1  # select best t along interpolation using leave-1-out on train set or any other method,
                     # for simplicity here we don't use t
    data_eps = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(train_data) < 10000:
        sample_ratio = 0.1
    else:
        sample_ratio = 0.01

    # evaluate semi-SDM with selected t on test set
    model = optimized_SSDM(n_components=n_components, labels_type='classification', data_eps=data_eps,
                           sample_ratio=sample_ratio, device=device)
    t0 = time.time()
    train_embeddings, test_embeddings = model.fit_transform(train_data, train_labels, test_data, t=selected_t)
    print(f"runtime: {time.time() - t0}")

    X_train, X_test = scale_data(train_embeddings, test_embeddings)
    clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, train_labels.flatten().astype(int))
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"Misclassification Rate for semi-SDM using n_components={n_components} and selected t={selected_t}: {1-accuracy}")
    plot_2d_visualization(train_embeddings, train_labels, test_embeddings, test_labels,
                          dataset_type='classification', algorithm_name='optimized SSDM')


if __name__ == '__main__':
    main()



