import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


from supervised_diffusion_maps.sdm import SDM


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    X_train = pd.DataFrame(train_data)
    X_test = pd.DataFrame(test_data)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def leave_1_out_sdm_select_optimal_t(train_data, train_labels, t_values, n_components, dataset_type):
    best_t = None
    best_error = np.inf

    for t_value in t_values:
        loo = LeaveOneOut()
        errors = []

        model = SDM(n_components=n_components, labels_type=dataset_type)
        train_reduced = model.fit_transform(train_data, train_labels, t=t_value)

        for i, (train_index, test_index) in enumerate(loo.split(train_data)):
            # if i == 100:
            #     break

            X_train, X_test = train_reduced[train_index], train_reduced[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]

            X_train, X_test = scale_data(X_train, X_test)
            if dataset_type == "classification":
                clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train.flatten().astype(int))
                y_predict = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_predict)
                errors.append(1 - accuracy)
            elif dataset_type == "regression":
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(X_train, y_train.flatten())
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                errors.append(mse)

        avg_error = sum(errors) / len(errors)
        print(f"leave-1-out train set error for t={t_value}: {avg_error}")

        if avg_error < best_error:
            best_error = avg_error
            best_t = t_value

    print(f"best t is {best_t} with error of {best_error}")
    return best_t


def plot_2d_visualization(train_data, train_labels, test_data, test_labels, dataset_type, algorithm_name):
    if dataset_type == 'classification':
        num_classes = len(np.unique(test_labels))
        cmap = plt.cm.get_cmap('viridis', num_classes)
        scatter_train = plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=cmap, marker='o')
        scatter_test = plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=cmap, marker='+')
        plt.legend(loc='upper right', handles=scatter_train.legend_elements()[0], labels=range(num_classes), title="Classes")
    elif dataset_type == 'regression':
        cmap = 'viridis'
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=cmap, marker='o')
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=cmap, marker='+')
        plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'2D for {algorithm_name}')
    plt.show()