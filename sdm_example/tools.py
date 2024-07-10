import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def scale_data(train_data, test_data):
    scaler = StandardScaler()
    X_train = pd.DataFrame(train_data)
    X_test = pd.DataFrame(test_data)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def normalized_mse(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    squared_errors = np.square(y_true - y_pred)
    sum_squared_labels = np.sum(np.square(y_true))
    nmse = np.sum(squared_errors) / sum_squared_labels
    return nmse

def plot_2d_visualization(train_data, train_labels, test_data, test_labels, dataset_type, algorithm_name):
    if dataset_type == 'classification':
        num_classes = len(np.unique(test_labels))
        cmap = plt.cm.get_cmap('viridis', num_classes)
        scatter_train = plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=cmap, marker='o')
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
