import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def fetch_dataset_UCI_ML_Repository_handle_server_exception(dataset_id):
    while True:
        try:
            dataset = fetch_ucirepo(id=dataset_id)
            return dataset
        except Exception as e:
            print(f"error in fetching dataset. Retrying in 10 seconds...")
            time.sleep(10)

def split_and_force_2_samples_for_class_and_nan_policy(random_state, X, y, dataset_type):
    unique_labels = np.unique(y)
    train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=random_state)
    if dataset_type == 'classification':
        # Ensure each class has at least 2 samples in the training set
        unique, counts = np.unique(train_labels, return_counts=True)
        while np.any(counts < 2) or len(unique) < len(unique_labels):
            random_state += 10000
            train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=random_state)
            unique, counts = np.unique(train_labels, return_counts=True)

    # fill nan values with the mean of the feature computed on the train set
    imputer = SimpleImputer(strategy='mean')
    train = imputer.fit_transform(train)
    test = imputer.transform(test)

    train_labels = train_labels.reshape((-1, 1))
    test_labels = test_labels.reshape((-1, 1))

    return train, test, train_labels, test_labels


def load_yacht_dataset(random_state=0):
    yacht_data = pd.read_csv(f"./yacht_hydrodynamics.data", header=None, delim_whitespace=True, usecols=range(7))
    yacht_data = yacht_data.to_numpy()
    X = yacht_data[:, :-1]
    y = yacht_data[:, -1]

    train, test, train_labels, test_labels = split_and_force_2_samples_for_class_and_nan_policy(random_state, X, y, dataset_type='regression')

    return train, test, train_labels, test_labels


def load_ionosphere(random_state=0):
    # fetch dataset
    ionosphere_id = 52
    dataset = fetch_dataset_UCI_ML_Repository_handle_server_exception(ionosphere_id)
    X = dataset.data.features
    y = dataset.data.targets
    label_encoder = LabelEncoder()

    X = X.values
    y = y.values.flatten()
    y = label_encoder.fit_transform(y)

    train, test, train_labels, test_labels = split_and_force_2_samples_for_class_and_nan_policy(random_state, X, y, dataset_type='classification')

    return train, test, train_labels, test_labels

def load_silhouettes_dataset(random_state=0):
    # fetch dataset
    statlog_id = 149
    dataset = fetch_dataset_UCI_ML_Repository_handle_server_exception(statlog_id)
    X = dataset.data.features
    y = dataset.data.targets
    label_encoder = LabelEncoder()

    ## REMOVE BAD SAMPLE ##
    X = X.drop(752)
    y = y.drop(752)
    #######################

    X = X.values
    y = y.values.flatten()
    y = label_encoder.fit_transform(y)

    train, test, train_labels, test_labels = split_and_force_2_samples_for_class_and_nan_policy(random_state, X, y, dataset_type='classification')

    return train, test, train_labels, test_labels