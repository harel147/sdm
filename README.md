# SDM and SSDM

Supervised Diffusion Maps (SDM) and Semi-Supervised Diffusion Maps (SSDM), transform the well-known unsupervised dimensionality reduction algorithm, Diffusion Maps, into supervised and semi-supervised learning tools.

More details can be found in our paper [TODO](https://arxiv.org/******)

*first author (TODO)*, *second author (TODO)*

## Installing
You can install all the required packages by executing the following command:

```bash
pip install -r requirements_python_3_08.txt
```

## How to use SDM and SSDM
Check out the examples for using SDM with regression and classification datasets in `supervised_regression_example.py` and
`supervised_classification_example.py` for the supervised setting, and SSDM in `semi_supervised_regression_example.py` and
`semi_supervised_classification_example.py` for the semi-supervised setting.

In general, our API is similar to the well-known sklearn transformers API. For SDM (supervised setting):

```python
train_data, test_data, train_labels, test_labels = ...
selected_t = 0.03  # in (0, 1)
n_components = 2
labels_type = 'regression'  # or 'classification'
model = SDM(n_components=n_components, labels_type=labels_type, setting='supervised')
sdm_train_embeddings = model.fit_transform(train_data, train_labels, t=selected_t)
sdm_test_embeddings = model.transform(test_data, t=selected_t)
```

For SSDM (semi-supervised setting):

```python
train_data, test_data, train_labels, test_labels = ...
selected_t = 0.93  # in (0, 1)
n_components = 2
labels_type = 'regression'  # or 'classification'
model = SDM(n_components=n_components, labels_type=labels_type, setting='semi-supervised')
sdm_train_embeddings, sdm_test_embeddings = model.fit_transform(train_data, train_labels, test_data, t=selected_t)
```

## Examples
Visualizations of two-dimensional data from the [Yacht Hydrodynamics dataset](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics) (continuous labels):

![Yacht_vizs](https://github.com/harel147/sdm/assets/63463677/1271d5e4-802d-4bfb-9f4f-771ee9675f15)

SDM (Supervised setting): Classification results for the
[Ionosphere dataset](https://archive.ics.uci.edu/dataset/52/ionosphere): Misclassification Rate after training a
KNN on the embeddings obtained from each dimension reduction algorithm:

<img src="https://github.com/user-attachments/assets/1a2e27e8-55ce-4382-99b6-562d2a8704ce" alt="Ionosphere_results_resized" width="600"/>

SSDM (Semi-supervised setting): Classification results for the
[Vehicle Silhouettes dataset](https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes): Misclassification Rate after training a
KNN on the embeddings obtained from each dimension reduction algorithm:

<img src="https://github.com/user-attachments/assets/d43a6cf2-b42b-4c01-b249-e6c9fe4a6046" alt="Silhouettes_results_resized" width="600"/>

