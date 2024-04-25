.. -*- mode: rst -*-
====
SDM
====

Supervised Diffusion Maps [SDM] is a novel approach that transforms the well-known
unsupervised dimension reduction algorithm, Diffusion Maps, into a supervised learning tool.

The details for the underlying mathematics can be found in
`our paper on ArXiv <https://arxiv.org/******>`_:

Mendelman, H, Talmon, R, *add paper name here*

----------
Installing
----------
You can install all necessary packages by running the following command:

.. code:: bash

    pip install -r requirements_python_3_08.txt


---------------
How to use SDM
---------------

See examples for using SDM for regression and classification datasets in `regression_example.py` and
`classification_example.py`.

In general, SDM has similar API to the well known sklearn transformers API:

.. code:: python

    train_data, test_data, train_labels, test_labels = ...
    best_t = 0.43  # selected using leave-1-out on the training set
    n_components = 2
    labels_type = 'regression'  # or 'classification'
    model = SDM(n_components=n_components, labels_type=labels_type)
    sdm_train_embeddings = model.fit_transform(train_data, train_labels, t=best_t)
    sdm_test_embeddings = model.transform(test_data, t=best_t)

------------------------
Examples
------------------------
Regression results for on the
`Yacht Hydrodynamics dataset <https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics>`_, NMSE after training
KNN Regressor on the embeddings obtained from each dimension reduction algorithm:

.. image:: images/Yacht_results.png
    :alt: Yacht_results

Classification results for on the
`Ionosphere dataset <https://archive.ics.uci.edu/dataset/52/ionosphere>`_, Misclassification Rate after training
KNN on the embeddings obtained from each dimension reduction algorithm:

.. image:: images/Ionosphere_results.png
    :alt: Ionosphere_results

-------
License
-------
The sdm package is 3-clause BSD licensed (to do).

