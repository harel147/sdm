.. -*- mode: rst -*-
====
SDM
====

Supervised Diffusion Maps [SDM] is a novel approach that transforms the well-known
unsupervised dimension reduction algorithm, Diffusion Maps, into a supervised learning tool.

The details for the underlying mathematics can be found in
`our paper on ArXiv <https://arxiv.org/******>`_:

----------
Installing
----------


---------------
How to use SDM
---------------

The umap package inherits from sklearn classes, and thus drops in neatly
next to other sklearn transformers with an identical calling API.

.. code:: python

    import umap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = umap.UMAP().fit_transform(digits.data)

------------------------
Performance and Examples
------------------------

UMAP is very efficient at embedding large high dimensional datasets. In
particular it scales well with both input dimension and embedding dimension.
For the best possible performance we recommend installing the nearest neighbor
computation library `pynndescent <https://github.com/lmcinnes/pynndescent>`_ .
UMAP will work without it, but if installed it will run faster, particularly on
multicore machines.

For a problem such as the 784-dimensional MNIST digits dataset with
70000 data samples, UMAP can complete the embedding in under a minute (as
compared with around 45 minutes for scikit-learn's t-SNE implementation).
Despite this runtime efficiency, UMAP still produces high quality embeddings.

The obligatory MNIST digits dataset, embedded in 42
seconds (with pynndescent installed and after numba jit warmup)
using a 3.1 GHz Intel Core i7 processor (n_neighbors=10, min_dist=0.001):

.. image:: images/umap_example_mnist1.png
    :alt: UMAP embedding of MNIST digits

The MNIST digits dataset is fairly straightforward, however. A better test is
the more recent "Fashion MNIST" dataset of images of fashion items (again
70000 data sample in 784 dimensions). UMAP
produced this embedding in 49 seconds (n_neighbors=5, min_dist=0.1):

.. image:: images/umap_example_fashion_mnist1.png
    :alt: UMAP embedding of "Fashion MNIST"

The UCI shuttle dataset (43500 sample in 8 dimensions) embeds well under
*correlation* distance in 44 seconds (note the longer time
required for correlation distance computations):

.. image:: images/umap_example_shuttle.png
    :alt: UMAP embedding the UCI Shuttle dataset

The following is a densMAP visualization of the MNIST digits dataset with 784 features
based on the same parameters as above (n_neighbors=10, min_dist=0.001). densMAP reveals
that the cluster corresponding to digit 1 is noticeably denser, suggesting that
there are fewer degrees of freedom in the images of 1 compared to other digits.

.. image:: images/densmap_example_mnist.png
    :alt: densMAP embedding of the MNIST dataset

-------
License
-------
The umap package is 3-clause BSD licensed.

