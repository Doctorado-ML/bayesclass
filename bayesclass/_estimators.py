"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from benchmark import Datasets


class TAN(ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.__train()
        # Return the classifier
        return self

    def __train(self):
        dt = Datasets()
        data = dt.load("balance-scale", dataframe=True)
        features = dt.dataset.features
        class_name = dt.dataset.class_name
        factorization, class_factors = pd.factorize(data[class_name])
        data[class_name] = factorization
        data.head()
        net = [(class_name, feature) for feature in features]
        model = BayesianNetwork(net)
        # 1st feature correlates with other features
        first_node = features[0]
        edges2 = [
            (first_node, feature)
            for feature in features
            if feature != first_node
        ]
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                edges.append((features[i], features[j]))
        print(edges2)
        model.add_edges_from(edges2)
        nx.draw_circular(
            model,
            with_labels=True,
            arrowsize=30,
            node_size=800,
            alpha=0.3,
            font_weight="bold",
        )
        plt.show()
        discretiz = MDLP()
        Xdisc = discretiz.fit_transform(
            data[features].to_numpy(), data[class_name].to_numpy()
        )
        features_discretized = pd.DataFrame(Xdisc, columns=features)
        dataset_discretized = features_discretized.copy()
        dataset_discretized[class_name] = data[class_name]
        dataset_discretized
        model.fit(dataset_discretized)
        from pgmpy.estimators import TreeSearch

        # learn graph structure
        est = TreeSearch(dataset_discretized, root_node=first_node)
        dag = est.estimate(estimator_type="tan", class_node=class_name)
        nx.draw_circular(
            dag,
            with_labels=True,
            arrowsize=30,
            node_size=800,
            alpha=0.3,
            font_weight="bold",
        )
        plt.show()

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
