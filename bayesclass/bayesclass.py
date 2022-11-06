"""
This is a module to be used as a reference for building other modules
"""
import random
from itertools import combinations
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import networkx as nx
from pgmpy.estimators import (
    TreeSearch,
    BayesianEstimator,
    MaximumLikelihoodEstimator,
)
from pgmpy.models import BayesianNetwork
import matplotlib.pyplot as plt


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

    def __init__(
        self, simple_init=False, show_progress=False, random_state=None
    ):
        self.simple_init = simple_init
        self.show_progress = show_progress
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        **kwargs : dict
            class_name : str (default='class') Name of the class column
            features: list (default=None) List of features
            head: int (default=None) Index of the head node. Default value
            gets the node with the highest sum of weights (mutual_info)
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Default values
        self.class_name_ = "class"
        self.features_ = [f"feature_{i}" for i in range(X.shape[1])]
        self.head_ = None
        expected_args = ["class_name", "features", "head"]
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        if self.random_state is not None:
            random.seed(self.random_state)
        if self.head_ == "random":
            self.head_ = random.randint(0, len(self.features_) - 1)
        if len(self.features_) != X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        if self.head_ is not None and self.head_ >= len(self.features_):
            raise ValueError("Head index out of range")

        self.X_ = X
        self.y_ = y
        self.__train()
        # Return the classifier
        return self

    def __initial_edges(self):
        """As with the naive Bayes, in a TAN structure, the class has no
        parents, while features must have the class as parent and are forced to
        have one other feature as parent too (except for one single feature,
        which has only the class as parent and is considered the root of the
        features' tree)
        Cassio P. de Campos, Giorgio Corani, Mauro Scanagatta, Marco Cuccu,
        Marco Zaffalon,
        Learning extended tree augmented naive structures,
        International Journal of Approximate Reasoning,
        Returns
        -------
        List
            List of edges
        """
        head = 0 if self.head_ is None else self.head_
        if self.simple_init:
            first_node = self.features_[head]
            return [
                (first_node, feature)
                for feature in self.features_
                if feature != first_node
            ]
        # initialize a complete network with all edges starting from head
        reordered = [
            self.features_[idx % len(self.features_)]
            for idx in range(head, len(self.features_) + head)
        ]
        return list(combinations(reordered, 2))

    def __train(self):
        # Initialize a Naive Bayes model
        net = [(self.class_name_, feature) for feature in self.features_]
        self.model_ = BayesianNetwork(net)
        # initialize a complete network with all edges
        self.model_.add_edges_from(self.__initial_edges())
        self.dataset_ = pd.DataFrame(self.X_, columns=self.features_)
        self.dataset_[self.class_name_] = self.y_
        # learn graph structure
        root_node = None if self.head_ is None else self.features_[self.head_]
        est = TreeSearch(self.dataset_, root_node=root_node)
        dag = est.estimate(
            estimator_type="tan",
            class_node=self.class_name_,
            show_progress=self.show_progress,
        )
        if self.head_ is None:
            self.head_ = est.root_node
        self.model_ = BayesianNetwork(dag.edges())
        self.model_.fit(
            self.dataset_,
            # estimator=MaximumLikelihoodEstimator,
            estimator=BayesianEstimator,
            prior_type="K2",
        )

    def plot(self, title=""):
        nx.draw_circular(
            self.model_,
            with_labels=True,
            arrowsize=30,
            node_size=800,
            alpha=0.3,
            font_weight="bold",
        )
        plt.title(title)
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
        dataset = pd.DataFrame(X, columns=self.features_)
        return self.model_.predict(dataset).to_numpy()
