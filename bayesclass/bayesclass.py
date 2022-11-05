"""
This is a module to be used as a reference for building other modules
"""
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import networkx as nx
from pgmpy.estimators import TreeSearch, BayesianEstimator
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

    def __init__(self, simple_init=False, show_progress=False):
        self.simple_init = simple_init
        self.show_progress = show_progress

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
            head: int (default=0) Index of the head node
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
        self.head_ = 0
        expected_args = ["class_name", "features", "head"]
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)

            else:
                raise ValueError(f"Unexpected argument: {key}")

        if len(self.features_) != X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        if self.head_ >= len(self.features_):
            raise ValueError("Head index out of range")

        self.X_ = X
        self.y_ = y
        self.__train()
        # Return the classifier
        return self

    def __initial_edges(self):
        if self.simple_init:
            first_node = self.features_[self.head_]
            return [
                (first_node, feature)
                for feature in self.features_
                if feature != first_node
            ]
        edges = []
        for i in range(len(self.features_)):
            for j in range(i + 1, len(self.features_)):
                edges.append((self.features_[i], self.features_[j]))
        return edges

    def __train(self):
        net = [(self.class_name_, feature) for feature in self.features_]
        self.model_ = BayesianNetwork(net)
        # initialize a complete network with all edges
        self.model_.add_edges_from(self.__initial_edges())

        self.dataset_ = pd.DataFrame(self.X_, columns=self.features_)
        self.dataset_[self.class_name_] = self.y_
        # learn graph structure
        est = TreeSearch(self.dataset_, root_node=self.features_[self.head_])
        dag = est.estimate(
            estimator_type="tan",
            class_node=self.class_name_,
            show_progress=self.show_progress,
        )
        self.model_ = BayesianNetwork(dag.edges())
        self.model_.fit(
            self.dataset_,
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
