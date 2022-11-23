import random
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import mutual_info_classif
import networkx as nx
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
import matplotlib.pyplot as plt
from ._version import __version__


class BayesBase(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state, show_progress):
        self.random_state = random_state
        self.show_progress = show_progress

    def _more_tags(self):
        return {
            "requires_positive_X": True,
            "requires_positive_y": True,
            "preserve_dtype": [np.int64, np.int32],
            "requires_y": True,
        }

    @staticmethod
    def version() -> str:
        """Return the version of the package."""
        return __version__

    def nodes_leaves(self):
        """To keep compatiblity with the benchmark platform"""
        return 0, 0

    def _check_params_fit(self, X, y, expected_args, kwargs):
        """Check the common parameters passed to fit"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        # Default values
        self.class_name_ = "class"
        self.features_ = [f"feature_{i}" for i in range(X.shape[1])]
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        if self.random_state is not None:
            random.seed(self.random_state)
        if len(self.features_) != X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        return X, y

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

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from bayesclass.clfs import TAN
        >>> features = ['A', 'B', 'C', 'D', 'E']
        >>> np.random.seed(17)
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2,
        ...                       size=(1000, 5)), columns=features)
        >>> train_data = values[:800]
        >>> train_y = train_data['E']
        >>> predict_data = values[800:]
        >>> train_data = train_data.drop('E', axis=1)
        >>> model = TAN(random_state=17)
        >>> features.remove('E')
        >>> model.fit(train_data, train_y, features=features, class_name='E')
        TAN(random_state=17)
        """
        X_, y_ = self._check_params(X, y, kwargs)
        # Store the information needed to build the model
        self.X_ = X_
        self.y_ = y_
        self.dataset_ = pd.DataFrame(self.X_, columns=self.features_)
        self.dataset_[self.class_name_] = self.y_
        # Build the DAG
        self._build()
        # Train the model
        self._train()
        self.fitted_ = True
        # Return the classifier
        return self

    def _train(self):
        self.model_ = BayesianNetwork(
            self.dag_.edges(), show_progress=self.show_progress
        )
        self.model_.fit(
            self.dataset_,
            estimator=BayesianEstimator,
            prior_type="K2",
        )

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

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from bayesclass.clfs import TAN
        >>> features = ['A', 'B', 'C', 'D', 'E']
        >>> np.random.seed(17)
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2,
        ...                       size=(1000, 5)), columns=features)
        >>> train_data = values[:800]
        >>> train_y = train_data['E']
        >>> predict_data = values[800:]
        >>> train_data = train_data.drop('E', axis=1)
        >>> model = TAN(random_state=17)
        >>> features.remove('E')
        >>> model.fit(train_data, train_y, features=features, class_name='E')
        TAN(random_state=17)
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop('E', axis=1, inplace=True)
        >>> y_pred = model.predict(predict_data)
        >>> y_pred[:10]
        array([[0],
               [0],
               [1],
               [1],
               [0],
               [1],
               [1],
               [1],
               [0],
               [1]])
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_", "fitted_"])

        # Input validation
        X = check_array(X)
        dataset = pd.DataFrame(X, columns=self.features_, dtype="int16")
        return self.model_.predict(dataset).values.ravel()

    def plot(self, title="", node_size=800):
        nx.draw_circular(
            self.model_,
            with_labels=True,
            arrowsize=20,
            node_size=node_size,
            alpha=0.3,
            font_weight="bold",
        )
        plt.title(title)
        plt.show()


class TAN(BayesBase):
    """Tree Augmented Naive Bayes

    Parameters
    ----------
    random_state: int, default=None
        Random state for reproducibility
    show_progress: bool, default=False
        used in pgmpy to show progress bars

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    class_name_ : str
        The name of the class column
    features_ : list
        The list of features names
    head_ : int
        The index of the node used as head for the initial DAG
    dataset_ : pd.DataFrame
        The dataset used to train the model (X_ + y_)
    dag_ : nx.DiGraph
        The TAN DAG
    model_ : BayesianNetwork
        The actual classifier
    """

    def __init__(self, show_progress=False, random_state=None):
        super().__init__(
            show_progress=show_progress, random_state=random_state
        )

    def _check_params(self, X, y, kwargs):
        self.head_ = 0
        expected_args = ["class_name", "features", "head"]
        X, y = self._check_params_fit(X, y, expected_args, kwargs)
        if self.head_ == "random":
            self.head_ = random.randint(0, len(self.features_) - 1)
        if self.head_ is not None and self.head_ >= len(self.features_):
            raise ValueError("Head index out of range")
        return X, y

    def _build(self):
        est = TreeSearch(self.dataset_, root_node=self.features_[self.head_])
        self.dag_ = est.estimate(
            estimator_type="tan",
            class_node=self.class_name_,
            show_progress=self.show_progress,
        )


class KDB(BayesBase):
    def __init__(self, k, theta=0.03, show_progress=False, random_state=None):
        self.k = k
        self.theta = theta
        super().__init__(
            show_progress=show_progress, random_state=random_state
        )

    def _check_params(self, X, y, kwargs):
        expected_args = ["class_name", "features"]
        return self._check_params_fit(X, y, expected_args, kwargs)

    def _build(self):
        """
        1. For each feature Xi, compute mutual information, I(X;;C), where C is the class.
        2. Compute class conditional mutual information I(Xi;XjIC), f or each pair of features Xi and Xj, where i#j.
        3. Let the used variable list, S, be empty.
        4. Let the Bayesian network being constructed, BN, begin with a single class node, C.
        5. Repeat until S includes all domain features
        5.1. Select feature Xmax which is not in S and has the largest value I(Xmax;C).
        5.2. Add a node to BN representing Xmax.
        5.3. Add an arc from C to Xmax in BN.
        5.4. Add m =min(lSl,/c) arcs from m distinct features Xj in S with the highest value for I(Xmax;X,jC).
        5.5. Add Xmax to S.
        Compute the conditional probabilility infered by the structure of BN by using counts from DB, and output BN.
        """

        def add_m_edges(dag, idx, S_nodes, conditional_weights):
            n_edges = min(self.k, len(S_nodes))
            cond_w = conditional_weights.copy()
            exit_cond = self.k == 0
            num = 0
            while not exit_cond:
                max_minfo = np.argmax(cond_w[idx, :])
                if (
                    max_minfo in S_nodes
                    and cond_w[idx, max_minfo] > self.theta
                ):
                    try:
                        dag.add_edge(
                            self.features_[max_minfo], self.features_[idx]
                        )
                        num += 1
                    except ValueError:
                        # Loops are not allowed
                        pass
                cond_w[idx, max_minfo] = -1
                exit_cond = num == n_edges or np.all(cond_w[idx, :] <= 0)

        # 1. get the mutual information between each feature and the class
        mutual = mutual_info_classif(self.X_, self.y_, discrete_features=True)
        # 2. symmetric matrix where each element represents I(X, Y| class_node)
        conditional_weights = TreeSearch(
            self.dataset_
        )._get_conditional_weights(
            self.dataset_, self.class_name_, show_progress=self.show_progress
        )
        # 3.
        S_nodes = []
        # 4.
        dag = BayesianNetwork()
        dag.add_node(self.class_name_)  # , state_names=self.classes_)
        # 5. 5.1
        for idx in np.argsort(mutual):
            # 5.2
            feature = self.features_[idx]
            dag.add_node(feature)
            # 5.3
            dag.add_edge(self.class_name_, feature)
            # 5.4
            add_m_edges(dag, idx, S_nodes, conditional_weights)
            # 5.5
            S_nodes.append(idx)
        self.dag_ = dag


class AODE(BayesBase, BaseEnsemble):
    def __init__(self, show_progress=False, random_state=None):
        super().__init__(
            show_progress=show_progress, random_state=random_state
        )

    def _check_params(self, X, y, kwargs):
        expected_args = ["class_name", "features"]
        return self._check_params_fit(X, y, expected_args, kwargs)

    def _build(self):

        self.dag_ = None

    def _train(self):
        """Build SPODE estimators (Super Parent One Dependent Estimator)"""
        self.models_ = []
        class_edges = [(self.class_name_, f) for f in self.features_]
        for idx in range(len(self.features_)):
            feature_edges = [
                (self.features_[idx], f)
                for f in self.features_
                if f != self.features_[idx]
            ]
            feature_edges.extend(class_edges)
            model = BayesianNetwork(
                feature_edges, show_progress=self.show_progress
            )
            model.fit(
                self.dataset_,
                estimator=BayesianEstimator,
                prior_type="K2",
            )
            self.models_.append(model)

    def plot(self, title=""):
        for idx, model in enumerate(self.models_):
            self.model_ = model
            super().plot(title=f"{idx} {title}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_", "y_", "fitted_"])
        # Input validation
        X = self._validate_data(X, reset=False)
        n_samples = X.shape[0]
        n_estimators = len(self.models_)
        result = np.empty((n_samples, n_estimators))
        dataset = pd.DataFrame(X, columns=self.features_, dtype="int16")
        for index, model in enumerate(self.models_):
            result[:, index] = model.predict(dataset).values.ravel()
        return mode(result, axis=1, keepdims=False).mode.ravel()
