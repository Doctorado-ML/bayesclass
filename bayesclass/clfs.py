import random
import warnings
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
from fimdlp.mdlp import FImdlp
from ._version import __version__


class BayesBase(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state, show_progress):
        self.random_state = random_state
        self.show_progress = show_progress

    def _more_tags(self):
        return {
            "requires_positive_X": True,
            "requires_positive_y": True,
            "preserve_dtype": [np.int32, np.int64],
            "requires_y": True,
        }

    @staticmethod
    def version() -> str:
        """Return the version of the package."""
        return __version__

    def nodes_edges(self):
        if hasattr(self, "dag_"):
            return len(self.dag_), len(self.dag_.edges())
        return 0, 0

    @staticmethod
    def default_feature_names(num_features):
        return [f"feature_{i}" for i in range(num_features)]

    @staticmethod
    def default_class_name():
        return "class"

    def _check_params_fit(self, X, y, expected_args, kwargs):
        """Check the common parameters passed to fit"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = self._validate_data(X, reset=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        # Default values
        self.class_name_ = self.default_class_name()
        self.features_ = self.default_feature_names(X.shape[1])
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        self.feature_names_in_ = self.features_
        if self.random_state is not None:
            random.seed(self.random_state)
        if len(self.feature_names_in_) != X.shape[1]:
            raise ValueError(
                "Number of features does not match the number of columns in X"
            )
        self.n_features_in_ = X.shape[1]
        return X, y

    @property
    def states_(self):
        if hasattr(self, "fitted_"):
            return sum([len(item) for _, item in self.model_.states.items()])
        return 0

    @property
    def depth_(self):
        return self.states_

    def fit(self, X, y, **kwargs):
        """Fit classifier

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
        self.dataset_ = pd.DataFrame(
            self.X_, columns=self.feature_names_in_, dtype=np.int32
        )
        self.dataset_[self.class_name_] = self.y_
        # Build the DAG
        self._build()
        # Train the model
        self._train(kwargs)
        self.fitted_ = True
        # To keep compatiblity with the benchmark platform
        self.nodes_leaves = self.nodes_edges
        # Return the classifier
        return self

    def _train(self, kwargs):
        self.model_ = BayesianNetwork(
            self.dag_.edges(), show_progress=self.show_progress
        )
        states = dict(state_names=kwargs.pop("state_names", []))
        self.model_.fit(
            self.dataset_,
            estimator=BayesianEstimator,
            prior_type="K2",
            **states,
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
        dataset = pd.DataFrame(
            X, columns=self.feature_names_in_, dtype=np.int32
        )
        return self.model_.predict(dataset).values.ravel()

    def plot(self, title="", node_size=800):
        warnings.simplefilter("ignore", UserWarning)
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
    feature_names_in_ : list
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
        expected_args = ["class_name", "features", "head", "state_names"]
        X, y = self._check_params_fit(X, y, expected_args, kwargs)
        if self.head_ == "random":
            self.head_ = random.randint(0, self.n_features_in_ - 1)
        if self.head_ is not None and self.head_ >= self.n_features_in_:
            raise ValueError("Head index out of range")
        return X, y

    def _build(self):
        est = TreeSearch(
            self.dataset_, root_node=self.feature_names_in_[self.head_]
        )
        self.dag_ = est.estimate(
            estimator_type="tan",
            class_node=self.class_name_,
            show_progress=self.show_progress,
        )
        # Code taken from pgmpy
        # n_jobs = -1
        # weights = TreeSearch._get_conditional_weights(
        #     self.dataset_,
        #     self.class_name_,
        #     "mutual_info",
        #     n_jobs,
        #     self.show_progress,
        # )
        # # Step 4.2: Construct chow-liu DAG on {data.columns - class_node}
        # class_node_idx = np.where(self.dataset_.columns == self.class_name_)[
        #     0
        # ][0]
        # weights = np.delete(weights, class_node_idx, axis=0)
        # weights = np.delete(weights, class_node_idx, axis=1)
        # reduced_columns = np.delete(self.dataset_.columns, class_node_idx)
        # D = TreeSearch._create_tree_and_dag(
        #     weights, reduced_columns, self.feature_names_in_[self.head_]
        # )
        # # Step 4.3: Add edges from class_node to all other nodes.
        # D.add_edges_from(
        #     [(self.class_name_, node) for node in reduced_columns]
        # )
        # self.dag_ = D


class KDB(BayesBase):
    def __init__(self, k, theta=0.03, show_progress=False, random_state=None):
        self.k = k
        self.theta = theta
        super().__init__(
            show_progress=show_progress, random_state=random_state
        )

    def _check_params(self, X, y, kwargs):
        expected_args = ["class_name", "features", "state_names"]
        return self._check_params_fit(X, y, expected_args, kwargs)

    def _add_m_edges(self, dag, idx, S_nodes, conditional_weights):
        n_edges = min(self.k, len(S_nodes))
        cond_w = conditional_weights.copy()
        exit_cond = self.k == 0
        num = 0
        while not exit_cond:
            max_minfo = np.argmax(cond_w[idx, :])
            if max_minfo in S_nodes and cond_w[idx, max_minfo] > self.theta:
                try:
                    dag.add_edge(
                        self.feature_names_in_[max_minfo],
                        self.feature_names_in_[idx],
                    )
                    num += 1
                except ValueError:
                    # Loops are not allowed
                    pass
            cond_w[idx, max_minfo] = -1
            exit_cond = num == n_edges or np.all(cond_w[idx, :] <= self.theta)

    def _build(self):
        """
        1. For each feature Xi, compute mutual information, I(X;;C),
        where C is the class.
        2. Compute class conditional mutual information I(Xi;XjIC), f or each
        pair of features Xi and Xj, where i#j.
        3. Let the used variable list, S, be empty.
        4. Let the Bayesian network being constructed, BN, begin with a single
        class node, C.
        5. Repeat until S includes all domain features
        5.1. Select feature Xmax which is not in S and has the largest value
        I(Xmax;C).
        5.2. Add a node to BN representing Xmax.
        5.3. Add an arc from C to Xmax in BN.
        5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
        the highest value for I(Xmax;X,jC).
        5.5. Add Xmax to S.
        Compute the conditional probabilility infered by the structure of BN by
        using counts from DB, and output BN.
        """
        # 1. get the mutual information between each feature and the class
        mutual = mutual_info_classif(self.X_, self.y_, discrete_features=True)
        # 2. symmetric matrix where each element represents I(X, Y| class_node)
        conditional_weights = TreeSearch(
            self.dataset_
        )._get_conditional_weights(
            self.dataset_, self.class_name_, show_progress=self.show_progress
        )
        # 3. Let the used variable list, S, be empty.
        S_nodes = []
        # 4. Let the BN being constructed, BN, begin with a single class node
        dag = BayesianNetwork()
        dag.add_node(self.class_name_)  # , state_names=self.classes_)
        # 5. Repeat until S includes all domain features
        # 5.1 Select feature Xmax which is not in S and has the largest value
        for idx in np.argsort(mutual):
            # 5.2 Add a node to BN representing Xmax.
            feature = self.feature_names_in_[idx]
            dag.add_node(feature)
            # 5.3 Add an arc from C to Xmax in BN.
            dag.add_edge(self.class_name_, feature)
            # 5.4 Add m = min(lSl,/c) arcs from m distinct features Xj in S
            self._add_m_edges(dag, idx, S_nodes, conditional_weights)
            # 5.5 Add Xmax to S.
            S_nodes.append(idx)
        self.dag_ = dag


def build_spodes(features, class_name):
    """Build SPODE estimators (Super Parent One Dependent Estimator)"""
    class_edges = [(class_name, f) for f in features]
    for idx in range(len(features)):
        feature_edges = [
            (features[idx], f) for f in features if f != features[idx]
        ]
        feature_edges.extend(class_edges)
        model = BayesianNetwork(feature_edges, show_progress=False)
        yield model


class AODE(ClassifierMixin, BaseEnsemble):
    def __init__(self, show_progress=False, random_state=None):
        self.base_model = BayesBase(
            show_progress=show_progress, random_state=random_state
        )
        self.show_progress = show_progress
        self.random_state = random_state

    def _check_params(self, X, y, kwargs):
        expected_args = ["class_name", "features", "state_names"]
        return self.base_model._check_params_fit(X, y, expected_args, kwargs)

    def nodes_edges(self):
        nodes = 0
        edges = 0
        if hasattr(self, "fitted_"):
            nodes = sum([len(x) for x in self.models_])
            edges = sum([len(x.edges()) for x in self.models_])
        return nodes, edges

    def version(self):
        return self.base_model.version()

    def fit(self, X, y, **kwargs):
        X_, y_ = self._check_params(X, y, kwargs)
        self.class_name_ = self.base_model.class_name_
        self.feature_names_in_ = self.base_model.feature_names_in_
        self.classes_ = self.base_model.classes_
        self.n_features_in_ = self.base_model.n_features_in_
        # Store the information needed to build the model
        self.X_ = X_
        self.y_ = y_
        self.dataset_ = pd.DataFrame(
            self.X_, columns=self.feature_names_in_, dtype=np.int32
        )
        self.dataset_[self.class_name_] = self.y_
        # Train the model
        self._train(kwargs)
        self.fitted_ = True
        # To keep compatiblity with the benchmark platform
        self.nodes_leaves = self.nodes_edges
        # Return the classifier
        return self

    @property
    def states_(self):
        if hasattr(self, "fitted_"):
            return sum(
                [
                    len(item)
                    for model in self.models_
                    for _, item in model.states.items()
                ]
            ) / len(self.models_)
        return 0

    @property
    def depth_(self):
        return self.states_

    def _train(self, kwargs):
        self.models_ = []
        states = dict(state_names=kwargs.pop("state_names", []))
        for model in build_spodes(self.feature_names_in_, self.class_name_):
            model.fit(
                self.dataset_,
                estimator=BayesianEstimator,
                prior_type="K2",
                **states,
            )
            self.models_.append(model)

    def plot(self, title=""):
        warnings.simplefilter("ignore", UserWarning)
        for idx, model in enumerate(self.models_):
            self.base_model.model_ = model
            self.base_model.plot(title=f"{idx} {title}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_", "y_", "fitted_"])
        # Input validation
        X = check_array(X)
        n_samples = X.shape[0]
        n_estimators = len(self.models_)
        result = np.empty((n_samples, n_estimators))
        dataset = pd.DataFrame(
            X, columns=self.feature_names_in_, dtype=np.int32
        )
        for index, model in enumerate(self.models_):
            result[:, index] = model.predict(dataset).values.ravel()
        return mode(result, axis=1, keepdims=False).mode.ravel()


class TANNew(TAN):
    def __init__(
        self,
        show_progress=False,
        random_state=None,
        discretizer_depth=1e6,
        discretizer_length=3,
        discretizer_cuts=0,
    ):
        self.discretizer_depth = discretizer_depth
        self.discretizer_length = discretizer_length
        self.discretizer_cuts = discretizer_cuts
        super().__init__(
            show_progress=show_progress, random_state=random_state
        )

    def fit(self, X, y, **kwargs):
        self.estimator = Proposal(self)
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        return self.estimator.predict(X)


class KDBNew(KDB):
    def __init__(
        self,
        k=2,
        show_progress=False,
        random_state=None,
        discretizer_depth=1e6,
        discretizer_length=3,
        discretizer_cuts=0,
    ):
        self.discretizer_depth = discretizer_depth
        self.discretizer_length = discretizer_length
        self.discretizer_cuts = discretizer_cuts
        super().__init__(
            k=k, show_progress=show_progress, random_state=random_state
        )

    def fit(self, X, y, **kwargs):
        self.estimator = Proposal(self)
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        return self.estimator.predict(X)


class SpodeNew(BayesBase):
    """This class implements a classifier for the SPODE algorithm similar to TANNew and KDBNew"""

    def __init__(self, random_state, show_progress, structure):
        super().__init__(
            random_state=random_state, show_progress=show_progress
        )
        self.structure = structure


class AODENew:
    def __init__(self, show_progress=False, random_state=None):
        self.show_progress = show_progress
        self.random_state = random_state

    def _train(self, kwargs):
        self.estimators_ = []
        states = dict(state_names=kwargs.pop("state_names", []))
        kwargs["states"] = states
        for spode in build_spodes(self.feature_names_in_, self.class_name_):
            model = SpodeNew(self.random_state, self.show_progress, spode)
            estimator = Proposal(spode)
            self.estimators_.append(estimator.fit(self.X_, self.y_, **kwargs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_", "y_", "fitted_"])
        # Input validation
        X = check_array(X)
        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        result = np.empty((n_samples, n_estimators))
        for index, model in enumerate(self.estimators_):
            result[:, index] = model.predict(X).values.ravel()
        return mode(result, axis=1, keepdims=False).mode.ravel()

    @property
    def states_(self):
        if hasattr(self, "fitted_"):
            return sum(
                [
                    len(item)
                    for model in self.models_
                    for _, item in model.states.items()
                ]
            ) / len(self.models_)
        return 0


class Proposal:
    def __init__(self, estimator):
        self.estimator = estimator
        self.class_type = estimator.__class__

    def fit(self, X, y, **kwargs):
        # Check parameters
        self.estimator._check_params(X, y, kwargs)

        # Discretize train data
        self.discretizer = FImdlp(
            n_jobs=1,
            max_depth=self.estimator.discretizer_depth,
            min_length=self.estimator.discretizer_length,
            max_cuts=self.estimator.discretizer_cuts,
        )
        self.Xd = self.discretizer.fit_transform(X, y)
        kwargs = self.update_kwargs(y, kwargs)
        # Build the model
        super(self.class_type, self.estimator).fit(self.Xd, y, **kwargs)
        # Local discretization based on the model
        features = kwargs["features"]
        # assign indices to feature names
        self.idx_features_ = dict(list(zip(features, range(len(features)))))
        upgraded, self.Xd = self._local_discretization()
        if upgraded:
            kwargs = self.update_kwargs(y, kwargs)
            super(self.class_type, self.estimator).fit(self.Xd, y, **kwargs)
        return self

    def predict(self, X):
        Xd = self.discretizer.transform(X)
        self.check_integrity("predict", Xd)
        return super(self.class_type, self.estimator).predict(Xd)

    def update_kwargs(self, y, kwargs):
        features = (
            kwargs["features"]
            if "features" in kwargs
            else self.estimator.default_feature_names(self.Xd.shape[1])
        )
        states = {
            features[i]: self.discretizer.get_states_feature(i)
            for i in range(self.Xd.shape[1])
        }
        class_name = (
            kwargs["class_name"]
            if "class_name" in kwargs
            else self.estimator.default_class_name()
        )
        states[class_name] = np.unique(y).tolist()
        kwargs["state_names"] = states
        self.state_names_ = states
        self.features_ = features
        kwargs["features"] = features
        kwargs["class_name"] = class_name
        return kwargs

    def _local_discretization(self):
        """Discretize each feature with its fathers and the class"""
        res = self.Xd.copy()
        upgraded = False
        # print("-" * 80)
        for idx, feature in enumerate(self.estimator.feature_names_in_):
            fathers = self.estimator.dag_.get_parents(feature)
            if len(fathers) > 1:
                # print(
                #     "Discretizing " + feature + " with " + str(fathers),
                #     end=" ",
                # )
                # First remove the class name as it will be added later
                fathers.remove(self.estimator.class_name_)
                # Get the fathers indices
                features = [self.idx_features_[f] for f in fathers]
                # Update the discretization of the feature
                res[:, idx] = self.discretizer.join_fit(
                    target=idx, features=features, data=self.Xd
                )
                # print(self.discretizer.y_join[:5])
                upgraded = True
        return upgraded, res

    def check_integrity(self, source, X):
        # print(f"Checking integrity of {source} data")
        for i in range(X.shape[1]):
            if not set(np.unique(X[:, i]).tolist()).issubset(
                set(self.state_names_[self.features_[i]])
            ):
                print(
                    "i",
                    i,
                    "features[i]",
                    self.features_[i],
                    "np.unique(X[:, i])",
                    np.unique(X[:, i]),
                    "np.array(state_names[features[i]])",
                    np.array(self.state_names_[self.features_[i]]),
                )
                raise ValueError("Discretization error")
