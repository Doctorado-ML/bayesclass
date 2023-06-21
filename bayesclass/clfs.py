import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import mutual_info_classif
import networkx as nx
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.base import DAG
import matplotlib.pyplot as plt
from fimdlp.mdlp import FImdlp
from .feature_selection import SelectKBestWeighted
from ._version import __version__


def default_feature_names(num_features):
    return [f"feature_{i}" for i in range(num_features)]


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
    def default_class_name():
        return "class"

    def build_dataset(self):
        self.dataset_ = pd.DataFrame(
            self.X_, columns=self.feature_names_in_, dtype=np.int32
        )
        self.dataset_[self.class_name_] = self.y_
        if self.sample_weight_ is not None:
            self.dataset_["_weight"] = self.sample_weight_

    def _check_params_fit(self, X, y, expected_args, kwargs):
        """Check the common parameters passed to fit"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = self._validate_data(X, reset=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        # Default values
        self.weighted_ = False
        self.sample_weight_ = None
        self.class_name_ = self.default_class_name()
        self.features_ = default_feature_names(X.shape[1])
        for key, value in kwargs.items():
            if key in expected_args:
                setattr(self, f"{key}_", value)
            else:
                raise ValueError(f"Unexpected argument: {key}")
        self.feature_names_in_ = self.features_
        # used for local discretization
        self.indexed_features_ = {
            feature: i for i, feature in enumerate(self.features_)
        }
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
        self.X_, self.y_ = self._check_params(X, y, kwargs)
        # Store the information needed to build the model
        self.build_dataset()
        # Build the DAG
        self._build()
        # Train the model
        self._train(kwargs)
        self.fitted_ = True
        # To keep compatiblity with the benchmark platform
        self.nodes_leaves = self.nodes_edges
        # Return the classifier
        return self

    def _build(self):
        """This method should be implemented by the subclasses to
        build the DAG
        """
        ...

    def _train(self, kwargs):
        """Build and train a BayesianNetwork from the DAG and the dataset

        Parameters
        ----------
        kwargs : dict
            fit parameters
        """
        self.model_ = BayesianNetwork(
            self.dag_.edges(), show_progress=self.show_progress
        )
        states = dict(state_names=kwargs.pop("state_names", []))
        self.model_.fit(
            self.dataset_,
            estimator=BayesianEstimator,
            prior_type="K2",
            weighted=self.weighted_,
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
        expected_args = [
            "class_name",
            "features",
            "state_names",
            "sample_weight",
            "weighted",
        ]
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
        4. Let the DAG network being constructed, BN, begin with a single
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
        # 4. Let the DAG being constructed, BN, begin with a single class node
        dag = BayesianNetwork(show_progress=self.show_progress)
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


class SPODE(BayesBase):
    def _check_params(self, X, y, kwargs):
        expected_args = [
            "class_name",
            "features",
            "state_names",
            "sample_weight",
            "weighted",
        ]
        return self._check_params_fit(X, y, expected_args, kwargs)


class AODE(ClassifierMixin, BaseEnsemble):
    def __init__(
        self,
        show_progress=False,
        random_state=None,
        estimator=None,
    ):
        self.show_progress = show_progress
        self.random_state = random_state
        super().__init__(estimator=estimator)

    def _validate_estimator(self) -> None:
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(
            default=SPODE(
                random_state=self.random_state,
                show_progress=self.show_progress,
            )
        )

    def fit(self, X, y, **kwargs):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = kwargs.get(
            "features", default_feature_names(self.n_features_in_)
        )
        self.class_name_ = kwargs.get("class_name", "class")
        # build estimator
        self._validate_estimator()
        self.X_ = X
        self.y_ = y
        self.n_samples_ = X.shape[0]
        self.estimators_ = []
        self._train(kwargs)
        self.fitted_ = True
        # To keep compatiblity with the benchmark platform
        self.nodes_leaves = self.nodes_edges
        return self

    def _train(self, kwargs):
        for dag in build_spodes(self.feature_names_in_, self.class_name_):
            estimator = clone(self.estimator_)
            estimator.dag_ = estimator.model_ = dag
            estimator.fit(self.X_, self.y_, **kwargs)
            self.estimators_.append(estimator)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        result = np.empty((n_samples, n_estimators))
        for index, estimator in enumerate(self.estimators_):
            result[:, index] = estimator.predict(X)
        return mode(result, axis=1, keepdims=False).mode.ravel()

    def version(self):
        if hasattr(self, "fitted_"):
            return self.estimator_.version()
        return SPODE(None, False).version()

    @property
    def states_(self):
        if hasattr(self, "fitted_"):
            return sum(
                [
                    len(item)
                    for model in self.estimators_
                    for _, item in model.model_.states.items()
                ]
            ) / len(self.estimators_)
        return 0

    @property
    def depth_(self):
        return self.states_

    def nodes_edges(self):
        nodes = 0
        edges = 0
        if hasattr(self, "fitted_"):
            nodes = sum([len(x.dag_) for x in self.estimators_])
            edges = sum([len(x.dag_.edges()) for x in self.estimators_])
        return nodes, edges

    def plot(self, title=""):
        warnings.simplefilter("ignore", UserWarning)
        for idx, model in enumerate(self.estimators_):
            model.plot(title=f"{idx} {title}")


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
        self.estimator_ = Proposal(self)
        self.estimator_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


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
        self.estimator_ = Proposal(self)
        self.estimator_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class SPODENew(SPODE):
    """This class implements a classifier for the SPODE algorithm similar to
    TANNew and KDBNew"""

    def __init__(
        self,
        random_state,
        show_progress,
        discretizer_depth=1e6,
        discretizer_length=3,
        discretizer_cuts=0,
    ):
        super().__init__(
            random_state=random_state, show_progress=show_progress
        )
        self.discretizer_depth = discretizer_depth
        self.discretizer_length = discretizer_length
        self.discretizer_cuts = discretizer_cuts


class AODENew(AODE):
    def __init__(
        self,
        random_state=None,
        show_progress=False,
        discretizer_depth=1e6,
        discretizer_length=3,
        discretizer_cuts=0,
    ):
        self.discretizer_depth = discretizer_depth
        self.discretizer_length = discretizer_length
        self.discretizer_cuts = discretizer_cuts
        super().__init__(
            random_state=random_state,
            show_progress=show_progress,
            estimator=Proposal(
                SPODENew(
                    random_state=random_state,
                    show_progress=show_progress,
                    discretizer_depth=discretizer_depth,
                    discretizer_length=discretizer_length,
                    discretizer_cuts=discretizer_cuts,
                )
            ),
        )

    def _train(self, kwargs):
        for dag in build_spodes(self.feature_names_in_, self.class_name_):
            proposal = clone(self.estimator_)
            proposal.estimator.dag_ = proposal.estimator.model_ = dag
            self.estimators_.append(proposal.fit(self.X_, self.y_, **kwargs))
        self.n_estimators_ = len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_", "y_", "fitted_"])
        # Input validation
        X = check_array(X)
        result = np.empty((X.shape[0], self.n_estimators_))
        for index, model in enumerate(self.estimators_):
            result[:, index] = model.predict(X)
        return mode(result, axis=1, keepdims=False).mode.ravel()

    @property
    def states_(self):
        if hasattr(self, "fitted_"):
            return sum(
                [
                    len(item)
                    for model in self.estimators_
                    for _, item in model.estimator.model_.states.items()
                ]
            ) / len(self.estimators_)
        return 0

    @property
    def depth_(self):
        return self.states_

    def nodes_edges(self):
        nodes = 0
        edges = 0
        if hasattr(self, "fitted_"):
            nodes = sum([len(x.estimator.dag_) for x in self.estimators_])
            edges = sum(
                [len(x.estimator.dag_.edges()) for x in self.estimators_]
            )
        return nodes, edges

    def plot(self, title=""):
        warnings.simplefilter("ignore", UserWarning)
        for idx, model in enumerate(self.estimators_):
            model.estimator.plot(title=f"{idx} {title}")

    def version(self):
        if hasattr(self, "fitted_"):
            return self.estimator_.estimator.version()
        return SPODENew(None, False).version()


class Proposal(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.class_type = estimator.__class__

    def fit(self, X, y, **kwargs):
        # Check parameters
        self.estimator._check_params(X, y, kwargs)
        # Discretize train data
        self.discretizer_ = FImdlp(
            n_jobs=1,
            max_depth=self.estimator.discretizer_depth,
            min_length=self.estimator.discretizer_length,
            max_cuts=self.estimator.discretizer_cuts,
        )
        self.Xd = self.discretizer_.fit_transform(X, y)
        kwargs = self.update_kwargs(y, kwargs)
        # Build the model
        super(self.class_type, self.estimator).fit(self.Xd, y, **kwargs)
        # Local discretization based on the model
        self._local_discretization()
        # self.check_integrity("fit", self.Xd)
        self.fitted_ = True
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["fitted_"])
        # Input validation
        X = check_array(X)
        Xd = self.discretizer_.transform(X)
        # self.check_integrity("predict", Xd)
        return super(self.class_type, self.estimator).predict(Xd)

    def update_kwargs(self, y, kwargs):
        features = (
            kwargs["features"]
            if "features" in kwargs
            else default_feature_names(self.Xd.shape[1])
        )
        states = {
            features[i]: self.discretizer_.get_states_feature(i)
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
        upgrade = False
        # order of local discretization is important. no good 0, 1, 2...
        ancestral_order = list(nx.topological_sort(self.estimator.dag_))
        for feature in ancestral_order:
            if feature == self.estimator.class_name_:
                continue
            idx = self.estimator.indexed_features_[feature]
            fathers = self.estimator.dag_.get_parents(feature)
            if len(fathers) > 1:
                # First remove the class name as it will be added later
                fathers.remove(self.estimator.class_name_)
                # Get the fathers indices
                features = [
                    self.estimator.indexed_features_[f] for f in fathers
                ]
                # Update the discretization of the feature
                self.Xd[:, idx] = self.discretizer_.join_fit(
                    # each feature has to use previous discretization data=res
                    target=idx,
                    features=features,
                    data=self.Xd,
                )
                upgrade = True
        if upgrade:
            # Update the dataset
            self.estimator.X_ = self.Xd
            self.estimator.build_dataset()
            self.state_names_ = {
                key: self.discretizer_.get_states_feature(value)
                for key, value in self.estimator.indexed_features_.items()
            }
            states = {"state_names": self.state_names_}
            # Update the model
            self.estimator.model_.fit(
                self.estimator.dataset_,
                estimator=BayesianEstimator,
                prior_type="K2",
                **states,
            )

    # def check_integrity(self, source, X):
    #     # print(f"Checking integrity of {source} data")
    #     for i in range(X.shape[1]):
    #         if not set(np.unique(X[:, i]).tolist()).issubset(
    #             set(self.state_names_[self.features_[i]])
    #         ):
    #             print(
    #                 "i",
    #                 i,
    #                 "features[i]",
    #                 self.features_[i],
    #                 "np.unique(X[:, i])",
    #                 np.unique(X[:, i]),
    #                 "np.array(state_names[features[i]])",
    #                 np.array(self.state_names_[self.features_[i]]),
    #             )
    #             raise ValueError("Discretization error")


class BoostSPODE(BayesBase):
    def _check_params(self, X, y, kwargs):
        expected_args = [
            "class_name",
            "features",
            "state_names",
            "sample_weight",
            "weighted",
            "sparent",
        ]
        return self._check_params_fit(X, y, expected_args, kwargs)

    def _build(self):
        class_edges = [(self.class_name_, f) for f in self.feature_names_in_]
        feature_edges = [
            (self.sparent_, f)
            for f in self.feature_names_in_
            if f != self.sparent_
        ]
        feature_edges.extend(class_edges)
        self.dag_ = DAG(feature_edges)

    def _train(self, kwargs):
        states = dict(state_names=kwargs.get("state_names", []))
        breakpoint()
        self.model_ = BayesianNetwork(self.dag_.edges(), show_progress=False)
        self.model_.fit(
            self.dataset_,
            estimator=BayesianEstimator,
            prior_type="K2",
            weighted=self.weighted_,
            **states,
        )


class BoostAODE(ClassifierMixin, BaseEnsemble):
    def __init__(
        self,
        show_progress=False,
        random_state=None,
        estimator=None,
        n_estimators=10,
    ):
        self.show_progress = show_progress
        self.random_state = random_state
        self.n_estimators = n_estimators
        super().__init__(estimator=estimator)

    def _validate_estimator(self) -> None:
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(
            default=BoostSPODE(
                random_state=self.random_state,
                show_progress=self.show_progress,
            )
        )

    def fit(self, X, y, **kwargs):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = kwargs.get(
            "features", default_feature_names(self.n_features_in_)
        )
        self.class_name_ = kwargs.get("class_name", "class")
        self.X_ = X
        self.y_ = y
        self.n_samples_ = X.shape[0]
        self.estimators_ = []
        self._validate_estimator()
        self._train(kwargs)
        self.fitted_ = True
        # To keep compatiblity with the benchmark platform
        self.nodes_leaves = self.nodes_edges
        return self

    def mutual_info_classif_weighted(X, y, sample_weight):
        # Compute the mutual information between each feature and the target
        mi = mutual_info_classif(X, y)

        # Multiply the mutual information scores with the sample weights
        mi_weighted = mi * sample_weight

        # Return the weighted mutual information scores
        return mi_weighted

    def _train(self, kwargs):
        """Build boosted SPODEs"""
        weights = [1 / self.n_samples_] * self.n_samples_
        # Step 0: Set the finish condition
        for num in range(self.n_estimators):
            # Step 1: Build ranking with mutual information
            # OJO MAL, ESTO NO ACTUALIZA EL RANKING CON LOS PESOS
            # SIEMPRE VA A SACAR LO MISMO
            feature = (
                SelectKBestWeighted(k=1)
                .fit(self.X_, self.y_, weights)
                .get_feature_names_out(self.feature_names_in_)
                .tolist()[0]
            )
            # Step 2: Build & train spode with the first feature as sparent
            estimator = clone(self.estimator_)
            _args = kwargs.copy()
            _args["sparent"] = feature
            _args["sample_weight"] = weights
            _args["weighted"] = True
            # Step 2.1: build dataset
            # Step 2.2: Train the model
            estimator.fit(self.X_, self.y_, **_args)
            # Step 3: Compute errors (epsilon sub m & alpha sub m)
            # Explanation in https://medium.datadriveninvestor.com/understanding-adaboost-and-scikit-learns-algorithm-c8d8af5ace10
            y_pred = estimator.predict(self.X_)
            em = np.sum(weights * (y_pred != self.y_)) / np.sum(weights)
            am = np.log((1 - em) / em) + np.log(estimator.n_classes_ - 1)
            # Step 3.2: Update weights for next classifier
            weights = [
                wm * np.exp(am * (ym != y_pred))
                for wm, ym in zip(weights, self.y_)
            ]
            # Step 4: Add the new model
            self.estimators_.append(estimator)
        """
        class_edges = [(self.class_name_, f) for f in self.feature_names_in_]
        feature_edges = [
            (sparent, f) for f in self.feature_names_in_ if f != sparent
        ]
        self.weights_ = weights.copy() if weights is not None else None
        feature_edges.extend(class_edges)
        self.model_ = BayesianNetwork(feature_edges, show_progress=False)
        return self.model_
        """
