import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_wine
from bayesclass.clfs import TAN
import warnings


# Warnings are not errors
warnings.simplefilter("ignore")
start = time.time()
random_state = 17
n_folds = 5
print(f"Accuracy in {n_folds} folds stratified crossvalidation")
dataset_start = time.time()
dataset = load_wine()
Xc = dataset.data
enc = KBinsDiscretizer(encode="ordinal")
X = enc.fit_transform(Xc)
y = dataset.target
clf = TAN(random_state=random_state)
fit_params = dict(features=dataset.feature_names, class_name="class", head=0)
kfold = StratifiedKFold(
    n_splits=n_folds, shuffle=True, random_state=random_state
)
score = cross_val_score(clf, X, y, cv=kfold, fit_params=fit_params)
print(
    f"wine {'.' * 10}{score.mean():9.7f} "
    f"({time.time()-dataset_start:7.2f} seconds)"
)
clf.fit(X, y, **fit_params)
clf.plot("TAN wine")
print(f"Took {time.time()-start:.2f} seconds")
