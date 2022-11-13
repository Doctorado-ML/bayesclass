import sys
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from benchmark import Datasets
from bayesclass import TAN
import warnings


if len(sys.argv) < 2:
    print("Usage: python3 example.py <dataset> [n_folds]")
    exit(1)
# Warnings are not errors
warnings.simplefilter("ignore")
start = time.time()
random_state = 17
name = sys.argv[1]
n_folds = int(sys.argv[2]) if len(sys.argv) == 3 else 5
dt = Datasets()
name_list = list(dt) if name == "all" else [name]
print(f"Accuracy in {n_folds} folds stratified crossvalidation")
for name in name_list:
    dataset_start = time.time()
    X, y = dt.load(name)
    clf = TAN(random_state=random_state)
    fit_params = dict(
        features=dt.get_features(), class_name=dt.get_class_name(), head=0
    )
    kfold = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    score = cross_val_score(clf, X, y, cv=kfold, fit_params=fit_params)
    print(
        f"{name:20s}{'.' * 10}{score.mean():9.7f} "
        f"({time.time()-dataset_start:7.2f} seconds)"
    )
print(f"Took {time.time()-start:.2f} seconds")
