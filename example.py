from benchmark import Discretizer
from bayesclass import TAN
import sys
from sklearn.model_selection import cross_val_score, StratifiedKFold


if len(sys.argv) < 2:
    print("Usage: python3 example.py <dataset> [n_folds]")
    exit(1)
random_state = 17
name = sys.argv[1]
n_folds = int(sys.argv[2]) if len(sys.argv) == 3 else 5
dt = Discretizer()
X, y = dt.load(name)
clf = TAN(random_state=random_state)
fit_params = dict(
    features=dt.get_features(), class_name=dt.get_class_name(), head=0
)
kfold = StratifiedKFold(
    n_splits=n_folds, shuffle=True, random_state=random_state
)
score = cross_val_score(clf, X, y, cv=kfold, fit_params=fit_params)
print(f"Accuracy in {n_folds} folds stratified crossvalidation")
print(f"{name}{'.' * 10}{score.mean():9.7f}")
