#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mdlp import MDLP
import pandas as pd
from benchmark import Datasets
from bayesclass import TAN
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    KFold,
    cross_val_score,
    train_test_split,
)
import numpy as np
import warnings
from stree import Stree

# In[2]:


# Get data as a dataset
dt = Datasets()
data = dt.load("glass", dataframe=True)
features = dt.dataset.features
class_name = dt.dataset.class_name
factorization, class_factors = pd.factorize(data[class_name])
data[class_name] = factorization
data.head()


# In[3]:


# Fayyad Irani
discretiz = MDLP()
Xdisc = discretiz.fit_transform(
    data[features].to_numpy(), data[class_name].to_numpy()
)
features_discretized = pd.DataFrame(Xdisc, columns=features)
dataset_discretized = features_discretized.copy()
dataset_discretized[class_name] = data[class_name]
X = dataset_discretized[features]
y = dataset_discretized[class_name]
dataset_discretized


# In[4]:


n_folds = 5
score_name = "accuracy"
random_state = 17
test_size = 0.3


def validate_classifier(model, X, y, stratified, fit_params):
    stratified_class = StratifiedKFold if stratified else KFold
    kfold = stratified_class(
        shuffle=True, random_state=random_state, n_splits=n_folds
    )
    # return cross_validate(model, X, y, cv=kfold, return_estimator=True,
    # scoring=score_name)
    return cross_val_score(model, X, y, fit_params=fit_params)


def split_data(X, y, stratified):
    if stratified:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=True,
        )
    else:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )


# In[5]:


warnings.filterwarnings("ignore")
for simple_init in [False, True]:
    model = TAN(simple_init=simple_init)
    for head in range(4):
        X_train, X_test, y_train, y_test = split_data(X, y, stratified=False)
        model.fit(
            X_train,
            y_train,
            head=head,
            features=features,
            class_name=class_name,
        )
        y = model.predict(X_test)
        model.plot()

# In[ ]:


model = TAN(simple_init=simple_init)
model.fit(X, y, features=features, class_name=class_name)
model.plot(
    f"**simple_init={simple_init} head={head}  score={model.score(X, y)}"
)
