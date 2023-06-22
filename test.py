from bayesclass.cppSelectFeatures import CSelectKBestWeighted


X = [[x for x in range(i, i + 3)] for i in range(1, 30, 3)]
weights = [25 / (i + 1) for i in range(10)]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test = CSelectKBestWeighted(X, labels, weights, 3)
test.fit()
for item in test.get_score():
    print(item)
