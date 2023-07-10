from sklearn.tree import DecisionTreeClassifier
def DecisionTree (X_train, y_train, X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

