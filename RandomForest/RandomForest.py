from sklearn.ensemble import RandomForestClassifier
def RandomForest(X_train, y_train, X_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    return y_pre