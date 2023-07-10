from sklearn.neighbors import KNeighborsClassifier
def KNN(X_train, y_train,X_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    return y_pre