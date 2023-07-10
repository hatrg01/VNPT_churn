from sklearn.svm import SVC
def SVM(X_train, y_train, X_test):
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    return y_pre