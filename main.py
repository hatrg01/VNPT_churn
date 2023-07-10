from preprocessing import preprocessing
from trainTestSplit import trainTestSplit
from DecisionTree.DecisionTree import DecisionTree
from RandomForest.RandomForest import RandomForest
from SVM.SVM import SVM
from KNN.KNN import KNN
from evaluation import evaluation
if __name__ == '__main__':
    #preprocessing
    X_smote, y_smote = preprocessing()
    #train test split
    X_train, X_test, y_train, y_test = trainTestSplit(X_smote,y_smote)

    # print(X_train, X_test, y_train, y_test)
    #model Decision Tree
    y_pre1 = DecisionTree(X_train, y_train, X_test)
    print(y_pre1)
    #model RandomForest
    y_pre2 = RandomForest(X_train,y_train,X_test)
    print(y_pre2)
    #moedl SVM
    y_pre3 = SVM(X_train, y_train, X_test)
    print(y_pre3)
    #model KNN
    y_pre4 = KNN(X_train,y_train, X_test)
    print(y_pre4)
    #evaluation Decison Tree
    print("--------Đánh giá model DecisionTree:--------")
    evaluation(y_test, y_pre1)
    #evaluation RandomForest
    print("--------Đánh giá model RandomForest:--------")
    evaluation(y_test,y_pre2)
    #evaluation SVM
    print("--------Đánh giá model SVM:--------")
    evaluation(y_test, y_pre3)
    #evaluation KNN
    print("--------Đánh giá model KNN:--------")
    evaluation(y_test, y_pre4)


