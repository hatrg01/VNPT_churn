from DecisionTree.DecisionTreeScratch import Node
from evaluation import evaluation
from preprocessing import preprocessing
from trainTestSplit import trainTestSplit

if __name__ == '__main__':
    # preprocessing
    X, y = preprocessing()
    # train test split
    X_train, X_test, y_train, y_test = trainTestSplit(X, y)
    # print(X_train, X_test, y_train, y_test)
    '''model Decision Tree'''
    hp = {
        'max_depth': 5,
        'min_samples_split': 200
    }
    # Khởi tạo node
    root = Node(y_train, X_train, **hp)
    # Split tốt nhất tạo cây quyết định
    root.grow_tree()
    # Print thông tin cây
    root.print_tree()
    # Dự đoán
    # results = X_train.copy()
    y_pre1 = root.predict(X_test)
    '''model RandomForest'''

    '''moedl SVM'''

    '''model KNN'''

    # evaluation Decison Tree
    print("--------Đánh giá model DecisionTree:--------")
    evaluation(y_test, y_pre1)
    # evaluation RandomForest
    print("--------Đánh giá model RandomForest:--------")
    # evaluation(y_test, y_pre2)
    # evaluation SVM
    print("--------Đánh giá model SVM:--------")
    # evaluation(y_test, y_pre3)
    # evaluation KNN
    print("--------Đánh giá model KNN:--------")
    # evaluation(y_test, y_pre4)