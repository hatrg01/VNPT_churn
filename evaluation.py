from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
def evaluation (y_test, y_pre):
    acc = accuracy_score(y_test, y_pre)
    recall = recall_score(y_test,y_pre)
    precision = precision_score(y_test,y_pre)
    f1 = f1_score(y_test,y_pre)
    print("accuracy:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1 score:", f1)
    return