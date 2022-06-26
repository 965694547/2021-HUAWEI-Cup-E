import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' %(tip, np.mean(acc)))

if __name__ == '__main__':
    csvPD=pd.read_csv("SVM_data.csv",header=None)
    csvPD= csvPD.iloc[0:9472,0:6]
    x, y = np.split(csvPD, (5,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.70)
    # 高斯核
    clf1 = svm.SVC(C=5, kernel='rbf', gamma=0.2, decision_function_shape='ovr')
    # 线性核
    # clf1 = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
    clf1.fit(x_train, y_train.values.ravel())
    y_hat = clf1.predict(x_train)
    show_accuracy(y_hat, y_train.values, 'traing data')
    y_hat_test = clf1.predict(x_test)
    show_accuracy(y_hat_test, y_test.values, 'testing data')
    a = clf1.decision_function(x_test)#  decision_function 计算得出样本点归属于某一类别时到分割超平面的函数距离
    distance = pd.DataFrame(a) #函数距离表
    b = clf1.predict(x_test)
    probability = pd.DataFrame(b) #可能性及预测类别
    # distance.to_csv('函数距离表.csv', header=0, index=0)
    # probability.to_csv('可能性及预测类别.csv', header=0, index=0)
