from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=30, stratify=data.target)

    #model = svm.SVC(C=0.5, kernel='rbf', gamma=5)
    #model.fit(X_train, y_train)
    #s = model.score(X_test, y_test)
    #print(s)

    parameters = {'kernel':['linear','rbf','sigmoid','poly'], 'C':np.linspace(0.1, 20, 50), 'gamma':np.linspace(0.1, 20, 20) }
    svc = svm.SVC()
    model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    print(model.best_params_)
    print(model.score(X_test, y_test))
