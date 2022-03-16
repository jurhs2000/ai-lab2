from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
import pandas as panda
from sklearn.model_selection import train_test_split
import numpy as numpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

setdatos = panda.read_csv('Oranges vs Grapefruit.csv')
setdatos = panda.DataFrame(setdatos)
setdatos['name'] = setdatos['name'].map(
{'orange': 0, 'grapefruit': 1}, na_action=None)
x = setdatos[setdatos.columns[1:3]].values
y = setdatos[setdatos.columns[0]].values

minx, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
miny, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
j = (xmax / minx)/100
xx, yy = numpy.meshgrid(numpy.arange(minx, xmax, j),numpy.arange(miny, ymax, j))
svmm = svm.SVC(kernel='RBF', C=1, gamma=10)
svmm.fit(x, y)

plt.subplot(1, 1, 1)
Z = svmm.predict(numpy.c_[xx.ravel(), yy.ravel()])
plt.contourf(xx, yy, Z, cmap=plt.cm.Accent, alpha=0.7,)
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Accent)
plt.xlabel('Diametro')
plt.title('SVM')
plt.ylabel('Peso')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    setdatos[setdatos.columns[:]].values, setdatos[setdatos.columns[0]].values, test_size=0.7, random_state=123)
model = svm.SVC(kernel='RBF', C=1)
model.fit(X_train, y_train)
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'GAMMA': [1, 0.1, 0.01, 0.001, 0.0001],
              'KERNEL': ['RBF']}
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

grid = GridSearchCV(svm.SVC(), param_grid, refit=True, cv=3, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))
