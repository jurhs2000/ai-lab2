import numpy as np
import matplotlib.pyplot as plt
import LoadData
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

def execute_PR():
    ds, train, test = LoadData.load_regresion_data()
    ds = pd.DataFrame(ds)

    # Variables independientes: Weekly_Sales, Unemployment, Fuel_Price, CPI
    # Variables dependetientes: Date
    # Variable a predecir Weekly_Sales
    ds['Date'] = pd.to_datetime(ds['Date'], dayfirst=True,)
    ds['Week'] = ds['Date'].dt.week
    y = ds.iloc[:, 2].values
    x = ds.iloc[:, 0:1].values

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    polinomio = PolynomialFeatures(degree=7)

    # Se transforma las caracteristicas
    poli_xTr = polinomio.fit_transform(train_x)
    poli_xT = polinomio.fit_transform(test_x)

    pr = linear_model.LinearRegression()
    pr.fit(poli_xTr, train_y)
    Y_pred_pr = pr.predict(poli_xT)

    plt.scatter(test_x, test_y)
    plt.plot(test_x, Y_pred_pr, color='green', linewidth=3)
    plt.show()

    print('Regresion polinomial\n')
    print('Valor de "a":\n')
    print(pr.coef_)
    print('Valor de "b":\n')
    print(pr.intercept_)
    print('Precisión:\n')
    print(pr.score(poli_xTr, train_y))