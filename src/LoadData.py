from sklearn import datasets
import pandas as pd

# Lo mejor seria que este metodo devuelva de una vez los datos de entrenamiento y de prueba separados
def load_data_PR_KNN():
  df = pd.read_csv('../data/Oranges vs Grapefruit.csv')
  iris = datasets.load_iris()
  X = iris.data[:, :2]
  Y = iris.target
  return df

def load_data_SVM():
  df = pd.read_csv('../data/Walmart.csv')
  return df
