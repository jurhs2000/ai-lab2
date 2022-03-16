import sklearn as skl
import pandas as pd

# Lo mejor seria que este metodo devuelva de una vez los datos de entrenamiento y de prueba separados
def load_regresion_data():
  df = pd.read_csv('../data/Oranges vs Grapefruit.csv')
  train, test = skl.train_test_split(df, test_size=0.7)
  return df, train, test

def load_KNNSVM_data():
  df = pd.read_csv('../data/Walmart.csv')
  train, test = skl.train_test_split(df, test_size=0.7)
  return df, train, test
