from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

# Lo mejor seria que este metodo devuelva de una vez los datos de entrenamiento y de prueba separados
def load_data_KNN():
  # Load the data
  df = pd.read_csv('../data/Oranges vs Grapefruit.csv')
  # split in train and test using train_test_split
  train, test = train_test_split(df, test_size=0.3, random_state=42)
  return train, test

def load_regresion_data():
  df = pd.read_csv('../data/Walmart.csv')
  train, test = train_test_split(df, test_size=0.3)
  return df, train, test
