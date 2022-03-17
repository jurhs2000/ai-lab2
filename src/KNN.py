import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

# Implementation of K Nearest Neighbors algorithm
class KNN:
  # train_data is a pandas dataframe
  def __init__(self, k, train_data, test_data):
    self.k = k
    self.train_data = train_data
    self.x_test = test_data.iloc[:, 1:]
    self.y_test = test_data.iloc[:, 0]

  # Euclidean distance
  def euclidean_distance(self, row1, row2):
    row1 = row1[1:-1]
    row2 = row2[2:]
    sum = 0
    for i in range(len(row1)):
      sum += (row1[i] - row2[i])**2
    return sum**0.5

  # return the k nearest neighbors of a row
  def get_k_nearest_neighbors(self, row):
    distances = []
    count = 0
    for train_row in self.train_data.itertuples():
      # store in tuple the distance and the index of panda row
      distances.append((self.euclidean_distance(row, train_row), count))
      count += 1
    distances.sort(key=lambda tup: tup[0])
    return distances[:self.k]

  # test_df is a pandas dataframe
  def predict(self, test_df):
    test_df['name'] = 'non predicted'
    for row in test_df.itertuples():
      k_nearest_neighbors = self.get_k_nearest_neighbors(row)
      # create a new dataframe and append the nearest neighbors on the train data
      neighbors = pd.DataFrame(columns=self.train_data.columns)
      for neighbor in k_nearest_neighbors:
        neighbors = neighbors.append(self.train_data.iloc[neighbor[1]])
      # get the name of the most common class
      test_df.loc[row.Index, 'name'] = neighbors['name'].value_counts().idxmax()
    return test_df

  # accuracy of the prediction
  def accuracy(self, prediction):
    count = 0
    for i in range(len(prediction)):
      if prediction.iloc[i]['name'] == self.y_test.iloc[i]:
        count += 1
    return count/len(prediction)

  # graph a confision matrix of the prediction
  def graph_confusion_matrix(self, prediction):
    y_test = self.y_test.apply(lambda x: 1 if x == 'orange' else 2)
    y_pred = prediction['name'].apply(lambda x: 1 if x == 'orange' else 2)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['oranges', 'grapefruit'], yticklabels=['oranges', 'grapefruit'])
    plt.title('Matriz de Confusion')
    plt.ylabel('Clasificación real')
    plt.xlabel('Clasificación predicha')
    plt.show()

  # Graph the train data in 2D using 'diameter' and 'weight' as the two features
  def graph_train_data(self, data):
    sns.scatterplot(self.train_data['diameter'], self.train_data['weight'], hue=self.train_data['name'], palette=['violet', 'yellow'])
    sns.scatterplot(data['diameter'], data['weight'], hue=data['name'], palette=['blue', 'orange'])
    plt.xlabel('diameter')
    plt.ylabel('weight')
    plt.show()

  # graph a color map of the train data
  def graph_hsv_rgb(self, data):
    V, H = np.mgrid[0:1:100j, 0:1:360j]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    plt.imshow(RGB, origin="lower", extent=[0, 360, 0, 1], aspect=150)
    self.train_data['H'] = self.train_data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[0], axis=1)
    self.train_data['S'] = self.train_data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[1], axis=1)
    self.train_data['V'] = self.train_data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[2], axis=1)
    sns.scatterplot(self.train_data['H'], self.train_data['V'], hue=self.train_data['name'], palette=['violet', 'yellow'])
    data['H'] = data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[0], axis=1)
    data['S'] = data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[1], axis=1)
    data['V'] = data.apply(lambda row: self.rgb_to_hsv(row['red'], row['green'], row['blue'])[2], axis=1)
    sns.scatterplot(data['H'], data['V'], hue=data['name'], palette=['blue', 'orange'])
    plt.show()

  # Convert RGB to HSV
  def rgb_to_hsv(self, r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
      h = 0
    elif mx == r:
      h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
      h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
      h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
      s = 0
    else:
      s = df/mx
    v = mx
    return h, s, v
