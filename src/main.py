from LoadData import load_data_KNN
from KNN import KNN

# return a pandas dataframe
knn_train, knn_test = load_data_KNN()
knn = KNN(3, knn_train, knn_test)

predictions = knn.predict(knn.x_test)

print(knn.accuracy(predictions))

knn.graph_confusion_matrix(predictions)

knn.graph_train_data(predictions)
knn.graph_hsv_rgb(predictions)