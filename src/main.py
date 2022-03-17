from LoadData import load_data_KNN
from KNN import KNN
from PR import execute_PR
from SVM import execute_SVM

option = 0
while option != 1 and option != 2 and option != 3:
    print('\nSeleccione el algoritmo de prediccion a utilizar:')
    print('1. Regresion Polinomial')
    print('2. K Nearest Neighbors')
    print('3. Support Vector Machine')
    print('4. Salir')
    option = int(input('Opcion: '))
    if option == 1:
        print('\nRegresion Polinomial')
        execute_PR()
    elif option == 2:
        print('\nK Nearest Neighbors')
        knn_train, knn_test = load_data_KNN()
        knn = KNN(3, knn_train, knn_test)
        predictions = knn.predict(knn.x_test)
        print(f'Score of the prediction: {knn.accuracy(predictions)}')
        knn.graph_confusion_matrix(predictions)
        knn.graph_train_data(predictions)
        knn.graph_hsv_rgb(predictions)
    elif option == 3:
        print('\nSupport Vector Machine')
        execute_SVM()
    elif option == 4:
        print('')
    else:
        print('\nOpcion incorrecta')
