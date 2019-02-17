#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Ejemplo basico de como llamar al clasificador de k vecinos mas cercanos usando scikit
y como filtrar caracteristicas usando test univariados (chi cuadrado en este caso)

"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Filtrado de caracteristicas usando un tes univariado (chi2)

iris = load_iris() #Cargamos los datos de ejemplo de iris
X,y = iris.data, iris.target #Asignamos la matriz de caracteristicas a la X y el vector respuesta a la Y
#X.shape #Nos muestra la informacion de las instancias y los atributos que hay en la matriz
X_new = SelectKBest(chi2, k=2).fit_transform(X,y) #Selecciona un subconjunto de dimension K(=2) de las variables basandose en el test de chi cuadrado

#Separamos los datos en un subconjunto de entrenamiento y otro de prueba usando el subconjunto de caracteristicas filtradas
X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.5)

#Clasificador kNN
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train) #Siendo X_train el conjunto de datos caracteristicos(matriz de caracteristicas) e y_Train el vector con los resultados(etiquetas) correspondientes
prediction = knn.predict(X_test) #siendo X_test la matriz de caracteristicas de los datos que vamos a usar para poner a prueba el modelo.
#La linea anterior nos devuelve la matriz respuesta (sera lo que usemos una vez ajustado el modelo para hacer predicciones sobre datos desconocidos)
precision = knn.score(X_test,y_test) #Calculamos el porcentaje de aciertos del modelo (parecido al R2)
print(precision)