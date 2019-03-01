# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""

@author: Cristo Suarez

Recopilación de clasificadores (Random forest, Decision tree, kNN, Naive Bayes)
y comparación de los resultados obtenidos para un conjunto de datos ( en este caso iris dataset)
permitiendo comparar entre los distintos sistemas y hacer una selección inicial de cual funciona 
mejor para nuestro dataset
"""


#Primer paso: importar librerías básicas
import numpy as np #gestión de arrays
from sklearn import datasets #conjuntos de datos
#Clasificadores, validacion y seleccion de caracteristicas
from sklearn.ensemble import RandomForestClassifier #un clasificador (debes usar más)
from sklearn.tree import DecisionTreeClassifier #Desicion tree
from sklearn.neighbors import KNeighborsClassifier #kNN
from sklearn.naive_bayes import GaussianNB #naive bayes

from sklearn.model_selection import train_test_split, cross_val_score #validación sencilla (¿usar cross validation?))




from sklearn.feature_selection import SelectKBest, chi2 #Por si queremos hacer selección de características
from sklearn.metrics import accuracy_score #métricas de calidad 

#Cargar los datos:
iris = datasets.load_iris() #Otras posibilidades: breast_cancer, diabetes, iris, wine
X,y = iris.data, iris.target

# Dividir los datos en 40% test y 60% training (mejor si validación cruzada - ver abajo)
X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.4, random_state=42)



# Crear un clasificador, por ejemplo random forest 
clf = RandomForestClassifier(n_estimators=100)
#Recuerda que otras opciones son knn, decision tree, naive bayes
#KNN
clf1 = KNeighborsClassifier(n_neighbors=20)
#Decision tree
clf2 = DecisionTreeClassifier(max_depth=5)
#Naive bayes
clf3 = GaussianNB()

#Para ello tendrás que importar las clases al comienzo del código

# Entrenar el clasificador con todas las características
clf.fit(X_tr, y_tr)
clf1.fit(X_tr, y_tr)
clf2.fit(X_tr, y_tr)
clf3.fit(X_tr, y_tr)
#También podemos entrenar con parámetros óptimos mediante GridSearch. Ejemplo:
"""
clf = RandomForestClassifier(n_estimators=100)
parameters = {'max_features':['sqrt', 'log2', 10],
              'max_depth':[5, 7, 9]}

clf_grid = GridSearchCV(rf, parameters, n_jobs=-1)
clf_grid.fit(X_tr, y_tr)
"""

#Chequeamos la calidad del modelo
print("Accuracy de los modelos: ")
print("")#mejorar el formato de los resultados 
y_pred = clf.predict(X_tst)
print("El accuracy del modelo RF es: ", accuracy_score(y_tst,y_pred))

y_pred1 = clf1.predict(X_tst)
print("El accuracy del modelo kNN es: ", accuracy_score(y_tst,y_pred1))

y_pred2 = clf2.predict(X_tst)
print("El accuracy del modelo  decision tree es: ", accuracy_score(y_tst,y_pred2))

y_pred3 = clf3.predict(X_tst)
print("El accuracy del modelo  Naive bayes es: ", accuracy_score(y_tst,y_pred3))
print("")
print("")
#¿Y si hubiésemos seleccionado características? usando chi2
select = SelectKBest(chi2, k=2).fit(X_tr, y_tr)
X_tr_fs = select.transform(X_tr)
X_tst_fs = select.transform(X_tst)
#Entrenar solo con las características seleccionadas
print("Accuracy seleccionando caracteristicas: ")
print("")

clf.fit(X_tr_fs, y_tr)
y_pred = clf.predict(X_tst_fs)
print("El accuracy del modelo RF con caracteristicas seleccionadas es: ", accuracy_score(y_tst,y_pred))

clf1.fit(X_tr_fs, y_tr)
y_pred1 = clf.predict(X_tst_fs)
print("El accuracy del modelo kNN con caracteristicas seleccionadas es: ", accuracy_score(y_tst,y_pred1))

clf2.fit(X_tr_fs, y_tr)
y_pred2 = clf.predict(X_tst_fs)
print("El accuracy del modelo  decision tree con caracteristicas seleccionadas es: ", accuracy_score(y_tst,y_pred2))

clf3.fit(X_tr_fs, y_tr)
y_pred3 = clf.predict(X_tst_fs)
print("El accuracy del modelo  Naive Bayes con caracteristicas seleccionadas es: ", accuracy_score(y_tst,y_pred3))
print("")
print("")
#Por último Recordemos la importancia de la validación cruzada
print("Accuracy con validación cruzada: ")
print("")
scores = cross_val_score(clf, X, y, cv = 10)
print("El accuracy del modelo RF en validacion cruzada es: ", scores, 
      " Promedio: ", np.mean(scores))

scores1 = cross_val_score(clf1, X, y, cv = 10)
print("El accuracy del modelo kNN en validacion cruzada es: ", scores1, 
      " Promedio: ", np.mean(scores1))

scores2 = cross_val_score(clf2, X, y, cv = 10)
print("El accuracy del modelo decision tree en validacion cruzada es: ", scores2, 
      " Promedio: ", np.mean(scores2))

scores3 = cross_val_score(clf3, X, y, cv = 10)
print("El accuracy del modelo Naive Bayes en validacion cruzada es: ", scores3, 
      " Promedio: ", np.mean(scores3))
