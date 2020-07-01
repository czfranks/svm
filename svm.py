import pickle
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
sns.set(font_scale=1.2)


recetas = pd.read_csv('receta.csv')
datos = pd.read_csv('receta.csv',header = None, skiprows=1)
datos = np.array(datos)

"""grafico de los ingredientes"""
sns.lmplot('Flour', 'Sugar', data=recetas, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})\
    .savefig("ingredientes.png")

"""Que caracteristicas(ingredientes) vamos a usar para el entrenamiento"""
# ingredients = recipes[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].to_numpy()
ingredientes = recetas[['Flour','Sugar']].to_numpy()
type_label = np.where(recetas['Type']=='Muffin', 0, 1)

"""Que carateristicas usamos"""
print("="*64)
caracteristicas = recetas.columns.values[1:].tolist()
print(caracteristicas)


"""Entrenando el SVM """
print("="*64)
model = svm.SVC(kernel='linear')
model.fit(ingredientes, type_label)

"""Hiperplano de generado segun los coeficientes de SVM"""
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

#rectas paralelas a el modelo
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# Grafico del hiperplano
sns.lmplot('Flour', 'Sugar', data=recetas, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70}).savefig("svm.png")
plt.plot(xx, yy, linewidth=2, color='black')

# Look at the margins and support vectors
sns.lmplot('Flour', 'Sugar', data=recetas, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none')

##plt.show()



X = datos[:,1:8]
Y = datos[:,0]

#muffin 0 , cupcake 1
le = LabelEncoder()
le.fit(Y)
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.20)
modelo2 = svm.SVC(kernel="linear")           #'linear', 'rbf', 'poly' , 'sigmoid'
modelo2.fit(train_x,train_y)

y_pred = modelo2.predict(train_x)
y_pred_test = modelo2.predict(test_x)

accuracy_train = accuracy_score(train_y, y_pred)
print("Accuracy Score de Entrenamiento SVM: %s " % (accuracy_train))

accuracy_test = accuracy_score(test_y, y_pred_test)
print("Accuracy Score de Test SVM: %s " % (accuracy_test))


logis = linear_model.LogisticRegression()
logis.fit(train_x,train_y)

y_pred_logis = modelo2.predict(train_x)
y_pred_test_logis = modelo2.predict(test_x)

accuracy_train_logis = accuracy_score(train_y, y_pred_logis)
print("Accuracy Score de Entrenamiento R. Logistica: %s " % (accuracy_train_logis))

accuracy_test_logis = accuracy_score(test_y, y_pred_test_logis)
print("Accuracy Score de Test R. Logistica: %s " % (accuracy_test_logis))


