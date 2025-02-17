from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np


def train_and_plot_decision_tree(max_depth=3, test_size=0.3, random_state=0):
    # Cargar el dataset de iris
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # Usamos solo largo y ancho del pétalo
    y = iris.target

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Inicializar y entrenar el modelo
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=random_state)
    tree.fit(X_train, y_train)

    # Combinar datos para la visualización
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # Graficar las regiones de decisión
    plt.figure(figsize=(10, 6))
    plot_decision_regions(X_combined, y_combined, clf=tree, legend=2)
    plt.title(f'Regiones de decisión - Árbol de decisión (max_depth={max_depth})')
    plt.xlabel('Largo del pétalo [cm]')
    plt.ylabel('Ancho del pétalo [cm]')
    plt.legend(loc='upper left', title='Clases')
    plt.tight_layout()
    plt.show()


# Llamada a la función con los parámetros deseados
train_and_plot_decision_tree(max_depth=3)
