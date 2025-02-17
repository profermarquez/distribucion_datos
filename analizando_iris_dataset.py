import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


def cargar_datos_iris():
    """Carga el conjunto de datos Iris y retorna características y etiquetas."""
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Solo las dos primeras características.
    y = iris.target
    datos_completos = iris.data
    return X, y, datos_completos


def graficar_2d(X, y):
    """Grafica los datos en 2D con las primeras dos características."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Longitud de sépalo")
    plt.ylabel("Ancho de sépalo")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.title("Iris Dataset - Proyección en 2D")
    plt.show()


def graficar_3d(datos, y):
    """Grafica los datos en 3D utilizando PCA para reducir dimensiones."""
    X_reduced = PCA(n_components=3).fit_transform(datos)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40
    )

    ax.set_title("Proyección en 3D - Primeras tres componentes PCA")
    ax.set_xlabel("1er componente principal")
    ax.set_ylabel("2do componente principal")
    ax.set_zlabel("3er componente principal")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()


def main():
    X, y, datos_completos = cargar_datos_iris()
    graficar_2d(X, y)
    graficar_3d(datos_completos, y)


if __name__ == "__main__":
    main()
