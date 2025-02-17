import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import category_scatter, heatmap


def load_custom_csv():
    """Carga un DataFrame desde un string CSV con datos de ejemplo."""
    csv_data = """label,x,y
    class1,10.0,8.04
    class1,10.5,7.30
    class2,8.3,5.5
    class2,8.1,5.9
    class3,3.5,3.5
    class3,3.8,5.1
    class1,10.0,5.5
    class2,10,6.3
    class1,7.9,5.2
    class3,1.2,5.9
    class3,10.0,0.3
    class1,7.2,0.4
    class2,5.0,0.2
    class1,5.0,7.0
    class2,10,0.9
    class3,10,8.9
    class2,5.3,1.9
    class1,2.0,9.8
    class3,3.0,4.2
    class1,12.0,9.04
    class1,4.5,6.30
    class2,6.3,3.5
    class2,7.1,4.9
    class3,8.5,2.5
    class3,2.8,4.1
    class1,9.0,4.5
    class2,9.1,9.3
    class1,8.2,3.2
    class3,10.2,1.9"""
    df = pd.read_csv(StringIO(csv_data))
    # Limpiar espacios innecesarios en las etiquetas
    df['label'] = df['label'].str.strip()
    print(df.head())
    return df


def plot_category_scatter(df):
    """Grafica un scatterplot categórico usando mlxtend."""
    fig = category_scatter(x='x', y='y', label_col='label', data=df, legend_loc='upper left')
    plt.title('Distribución de Clases en el Plano XY')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    fig.show()


def load_and_plot_heatmap():
    """Carga un dataset de ejemplo desde una URL y grafica un heatmap de correlación."""
    url = 'https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt'
    column_names = [f'sample_{i}' for i in range(1, 15)]

    # Cargar el DataFrame
    df = pd.read_csv(url, header=None, sep='\s+')
    df.columns = column_names

    # Seleccionar columnas relevantes para el heatmap
    selected_cols = ['sample_1', 'sample_5', 'sample_9', 'sample_12', 'sample_14']
    correlation_matrix = np.corrcoef(df[selected_cols].values.T)

    # Graficar el heatmap
    heatmap(correlation_matrix, 
            column_names=selected_cols, 
            row_names=selected_cols,
            cmap='magma',
            figsize=(7.5, 7.5),
            cell_font_size=15)
    plt.title('Mapa de Calor de Correlación de Variables Seleccionadas')
    plt.show()


# Ejecutar las funciones
df = load_custom_csv()
plot_category_scatter(df)
load_and_plot_heatmap()
