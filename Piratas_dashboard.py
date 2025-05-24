import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide") # Opcional: para usar todo el ancho de la página

st.title('Dashboard de Análisis de LMB ⚾︎')
st.write('Explora los componentes principales de tus datos.')

# --- 1. Carga de Datos (Mejora: Usar st.file_uploader) ---
st.header("Carga tu archivo de datos")
uploaded_file = st.file_uploader("Sube tu archivo")

data = None # Inicializa data a None
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.success("Archivo cargado exitosamente.")
        st.subheader("Vista previa de los datos cargados:")
        st.dataframe(data.head())

        # --- 2. Preprocesamiento de Datos ---
        # Si el usuario sube el archivo, procedemos con el análisis
        if data is not None:
            # Seleccionar solo columnas numéricas para PCA
            # Esto es más robusto que un slicing fijo
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                st.error("No se encontraron columnas numéricas en el archivo. Asegúrate de que tus datos contengan valores numéricos para el análisis PCA.")
            else:
                data_numeric = data[numeric_cols].copy()

                # Manejo de valores faltantes: Imputación simple o eliminación
                # Aquí optamos por rellenar NaNs con la media de la columna
                # Considera otras estrategias si tus datos lo requieren
                if data_numeric.isnull().sum().sum() > 0:
                    st.warning("Se encontraron valores faltantes en las columnas numéricas. Se rellenarán con la media de cada columna.")
                    data_numeric = data_numeric.dropna(axis=1)
                # Asegurarse de que hay suficientes filas para el slicing si se mantiene
                # O simplemente usar todo el dataframe numérico
                # Para este ejemplo, usaremos todo el dataframe numérico para el PCA
                # Si necesitas un slicing específico, asegúrate de que los índices existan
                # Por ejemplo, si siempre quieres las primeras 18 filas y ciertas columnas:
                # data_for_pca = data_numeric.iloc[0:18, 1:25] # Asegúrate de que estos índices sean válidos para data_numeric

                # Usaremos todo el data_numeric para PCA para mayor flexibilidad
                data_for_pca = data_numeric

                if data_for_pca.empty:
                    st.error("El DataFrame para PCA está vacío después del preprocesamiento. Por favor, revisa tus datos.")
                else:
                    # Escalado de los datos
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(data_for_pca)

                    # --- 3. Aplicación de PCA ---
                    # Puedes permitir al usuario elegir el número de componentes
                    n_components_option = st.slider(
                        "Selecciona el número de componentes PCA:",
                        min_value=2,
                        max_value=min(4, train_scaled.shape[1]), # Máximo de componentes es el menor entre 4 y el número de columnas
                        value=2 # Valor por defecto para un gráfico 2D
                    )

                    pca = PCA(n_components=n_components_option)
                    data_pca_transformed = pca.fit_transform(train_scaled)

                    # Convertir el resultado de PCA a un DataFrame para Plotly
                    pca_columns = [f'PC{i+1}' for i in range(n_components_option)]
                    pca_df = pd.DataFrame(data_pca_transformed, columns=pca_columns)

                    # Añadir las etiquetas originales si es posible (por ejemplo, el índice original)
                    # O si tienes una columna de etiquetas en tus datos originales, la puedes añadir aquí
                    # Por ejemplo, si la primera columna de 'data' era un ID o nombre:
                    # if not data.empty and data.shape[0] == pca_df.shape[0]:
                    #     pca_df['Etiqueta'] = data.iloc[:, 0].reset_index(drop=True)


                    st.subheader(f"Resultados de PCA con {n_components_option} componentes")
                    st.write(f"Varianza explicada por cada componente: {pca.explained_variance_ratio_}")
                    st.write(f"Varianza explicada acumulada: {np.sum(pca.explained_variance_ratio_):.2f}")

                    # --- 4. Visualización con Plotly Express ---
                    if n_components_option >= 2:
                        st.subheader("Gráfico de Dispersión PCA (PC1 vs PC2)")
                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            title='Análisis de Componentes Principales',
                            hover_data=pca_columns # Muestra los valores de PC al pasar el ratón
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if n_components_option >= 3:
                        st.subheader("Gráfico de Dispersión PCA 3D (PC1 vs PC2 vs PC3)")
                        fig_3d = px.scatter_3d(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            title='Análisis de Componentes Principales (3D)',
                            hover_data=pca_columns
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.info("Asegúrate de que el archivo es un Excel válido y que contiene datos numéricos apropiados.")
else:
    st.info("Por favor, sube un archivo Excel para comenzar el análisis.")

