import pandas as pd
import numpy as np
# import seaborn as sns # No se usa directamente en el código actual, puedes removerlo si no lo necesitas
# import matplotlib.pyplot as plt # No se usa directamente, puedes removerlo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide") # Opcional: para usar todo el ancho de la página

st.title('Dashboard de Análisis de LMB ⚾')
st.write('Explora los componentes principales de tus datos.')

# --- 1. Carga de Datos ---
st.header("Carga tu archivo de datos")
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

data = None
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.success("Archivo cargado exitosamente.")
        st.subheader("Vista previa de los datos cargados:")
        st.dataframe(data.head())

        # --- 2. Preprocesamiento de Datos ---
        if data is not None:
            # Seleccionar solo columnas numéricas para PCA
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                st.error("No se encontraron columnas numéricas en el archivo. Asegúrate de que tus datos contengan valores numéricos para el análisis PCA.")
            else:
                data_numeric = data[numeric_cols].copy()

                if data_numeric.isnull().sum().sum() > 0:
                    st.warning("Se encontraron valores faltantes en las columnas numéricas. Se rellenarán con la media de cada columna.")
                    data_numeric = data_numeric.dropna(axis=1)

                data_for_pca = data_numeric

                if data_for_pca.empty:
                    st.error("El DataFrame para PCA está vacío después del preprocesamiento. Por favor, revisa tus datos.")
                else:
                    # --- Obtener las etiquetas/nombres de los puntos ---
                    st.subheader("Configuración de Nombres de Jugadores")
                    player_id_column_name = st.text_input(
                        "Introduce el nombre de la columna que contiene los nombres/IDs de los jugadores (deja vacío si no hay una):",
                        value='' # Valor por defecto vacío
                    )

                    player_labels = []
                    if player_id_column_name and player_id_column_name in data.columns:
                        # Asegúrate de que las etiquetas corresponden a las filas usadas para PCA
                        # Esto es clave para que los nombres y los puntos coincidan
                        player_labels = data[player_id_column_name].loc[data_for_pca.index].reset_index(drop=True)
                        if len(player_labels) != data_for_pca.shape[0]:
                            st.warning("El número de etiquetas de jugadores no coincide con el número de filas procesadas para PCA. Usando etiquetas genéricas.")
                            player_labels = [f"Punto {i+1}" for i in range(data_for_pca.shape[0])]
                    else:
                        st.info("No se especificó una columna de nombres o no se encontró. Los puntos se etiquetarán genéricamente (Punto 1, Punto 2...).")
                        player_labels = [f"Punto {i+1}" for i in range(data_for_pca.shape[0])]


                    # Escalado de los datos
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(data_for_pca)

                    # --- 3. Aplicación de PCA ---
                    n_components_option = st.slider(
                        "Selecciona el número de componentes PCA:",
                        min_value=2,
                        max_value=min(4, train_scaled.shape[1]),
                        value=2
                    )

                    pca = PCA(n_components=n_components_option)
                    data_pca_transformed = pca.fit_transform(train_scaled)

                    # Convertir el resultado de PCA a un DataFrame para Plotly
                    pca_columns = [f'PC{i+1}' for i in range(n_components_option)]
                    pca_df = pd.DataFrame(data_pca_transformed, columns=pca_columns)

                    # --- Añadir las etiquetas de los jugadores al DataFrame PCA ---
                    pca_df['Etiqueta del Punto'] = player_labels

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
                            hover_data=pca_columns,
                            hover_name='Etiqueta del Punto' # <--- ¡Aquí se usa la columna de nombres!
                            # Si quieres el texto directamente en el punto (puede saturar):
                            # text='Etiqueta del Punto'
                        )
                        # Si usas 'text', podrías querer ajustar la posición:
                        # fig.update_traces(textposition='top center')
                        st.plotly_chart(fig, use_container_width=True)

                    if n_components_option >= 3:
                        st.subheader("Gráfico de Dispersión PCA 3D (PC1 vs PC2 vs PC3)")
                        fig_3d = px.scatter_3d(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            title='Análisis de Componentes Principales (3D)',
                            hover_data=pca_columns,
                            hover_name='Etiqueta del Punto' # <--- También para el gráfico 3D
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
        st.info("Asegúrate de que el archivo es un Excel válido y que contiene datos numéricos apropiados.")
else:
    st.info("Por favor, sube un archivo Excel para comenzar el análisis.")