import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer

sns.set(style="whitegrid")

# ============================================================
# CARGAR ARTEFACTOS ENTRENADOS
# ============================================================
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load("gbr_mejor_modelo_tc.pkl")
    selected_vars = joblib.load("selected_vars_volatilidad.pkl")
    imputer = joblib.load("imputer_volatilidad.pkl")
    scaler = joblib.load("scaler_volatilidad.pkl")
    return modelo, selected_vars, imputer, scaler

gbr, selected_vars, imputer, scaler = cargar_recursos()

# ============================================================
# CARGAR DATASET HISTÓRICO (YA TRATADO)
# ============================================================
df = pd.read_csv("df_tratado_eda.csv")

# Crear columna mes_num para orden
mes_dict = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,
            'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}

df["mes_num"] = df["mes"].map(mes_dict)
df = df.sort_values(by=["anio", "mes_num"]).reset_index(drop=True)

# ============================================================
# MENÚ
# ============================================================
st.sidebar.title("Menú de Navegación")
pagina = st.sidebar.radio("Selecciona una página:",
    ["Análisis Exploratorio", "Procesamiento de Datos", 
     "Modelado y Predicciones", "Inputs y Predicciones"]
)

# ============================================================
# 1️⃣ ANÁLISIS EXPLORATORIO
# ============================================================
if pagina == "Análisis Exploratorio":

    st.header("Análisis Exploratorio de Datos")

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    st.subheader("Estadísticas descriptivas")
    st.write(df.describe())

    st.subheader("Gráfico del tipo de cambio histórico")
    st.line_chart(df["TC"])


# ============================================================
# 2️⃣ PROCESAMIENTO DE DATOS
# ============================================================
elif pagina == "Procesamiento de Datos":

    st.header("Procesamiento de Datos")

    df_proc = df.copy()
    df_proc["Rendimientos_log"] = np.log(df_proc["TC"] / df_proc["TC"].shift(1))
    df_proc = df_proc.dropna(subset=["Rendimientos_log"])

    st.subheader("Dataset con rendimientos logarítmicos")
    st.dataframe(df_proc.head())


# ============================================================
# 3️⃣ MODELO Y METRICAS
# ============================================================
elif pagina == "Modelado y Predicciones":

    st.header("Modelo usado: Gradient Boosting Regressor")
    st.write("Carga completa y correcta de artefactos entrenados.")
    st.write("No es necesario reentrenar pues es un modelo optimizado con CV-5.")

    st.subheader("Variables usadas en el modelo")
    st.write(selected_vars)


# ============================================================
# 4️⃣ INPUTS Y PREDICCIONES
# ============================================================

elif pagina == "Inputs y Predicciones":

    st.header("Predicciones de Tipo de Cambio")

    # Inputs del usuario
    anio_input = st.number_input("Año inicio", min_value=2000, max_value=2100, value=2025)

    mes_inicio = st.selectbox(
        "Mes inicio",
        list(range(1,13)),
        index=0,
        format_func=lambda x: ["Ene","Feb","Mar","Abr","May","Jun","Jul",
                               "Ago","Sep","Oct","Nov","Dic"][x-1]
    )

    num_meses = st.number_input("Meses a predecir", min_value=1, max_value=36, value=12)

    ejecutar = st.button("PREDICIR")

    if ejecutar:

        # ----------------------------------------------------
        # 1. Construir DF futuro SOLO con selected_vars
        # ----------------------------------------------------
        ultimo_row = df[selected_vars].iloc[-1].copy()
        ultimo_tc = df["TC"].iloc[-1]

        meses_futuro = []
        mes_actual, anio_actual = mes_inicio, anio_input

        for _ in range(num_meses):
            meses_futuro.append((anio_actual, mes_actual))
            mes_actual += 1
            if mes_actual > 12:
                mes_actual = 1
                anio_actual += 1

        # df_futuro SOLO para mostrar (con mes/ año)
        df_futuro = pd.DataFrame(meses_futuro, columns=["anio", "mes_num"])

        # ----------------------------------------------------
        # 2. Crear matriz EXACTA de entrada para el modelo
        # ----------------------------------------------------
        X_futuro = pd.DataFrame(columns=selected_vars)

        for anio_val, mes_val in meses_futuro:
            nueva_fila = ultimo_row.copy()
            nueva_fila["anio"] = anio_val   # reemplazar el año futuro
            X_futuro.loc[len(X_futuro)] = nueva_fila

        # ----------------------------------------------------
        # 3. Imputar y escalar
        # ----------------------------------------------------
        # Convertir tipos a float (SimpleImputer requiere dtype uniforme)
        X_futuro = X_futuro.astype(float)
        
        # Convertir a la forma exacta que espera el imputer (mismo orden)
        X_futuro = X_futuro[selected_vars]
        
        # Aplicar imputer
        X_futuro = imputer.transform(X_futuro)
        
        # Aplicar scaler
        X_futuro = scaler.transform(X_futuro)


        # ----------------------------------------------------
        # 4. Predicción
        # ----------------------------------------------------
        rendimientos_pred = gbr.predict(X_futuro)

        # reconstruir tipo de cambio
        tc_pred = [ultimo_tc * np.exp(rendimientos_pred[0])]
        for r in rendimientos_pred[1:]:
            tc_pred.append(tc_pred[-1] * np.exp(r))

        df_futuro["TC_predicho"] = tc_pred

        # Mes en texto
        mes_dict_inv = {v:k for k,v in mes_dict.items()}
        df_futuro["mes"] = df_futuro["mes_num"].map(mes_dict_inv)

        # ----------------------------------------------------
        # 5. Mostrar tabla
        # ----------------------------------------------------
        st.subheader("Tabla de Predicciones")
        st.dataframe(df_futuro[["anio", "mes", "TC_predicho"]].round(4))

        # ----------------------------------------------------
        # 6. Gráfico
        # ----------------------------------------------------
        st.subheader("Gráfico Histórico + Predicciones")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(range(len(df)), df["TC"], label="TC histórico")
        ax.plot(range(len(df), len(df)+num_meses), df_futuro["TC_predicho"],
                marker="o", label="TC Predicho")
        ax.legend()
        st.pyplot(fig)
