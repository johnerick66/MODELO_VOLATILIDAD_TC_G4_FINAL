import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
sns.set(style="whitegrid")

# Cargar artefactos entrenados
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load("gbr_mejor_modelo_tc.pkl")
    selected_vars = joblib.load("selected_vars_volatilidad.pkl")
    imputer = joblib.load("imputer_volatilidad.pkl")
    scaler = joblib.load("scaler_volatilidad.pkl")
    return modelo, selected_vars, imputer, scaler

gbr, selected_vars, imputer, scaler = cargar_recursos()

# Cargar CSV
url = "https://raw.githubusercontent.com/johnerick66/modelo-volatilidad/main/tipo_cambio%202.csv"
df = pd.read_csv(url, sep=",", quotechar='"', thousands=",")
df = df.rename(columns={
    "Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta": "TC",
    "mes": "mes",
    "año": "anio",
    "Precio Cobre": "precio_cobre",
    "Precio Oro": "precio_oro",
    "Precio Zinc": "precio_zinc",
    "PIB": "pbi",
    "Reservas internacionales": "reservas",
    "Intervenciones del BCRP": "interv_bcrp",
    "Inflación EEUU": "inflacion_usa"
})


st.sidebar.title("Menú de Navegación")
pagina = st.sidebar.radio("Selecciona una página:",
                          ["Análisis Exploratorio", "Procesamiento de Datos", 
                           "Modelado y Predicciones", "Inputs y Predicciones"])

# Página 1: EDA
if pagina == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    st.write(df.head())
    st.write(df.describe())
    st.bar_chart(df["TC"])
    # Agregar más gráficos si quieres

# Página 2: Procesamiento de Datos
elif pagina == "Procesamiento de Datos":
    st.header("Procesamiento de Datos")
    df_proc = df.copy()
    df_proc['Rendimientos_log'] = np.log(df["TC"] / df["TC"].shift(1))
    df_proc.dropna(subset=['Rendimientos_log'], inplace=True)
    st.write(df_proc.head())

# Página 3: Modelado y Predicciones
elif pagina == "Modelado y Predicciones":
    st.header("Modelo de Volatilidad")
    st.write("Usando Gradient Boosting Regressor optimizado")
    # Mostrar métricas guardadas (si las guardaste en Colab)
    st.write("Modelo cargado con éxito. No es necesario reentrenar.")

# Página 4: Inputs y Predicciones
elif pagina == "Inputs y Predicciones":
    st.header("Predicciones de Tipo de Cambio")
    anio_input = st.number_input("Año inicio", min_value=2025, max_value=2030, value=2025)
    mes_inicio = st.selectbox("Mes inicio", list(range(1,13)), index=0)
    num_meses = st.number_input("Número de meses a predecir", min_value=1, max_value=36, value=12)
    
    # Preparar DataFrame futuro
    mes_dict = {'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,'Jul':7,'Ago':8,'Sep':9,'Oct':10,'Nov':11,'Dic':12}
    df['mes_num'] = df['mes'].map(mes_dict)
    df = df.sort_values(by=['anio','mes_num']).reset_index(drop=True)
    ultimo_X = df[selected_vars].iloc[-1].copy()
    ultimo_tc = df['TC'].iloc[-1]

    meses_futuro = []
    mes_actual = mes_inicio
    anio_actual = anio_input
    for _ in range(num_meses):
        meses_futuro.append((anio_actual, mes_actual))
        mes_actual += 1
        if mes_actual > 12:
            mes_actual = 1
            anio_actual += 1
    df_futuro = pd.DataFrame(meses_futuro, columns=['anio','mes_num'])
    for col in selected_vars:
        if col not in ['anio','mes_num']:
            df_futuro[col] = ultimo_X[col]
    df_futuro_scaled = scaler.transform(df_futuro[selected_vars])
    rendimientos_pred = gbr.predict(df_futuro_scaled)
    tc_pred = [ultimo_tc * np.exp(rendimientos_pred[0])]
    for r in rendimientos_pred[1:]:
        tc_pred.append(tc_pred[-1] * np.exp(r))
    df_futuro['TC_predicho'] = tc_pred

    # Mostrar resultados
    st.write(df_futuro[['anio','mes_num','TC_predicho']])
    st.line_chart(df_futuro['TC_predicho'])
