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
