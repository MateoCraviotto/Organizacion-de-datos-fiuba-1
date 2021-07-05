import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

def definir_barrio(barrio):
    if barrio == "Palermo":
        return "Palermo"
    else:
        return "Otro"
    
def definir_estado_marital(estado_marital):
    if estado_marital.startswith('matrimonio'):
        return 'Con matrimonio'
    else:
        return 'Sin matrimonio'
    
def definir_nivel_educativo(nivel):
    if nivel.startswith("uni"):
        return "Universitario"
    elif nivel.endswith("anio"):
        return "Secundario"
    elif nivel.endswith("grado"):
        return "Primario"
    else:
        return "Jardin"
    
def feature_engineering(df):
    df['trabajo'] = np.where(df['categoria_de_trabajo'] == 'sin_trabajo', 'desempleado', df['trabajo'])
    df.reset_index(drop=True, inplace=True)

    df = df.replace(np.nan, {'categoria_de_trabajo': 'no_responde', 'trabajo': 'no_responde'})
    df.reset_index(drop=True, inplace=True)

    df = df.replace(np.nan, {'barrio': 'No responde'})
    df.reset_index(drop=True, inplace=True)

    return df
    
def preparar_dataset(df):
    feature_engineering(df)    
    df['barrio'] = df['barrio'].apply(definir_barrio)
    df['educacion_alcanzada'] = df['educacion_alcanzada'].apply(definir_nivel_educativo)
    df['estado_marital'] = df['estado_marital'].apply(definir_estado_marital)

    df_preparado = pd.get_dummies(df, dummy_na=True, drop_first=True)
    
    return df_preparado

def dividir_dataset(df):
    y = df['tiene_alto_valor_adquisitivo'].copy()
    df.drop(columns=['tiene_alto_valor_adquisitivo'],inplace=True)
    df.reset_index()
    X = df
    
    return X, y

def normalizar_datos(X_train, X_test):
    columnas_numericas = ['anios_estudiados', 'edad', 'ganancia_perdida_declarada_bolsa_argentina', 'horas_trabajo_registradas']
    for col in columnas_numericas:
        mean = X_train[col].mean()
        std = X_train[col].std()
        X_train[col].apply(lambda x: (x - mean) / std)
        X_test[col].apply(lambda x: (x - mean) / std)
        
    return X_train, X_test