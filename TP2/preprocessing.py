import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
def preparar_dataset(df):
    
    df.loc[df['categoria_de_trabajo'] == 'sin_trabajo', 'trabajo'] = 'desempleado'
    df.loc[pd.isna(df['trabajo']), 'trabajo'] = 'no_responde'   
    df.loc[pd.isna(df['categoria_de_trabajo']), 'categoria_de_trabajo'] = 'no_responde'
    df.loc[pd.isna(df['barrio']), 'barrio'] = 'No responde'    
    
    df['barrio'] = df['barrio'].apply(definir_barrio)
    df['educacion_alcanzada'] = df['educacion_alcanzada'].apply(definir_nivel_educativo)
    df['estado_marital'] = df['estado_marital'].apply(definir_estado_marital)  
    return df


def aplicar_one_hot_encoding(df):
    columnas_one_hot = ['barrio', 'categoria_de_trabajo', 'educacion_alcanzada', 'estado_marital', 'genero', 'religion', 'rol_familiar_registrado', 'trabajo']
    df_preparado = pd.get_dummies(df, dummy_na=True, drop_first=True, columns=columnas_one_hot)    
    return df_preparado


def dividir_dataset(df):
    y = df['tiene_alto_valor_adquisitivo'].copy()
    df.drop(columns=['tiene_alto_valor_adquisitivo'],inplace=True)
    df.reset_index()
    X = df
    
    return X, y

def normalizar_datos(X_train, X_test):
    X_train_normalizado = X_train.copy()
    X_test_normalizado = X_test.copy()

    scaler = StandardScaler()
    scaler.fit(X_train_normalizado)
    
    X_train_normalizado = scaler.transform(X_train_normalizado)
    X_test_normalizado = scaler.transform(X_test_normalizado)

    return X_train_normalizado, X_test_normalizado

def traer_variables_categoricas(df):
    df = df[['barrio', 'categoria_de_trabajo', 'educacion_alcanzada', 'estado_marital','genero','religion','rol_familiar_registrado','trabajo']]
    return df


def traer_variables_discretas(df):
    df = df[['anios_estudiados', 'edad', 'horas_trabajo_registradas']]
    return df

def traer_variables_numericas(df):
    df = df[['anios_estudiados', 'edad', 'horas_trabajo_registradas','ganancia_perdida_declarada_bolsa_argentina']]
    return df

def expandir_dataset(X):
    X2 = X.copy()
    
    X2['clustering_2'] = KMeans(n_clusters = 2).fit_predict(X2)
    X2['clustering_4'] = KMeans(n_clusters = 4).fit_predict(X2)
    X2['clustering_6'] = KMeans(n_clusters = 6).fit_predict(X2)
    X2['clustering_10'] = KMeans(n_clusters = 10).fit_predict(X2)
    
    return X2

        
def preparar_df_predicciones(df_predicciones):
    
    id = df_predicciones['id'].copy()
    
    df_predicciones.drop(columns=['id'],inplace=True)
    df_predicciones.reset_index()
    df_predicciones.drop(columns=['representatividad_poblacional'],inplace=True)
    df_predicciones.reset_index()
    
    df_predicciones = preparar_dataset(df_predicciones)
    
    return id, df_predicciones


    