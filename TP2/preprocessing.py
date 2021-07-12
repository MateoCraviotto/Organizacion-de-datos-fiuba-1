import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


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
    return df


def aplicar_one_hot_encoding(df):
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
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    for col in columnas_numericas:
        mean = X_train_norm[col].mean()
        std = X_train_norm[col].std()
        
        X_train_norm[col] = (X_train_norm[col] - mean) / std
        X_test_norm[col] = (X_test_norm[col] - mean) / std
        
    return X_train_norm, X_test_norm


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
    X = X.copy()
    X2 = aplicar_one_hot_encoding(X)
    
    X2['clustering_2'] = KMeans(n_clusters = 2).fit_predict(X2)
    X2['clustering_4'] = KMeans(n_clusters = 4).fit_predict(X2)
    X2['clustering_6'] = KMeans(n_clusters = 6).fit_predict(X2)
    X2['clustering_10'] = KMeans(n_clusters = 10).fit_predict(X2)
    
    return X2

        
def preparar_holdout(holdout):
    
    id = holdout['id'].copy()
    
    holdout.drop(columns=['id'],inplace=True)
    holdout.reset_index()
    holdout.drop(columns=['representatividad_poblacional'],inplace=True)
    holdout.reset_index()
    
    holdout = preparar_dataset(holdout)
    
    return id, holdout