from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


def traer_df():    
    return pd.read_csv("https://docs.google.com/spreadsheets/d/1-DWTP8uwVS-dZY402-dm0F9ICw_6PNqDGLmH0u8Eqa0/export?format=csv")


def traer_df_predicciones():     
    return pd.read_csv("https://docs.google.com/spreadsheets/d/1ObsojtXfzvwicsFieGINPx500oGbUoaVTERTc69pzxE/export?format=csv")


def plot_roc(_fpr, _tpr, x):

    roc_auc = auc(_fpr, _tpr)

    plt.figure(figsize=(15, 10))
    plt.plot(
        _fpr, _tpr, color='darkorange', lw=2, label=f'AUC score: {roc_auc:.2f}'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    
def graficar_auc_roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr, thresholds)
    print(f"El valor de la metrica AUC-ROC para este modelo es: {roc_auc_score(y_test, y_pred)}")

    
def graficar_matriz_confusion(y_true, y_pred):
    fig, ax = plt.subplots(dpi=100)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, ax=ax,fmt="d",square = True,cmap=plt.cm.Blues)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

    
def escribir_predicciones(ids, predicciones, nombre_archivo):
    
    with open("predicciones/"+nombre_archivo+".csv", "w") as archivo:
        
        archivo.write("id,tiene_alto_valor_adquisitivo\n")       
        for persona in range(len(ids)):
            archivo.write(str(ids[persona]) + "," + str(predicciones[persona]) + "\n")
 