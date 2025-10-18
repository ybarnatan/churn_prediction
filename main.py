import pandas as pd
import os
import datetime
import logging
from src.data_loader import cargar_datos, convertir_clase_ternaria_a_target, agregar_clase_ternaria_duckdb
import src.feature_engeneering as FeatEng
from src.conf import *
from src.optimization import *
import optuna
import lightgbm as lgb
import numpy as np
import logging
import json
from src.gain_function import calcular_ganancia, ganancia_lgb_binary


#Log para el main
os.makedirs("logs", exist_ok=True) #Corroboro que exista la carpeta logs
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre = f"log_fecha_{fecha}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
        ])

logger = logging.getLogger(__name__)

#Configuraciones del YAML en log:
logger.info("==Configuracion cargada desde YAML==")
logger.info(f"Study name: {STUDY_NAME}")
logger.info(f"Data path: {DATA_PATH}")
logger.info(f"Semilla: {SEMILLA}")
logger.info(f"Mes train: {MES_TRAIN}")      
logger.info(f"Mes validacion: {MES_VALIDACION}")
logger.info(f"Mes test: {MES_TEST}")
logger.info(f"Ganancia acierto: {GANANCIA_ACIERTO}")
logger.info(f"Costo estimulo: {COSTO_ESTIMULO}")


def main():
    logger.info("Inicio de ejecucion del programa")
    # Carga de datos
    #path = "C:/Users/ybbar/OneDrive/Desktop/DMEyF2025_Competencia01_Proyect_Wedesnday/data/competencia_01_crudo.csv"
    df_original = cargar_datos(DATA_PATH)
    print("Distribución datos x foto_mes:")
    print(df_original['foto_mes'].value_counts().sort_index())
    
    #Crear clase_ternaria 
    logger.info("Inicio de creacion clase ternaria")
    df = agregar_clase_ternaria_duckdb(df_original)
    print("Distribución datos x foto_mes ternaria duckdb:")
    print(df['foto_mes'].value_counts().sort_index())
    
    print(df.head(5))

    # Feature engineering | Lags
    # atributos_lags = ["ctrx_quarter"] #Atributos para los cuales quiero crear lags
    # cant_lags = 5 #Cantidad de lags a crear (1 mes atras, 2 meses atras, etc)
    # df = FeatEng.crear_lags(df, columnas = atributos_lags, cant_lag = cant_lags)
    
    # Feature engineering | Deltas
    #atributos_deltas = ["ctrx_quarter"]
    #cant_deltas = 3
    #df = FeatEng.crear_deltas(df, columnas=atributos_deltas, cant_deltas=cant_deltas)

        
    
    # Convertir clase_ternaria a target binario (CONT=0, BAJA+1 y BAJA+2 = 1)
    df = convertir_clase_ternaria_a_target(df)
    print("Distribución datos x foto_mes df desp de ternaria a target:")
    print(df_original['foto_mes'].value_counts().sort_index())
    
    print(df.head(5))
    
    # Optimizacion de hiperparametros
    study = optimizacion_bayesiana(df, n_trials = 100)
    
    # Análisis de la bayesiana
    logger.info("=== ANALISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    # Guardar el df resultante
    logger.info("Guardando el DataFrame resultante en un archivo CSV")
    output_path = "data/competencia_01.csv"
    df.to_csv(output_path, index=False)    
    
    print("Semillas que vienen del config.yaml:", SEMILLA)
    
    logger.info(f"Fin de ejecucion del programa. Revisar detalle en logs/{nombre}")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()