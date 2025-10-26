import pandas as pd
import os
import datetime
import logging
from src.data_loader import cargar_datos, convertir_clase_ternaria_a_target, agregar_clase_ternaria_duckdb
from src.optimization import * 
from src.graficos_optuna import generar_grafico_html_optuna , bayesiana_top5_ganancia
import src.feature_engeneering as FeatEng
from src.config import *
import optuna
import lightgbm as lgb
import numpy as np
import logging
import json
from src.gain_function import calcular_ganancia, ganancia_lgb_binary
from src.crear_carpetas_proyecto import crear_carpetas_proyecto # Importamos la función
from src.best_params import *
from src.final_training import *

# Creo carpetas del proyecto si no existen
crear_carpetas_proyecto()

# Logging configuration
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre = f"log_{STUDY_NAME}_{fecha}.log"
log_filepath = os.path.join(LOGS_DIR, nombre)

logging.basicConfig(
    level=logging.INFO, # Nivel de log (INFO, DEBUG, etc.)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # 1. Handler para la consola (salida por terminal)
        logging.StreamHandler(),
        
        # 2. Handler para el archivo de log (solución al problema de archivos vacíos)
        # El archivo se crea y se escriben los logs en él.
        logging.FileHandler(log_filepath, mode='a', encoding='utf-8')])

logger = logging.getLogger(__name__)
logger.info("✅ Sistema de Logging Inicializado.")
logger.info(f"Los logs se guardarán en: {os.path.abspath(log_filepath)}")


def main():
    logger.info("Inicio de ejecucion del programa")
    
    #00  | Cargar datos y crear clase ternaria
    df_original = cargar_datos(DATA_PATH)    #path = "C:/Users/ybbar/OneDrive/Desktop/DMEyF2025_Competencia01_Proyect_Wedesnday/data/competencia_01_crudo.csv"
    print("Distribución datos x foto_mes:")
    print(df_original['foto_mes'].value_counts().sort_index())
    logger.info("Inicio de creacion clase ternaria")
    df = agregar_clase_ternaria_duckdb(df_original)
   
    # 01  | Feature Engeneering
    df = FeatEng.crear_lags(df, FE_ATRIBUTOS, CANT_LAGS)
    df = FeatEng.crear_deltas(df, FE_ATRIBUTOS, CANT_DELTAS)

    #02  | Convertir clase_ternaria a target binario
    df = convertir_clase_ternaria_a_target(df)    # Convertir clase_ternaria a target binario (CONT=0, BAJA+1 y BAJA+2 = 1)
    print("Distribución datos x foto_mes df desp de ternaria a target:")
    print(df_original['foto_mes'].value_counts().sort_index())
    
    # 03  | Optimizacion de hiperparametros
    study = optimizacion_bayesiana(df, n_trials = N_TRIALS_OPTUNA)
    
    #Guardando el grafico .html de la bayesiana
    FILE_JSON = f'{STUDY_NAME}_iteraciones' 
    NOMBRE_ESTUDIO = FILE_JSON.replace('_iteraciones', '') # Usa el nombre del est
    generar_grafico_html_optuna(NOMBRE_ESTUDIO, FILE_JSON)
    bayesiana_top5_ganancia(NOMBRE_ESTUDIO, FILE_JSON)
    
    # 04  |  Adicional  Análisis de la bayesiana
    logger.info("=== ANALISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    # 05  | Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    mejores_hiperparam = cargar_mejores_hiperparametros()    # Cargar mejores hiperparámetros  
    resultados_test = evaluar_en_test(df, mejores_hiperparam)    #Evaluo en test
    guardar_resultados_test(resultados_test)    #Guardar resultados de test
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")
    
    # 06 |  Entrenamiento modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df)
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_hiperparam)
    # Guardo el modelo final
    ruta_modelo_final = os.path.join(MODELS_DIR, f"{STUDY_NAME}_modelo_final.txt")
    modelo_final.save_model(ruta_modelo_final)
    logger.info(f"Modelo final guardado en: {ruta_modelo_final}")
    
    #07 | Generar predicciones 
    logger.info("Generar predicciones finales")
    envios = cargar_mejores_envios()
    predicciones = generar_predicciones_finales(X_predict, clientes_predict, envios) #predicciones = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)
    
    # Guardo las predicciones
    logger.info("Guardar predicciones")
    salida_kaggle = guardar_predicciones(predicciones)
    logger.info(f"Predicciones kaggle en: {salida_kaggle}")
    
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_hiperparam}")
    logger.info(f"🎯 Períodos de entrenamiento final: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDICT}")
    logger.info(f"📁 Archivo de salida: {salida_kaggle}")
    logger.info(f"📝 Log detallado: logs/{nombre}")
    
    # Guardar el df resultante post Feat Eng, post ternaria, post todo.
    logger.info("Guardando el DataFrame resultante en un archivo CSV")
    output_path = "data/competencia_01.csv"
    df.to_csv(output_path, index=False)    
    

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()