import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
import gc

from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval

logger = logging.getLogger(__name__)


def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación para el LIGHTGBM
    Utiliza configuración YAML para períodos y semilla.
    1. Define parametros para el modelo LightGBM
    2. Preparar dataset para entrenamiento y validación
    3. Entrena modelo con función de ganancia personalizada (CV)
    4. Ganancia promedio del CV
    5 .Guardar cada iteración en JSON

    Returns:
    float: ganancia total
    """
 
    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', PARAMETROS_LGB['min_data_in_leaf'][0], PARAMETROS_LGB['min_data_in_leaf'][1]),
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        'seed': SEMILLA[0],
        'verbose': -1
    }

    # Preparo datasets para train y validacion -> Filtro segun periodos
    if isinstance(MES_TRAIN, list): #Si df_train es una lista
        meses_train_str = [str(m) for m in MES_TRAIN]
        df_train = df[df['foto_mes'].astype(str).isin(meses_train_str)]   
    else: #Si es un string i.e. solo un mes.
        mes_train_str = str(MES_TRAIN)       
        df_train = df[df['foto_mes'].astype(str) == mes_train_str]
    
    df_val = df[df['foto_mes'] == MES_VALIDACION]
    
    # Validaciones tempranas por posibles errores de filtrado en fechas/formato de fechas.
    if df_train.empty:
        raise ValueError("df_train está vacío. Revisá PERIODO_ENTRENAMIENTO y que existan datos.")
    if df_val.empty:
        raise ValueError("df_val está vacío. Revisá PERIODO_VALIDACION y que existan datos.")
    if df_train['clase_ternaria'].nunique() < 2:
        raise ValueError("df_train no contiene ambas clases (0 y 1).")

    # Usar target (con clase ternaria ya convertida a binaria)
    y_train = df_train['clase_ternaria'].values
    y_val = df_val['clase_ternaria'].values

    # Features (excluir target)
    X_train = df_train.drop(columns=['clase_ternaria'])
    X_val = df_val.drop(columns=['clase_ternaria'])

    # Entrenar modelo con función de ganancia personalizada
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
     #Aquí se entrena y valida el modelo con 5-fold cross-validation, usando los hiperparámetros definidos arriba.
    cv_result = lgb.cv(         # Uso LGBM CV = dividir el data en k folds -> promedia los resultados de mis k modelos.
        params,
        train_data,
        num_boost_round = 300,      # Modificar, subir y subir... y descomentar la línea inferior
        #early_stopping_rounds= int(50+5 / (param['learning_rate']))  ,   # Corta el entrenamiento si el modelo deja de mejorar durante N rondas consecutivas.
        feval=lgb_gan_eval,
        stratified=True,
        nfold=5,
        seed=SEMILLA[0],
        callbacks=[
                lgb.early_stopping(stopping_rounds=int(50+5 / (params['learning_rate'])), verbose=False),
                #lgb.log_evaluation(period=200),
        ]
    )

    max_ganancia = max(cv_result['valid gan_eval-mean']) # Toma la máxima ganancia promedio alcanzada entre los 5 folds.
    ganancia_total = max_ganancia * 5#Como la ganancia es promedio entre los k=5 folds, se multiplica por k=5 para simular la ganancia total estimada en datos reales (sin dividirlos en k=5 folds).
    best_iter = cv_result['valid gan_eval-mean'].index(max_ganancia) + 1 # Encuentra en qué iteración se logró esa ganancia máxima.
    trial.set_user_attr("best_iter", best_iter) #Guarda ese dato dentro del trial, así lo podés recuperar después.
    
    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
    
    return ganancia_total  

    
    '''
    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=ganancia_lgb_binary, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)  # Usar mismo umbral que en ganancia_lgb_binary  
    '''
                    
    # ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # # Guardar cada iteración en JSON
    # guardar_iteracion(trial, ganancia_total)
  
    # logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    # return ganancia_total



def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
        
    if archivo_base is None:
        archivo_base = STUDY_NAME
    # Construir ruta: resultados/[STUDY_NAME]_iteraciones.json
    archivo = os.path.join(RESULTS_DIR, f"{archivo_base}_iteraciones.json")
    
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA[0],
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")



def optimizacion_bayesiana(df, n_trials=100, n_jobs=1) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)

    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado.
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME
    storage_path = os.path.join(OPTUNA_DIR, f"{study_name}.db")
    storage_url = f"sqlite:///{storage_path}"

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN = {MES_TRAIN}, VALID = {MES_VALIDACION}, SEMILLA = {SEMILLA}")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,    
        sampler=optuna.samplers.TPESampler(seed=SEMILLA)
        # storage=storage_name,
        # load_if_exists=True,
    )

    #Corro la optimización
    study.optimize(lambda t: objetivo_ganancia(t, df), n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)
    
    #Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f} en el trial {study.best_trial.number}")
    logger.info(f"Mejores hiperparámetros: {study.best_trial.params}")

    # Generar y guardar los gráficos
    
    # Importancia de parámetros
    fig_importancia = optuna.visualization.plot_param_importances(study)
    fig_importancia.update_layout(title=f"Importancia de Parámetros - {STUDY_NAME}")
    importance_path = os.path.join(GRAPHICS_DIR, f"{STUDY_NAME}_FeatImportance.html")
    fig_importancia.write_html(importance_path) # <-- Usamos GRAPHICS_DIR
    logger.info(f"Gráfico de Importancia guardado en: {importance_path}")

    # Gráfico de Contorno
    fig_contour = optuna.visualization.plot_contour(study, params=['num_leaves', 'min_data_in_leaf'])
    fig_contour.update_layout(title=f"Gráfico Contorno - {STUDY_NAME}")
    contour_path = os.path.join(GRAPHICS_DIR, f"{STUDY_NAME}_ContourPlot.html")
    fig_contour.write_html(contour_path) # <-- Usamos GRAPHICS_DIR
    logger.info(f"Gráfico de Contorno guardado en: {contour_path}")

    # Gráfico de Slice
    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.update_layout(title=f"Gráfico Slice - {STUDY_NAME}")
    slice_path = os.path.join(GRAPHICS_DIR, f"{STUDY_NAME}_SlicePlot.html")
    fig_slice.write_html(slice_path) # <-- Usamos GRAPHICS_DIR
    logger.info(f"Gráfico de Slice guardado en: {slice_path}")

    return study

'''
def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
        * Entrenamiento: MES_TRAIN + MES_VALIDACION
        * Evaluación: MES_TEST 
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]
  
    # Entrenar modelo con mejores parámetros
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.shape}, Dimensiones df_test: {df_test.shape}')


    # # Preparar dataseT de LightGBM
    train_data = lgb.Dataset(df_train_completo.drop(columns=['clase_ternaria']), 
                             label=df_train_completo['clase_ternaria'].values)
    # chequeo si train_data está ok
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")

    #Se entrena el modelo usando los mejores parámetros y una función de evaluación ganancia_evaluator
    model = lgb.train(
        mejores_params,
        train_data,
        #num_boost_round=1000,
        #valid_sets=[test_data],
        #feval=ganancia_lgb_binary,
        feval=ganancia_evaluator
      #  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # #Se generan predicciones probabilísticas  en test (y_pred_proba) del modelo LightGBM.
    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria'].values
    y_pred_proba = model.predict(X_test)

    # # Buscar el umbral que maximiza la ganancia >LightGBM devuelve probabilidades; aquí se busca el umbral que maximiza la ganancia.
    # #Para cada umbral: Se convierte la predicción en binaria. => Se calcula la ganancia usando calcular_ganancia. => Se guarda el umbral y predicción que dan la mayor ganancia
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # Guardar predicción óptima

    ganancia_test = mejor_ganancia

    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': SEMILLA[0]
    }
  
    return resultados
'''

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: dict - mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # === 1. Preparar datos de entrenamiento (TRAIN + VALIDACION) ===
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)].copy()
    df_test = df[df['foto_mes'] == MES_TEST].copy()

    logger.info(f"Dimensiones entrenamiento: {df_train_completo.shape}, test: {df_test.shape}")

    # === 2. Preparar datasets para LightGBM ===
    X_train = df_train_completo.drop(columns=['clase_ternaria'])
    y_train = df_train_completo['clase_ternaria'].values

    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria'].values

    train_data = lgb.Dataset(X_train, label=y_train)

    logger.info("Entrenando modelo con los mejores hiperparámetros...")

    # === 3. Entrenar modelo ===
    model = lgb.train(
        mejores_params,
        train_data,
        feval=ganancia_evaluator  # función de evaluación personalizada
    )

    # === 4. Predicciones en test ===
    y_pred_proba = model.predict(X_test)

    # === 5. Buscar el umbral óptimo que maximiza la ganancia ===
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    umbrales = np.linspace(0, 1, 201)  # de 0.0 a 1.0 en pasos de 0.005

    for umbral in umbrales:
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin  # guardar la mejor predicción

    ganancia_test = mejor_ganancia

    # === 6. Estadísticas básicas ===
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100

    # === 7. Crear diccionario de resultados ===
    resultados = {
        'ganancia_test': float(ganancia_test),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'semilla': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
    }

    logger.info(f"Ganancia en test: {ganancia_test:.2f}, Umbral óptimo: {mejor_umbral:.3f}")
    logger.info(f"Porcentaje de positivos: {porcentaje_positivas:.2f}%")

    # === 8. Liberar memoria ===
    del df_train_completo, df_test, X_train, X_test, y_train, y_test, train_data, model
    gc.collect()

    return resultados


import os
import json
import logging
from datetime import datetime
# Importaciones necesarias de config.py (asumiendo que config.py es ejecutado y exporta estas variables)
from .config import (
    STUDY_NAME,
    RESULTS_DIR,
    SEMILLA,
    FINAL_PREDICT
)

logger = logging.getLogger(__name__)




def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    """
    Args:
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_Resultado_Test.json"
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
    

    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    iteracion_data = {
        'Mes_test': MES_TEST,
        'ganancia_test': float(resultados_test['ganancia_test']),
        'date_time': datetime.now().isoformat(),
        'state': 'COMPLETE',
        'configuracion':{
            'semilla': resultados_test['semilla'],
            'meses_train': periodos_entrenamiento
        },
        'resultados':resultados_test
    }

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    #logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {resultados_test['ganancia_test']:,.0f}" + "---" + f"Total Predicciones positivas: {resultados_test['predicciones_positivas']:,.0f}")
