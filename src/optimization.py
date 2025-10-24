import optuna
import lightgbm as lgb
# from lightgbm import early_stopping, log_evaluation
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


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

    # #Preparo datasets para train y validacion -> Filtro segun periodos
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
    cv_result = lgb.cv(
        params,
        train_data,
        num_boost_round = 300,      # Modificar, subit y subir... y descomentar la línea inferior
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
    best_iter = cv_result['valid gan_eval-mean'].index(max_ganancia) + 1 # Encuentra en qué iteración se logró esa ganancia máxima.
    trial.set_user_attr("best_iter", best_iter) #Guarda ese dato dentro del trial, así lo podés recuperar después.
    
    # Guardar cada iteración en JSON
    guardar_iteracion(trial, max_ganancia * 5)
    

    return max_ganancia * 5  #Como la ganancia es promedio entre los 5 folds, se multiplica por 5 para simular la ganancia total estimada en datos reales (sin dividirlos en folds).

    
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

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0])
        # storage=storage_name,
        # load_if_exists=True,
    )

    #Corro la optimización
    study.optimize(lambda t: objetivo_ganancia(t, df), n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs, gc_after_trial=True)

    # Generar y guardar los gráficos
    # Usamos GRAPHICS_DIR en lugar de "graficos_resultados/"
    
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
