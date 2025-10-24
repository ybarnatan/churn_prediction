import yaml
import os
import logging

# ====================
# Tomo los datos de conf.yaml
# ====================
logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try: 
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f) #Leo el archivo conf.yaml
        _cfg = _cfgGeneral["competencia01"] #Me quedo con la seccion competencia01
        _dirs_cfg = _cfgGeneral.get("PROJECT_DIRS", {}) # Cargo la nueva sección
        
        # Asigno las variables de configuracion:
        #En c/u, el  2do's parametro es el valor por defecto para el yaml.
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wednesday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
        PARAMETROS_LGB = _cfgGeneral.get("PARAMETROS_LGB", {})
        UMBRAL = _cfg.get("UMBRAL", 0.025)
        N_TRIALS_OPTUNA = PARAMETROS_LGB.get("n_trial", 100)
        
        # --- Variables de carpetas ---
        DIRS_TO_CREATE = list(_dirs_cfg.values())
        if "src" not in DIRS_TO_CREATE:
            DIRS_TO_CREATE.append("src")

        DATA_DIR = _dirs_cfg.get("DATA_DIR", "data")
        LOGS_DIR = _dirs_cfg.get("LOGS_DIR", "logs")
        RESULTS_DIR = _dirs_cfg.get("RESULTS_DIR", "resultados")
        GRAPHICS_DIR = _dirs_cfg.get("GRAPHICS_DIR", "graficos_resultados")
        MODELS_DIR = _dirs_cfg.get("MODELS_DIR", "models")
        OPTUNA_DIR = _dirs_cfg.get("OPTUNA_DIR", "optimization_optuna")
        REPORTS_DIR = _dirs_cfg.get("REPORTS_DIR", "reportes")
        KAGGLE_DIR = _dirs_cfg.get("KAGGLE_DIR", "kaggle")


except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise

