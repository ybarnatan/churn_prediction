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

        # Asigno las variables de configuracion:
        #Los 2do's parametro es el valor por defecto para el yaml.
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wendsday")
        DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
        MES_TEST = _cfg.get("MES_TEST", "202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise