import pandas as pd
import logging

logger = logging.getLogger(__name__)



#Funcion para  cargar  datios

def cargar_datos(path: str) -> pd.DataFrame | None:
    logger.info("Inicio de la funcion cargar_datos")
    try:
        df = pd.read_csv(path)
        logger.info("Datos cargados correctamente con filas: %d y columnas: %d", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.error("Error al cargar los datos: %s", e)
        raise # Crashear el programa si no cargo el df.
            
