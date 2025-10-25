# src/output_manager.py
import pandas as pd
import os
import logging
from datetime import datetime
# Asumo que config está en el mismo nivel o el nivel superior
from .config import STUDY_NAME 
from .config import LOGS_DIR # Usamos LOGS_DIR si necesitas acceder a él.

logger = logging.getLogger(__name__)


def guardar_predicciones_finales(resultados_df: pd.DataFrame, nombre_archivo: str = None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta 'predict'.
    Realiza validaciones de formato y tipo de datos.

    Args:
        resultados_df: DataFrame con numero_cliente y predict (valores binarios 0 o 1).
        nombre_archivo: Nombre base del archivo (si es None, usa STUDY_NAME).

    Returns:
        str: Ruta absoluta del archivo guardado.
    """

    # --- 1. Preparación de la ruta y carpeta ---
    
    # Crear carpeta 'predict' si no existe (asumo que 'predict' es la carpeta de destino)
    os.makedirs("predict", exist_ok=True)

    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME

    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = os.path.join("predict", f"{nombre_archivo}_{timestamp}.csv")

    
    try:
        # --- 2. Validaciones del DataFrame ---
        
        # Renombrar columnas a un estándar, si aún no lo están
        # Usamos .copy() para evitar SettingWithCopyWarning
        df_a_guardar = resultados_df.copy() 
        
        # Validar formato del DataFrame (Columnas requeridas)
        required_cols = {'numero_de_cliente', 'predict'}
        if not required_cols.issubset(df_a_guardar.columns):
            missing = required_cols - set(df_a_guardar.columns)
            logger.error(f"❌ Error de formato: Faltan las columnas requeridas: {missing}")
            raise ValueError(f"DataFrame incompleto. Faltan: {missing}")
            
        # Seleccionar y reordenar solo las columnas a guardar
        df_a_guardar = df_a_guardar[['numero_de_cliente', 'predict']]

        # Validar tipos de datos
        # numero_de_cliente debe ser un tipo entero (int)
        # predict puede ser entero (int) o booleano (bool)
        try:
            df_a_guardar['numero_de_cliente'] = df_a_guardar['numero_de_cliente'].astype(int)
            # Convertir 'predict' a entero (0 o 1) si aún no lo es
            df_a_guardar['predict'] = df_a_guardar['predict'].astype(int)
        except Exception as e:
            logger.error(f"❌ Error de tipo de dato al convertir columnas: {e}")
            raise TypeError(f"Asegúrese de que 'numero_de_cliente' y 'predict' contengan valores numéricos válidos. Detalle: {e}")

        # Validar valores de predict (deben ser 0 o 1)
        valid_values = df_a_guardar['predict'].isin([0, 1]).all()
        if not valid_values:
            invalid_count = df_a_guardar['predict'].isin([0, 1]).sum()
            total_count = len(df_a_guardar)
            logger.warning(f"⚠️ Alerta de valores: La columna 'predict' contiene valores distintos de 0 o 1. ({total_count - invalid_count} valores inválidos)")
            
            # Forzar la binarización si hay valores intermedios (ej: umbral con 0.5)
            # Esto asume que si no es 0 o 1, podría ser una probabilidad que debe ser binarizada
            if df_a_guardar['predict'].max() > 1 or df_a_guardar['predict'].min() < 0:
                 # Aplicar un umbral simple (ej: >= 0.5) como medida de seguridad si se pasó una probabilidad
                 df_a_guardar['predict'] = (df_a_guardar['predict'] >= 0.5).astype(int)
                 logger.info("ℹ️ Se aplicó una binarización forzada de 'predict' a 0/1 con umbral 0.5.")
            
        # --- 3. Guardar archivo ---
        
        # Guardar archivo CSV
        # index=False es fundamental para no incluir el índice de Pandas en el archivo final
        df_a_guardar.to_csv(ruta_archivo, index=False)

        # --- 4. Log y Retorno ---
        logger.info(f"✅ Predicciones guardadas en: {ruta_archivo}")
        logger.info(f"Formato del archivo:")
        logger.info(f"  Columnas: {list(df_a_guardar.columns)}")
        logger.info(f"  Registros: {len(df_a_guardar):,}")
        logger.info(f"  Primeras filas:")
        logger.info(f"\n{df_a_guardar.head()}")

    except Exception as e:
        logger.error(f"❌ Falla crítica al guardar predicciones: {e}")
        # En un entorno de producción, es mejor relanzar la excepción para detener el proceso
        raise

    return os.path.abspath(ruta_archivo)