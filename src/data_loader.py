import pandas as pd
import logging
import duckdb

logger = logging.getLogger(__name__)

def cargar_datos(path: str) -> pd.DataFrame | None:
    """
    Carga los datos desde un archivo CSV a un DataFrame de pandas.
    Parameters:
    -----------
    path : str
        Ruta al archivo CSV
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos cargados
    """
    logger.info("Inicio de la funcion cargar_datos")
    try:
        df = pd.read_csv(path)
        logger.info("Datos cargados correctamente con filas: %d y columnas: %d", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.error("Error al cargar los datos: %s", e)
        raise # Crashear el programa si no cargo el df.



def agregar_clase_ternaria_duckdb(df_original):
    """
    Usa DuckDB y SQL para agregar la columna clase_ternaria al DataFrame original.

    Parámetros:
        df_original (pd.DataFrame): DataFrame original que contiene al menos
                                    numero_de_cliente y foto_mes

    Retorna:
        pd.DataFrame: DataFrame con clase_ternaria agregada
    """
    con = duckdb.connect()

    # Registrar df_original como tabla temporal
    con.register('df_original', df_original)

    # Crear tabla temporal con la clase_ternaria usando SQL
    con.execute("""
        CREATE OR REPLACE TEMP TABLE base_target AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            CASE 
                WHEN foto_mes_1 IS NULL THEN 'BAJA+1'
                WHEN foto_mes_2 IS NULL THEN 'BAJA+2'
                ELSE 'CONTINUA'
            END AS clase_ternaria
        FROM (
            SELECT 
                numero_de_cliente,
                foto_mes,
                LEAD(foto_mes, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS foto_mes_1,
                LEAD(foto_mes, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS foto_mes_2
            FROM df_original
        ) sub
        WHERE foto_mes <= 202104
    """)

    # Hacer el join con el DataFrame original
    result = con.execute("""
        SELECT 
            o.*, 
            t.clase_ternaria
        FROM df_original o
        LEFT JOIN base_target t
        ON o.numero_de_cliente = t.numero_de_cliente
           AND o.foto_mes = t.foto_mes
    """).fetchdf()

    con.close()
    return result

            
            
def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
        - CONTINUA = 0
        - BAJA+1 y BAJA+2 = 1
    La optimización requiere convertir clase_ternaria a formato binario: Aca esta en cada uno la decision que toma, 
    creo lo mas sano es probar incluyendo los BAJA+1 como target igual a CONTINUA y luego como los BAJA+2.
  
    Args:
        df: DataFrame con columna 'clase_ternaria'
  
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()
  
    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()
  
    # Convertir clase_ternaria a binario en el mismo atributo
    df_result['clase_ternaria'] = df_result['clase_ternaria'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })
  
    # Log de la conversión
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] == 1).sum()
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos (i.e. 1's / total)")
  
    return df_result