import pandas as pd
import logging
import duckdb

logger = logging.getLogger(__name__)

def crear_lags(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
    
    logger.debug(f"Consulta SQL: {sql}")
    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            #logger.warning(f"El atributo {attr} no existe en el DataFrame") 
            print("El atributo no existe en el DF")


    # Completar la consulta
    sql += " FROM df"


    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df() # Asigno la consulta al df.
    con.close()
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


def crear_deltas(df: pd.DataFrame, columnas: list[str], cant_deltas: int = 1) -> pd.DataFrame:
    """
    Genera variables de delta (diferencia con meses anteriores) para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar deltas. Si es None o vacía, no se generan.
    cant_deltas : int, default=1
        Cantidad de deltas (meses hacia atrás) a calcular para cada atributo

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de delta agregadas
    """
     logger.info(f"Realizando feature engineering con {cant_deltas} deltas para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar deltas")
        return df


    # Construir la consulta SQL
    sql = "SELECT *"
    logger.debug(f"Consulta SQL: {sql}")

    # Agregar los deltas para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_deltas + 1):
                sql += f", {attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_delta_{i}"
        else:
            #logger.warning(f"El atributo {attr} no existe en el DataFrame") 
            print("El atributo no existe en el DF")

    # Completar la consulta
    sql += " FROM df"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()
    
    logger.info(f"Feature engineering (deltas) completado. DataFrame resultante con {df.shape[1]} columnas")

    return df



