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
    #Nro columnas antes de este F.E.
    n_inicial = df.shape[1]
    logger.info(f"[LAGS] Columnas iniciales: {n_inicial}")

    # Calcular cuántas columnas nuevas se van a crear (solo las que existen)
    columnas_validas = [c for c in columnas if c in df.columns]
    n_nuevas = len(columnas_validas) * cant_lag
    logger.info(f"[LAGS] Se agregarán {n_nuevas} columnas nuevas ({cant_lag} por cada uno de {len(columnas_validas)} atributos válidos)")

    
    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    #Nro columnas desp de este F.E.
    n_final = df.shape[1]
    logger.info(f"[LAGS] Feature engineering completado. Columnas finales: {n_final} (se agregaron {n_final - n_inicial})")
    return df


def crear_deltas(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de delta (diferencia) entre el valor actual y el valor lag para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar deltas de lag. Si es None, no se generan deltas.
    cant_lag : int, default=1
        Cantidad de lags a considerar para calcular el delta

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de delta de lag agregadas
    """

    logger.info(f"Realizando feature engineering de delta lag con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    n_inicial = df.shape[1]
    logger.info(f"[DELTAS] Columnas iniciales: {n_inicial}")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar delta lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"
    
    # Agregar los delta lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", {attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) "
                    f"AS {attr}_delta_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    n_final = df.shape[1]
    logger.info(f"[DELTAS] Feature engineering de delta lag completado. Columnas finales: {n_final} (se agregaron {n_final - n_inicial})")

    return df




def crear_percentil(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de percentil aproximado para los atributos especificados,
    calculando previamente los límites de percentil por grupo (foto_mes)
    y luego asignándolos mediante un JOIN.
  
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar los percentiles.
  
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de percentil agregadas
    
    Autor: Guillermo Teran    
    """

    logger.info(f"Realizando feature engineering con percentiles aproximados para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar percentiles")
        return df

         # para cada columna, generamos un bloque SQL
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
            continue

        # número de percentiles a calcular (por ejemplo 100)
        n_percentiles = 100
        percentiles = [round(i / n_percentiles, 2) for i in range(1, n_percentiles)]

        # CTE que calcula los límites
        sql_limites = f"""
        WITH limites AS (
            SELECT 
                foto_mes,
                unnest(quantile_cont({attr}, {percentiles})) AS valor_limite,
                unnest(range(1, {n_percentiles})) AS percentil
            FROM df
            GROUP BY foto_mes
        )
        """

        # Join para asignar el percentil a cada registro
        sql_join = f"""
        SELECT 
            d.*, 
            MAX(l.percentil) AS {attr}_percentil
        FROM df d
        JOIN limites l
            ON d.foto_mes = l.foto_mes
           AND d.{attr} >= l.valor_limite
        GROUP BY ALL
        """

        # Ejecutar la consulta SQL
        con = duckdb.connect(database=":memory:")
        con.register("df", df)
        df = con.execute(sql_limites + sql_join).df()
        con.close()

        logger.debug(f"Consulta SQL: {sql_limites + sql_join}")

    # con.close()
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


#######################################################################################
def crear_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking normalizado (0 a 1) para los atributos especificados utilizando SQL.
    Parameters:
    -----------
    df : pd.DataFrame

        DataFrame con los datos
    columnas : list 
        Lista de atributos para los cuales generar los rankings. Si es None, no se generan.
    Returns:
    --------
    pd.DataFrame    
        DataFrame con las variables de ranking agregadas
        
    Autor: Guillermo Teran    
    """

    logger.info(f"Realizando feature engineering con ranking normalizado para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar rankings")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los rankings para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f"\n, (DENSE_RANK() OVER (PARTITION BY foto_mes ORDER BY {attr}) - 1) * 1.0 / (COUNT(*) OVER (PARTITION BY foto_mes) - 1) AS {attr}_rank"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def feature_engineering_drop(df: pd.DataFrame, columnas_a_eliminar: list[str]) -> pd.DataFrame:
    """
    Elimina las columnas especificadas del DataFrame.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas_a_eliminar : list
        Lista de nombres de columnas a eliminar
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las columnas eliminadas
        
    Autor: Guillermo Teran    
    """
  
    logger.info(f"Eliminando {len(columnas_a_eliminar) if columnas_a_eliminar else 0} columnas")
  
    if not columnas_a_eliminar:
        logger.warning("No se especificaron columnas para eliminar")
        return df
  
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    columnas_no_existentes = [col for col in columnas_a_eliminar if col not in df.columns]
  
    if columnas_no_existentes:
        logger.warning(f"Las siguientes columnas no existen en el DataFrame y no se pueden eliminar: {columnas_no_existentes}")
  
    df = df.drop(columns=columnas_existentes)
  
    logger.info(f"Columnas eliminadas. DataFrame resultante con {df.shape[1]} columnas")
  
    return df


def crear_max_ultimos_n_meses(df: pd.DataFrame, columnas: list[str], n_meses: int = 3) -> pd.DataFrame:
    """
    Genera variables con el máximo de los últimos n meses para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar el máximo de los últimos n meses. Si es None, no se generan.
    n_meses : int, default=3
        Cantidad de meses a considerar para calcular el máximo (incluye el mes actual).

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de máximo de los últimos n meses agregadas
        
    Autor: Guillermo Teran   
    """

    logger.info(f"Realizando feature engineering con máximo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar máximos")
        return df

    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar el máximo de los últimos n meses para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, max({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_max_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

def crear_min_ultimos_n_meses(df: pd.DataFrame, columnas: list[str], n_meses: int = 3) -> pd.DataFrame:
    """
    Genera variables con el mínimo de los últimos n meses para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar el mínimo de los últimos n meses. Si es None, no se generan.
    n_meses : int, default=3
        Cantidad de meses a considerar para calcular el mínimo (incluye el mes actual).

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de mínimo de los últimos n meses agregadas
        
    Autor: Guillermo Teran    
    """

    logger.info(f"Realizando feature engineering con mínimo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar mínimos")
        return df

    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar el mínimo de los últimos n meses para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, min({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_min_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df



def crear_ratios(df: pd.DataFrame, ratios: list[dict]) -> pd.DataFrame:
    """
    Genera variables de ratio entre columnas especificadas utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    ratios : list[dict]
        Lista de diccionarios con las claves 'numerador', 'denominador' y 'nombre' para cada ratio.
        Ejemplo: [{'numerador': 'col1', 'denominador': 'col2', 'nombre': 'col1_col2_ratio'}]

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ratio agregadas
        
    Autor: Guillermo Teran    
    """
    logger.info(f"Realizando feature engineering de ratios para {len(ratios) if ratios else 0} combinaciones")

    if not ratios:
        logger.warning("No se especificaron ratios para generar")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"
    for ratio in ratios:
        numerador = ratio.get("numerador")
        denominador = ratio.get("denominador")
        nombre = ratio.get("nombre", f"{numerador}_{denominador}_ratio")
        if numerador not in df.columns or denominador not in df.columns:
            logger.warning(f"Alguna columna no existe en el DataFrame: {numerador}, {denominador}")
            continue
        # Usar NULLIF para evitar división por cero
        sql += f", CASE WHEN {numerador} IS NULL OR {denominador} IS NULL THEN NULL ELSE {numerador} / NULLIF({denominador}, 0) END AS {nombre}"

    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering de ratios completado. DataFrame resultante con {df_result.shape[1]} columnas")
    return df_result


def cambio_estado(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables binarias que indican si hubo un cambio en variables categóricas respecto al lag especificado.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos categóricos para los cuales detectar cambios.
    cant_lag : int, default=1
        Cantidad de lags a comparar (por defecto 1)

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de cambio categórico agregadas
    
    Autor: Guillermo Teran
    """
    logger.info(f"Realizando feature engineering de cambio categórico con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar cambios categóricos")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", CASE WHEN {attr} IS NULL OR lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) IS NULL "
                    f"THEN NULL ELSE CAST({attr} != lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS INTEGER) END "
                    f"AS {attr}_cambio_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering de cambio categórico completado. DataFrame resultante con {df_result.shape[1]} columnas")

    return df_result



#######################################################################################
# ######################## MIAS  no usadas  ####################################################
#######################################################################################

def crear_lags_YBB(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
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


def crear_deltas_YBB(df: pd.DataFrame, columnas: list[str], cant_deltas: int = 1) -> pd.DataFrame:
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



