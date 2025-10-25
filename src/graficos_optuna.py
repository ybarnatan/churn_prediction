import pandas as pd
import os
import plotly.express as px
import logging
from .config import STUDY_NAME, RESULTS_DIR, GRAPHICS_DIR # Importamos las constantes


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def generar_grafico_html_optuna(study_name_yaml: str, file_json_name: str):
    """
    Carga el historial de trials desde el JSON, genera el gráfico de evolución 
    interactivo (Plotly) y lo guarda en la carpeta de gráficos.

    Args:
        study_name_yaml (str): El nombre del estudio, usualmente obtenido del archivo YAML/config.
        file_json_name (str): El nombre base del archivo JSON (ej: 'Comp01_Exp001_LGBM_iteraciones').
    """
    
    nombre_archivo_json = f"{file_json_name}.json"
    # Utiliza la variable global RESULTADOS_DIR
    ruta_json = os.path.join(RESULTS_DIR, nombre_archivo_json)
    logger.info(f"Cargando datos desde: {ruta_json}")

    # 1. Cargar datos
    try:
        df = pd.read_json(ruta_json)
        
    except FileNotFoundError:
        logger.error(f"❌ Error: No se encontró el archivo de log: {ruta_json}")
        return
    except Exception as e:
        logger.error(f"❌ Error al cargar o leer el JSON de Optuna ({ruta_json}): {e}")
        return

    # 2. Pre-procesamiento: Manejar múltiples valores por trial_number
    df_procesado = df.groupby('trial_number', as_index=False)['value'].max()
    logger.info(f"Datos procesados: Reducidos a {len(df_procesado)} puntos (máx. ganancia por trial).")

    # Usa el study_name_yaml pasado como parámetro para el título
    titulo = f"Evolucion de ganancia por trial de optuna.<br>{study_name_yaml}"
    
    fig = px.line(df_procesado, x='trial_number', y='value', title=titulo,markers=True,
        labels={
            "trial_number": "Número de Trial",
            "value": "Ganancia Total (Máxima por Trial)"
        }
    )
    
    # Personalización (igual que el original)
    fig.update_traces(line=dict(width=2), marker=dict(size=8, opacity=0.9))
    fig.update_layout(title_font_size=18,yaxis_tickformat=',.0f',hovermode="x unified")

    # 4. Guardar el gráfico como HTML
    nombre_archivo_html = f"{study_name_yaml}_Evolucion.html"
    # Utiliza la variable global GRAPHICS_DIR (importada de config)
    ruta_html = os.path.join(GRAPHICS_DIR, nombre_archivo_html)

    try:
        fig.write_html(ruta_html)
        logger.info(f"✅ Gráfico de evolución guardado en: {ruta_html}")
    except Exception as e:
        logger.error(f"❌ Error al guardar el archivo HTML en {ruta_html}: {e}")



def bayesiana_top5_ganancia(study_name_yaml: str, file_json_name: str):
    """
    Carga el historial de trials, identifica los 5 con mayor 'value' (ganancia), 
    calcula el promedio y genera un barplot interactivo con Plotly.

    Args:
        study_name_yaml (str): El nombre del estudio.
        file_json_name (str): El nombre base del archivo JSON.
    """
    
    nombre_archivo_json = f"{file_json_name}.json"
    ruta_json = os.path.join(RESULTS_DIR, nombre_archivo_json) # <-- Usamos RESULTS_DIR
    
    # 1. Cargar datos
    try:
        df = pd.read_json(ruta_json)
    except FileNotFoundError:
        logger.error(f"❌ Error: No se encontró el archivo de log: {ruta_json}")
        return
    except Exception as e:
        logger.error(f"❌ Error al cargar o leer el JSON de Optuna ({ruta_json}): {e}")
        return

    # 2. Pre-procesamiento: Obtener el valor máximo por trial_number y seleccionar el Top 5
    
    # Obtener la ganancia máxima por trial (si hubiese múltiples valores)
    df_trials = df.groupby('trial_number', as_index=False).agg(value=('value', 'max'), # Columna de ganancia
        # Podemos incluir otros datos para el hover si es necesario
    ).sort_values(by='value', ascending=False)
    
    # Seleccionar el top 5
    df_top5 = df_trials.head(5)
    
    if df_top5.empty:
        logger.warning("⚠️ No se encontraron trials 'COMPLETE' con valor para generar el Top 5.")
        return

    # 3. Calcular el promedio de ganancia del Top 5
    promedio_ganancia = df_trials['value'].mean() # Calcular el promedio de *todos* los trials
    logger.info(f"Promedio de ganancia de TODOS los trials: {promedio_ganancia:,.0f}")
    logger.info(f"Datos Top 5 procesados: {df_top5.shape[0]} trials.")

    # 4. Generar el gráfico interactivo (Barplot)
    titulo = f"5 mejores modelos y su promedio<br>{study_name_yaml}"
    
    # Convertir trial_number a string para que Plotly lo trate como categoría en el eje X (barplot)
    df_top5['trial_number_str'] = df_top5['trial_number'].astype(str)
    
    fig = px.bar(df_top5, x='trial_number_str', y='value', title=titulo,color='value', # Colorear por ganancia
        color_continuous_scale=px.colors.sequential.Viridis, # Escala de color
        labels={
            "trial_number_str": "Número de Trial",
            "value": "Ganancia Total (Máxima por Trial)",
            "color": "Ganancia"
        },template="plotly_white")
    
    # Añadir el promedio de TODOS los trials como una línea horizontal
    fig.add_hline(y=promedio_ganancia, line_dash="dot", line_color="red",annotation_text=f"Promedio ({promedio_ganancia:,.0f})", annotation_position="top right")

    # Personalización
    fig.update_layout(
        title_font_size=16,
        xaxis_title="Número de Trial",
        yaxis_title="Ganancia Total (Máxima por Trial)",
        yaxis_tickformat=',.0f',  # Formato de miles para el eje Y
        hovermode="x unified"     # Asegura que se muestre el hover para todo el eje X
    )
    
    # Mejorar la apariencia de las barras
    fig.update_traces(marker_line_width=1.5, marker_line_color='black', opacity=0.9)


    # 5. Guardar el gráfico como HTML
    nombre_archivo_html = f"{study_name_yaml}_Top5_Ganancia.html"
    ruta_html = os.path.join(GRAPHICS_DIR, nombre_archivo_html) # <-- Usamos GRAPHICS_DIR
    
    try:
        fig.write_html(ruta_html)
        logger.info(f"✅ Gráfico Top 5 guardado en: {ruta_html}")
    except Exception as e:
        logger.error(f"❌ Error al guardar el archivo HTML en {ruta_html}: {e}")