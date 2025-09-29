import pandas as pd
import os
import datetime
import logging
from src.data_loader import cargar_datos
import src.feature_engeneering as FeatEng

#Log para el main
os.makedirs("logs", exist_ok=True) #Corroboro que exista la carpeta logs
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre = f"log_fecha_{fecha}.log"


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
        ])

logger = logging.getLogger(__name__)



def main():
    logger.info("Inicio de ejecucion del programa")
    # Carga de datos
    path = "C:/Users/ybbar/OneDrive/Desktop/DMEyF2025_Competencia01_Proyect_Wedesnday/data/competencia_01_crudo.csv"
    df = cargar_datos(path)
    
    # Feature engineering | Lags
    atributos = ["ctrx_quarter"] #Atributos para los cuales quiero crear lags
    cant_lags = 5 #Cantidad de lags a crear (1 mes atras, 2 meses atras, etc)
    df = FeatEng.crear_lags(df, columnas = atributos, cant_lag = cant_lags)
    
    # Guardar el df resultante
    logger.info("Guardando el DataFrame resultante en un archivo CSV")
    output_path = "data/competencia_01_conLags.csv"
    df.to_csv(output_path, index=False)    
    
    
    
    
    logger.info(f"Fin de ejecucion del programa. Revisar detalle en logs/{nombre}")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()