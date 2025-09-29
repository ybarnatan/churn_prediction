import pandas as pd
import os
import datetime
import logging


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

#Funcion apr cargar  datios
path = "C:/Users/ybbar/OneDrive/Desktop/DMEyF2025_Competencia01_Proyect_Wedesnday/data/competencia_01_crudo.csv"

def cargar_datos(path: str) -> pd.DataFrame | None:
    logger.info("Inicio de la funcion cargar_datos")
    try:
        df = pd.read_csv(path)
        logger.info("Datos cargados correctamente con filas: %d y columnas: %d", df.shape[0], df.shape[1])
        return df
    except Exception as e:
        logger.error("Error al cargar los datos: %s", e)
        raise # Crashear el programa si no cargo el df.
            


def main():
    logger.info("Inicio de ejecucion del programa")
    print("Hola mundo desde main!")
    df = cargar_datos(path)
    print(df.head(4))
    logger.info("Fin de ejecucion del programa")

if __name__ == "__main__": # Asegura que solo se ejecute main() si corremos "python main.py" en terminal, pero no ejecute main() si lo importamos en otro archivo.
    main()